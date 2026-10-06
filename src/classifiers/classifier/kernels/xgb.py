import hashlib
import json
import math
import os
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cupy as cp
import cutlass
import cutlass.cute as cute
import numpy as np
import torch
import triton
from cuda.bindings import driver as cuda
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from cutlass.memory import SmemAllocator
from quack.autotuner import AutotuneConfig, autotune
from quack.reduce import row_reduce

_SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@cute.kernel
def grouped_kernel(
    x: cute.Tensor,
    codes: cute.Tensor,
    values: cute.Tensor,
    out: cute.Tensor,
    base: cutlass.Float32,
    missing: cutlass.Float32,
    depth: cutlass.Constexpr,
    trees: cutlass.Constexpr,
    sigmoid: cutlass.Constexpr,
    group: cutlass.Constexpr,
    block: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    rows = block // group
    local_row, lane = tid // group, tid % group
    row = cutlass.Int64(bid) * rows + local_row
    cache = SmemAllocator().allocate_tensor(
        cutlass.Float32,
        cute.make_layout((rows, x.shape[1]), stride=(x.shape[1] + 1, 1)),
        byte_alignment=16,
    )
    for i in range((rows * x.shape[1] + block - 1) // block):
        k = tid + i * block
        if k < rows * x.shape[1]:
            r, f = k // x.shape[1], k % x.shape[1]
            source = cutlass.Int64(bid) * rows + r
            value = cutlass.Float32(0)
            if source < x.shape[0]:
                value = x[source, f]
            cache[r, f] = value
    cute.arch.sync_threads()
    total = cutlass.Float32(0)
    for batch in range((trees + group - 1) // group):
        tree = batch * group + lane
        node = cutlass.Int32(0)
        for _ in cutlass.range_constexpr(depth):
            code = codes[tree, node]
            value = cache[local_row, code & 0x3FFFFFFF]
            go_left = value < values[tree, node]
            if (value != value) | (value == missing):
                go_left = (code & 0x40000000) != 0
            node = node * 2 + 2 - cutlass.Int32(go_left)
        total += values[tree, node]
    total = row_reduce(total, cute.ReductionOp.ADD, group)
    if (lane == 0) & (row < x.shape[0]):
        total += base
        if cutlass.const_expr(sigmoid):
            total = 1.0 / (1.0 + cute.exp(-total))
        out[row] = total


@cute.jit
def launch_fused(
    x: cute.Tensor,
    codes: cute.Tensor,
    values: cute.Tensor,
    out: cute.Tensor,
    base: cutlass.Float32,
    missing: cutlass.Float32,
    stream: cuda.CUstream,
    depth: cutlass.Constexpr,
    trees: cutlass.Constexpr,
    sigmoid: cutlass.Constexpr,
    group: cutlass.Constexpr,
    block: cutlass.Constexpr,
):
    rows = block // group
    grouped_kernel(
        x, codes, values, out, base, missing, depth, trees, sigmoid, group, block
    ).launch(
        grid=((x.shape[0] + rows - 1) // rows, 1, 1),
        block=(block, 1, 1),
        smem=rows * (x.shape[1] + 1) * 4,
        stream=stream,
    )


def pack_heap(trees, max_bytes=128 * 2**20):
    depth = 0
    for tree in trees:
        stack = [(0, 0)]
        while stack:
            node, level = stack.pop()
            depth = max(depth, level)
            left = tree["left_children"][node]
            if left >= 0:
                stack.extend(
                    ((left, level + 1), (tree["right_children"][node], level + 1))
                )
    padded = (len(trees) + 31) // 32 * 32
    if depth > 12 or padded * ((1 << (depth + 1)) - 1) * 8 > max_bytes:
        raise ValueError("Model exceeds depth 12 or the packed-tree memory limit")
    width = (1 << (depth + 1)) - 1
    codes = np.zeros((padded, width), dtype=np.int32)
    values = np.zeros((padded, width), dtype=np.float32)
    for t, tree in enumerate(trees):
        stack = [(0, 0)]
        while stack:
            source, dest = stack.pop()
            values[t, dest] = tree["split_conditions"][source]
            left = tree["left_children"][source]
            if left >= 0:
                codes[t, dest] = int(tree["split_indices"][source]) | (
                    int(tree["default_left"][source]) << 30
                )
                stack.extend(
                    (
                        (left, dest * 2 + 1),
                        (tree["right_children"][source], dest * 2 + 2),
                    )
                )
            elif dest * 2 + 2 < width:
                stack.extend(((source, dest * 2 + 1), (source, dest * 2 + 2)))
    return codes, values, depth


def configurations(features, shared_bytes, max_threads):
    return [
        (g, b)
        for g in (8, 16, 32)
        for b in (128, 256)
        if b <= max_threads and (b // g) * (features + 1) * 4 <= shared_bytes
    ]


class CuTeXGBoost:
    def __init__(self, booster, device="cuda:0"):
        raw = bytes(booster.save_raw(raw_format="json"))
        learner = json.loads(raw)["learner"]
        params, gb = learner["learner_model_param"], learner["gradient_booster"]
        self.objective = learner["objective"]["name"]
        if (
            gb["name"] != "gbtree"
            or int(params["num_class"]) != 0
            or int(params.get("num_target", "1")) != 1
        ):
            raise ValueError("Requires a single-output gbtree model")
        if self.objective not in ("reg:squarederror", "binary:logistic"):
            raise ValueError("Requires reg:squarederror or binary:logistic")
        trees = gb["model"]["trees"]
        if not trees or any(
            any(t.get("split_type", []))
            or int(t["tree_param"].get("size_leaf_vector", "0")) > 1
            for t in trees
        ):
            raise ValueError("Requires numeric trees with scalar leaves")
        self.num_features, self.num_trees = int(params["num_feature"]), len(trees)
        if not 0 < self.num_features < 2**30:
            raise ValueError("Invalid feature count")
        base = json.loads(params["base_score"])
        self.base = float(base[0] if isinstance(base, list) else base)
        if self.objective == "binary:logistic":
            self.base = math.log(self.base / (1.0 - self.base))
        with torch.cuda.device(device):
            self.device = torch.device("cuda", torch.cuda.current_device())
            props = cp.cuda.runtime.getDeviceProperties(self.device.index)
            if props["major"] < 8:
                raise ValueError(
                    "This Quack tuning path requires an Ampere or newer NVIDIA GPU"
                )
            self.configs = configurations(
                self.num_features,
                props["sharedMemPerBlock"],
                props["maxThreadsPerBlock"],
            )
            if not self.configs:
                raise ValueError(
                    "No configuration fits this GPU's shared-memory/thread limits"
                )
            codes, values, self.depth = pack_heap(trees)
            self.arrays = tuple(
                torch.as_tensor(a.T.copy(), device=self.device).t()
                for a in (codes, values)
            )
        self.default_config = (
            (16, 256) if (16, 256) in self.configs else self.configs[0]
        )
        hardware = tuple(
            str(props.get(k))
            for k in (
                "uuid",
                "name",
                "major",
                "minor",
                "multiProcessorCount",
                "pciDomainID",
                "pciBusID",
                "pciDeviceID",
            )
        )
        software = tuple(
            version(p)
            for p in ("quack-kernels", "nvidia-cutlass-dsl", "apache-tvm-ffi")
        )
        self._identity = (
            _SOURCE,
            hashlib.sha256(raw).hexdigest(),
            platform.node(),
            platform.processor(),
            hardware,
            software,
            torch.__version__,
            torch.version.cuda,
            triton.__version__,
            cp.__version__,
            np.__version__,
            cp.cuda.runtime.driverGetVersion(),
            cp.cuda.runtime.runtimeGetVersion(),
        )
        self._compiled, self._selected = {}, {}

    @torch.no_grad()
    def __call__(self, x, *, missing=float("nan"), output_margin=False, config=None):
        if (
            not x.is_cuda
            or x.dtype != torch.float32
            or x.ndim != 2
            or x.shape[1] != self.num_features
            or x.device != self.device
        ):
            raise ValueError(
                "Expected float32 CUDA [rows, features] on the model device"
            )
        sigmoid = self.objective == "binary:logistic" and not output_margin
        config = self.default_config if config is None else tuple(config)
        if config not in self.configs:
            raise ValueError(f"Unsupported configuration: {config}")
        key = (tuple(x.stride()), sigmoid, config)
        with torch.cuda.device(x.device):
            out = torch.empty((x.shape[0],), device=x.device, dtype=torch.float32)
            if not x.shape[0]:
                return out
            compiled = self._compiled.get(key)
            if compiled is None:
                rows = cute.sym_int()
                fx = make_fake_tensor(
                    cutlass.Float32, (rows, self.num_features), key[0], assumed_align=4
                )
                arrays = [
                    make_fake_tensor(
                        dtype, tuple(t.shape), tuple(t.stride()), assumed_align=4
                    )
                    for dtype, t in zip((cutlass.Int32, cutlass.Float32), self.arrays)
                ]
                fy = make_fake_tensor(cutlass.Float32, (rows,), (1,), assumed_align=4)
                compiled = cute.compile(
                    launch_fused,
                    fx,
                    *arrays,
                    fy,
                    cutlass.Float32(0),
                    cutlass.Float32(0),
                    make_fake_stream(use_tvm_ffi_env_stream=True),
                    self.depth,
                    self.num_trees,
                    sigmoid,
                    *config,
                    options="--enable-tvm-ffi",
                )
                self._compiled[key] = compiled
            compiled(
                x.detach() if x.requires_grad else x,
                *self.arrays,
                out,
                self.base,
                float(missing),
            )
            return out

    def _host_key(self, x, missing, output_margin):
        if (
            not isinstance(x, np.ndarray)
            or x.ndim != 2
            or x.shape[1] != self.num_features
            or x.dtype.kind not in "fiu"
        ):
            raise ValueError("Expected numeric NumPy [rows, features]")
        return (
            tuple(x.shape),
            x.dtype.str,
            tuple(x.strides),
            float(missing).hex(),
            bool(output_margin),
        )

    def predict_numpy(
        self, x, *, missing=float("nan"), output_margin=False, config=None
    ):
        key = self._host_key(x, missing, output_margin)
        if config is None:
            config = self._selected.get(key, self.default_config)
        if not len(x):
            return np.empty(0, dtype=np.float32)
        with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
            dx = cp.asarray(np.ascontiguousarray(x, dtype=np.float32), blocking=True)
            dy = self(
                torch.from_dlpack(dx),
                missing=missing,
                output_margin=output_margin,
                config=config,
            )
            return cp.asnumpy(cp.from_dlpack(dy), blocking=True)

    def tune(
        self,
        x,
        expected,
        *,
        repeats=15,
        warmup=3,
        retune=False,
        missing=float("nan"),
        output_margin=False,
    ):
        key = self._host_key(x, missing, output_margin)
        if not len(x) or min(repeats, warmup) < 1:
            raise ValueError(
                "Tuning requires nonempty input and positive repetition counts"
            )
        expected = np.asarray(expected)
        if expected.shape != (len(x),):
            raise ValueError("Expected one XGBoost reference prediction per row")
        for config in self.configs:
            actual = self.predict_numpy(
                x, missing=missing, output_margin=output_margin, config=config
            )
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-4,
                atol=1e-5,
                err_msg=f"Configuration {config}",
            )

        def run(*, signature, config):
            return self.predict_numpy(
                x, missing=missing, output_margin=output_margin, config=config
            )

        def bench(fn, quantiles):
            for _ in range(warmup):
                fn()
            samples = []
            for _ in range(repeats):
                torch.cuda.synchronize(self.device)
                start = perf_counter()
                out = fn()
                samples.append((perf_counter() - start) * 1000)
                del out
            return np.quantile(samples, quantiles).tolist()

        tuner = autotune(
            configs=[AutotuneConfig(config=c) for c in self.configs],
            key=["signature"],
            do_bench=bench,
            cache_results=True,
        )(run)
        previous = os.environ.get("QUACK_FORCE_CACHE_UPDATE")
        try:
            if retune:
                os.environ["QUACK_FORCE_CACHE_UPDATE"] = "1"
            with torch.cuda.device(self.device), cp.cuda.Device(self.device.index):
                tuner(signature=(self._identity, key, repeats, warmup))
        finally:
            if retune:
                if previous is None:
                    os.environ.pop("QUACK_FORCE_CACHE_UPDATE", None)
                else:
                    os.environ["QUACK_FORCE_CACHE_UPDATE"] = previous
        timings = getattr(tuner, "configs_timings", {})
        if any(not np.isfinite(t[0]) for t in timings.values()):
            raise RuntimeError(
                "A tuning measurement failed; enable QUACK_PRINT_AUTOTUNING=1"
            )
        config = tuple(tuner.best_config.all_kwargs()["config"])
        self._selected[key] = config
        return config
