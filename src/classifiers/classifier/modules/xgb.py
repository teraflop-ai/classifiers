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
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor
from quack.autotuner import AutotuneConfig, autotune

_SOURCE = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

from classifiers.classifier.kernels.xgb import configurations, launch_fused, pack_heap


class XGBoost:
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
