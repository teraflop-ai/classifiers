import cutlass
import cutlass.cute as cute
import numpy as np
from cuda.bindings import driver as cuda
from cutlass.memory import SmemAllocator
from quack.reduce import row_reduce


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
