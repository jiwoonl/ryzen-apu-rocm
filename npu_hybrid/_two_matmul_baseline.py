#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Single-dispatch ReLU(A @ B) @ D on all 16 XDNA1 compute tiles.

Both phases reuse the same matmul dataflow. The first phase's output DMA
writes a shared-memory scratch tensor. finish_task_group emits a device-side
DMA completion barrier before phase two can read that tensor. The entire
runtime sequence is compiled into one instruction buffer and one XRT launch.
The intermediate is not kept wholly in tile SRAM.
"""

import numpy as np

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    InOut,
    Program,
    Runtime,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2
from aie.helpers.taplib import TensorTiler2D

def _build_design(
    dev,
    M,
    K,
    N,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    b_col_maj,
    c_col_maj,
    emulate_bf16_mmul_with_bfp16,
    use_chess,
    scalar,
):
    """Build both phases and their DMA barrier in one runtime sequence."""
    assert M == K == N, "Two-stage implementation requires square matrices"
    dev_str = "npu2" if isinstance(dev, NPU2) else "npu"

    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols

    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        use_chess=use_chess,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
        vectorized=not scalar,
    )
    zero_kernel = matmul_kernel.zero
    assert dtype_out_str == "bf16" and m * n == 1024, "Fused ReLU requires a 1024-element bf16 output tile"
    relu_kernel = kernels.relu(tile_size=1024)
    r, s, t = matmul_kernel.mac_dims

    if dev_str == "npu" and n_aie_cols > 4:
        raise AssertionError("Invalid configuration: NPU (Phoenix/Hawk) has 4 columns")
    if dev_str == "npu2" and n_aie_cols > 8:
        raise AssertionError(
            "Invalid configuration: NPU2 (Strix/Strix Halo/Krackan) has 8 columns"
        )

    assert (
        M % (m * n_aie_rows) == 0
    ), "A must be tileable into (m * n_aie_rows, k)-sized blocks"
    assert K % k == 0
    assert (
        N % (n * n_aie_cols) == 0
    ), "B must be tileable into (k, n * n_aie_cols)-sized blocks"
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    fifo_depth = 2
    n_tiles_per_core = (M // m) * (N // n) // n_aie_cores

    if n_aie_cols > n_aie_rows:
        n_shim_mem_A = n_aie_rows
    else:
        n_shim_mem_A = n_aie_cols

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    tiles = [[(col, row) for col in range(0, n_aie_cols)] for row in range(0, 6)]
    core_tiles = tiles[2:]

    A_l3l2_fifos = [None] * n_shim_mem_A
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    for i in range(n_shim_mem_A):
        A_l3l2_fifos[i] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        of_offsets = [m * k * j for j in range(stop_row - start_row)]
        dims_to_stream = [
            [
                (m // r, r * k),
                (k // s, s),
                (r, k),
                (s, 1),
            ]
        ] * (stop_row - start_row)
        a_tmp_fifos = (
            A_l3l2_fifos[i]
            .cons()
            .split(
                of_offsets,
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                dims_to_stream=dims_to_stream,
            )
        )
        for j in range(stop_row - start_row):
            A_l2l1_fifos[j + start_row] = a_tmp_fifos[j]

    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        if b_col_maj:
            dims_to_stream = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
        else:
            dims_to_stream = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=dims_to_stream,
            )
        )

        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=(
                [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
                if not c_col_maj
                else [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
            ),
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]

        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    def core_fn(in_a, in_b, out_c, zero, matmul, relu):
        for phase in range(2):
            loop = range(1)  # Workaround for issue #1547
            if n_tiles_per_core > 1:
                loop = range_(n_tiles_per_core)
            for _ in loop:
                elem_out = out_c.acquire(1)
                zero(elem_out)

                for _ in range_(K // k):
                    elem_in_a = in_a.acquire(1)
                    elem_in_b = in_b.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out)
                    in_a.release(1)
                    in_b.release(1)
                if phase == 0:
                    relu(elem_out, elem_out)
                out_c.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
                relu_kernel,
            ],
            stack_size=0xD00,
        ),
    )

    tb_max_n_rows = 4 if not c_col_maj else 2
    tb_n_rows = tb_max_n_rows // 2

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
        prune_step=False,
    )
    if b_col_maj:
        B_tiles = TensorTiler2D.step_tiler(
            (N, K),
            (n, k),
            tile_group_repeats=(N // n // n_aie_cols, K // k),
            tile_group_steps=(n_aie_cols, 1),
            prune_step=False,
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),
            (k, n),
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True,
            prune_step=False,
        )
    if c_col_maj:
        # Splitting n_aie_rows out of the tile dim is what lets TensorTiler emit
        # the (col-fast, row_block-slow) DMA pattern; iter_col_major matches it.
        C_tiles = TensorTiler2D.step_tiler(
            (N, M),
            (n, m),
            tile_group_repeats=(N // n // n_aie_cols, n_aie_rows),
            tile_group_steps=(n_aie_cols, 1),
            iter_col_major=True,
            prune_step=False,
        )
    else:
        C_tiles = TensorTiler2D.step_tiler(
            (M, N),
            (m * n_aie_rows, n),
            tile_group_repeats=(tb_n_rows, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            prune_step=False,
        )
    rt = Runtime()
    with rt.sequence(A_ty, B_ty, B_ty, C_ty, C_ty) as (input_a, weight_b, weight_d, hidden, output):
        rt.start(*[w for row in workers for w in row])
        # Each phase ends with completion of all output DMA before the next
        # phase can read hidden. Both phases remain in ONE runtime sequence.
        for A, B, C in [(input_a, weight_b, hidden), (hidden, weight_d, output)]:
            c_index = 0
            tg = rt.task_group()
            for tb in range(iron.ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
                for pingpong in [0, 1]:
                    if c_index >= len(C_tiles):
                        break

                    row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                    current_tb_n_rows = min(
                        [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                    )

                    for col in range(n_aie_cols):
                        rt.drain(
                            C_l2l3_fifos[col].cons(),
                            C,
                            tap=C_tiles[c_index],
                            wait=True,
                            task_group=tg,
                        )
                        c_index += 1

                        for tile_row in range(current_tb_n_rows):
                            tile_offset = (
                                (row_base + tile_row) * n_shim_mem_A + col
                            ) % len(A_tiles)
                            if col < n_aie_rows:
                                rt.fill(
                                    A_l3l2_fifos[col].prod(),
                                    A,
                                    tap=A_tiles[tile_offset],
                                    task_group=tg,
                                )
                            rt.fill(
                                B_l3l2_fifos[col].prod(),
                                B,
                                tap=B_tiles[col],
                                task_group=tg,
                            )

                    if tb > 0 or (tb == 0 and pingpong > 0):
                        rt.finish_task_group(tg)
                        tg = rt.task_group()
            rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def two_matmul(
    A: In,
    B: In,
    D: In,
    H: InOut,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 0,
    c_col_maj: CompileTime[int] = 0,
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    use_chess: CompileTime[bool] = False,
    scalar: CompileTime[bool] = False,
):
    return _build_design(
        iron.get_current_device(),
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        dtype_in_str,
        dtype_out_str,
        b_col_maj,
        c_col_maj,
        emulate_bf16_mmul_with_bfp16,
        use_chess,
        scalar,
    )
