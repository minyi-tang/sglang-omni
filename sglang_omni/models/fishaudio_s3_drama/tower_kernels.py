# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses
from typing import Protocol

import torch
from torch import nn

try:
    import triton
    import triton.language as tl
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, MHATokenToKVPool
    from sglang.srt.runtime_context import get_exec
    from triton.language.extra import libdevice

    from sglang_omni.platforms import current_platform

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    HybridLinearKVPool = None
    MHATokenToKVPool = None
    get_exec = None
    current_platform = None
    HAS_TRITON = False

MAX_FUSED_ROWS = 16
FUSED_BLOCK_M = MAX_FUSED_ROWS
MIX_BLOCK_N = 64
O_PROJ_BLOCK_N = 64
GATE_UP_BLOCK_N = 32
GEMV_BLOCK_K = 128
QKV_BLOCK_K = 64
SPLIT_CANDIDATES = (1, 2, 4, 8, 16)
GEMV_WARPS = 4
QKV_WARPS = 8
PIPELINE_STAGES = 3
FUSED_KERNELS = (
    "mix_eh_proj",
    "norm_qkv_rope_store",
    "gate_o_proj_add",
    "norm_gate_up_silu",
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class NormWeights:
    weight: torch.Tensor
    eps: float


@dataclasses.dataclass(frozen=True, kw_only=True)
class TowerLayerWeights:
    token_embedding: torch.Tensor
    enorm: NormWeights
    hnorm: NormWeights
    eh_proj: nn.Linear
    input_norm: NormWeights
    qkv_proj: nn.Linear
    q_norm: NormWeights
    k_norm: NormWeights
    o_proj: nn.Linear
    post_norm: NormWeights
    gate_up_proj: nn.Linear


@dataclasses.dataclass(frozen=True, kw_only=True)
class TowerLayerShape:
    dim: int
    head_count: int
    kv_head_count: int
    head_dim: int
    rotary_dim: int
    intermediate_size: int
    codebook_size: int
    mix_half_split: int
    qkv_split: int
    o_proj_split: int
    gate_up_split: int

    @property
    def query_width(self) -> int:
        return self.head_count * self.head_dim

    @property
    def qkv_part_count(self) -> int:
        return 2 * (self.head_count + self.kv_head_count)


@dataclasses.dataclass(frozen=True, kw_only=True)
class SplitScratch:
    partials: torch.Tensor
    sum_of_squares: torch.Tensor
    norm_sums: torch.Tensor
    counters: torch.Tensor


class QueryAttention(Protocol):
    def __call__(self, layer_index: int, query: torch.Tensor) -> torch.Tensor: ...


if HAS_TRITON:

    @triton.jit
    def gemma_scaled(inputs, norm_weight_pointer, first_column, BLOCK_K: tl.constexpr):
        columns = first_column + tl.arange(0, BLOCK_K)
        gains = 1.0 + tl.load(norm_weight_pointer + columns).to(tl.float32)
        return (inputs.to(tl.float32) * gains[None, :]).to(tl.bfloat16)

    @triton.jit
    def weight_tile(
        weight_pointer,
        weight_rows,
        first_column,
        ROW_WIDTH: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        columns = first_column + tl.arange(0, BLOCK_K)
        return tl.load(
            weight_pointer + weight_rows[:, None] * ROW_WIDTH + columns[None, :]
        )

    @triton.jit
    def row_tile(
        values_pointer,
        row_stride,
        is_row,
        first_column,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        rows = tl.arange(0, BLOCK_M)
        columns = first_column + tl.arange(0, BLOCK_K)
        return tl.load(
            values_pointer + rows[:, None] * row_stride + columns[None, :],
            mask=is_row[:, None],
            other=0.0,
        )

    @triton.jit
    def tile_sums(
        norm_sums_pointer,
        TILES: tl.constexpr,
        TILE_BLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        tiles = tl.arange(0, TILE_BLOCK)
        rows = tl.arange(0, BLOCK_M)
        return tl.sum(
            tl.load(
                norm_sums_pointer + tiles[:, None] * BLOCK_M + rows[None, :],
                mask=(tiles < TILES)[:, None],
                other=0.0,
                cache_modifier=".cg",
            ),
            0,
        )

    @triton.jit
    def sigmoid(values):
        return tl.math.div_rn(
            tl.full(values.shape, 1.0, tl.float32), 1.0 + libdevice.exp(-values)
        )

    @triton.jit
    def normed_rotated(
        even,
        odd,
        norm_weight_pointer,
        norm_eps,
        cos_sin_pointer,
        positions,
        is_row,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
    ):
        pairs = tl.arange(0, HEAD_DIM // 2)
        head_rstd = libdevice.rsqrt(
            (tl.sum(even * even, 1) + tl.sum(odd * odd, 1)) / HEAD_DIM + norm_eps
        )
        even_gains = 1.0 + tl.load(norm_weight_pointer + 2 * pairs).to(tl.float32)
        odd_gains = 1.0 + tl.load(norm_weight_pointer + 2 * pairs + 1).to(tl.float32)
        even = (even * head_rstd[:, None] * even_gains[None, :]).to(tl.bfloat16)
        odd = (odd * head_rstd[:, None] * odd_gains[None, :]).to(tl.bfloat16)
        even = even.to(tl.float32)
        odd = odd.to(tl.float32)
        is_rotary = pairs < ROTARY_DIM // 2
        table_mask = is_row[:, None] & is_rotary[None, :]
        table_rows = positions[:, None] * ROTARY_DIM
        cosine = tl.load(
            cos_sin_pointer + table_rows + pairs[None, :], mask=table_mask, other=1.0
        )
        sine = tl.load(
            cos_sin_pointer + table_rows + ROTARY_DIM // 2 + pairs[None, :],
            mask=table_mask,
            other=0.0,
        )
        rotated_even = tl.where(is_rotary[None, :], even * cosine - odd * sine, even)
        rotated_odd = tl.where(is_rotary[None, :], odd * cosine + even * sine, odd)
        return rotated_even.to(tl.bfloat16), rotated_odd.to(tl.bfloat16)

    @triton.jit(do_not_specialize=["row_count"])
    def mix_eh_proj_kernel(
        code_ids_pointer,
        label_is_semantic_pointer,
        token_embedding_pointer,
        shared_input_pointer,
        hidden_states_pointer,
        enorm_weight_pointer,
        hnorm_weight_pointer,
        weight_pointer,
        input_norm_weight_pointer,
        mixed_pointer,
        scaled_pointer,
        partials_pointer,
        sum_of_squares_pointer,
        norm_sums_pointer,
        counters_pointer,
        row_count,
        enorm_eps,
        hnorm_eps,
        CODEBOOK_SIZE: tl.constexpr,
        DIM: tl.constexpr,
        HAS_SHARED_INPUT: tl.constexpr,
        HALF_SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        tile = tl.program_id(0)
        split = tl.program_id(1)
        SPLIT: tl.constexpr = 2 * HALF_SPLIT
        SPLIT_WIDTH: tl.constexpr = DIM // HALF_SPLIT
        rows = tl.arange(0, BLOCK_M)
        block_columns = tl.arange(0, BLOCK_K)
        output_columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        is_row = rows < row_count
        first_column = split % HALF_SPLIT * SPLIT_WIDTH
        square = tl.zeros([BLOCK_M, BLOCK_M], dtype=tl.float32)
        product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if split < HALF_SPLIT:
            code_ids = tl.load(code_ids_pointer + rows, mask=is_row, other=0)
            table_rows = tl.minimum(tl.maximum(code_ids, 0), CODEBOOK_SIZE - 1)
            is_semantic = (
                tl.load(label_is_semantic_pointer + rows, mask=is_row, other=0) != 0
            )
            for column in range(first_column, first_column + SPLIT_WIDTH, BLOCK_K):
                embeddings = tl.load(
                    token_embedding_pointer
                    + table_rows[:, None] * DIM
                    + (column + block_columns)[None, :],
                    mask=(is_row & is_semantic)[:, None],
                    other=0.0,
                ).to(tl.float32)
                if HAS_SHARED_INPUT:
                    embeddings += row_tile(
                        shared_input_pointer, DIM, is_row, column, BLOCK_M, BLOCK_K
                    ).to(tl.float32)
                else:
                    pass
                inputs = embeddings.to(tl.bfloat16)
                square = tl.dot(inputs, tl.trans(inputs), square)
                product = tl.dot(
                    gemma_scaled(inputs, enorm_weight_pointer, column, BLOCK_K),
                    tl.trans(
                        weight_tile(
                            weight_pointer, output_columns, column, 2 * DIM, BLOCK_K
                        )
                    ),
                    product,
                )
        else:
            for column in range(first_column, first_column + SPLIT_WIDTH, BLOCK_K):
                inputs = row_tile(
                    hidden_states_pointer, DIM, is_row, column, BLOCK_M, BLOCK_K
                )
                square = tl.dot(inputs, tl.trans(inputs), square)
                product = tl.dot(
                    gemma_scaled(inputs, hnorm_weight_pointer, column, BLOCK_K),
                    tl.trans(
                        weight_tile(
                            weight_pointer,
                            output_columns,
                            DIM + column,
                            2 * DIM,
                            BLOCK_K,
                        )
                    ),
                    product,
                )
        partial_rows = (split * BLOCK_M + rows)[:, None] * DIM
        tl.store(
            partials_pointer + partial_rows + output_columns[None, :],
            product,
            mask=is_row[:, None],
        )
        tl.store(
            sum_of_squares_pointer + (tile * SPLIT + split) * BLOCK_M + rows,
            tl.sum(tl.where(rows[:, None] == rows[None, :], square, 0.0), 1),
        )
        tl.debug_barrier()
        arrived = tl.atomic_add(counters_pointer + tile, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT - 1:
            embedding_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            hidden_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            embedding_sum_of_squares = tl.zeros([BLOCK_M], dtype=tl.float32)
            hidden_sum_of_squares = tl.zeros([BLOCK_M], dtype=tl.float32)
            for part in tl.static_range(SPLIT):
                partial_rows = (part * BLOCK_M + rows)[:, None] * DIM
                partial = tl.load(
                    partials_pointer + partial_rows + output_columns[None, :],
                    mask=is_row[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                partial_sum_of_squares = tl.load(
                    sum_of_squares_pointer + (tile * SPLIT + part) * BLOCK_M + rows,
                    cache_modifier=".cg",
                )
                if part < HALF_SPLIT:
                    embedding_product += partial
                    embedding_sum_of_squares += partial_sum_of_squares
                else:
                    hidden_product += partial
                    hidden_sum_of_squares += partial_sum_of_squares
            tl.atomic_xchg(counters_pointer + tile, 0)
            embedding_rstd = libdevice.rsqrt(embedding_sum_of_squares / DIM + enorm_eps)
            hidden_rstd = libdevice.rsqrt(hidden_sum_of_squares / DIM + hnorm_eps)
            mixed = (
                embedding_product * embedding_rstd[:, None]
                + hidden_product * hidden_rstd[:, None]
            ).to(tl.bfloat16)
            row_columns = rows[:, None] * DIM + output_columns[None, :]
            tl.store(mixed_pointer + row_columns, mixed, mask=is_row[:, None])
            mixed = mixed.to(tl.float32)
            gains = 1.0 + tl.load(input_norm_weight_pointer + output_columns).to(
                tl.float32
            )
            tl.store(
                scaled_pointer + row_columns,
                (mixed * gains[None, :]).to(tl.bfloat16),
                mask=is_row[:, None],
            )
            tl.store(
                norm_sums_pointer + tile * BLOCK_M + rows, tl.sum(mixed * mixed, 1)
            )
        else:
            pass

    @triton.jit(do_not_specialize=["row_count"])
    def norm_qkv_rope_store_kernel(
        scaled_pointer,
        norm_sums_pointer,
        weight_pointer,
        q_norm_weight_pointer,
        k_norm_weight_pointer,
        cos_sin_pointer,
        positions_pointer,
        cache_slots_pointer,
        query_pointer,
        gate_scale_pointer,
        key_cache_pointer,
        value_cache_pointer,
        cache_slot_stride,
        cache_head_stride,
        partials_pointer,
        counters_pointer,
        row_count,
        input_norm_eps,
        q_norm_eps,
        k_norm_eps,
        DIM: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        ROTARY_DIM: tl.constexpr,
        HEAD_COUNT: tl.constexpr,
        KV_HEAD_COUNT: tl.constexpr,
        NORM_TILES: tl.constexpr,
        NORM_TILE_BLOCK: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        part = tl.program_id(0)
        split = tl.program_id(1)
        PAIRS: tl.constexpr = HEAD_DIM // 2
        PART_COUNT: tl.constexpr = 2 * (HEAD_COUNT + KV_HEAD_COUNT)
        QUERY_WIDTH: tl.constexpr = HEAD_COUNT * HEAD_DIM
        SPLIT_WIDTH: tl.constexpr = DIM // SPLIT
        rows = tl.arange(0, BLOCK_M)
        pairs = tl.arange(0, PAIRS)
        is_row = rows < row_count
        if part < HEAD_COUNT:
            first_weight_row = 2 * part * HEAD_DIM
        elif part < 2 * HEAD_COUNT:
            first_weight_row = (2 * (part - HEAD_COUNT) + 1) * HEAD_DIM
        else:
            first_weight_row = part * HEAD_DIM
        even_weight_rows = first_weight_row + 2 * pairs
        odd_weight_rows = even_weight_rows + 1
        even_product = tl.zeros([BLOCK_M, PAIRS], dtype=tl.float32)
        odd_product = tl.zeros([BLOCK_M, PAIRS], dtype=tl.float32)
        for column in range(split * SPLIT_WIDTH, (split + 1) * SPLIT_WIDTH, BLOCK_K):
            scaled = row_tile(scaled_pointer, DIM, is_row, column, BLOCK_M, BLOCK_K)
            even_product = tl.dot(
                scaled,
                tl.trans(
                    weight_tile(weight_pointer, even_weight_rows, column, DIM, BLOCK_K)
                ),
                even_product,
            )
            odd_product = tl.dot(
                scaled,
                tl.trans(
                    weight_tile(weight_pointer, odd_weight_rows, column, DIM, BLOCK_K)
                ),
                odd_product,
            )
        partial_rows = ((split * BLOCK_M + rows) * PART_COUNT + part) * HEAD_DIM
        tl.store(
            partials_pointer + partial_rows[:, None] + pairs[None, :],
            even_product,
            mask=is_row[:, None],
        )
        tl.store(
            partials_pointer + partial_rows[:, None] + PAIRS + pairs[None, :],
            odd_product,
            mask=is_row[:, None],
        )
        tl.debug_barrier()
        arrived = tl.atomic_add(counters_pointer + part, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLIT - 1:
            even = tl.zeros([BLOCK_M, PAIRS], dtype=tl.float32)
            odd = tl.zeros([BLOCK_M, PAIRS], dtype=tl.float32)
            for other_split in tl.static_range(SPLIT):
                partial_rows = (
                    (other_split * BLOCK_M + rows) * PART_COUNT + part
                ) * HEAD_DIM
                even += tl.load(
                    partials_pointer + partial_rows[:, None] + pairs[None, :],
                    mask=is_row[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
                odd += tl.load(
                    partials_pointer + partial_rows[:, None] + PAIRS + pairs[None, :],
                    mask=is_row[:, None],
                    other=0.0,
                    cache_modifier=".cg",
                )
            tl.atomic_xchg(counters_pointer + part, 0)
            input_rstd = libdevice.rsqrt(
                tile_sums(norm_sums_pointer, NORM_TILES, NORM_TILE_BLOCK, BLOCK_M) / DIM
                + input_norm_eps
            )
            even = (even * input_rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            odd = (odd * input_rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            positions = tl.load(positions_pointer + rows, mask=is_row, other=0)
            cache_slots = tl.load(cache_slots_pointer + rows, mask=is_row, other=0)
            is_cached_row = is_row & (cache_slots != 0)
            cache_rows = cache_slots[:, None] * cache_slot_stride + 2 * pairs[None, :]
            if part < HEAD_COUNT:
                query_even, query_odd = normed_rotated(
                    even,
                    odd,
                    q_norm_weight_pointer,
                    q_norm_eps,
                    cos_sin_pointer,
                    positions,
                    is_row,
                    HEAD_DIM,
                    ROTARY_DIM,
                )
                query_columns = (
                    rows[:, None] * QUERY_WIDTH + part * HEAD_DIM + 2 * pairs[None, :]
                )
                tl.store(
                    query_pointer + query_columns, query_even, mask=is_row[:, None]
                )
                tl.store(
                    query_pointer + query_columns + 1, query_odd, mask=is_row[:, None]
                )
            elif part < 2 * HEAD_COUNT:
                gate_columns = (
                    rows[:, None] * QUERY_WIDTH
                    + (part - HEAD_COUNT) * HEAD_DIM
                    + 2 * pairs[None, :]
                )
                tl.store(
                    gate_scale_pointer + gate_columns,
                    sigmoid(even).to(tl.bfloat16),
                    mask=is_row[:, None],
                )
                tl.store(
                    gate_scale_pointer + gate_columns + 1,
                    sigmoid(odd).to(tl.bfloat16),
                    mask=is_row[:, None],
                )
            elif part < 2 * HEAD_COUNT + KV_HEAD_COUNT:
                key_even, key_odd = normed_rotated(
                    even,
                    odd,
                    k_norm_weight_pointer,
                    k_norm_eps,
                    cos_sin_pointer,
                    positions,
                    is_row,
                    HEAD_DIM,
                    ROTARY_DIM,
                )
                key_columns = cache_rows + (part - 2 * HEAD_COUNT) * cache_head_stride
                tl.store(
                    key_cache_pointer + key_columns,
                    key_even,
                    mask=is_cached_row[:, None],
                )
                tl.store(
                    key_cache_pointer + key_columns + 1,
                    key_odd,
                    mask=is_cached_row[:, None],
                )
            else:
                value_columns = (
                    cache_rows
                    + (part - 2 * HEAD_COUNT - KV_HEAD_COUNT) * cache_head_stride
                )
                tl.store(
                    value_cache_pointer + value_columns,
                    even.to(tl.bfloat16),
                    mask=is_cached_row[:, None],
                )
                tl.store(
                    value_cache_pointer + value_columns + 1,
                    odd.to(tl.bfloat16),
                    mask=is_cached_row[:, None],
                )
        else:
            pass

    @triton.jit(do_not_specialize=["row_count"])
    def gate_o_proj_add_kernel(
        attended_pointer,
        attended_row_stride,
        gate_scale_pointer,
        weight_pointer,
        residual_in_pointer,
        norm_weight_pointer,
        residual_out_pointer,
        scaled_out_pointer,
        partials_pointer,
        norm_sums_pointer,
        counters_pointer,
        row_count,
        DIM: tl.constexpr,
        QUERY_WIDTH: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        tile = tl.program_id(0)
        split = tl.program_id(1)
        SPLIT_WIDTH: tl.constexpr = QUERY_WIDTH // SPLIT
        rows = tl.arange(0, BLOCK_M)
        output_columns = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        is_row = rows < row_count
        product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for column in range(split * SPLIT_WIDTH, (split + 1) * SPLIT_WIDTH, BLOCK_K):
            attended = row_tile(
                attended_pointer, attended_row_stride, is_row, column, BLOCK_M, BLOCK_K
            ).to(tl.float32)
            gate_scale = row_tile(
                gate_scale_pointer, QUERY_WIDTH, is_row, column, BLOCK_M, BLOCK_K
            ).to(tl.float32)
            inputs = (attended * gate_scale).to(tl.bfloat16)
            product = tl.dot(
                inputs,
                tl.trans(
                    weight_tile(
                        weight_pointer, output_columns, column, QUERY_WIDTH, BLOCK_K
                    )
                ),
                product,
            )
        partial_rows = (split * BLOCK_M + rows)[:, None] * DIM
        if SPLIT > 1:
            tl.store(
                partials_pointer + partial_rows + output_columns[None, :],
                product,
                mask=is_row[:, None],
            )
            tl.debug_barrier()
            arrived = tl.atomic_add(
                counters_pointer + tile, 1, sem="acq_rel", scope="gpu"
            )
            is_last = arrived == SPLIT - 1
        else:
            is_last = split == 0
        if is_last:
            if SPLIT > 1:
                product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for other_split in tl.static_range(SPLIT):
                    partial_rows = (other_split * BLOCK_M + rows)[:, None] * DIM
                    product += tl.load(
                        partials_pointer + partial_rows + output_columns[None, :],
                        mask=is_row[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                tl.atomic_xchg(counters_pointer + tile, 0)
            else:
                pass
            residual_columns = rows[:, None] * DIM + output_columns[None, :]
            residual = tl.load(
                residual_in_pointer + residual_columns,
                mask=is_row[:, None],
                other=0.0,
            ).to(tl.float32)
            summed = product.to(tl.bfloat16).to(tl.float32) + residual
            gains = 1.0 + tl.load(norm_weight_pointer + output_columns).to(tl.float32)
            tl.store(
                residual_out_pointer + residual_columns,
                summed.to(tl.bfloat16),
                mask=is_row[:, None],
            )
            tl.store(
                scaled_out_pointer + residual_columns,
                (summed * gains[None, :]).to(tl.bfloat16),
                mask=is_row[:, None],
            )
            tl.store(
                norm_sums_pointer + tile * BLOCK_M + rows,
                tl.sum(summed * summed, 1),
            )
        else:
            pass

    @triton.jit(do_not_specialize=["row_count"])
    def norm_gate_up_silu_kernel(
        scaled_pointer,
        norm_sums_pointer,
        weight_pointer,
        activated_pointer,
        partials_pointer,
        counters_pointer,
        row_count,
        norm_eps,
        DIM: tl.constexpr,
        INTERMEDIATE: tl.constexpr,
        NORM_TILES: tl.constexpr,
        NORM_TILE_BLOCK: tl.constexpr,
        SPLIT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        tile = tl.program_id(0)
        split = tl.program_id(1)
        SPLIT_WIDTH: tl.constexpr = DIM // SPLIT
        rows = tl.arange(0, BLOCK_M)
        gate_rows = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        up_rows = INTERMEDIATE + gate_rows
        is_row = rows < row_count
        gate_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        up_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for column in range(split * SPLIT_WIDTH, (split + 1) * SPLIT_WIDTH, BLOCK_K):
            inputs = row_tile(scaled_pointer, DIM, is_row, column, BLOCK_M, BLOCK_K)
            gate_product = tl.dot(
                inputs,
                tl.trans(weight_tile(weight_pointer, gate_rows, column, DIM, BLOCK_K)),
                gate_product,
            )
            up_product = tl.dot(
                inputs,
                tl.trans(weight_tile(weight_pointer, up_rows, column, DIM, BLOCK_K)),
                up_product,
            )
        partial_rows = (split * BLOCK_M + rows)[:, None] * (2 * INTERMEDIATE)
        if SPLIT > 1:
            tl.store(
                partials_pointer + partial_rows + gate_rows[None, :],
                gate_product,
                mask=is_row[:, None],
            )
            tl.store(
                partials_pointer + partial_rows + up_rows[None, :],
                up_product,
                mask=is_row[:, None],
            )
            tl.debug_barrier()
            arrived = tl.atomic_add(
                counters_pointer + tile, 1, sem="acq_rel", scope="gpu"
            )
            is_last = arrived == SPLIT - 1
        else:
            is_last = split == 0
        if is_last:
            if SPLIT > 1:
                gate_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                up_product = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
                for other_split in tl.static_range(SPLIT):
                    partial_rows = (other_split * BLOCK_M + rows)[:, None] * (
                        2 * INTERMEDIATE
                    )
                    gate_product += tl.load(
                        partials_pointer + partial_rows + gate_rows[None, :],
                        mask=is_row[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                    up_product += tl.load(
                        partials_pointer + partial_rows + up_rows[None, :],
                        mask=is_row[:, None],
                        other=0.0,
                        cache_modifier=".cg",
                    )
                tl.atomic_xchg(counters_pointer + tile, 0)
            else:
                pass
            rstd = libdevice.rsqrt(
                tile_sums(norm_sums_pointer, NORM_TILES, NORM_TILE_BLOCK, BLOCK_M) / DIM
                + norm_eps
            )
            gate = (gate_product * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            up = (up_product * rstd[:, None]).to(tl.bfloat16).to(tl.float32)
            activated = tl.math.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up
            tl.store(
                activated_pointer + rows[:, None] * INTERMEDIATE + gate_rows[None, :],
                activated.to(tl.bfloat16),
                mask=is_row[:, None],
            )
        else:
            pass

else:
    pass


def mix_eh_proj(
    *,
    weights: TowerLayerWeights,
    hidden_states: torch.Tensor,
    code_ids: torch.Tensor,
    label_is_semantic: torch.Tensor,
    shared_input: torch.Tensor | None,
    mixed: torch.Tensor,
    scaled: torch.Tensor,
    scratch: SplitScratch,
    shape: TowerLayerShape,
) -> None:
    row_count = hidden_states.shape[0]
    mix_eh_proj_kernel[(shape.dim // MIX_BLOCK_N, 2 * shape.mix_half_split)](
        code_ids,
        label_is_semantic,
        weights.token_embedding,
        hidden_states if shared_input is None else shared_input,
        hidden_states,
        weights.enorm.weight,
        weights.hnorm.weight,
        weights.eh_proj.weight,
        weights.input_norm.weight,
        mixed,
        scaled,
        scratch.partials,
        scratch.sum_of_squares,
        scratch.norm_sums,
        scratch.counters,
        row_count,
        weights.enorm.eps,
        weights.hnorm.eps,
        CODEBOOK_SIZE=shape.codebook_size,
        DIM=shape.dim,
        HAS_SHARED_INPUT=shared_input is not None,
        HALF_SPLIT=shape.mix_half_split,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_N=MIX_BLOCK_N,
        BLOCK_K=GEMV_BLOCK_K,
        num_warps=GEMV_WARPS,
        num_stages=PIPELINE_STAGES,
    )


def norm_qkv_rope_store(
    *,
    weights: TowerLayerWeights,
    scaled: torch.Tensor,
    rotary_cos_sin: torch.Tensor,
    positions: torch.Tensor,
    cache_slots: torch.Tensor,
    query: torch.Tensor,
    gate_scale: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    scratch: SplitScratch,
    shape: TowerLayerShape,
) -> None:
    norm_tiles = shape.dim // MIX_BLOCK_N
    norm_qkv_rope_store_kernel[(shape.qkv_part_count, shape.qkv_split)](
        scaled,
        scratch.norm_sums,
        weights.qkv_proj.weight,
        weights.q_norm.weight,
        weights.k_norm.weight,
        rotary_cos_sin,
        positions,
        cache_slots,
        query,
        gate_scale,
        key_cache,
        value_cache,
        key_cache.stride(0),
        key_cache.stride(1),
        scratch.partials,
        scratch.counters,
        scaled.shape[0],
        weights.input_norm.eps,
        weights.q_norm.eps,
        weights.k_norm.eps,
        DIM=shape.dim,
        HEAD_DIM=shape.head_dim,
        ROTARY_DIM=shape.rotary_dim,
        HEAD_COUNT=shape.head_count,
        KV_HEAD_COUNT=shape.kv_head_count,
        NORM_TILES=norm_tiles,
        NORM_TILE_BLOCK=triton.next_power_of_2(norm_tiles),
        SPLIT=shape.qkv_split,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_K=QKV_BLOCK_K,
        num_warps=QKV_WARPS,
        num_stages=PIPELINE_STAGES,
    )


def gate_o_proj_add(
    *,
    weights: TowerLayerWeights,
    attended: torch.Tensor,
    gate_scale: torch.Tensor,
    residual_in: torch.Tensor,
    residual_out: torch.Tensor,
    scaled_out: torch.Tensor,
    scratch: SplitScratch,
    shape: TowerLayerShape,
) -> None:
    gate_o_proj_add_kernel[(shape.dim // O_PROJ_BLOCK_N, shape.o_proj_split)](
        attended,
        attended.stride(0),
        gate_scale,
        weights.o_proj.weight,
        residual_in,
        weights.post_norm.weight,
        residual_out,
        scaled_out,
        scratch.partials,
        scratch.norm_sums,
        scratch.counters,
        residual_out.shape[0],
        DIM=shape.dim,
        QUERY_WIDTH=shape.query_width,
        SPLIT=shape.o_proj_split,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_N=O_PROJ_BLOCK_N,
        BLOCK_K=GEMV_BLOCK_K,
        num_warps=GEMV_WARPS,
        num_stages=PIPELINE_STAGES,
    )


def norm_gate_up_silu(
    *,
    weights: TowerLayerWeights,
    scaled: torch.Tensor,
    activated: torch.Tensor,
    scratch: SplitScratch,
    shape: TowerLayerShape,
) -> None:
    norm_tiles = shape.dim // O_PROJ_BLOCK_N
    norm_gate_up_silu_kernel[
        (shape.intermediate_size // GATE_UP_BLOCK_N, shape.gate_up_split)
    ](
        scaled,
        scratch.norm_sums,
        weights.gate_up_proj.weight,
        activated,
        scratch.partials,
        scratch.counters,
        scaled.shape[0],
        weights.post_norm.eps,
        DIM=shape.dim,
        INTERMEDIATE=shape.intermediate_size,
        NORM_TILES=norm_tiles,
        NORM_TILE_BLOCK=triton.next_power_of_2(norm_tiles),
        SPLIT=shape.gate_up_split,
        BLOCK_M=FUSED_BLOCK_M,
        BLOCK_N=GATE_UP_BLOCK_N,
        BLOCK_K=GEMV_BLOCK_K,
        num_warps=GEMV_WARPS,
        num_stages=PIPELINE_STAGES,
    )


class FusedTowerLayers:
    def __init__(
        self,
        *,
        shape: TowerLayerShape,
        rotary_cos_sin: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.shape = shape
        self.max_rows = MAX_FUSED_ROWS
        self.rotary_cos_sin = rotary_cos_sin

        def activations(width: int) -> torch.Tensor:
            return torch.zeros(
                (MAX_FUSED_ROWS, width), dtype=torch.bfloat16, device=device
            )

        self.mixed = activations(shape.dim)
        self.query = activations(shape.query_width)
        self.gate_scale = activations(shape.query_width)
        self.residual = activations(shape.dim)
        self.scaled = activations(shape.dim)
        self.activated = activations(shape.intermediate_size)
        mix_tiles = shape.dim // MIX_BLOCK_N
        o_proj_tiles = shape.dim // O_PROJ_BLOCK_N
        partial_count = FUSED_BLOCK_M * max(
            2 * shape.mix_half_split * shape.dim,
            shape.qkv_split * shape.qkv_part_count * shape.head_dim,
            shape.o_proj_split * shape.dim,
            shape.gate_up_split * 2 * shape.intermediate_size,
        )
        self.scratch = SplitScratch(
            partials=torch.zeros(partial_count, dtype=torch.float32, device=device),
            sum_of_squares=torch.zeros(
                FUSED_BLOCK_M * mix_tiles * 2 * shape.mix_half_split,
                dtype=torch.float32,
                device=device,
            ),
            norm_sums=torch.zeros(
                FUSED_BLOCK_M * max(mix_tiles, o_proj_tiles),
                dtype=torch.float32,
                device=device,
            ),
            counters=torch.zeros(
                max(
                    mix_tiles,
                    o_proj_tiles,
                    shape.intermediate_size // GATE_UP_BLOCK_N,
                    shape.qkv_part_count,
                ),
                dtype=torch.int32,
                device=device,
            ),
        )

    def covers(self, row_count: int) -> bool:
        return row_count <= self.max_rows

    def forward_layer(
        self,
        *,
        weights: TowerLayerWeights,
        layer_index: int,
        hidden_states: torch.Tensor,
        code_ids: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor | None,
        positions: torch.Tensor,
        cache_slots: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attention: QueryAttention,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_count = hidden_states.shape[0]
        shape = self.shape
        mixed = self.mixed[:row_count]
        scaled = self.scaled[:row_count]
        query = self.query[:row_count]
        gate_scale = self.gate_scale[:row_count]
        residual = self.residual[:row_count]
        activated = self.activated[:row_count]
        mix_eh_proj(
            weights=weights,
            hidden_states=hidden_states,
            code_ids=code_ids,
            label_is_semantic=label_is_semantic,
            shared_input=shared_input,
            mixed=mixed,
            scaled=scaled,
            scratch=self.scratch,
            shape=shape,
        )
        norm_qkv_rope_store(
            weights=weights,
            scaled=scaled,
            rotary_cos_sin=self.rotary_cos_sin,
            positions=positions,
            cache_slots=cache_slots,
            query=query,
            gate_scale=gate_scale,
            key_cache=key_cache,
            value_cache=value_cache,
            scratch=self.scratch,
            shape=shape,
        )
        gate_o_proj_add(
            weights=weights,
            attended=attention(layer_index, query),
            gate_scale=gate_scale,
            residual_in=mixed,
            residual_out=residual,
            scaled_out=scaled,
            scratch=self.scratch,
            shape=shape,
        )
        norm_gate_up_silu(
            weights=weights,
            scaled=scaled,
            activated=activated,
            scratch=self.scratch,
            shape=shape,
        )
        return activated, residual


def split_count(
    output_tiles: int, reduction_blocks: int, multiprocessor_count: int
) -> int:
    candidates = [split for split in SPLIT_CANDIDATES if reduction_blocks % split == 0]
    return min(
        candidates, key=lambda split: abs(output_tiles * split - multiprocessor_count)
    )


def tower_layer_shape(
    *,
    weights: TowerLayerWeights,
    head_count: int,
    kv_head_count: int,
    rotary_dim: int,
    multiprocessor_count: int,
) -> TowerLayerShape:
    dim = weights.eh_proj.weight.shape[0]
    head_dim = weights.q_norm.weight.shape[0]
    intermediate_size = weights.gate_up_proj.weight.shape[0] // 2
    return TowerLayerShape(
        dim=dim,
        head_count=head_count,
        kv_head_count=kv_head_count,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        intermediate_size=intermediate_size,
        codebook_size=weights.token_embedding.shape[0],
        mix_half_split=split_count(
            2 * (dim // MIX_BLOCK_N), dim // GEMV_BLOCK_K, multiprocessor_count
        ),
        qkv_split=split_count(
            2 * (head_count + kv_head_count), dim // QKV_BLOCK_K, multiprocessor_count
        ),
        o_proj_split=split_count(
            dim // O_PROJ_BLOCK_N,
            head_count * head_dim // GEMV_BLOCK_K,
            multiprocessor_count,
        ),
        gate_up_split=split_count(
            intermediate_size // GATE_UP_BLOCK_N,
            dim // GEMV_BLOCK_K,
            multiprocessor_count,
        ),
    )


def fused_tower_layers_unsupported_reason(
    *,
    layers: list[TowerLayerWeights],
    head_count: int,
    kv_head_count: int,
    rotary_dim: int,
    rotary_cos_sin: torch.Tensor,
    context_length: int,
    kv_pool: HybridLinearKVPool,
    tower_layer_ids: list[int],
    device: torch.device,
) -> str | None:
    if not HAS_TRITON:
        return "Triton is not installed"
    elif device.type != "cuda" or not current_platform.is_cuda():
        return f"device {device} is not a CUDA device of the CUDA platform"
    elif get_exec().deterministic.enable_deterministic_inference:
        return (
            f"deterministic inference is on, and passes above {MAX_FUSED_ROWS} rows "
            f"keep the plain path"
        )
    else:
        pass
    first_layer = layers[0]
    dim = first_layer.eh_proj.weight.shape[0]
    head_dim = first_layer.q_norm.weight.shape[0]
    intermediate_size = first_layer.gate_up_proj.weight.shape[0] // 2
    codebook_size = first_layer.token_embedding.shape[0]
    query_width = head_count * head_dim
    expected_shapes = {
        "eh_proj": (dim, 2 * dim),
        "qkv_proj": (2 * (head_count + kv_head_count) * head_dim, dim),
        "o_proj": (dim, query_width),
        "gate_up_proj": (2 * intermediate_size, dim),
    }
    if (
        dim % MIX_BLOCK_N != 0
        or dim % O_PROJ_BLOCK_N != 0
        or dim % GEMV_BLOCK_K != 0
        or dim % QKV_BLOCK_K != 0
        or intermediate_size % GATE_UP_BLOCK_N != 0
        or query_width % GEMV_BLOCK_K != 0
        or head_dim < 32
        or head_dim & (head_dim - 1) != 0
    ):
        return (
            f"dim {dim}, intermediate size {intermediate_size}, head dim {head_dim} "
            f"and {head_count} heads do not fit the kernels' blocks"
        )
    elif not (0 < rotary_dim <= head_dim and rotary_dim % 2 == 0):
        return (
            f"rotary dim {rotary_dim} is not an even width within head dim {head_dim}"
        )
    elif (
        rotary_cos_sin.dtype != torch.float32
        or not rotary_cos_sin.is_contiguous()
        or rotary_cos_sin.shape[1] != rotary_dim
        or rotary_cos_sin.shape[0] < context_length
        or rotary_cos_sin.device != device
    ):
        return (
            f"the rotary table {tuple(rotary_cos_sin.shape)} {rotary_cos_sin.dtype} is "
            f"not a contiguous float32 table of {context_length} positions and "
            f"{rotary_dim} columns on {device}"
        )
    else:
        pass
    for layer_index, layer in enumerate(layers):
        linears = {
            "eh_proj": layer.eh_proj,
            "qkv_proj": layer.qkv_proj,
            "o_proj": layer.o_proj,
            "gate_up_proj": layer.gate_up_proj,
        }
        norms = {
            "enorm": (layer.enorm, dim),
            "hnorm": (layer.hnorm, dim),
            "input_layernorm": (layer.input_norm, dim),
            "q_norm": (layer.q_norm, head_dim),
            "k_norm": (layer.k_norm, head_dim),
            "post_attention_layernorm": (layer.post_norm, dim),
        }
        token_embedding = layer.token_embedding
        for name, linear in linears.items():
            weight = linear.weight
            if type(linear) is not nn.Linear or linear.bias is not None:
                return (
                    f"tower layer {layer_index} {name} is not a bias-free torch Linear"
                )
            elif weight.dtype != torch.bfloat16 or not weight.is_contiguous():
                return f"tower layer {layer_index} {name} weight is not contiguous bfloat16"
            elif weight.device != device:
                return (
                    f"tower layer {layer_index} {name} weight is on {weight.device}, "
                    f"not {device}"
                )
            elif tuple(weight.shape) != expected_shapes[name]:
                return (
                    f"tower layer {layer_index} {name} weight is "
                    f"{tuple(weight.shape)}, not {expected_shapes[name]}"
                )
            else:
                pass
        for name, (norm, width) in norms.items():
            if (
                tuple(norm.weight.shape) != (width,)
                or not norm.weight.is_contiguous()
                or norm.weight.device != device
            ):
                return (
                    f"tower layer {layer_index} {name} weight is not a contiguous "
                    f"vector of {width} on {device}"
                )
            else:
                pass
        if (
            token_embedding.dtype != torch.bfloat16
            or not token_embedding.is_contiguous()
            or tuple(token_embedding.shape) != (codebook_size, dim)
            or token_embedding.device != device
        ):
            return (
                f"tower layer {layer_index} token embedding is not a contiguous "
                f"bfloat16 table of {codebook_size} x {dim} on {device}"
            )
        else:
            pass
    full_pool = kv_pool.full_kv_pool
    if type(full_pool) is not MHATokenToKVPool:
        return (
            f"the full-attention KV pool is {type(full_pool).__name__}, not "
            f"MHATokenToKVPool"
        )
    elif (
        full_pool.is_quantized_kv_cache
        or full_pool.use_hnd
        or full_pool.kv_cache_layout != "nhd"
        or full_pool.dtype != torch.bfloat16
        or full_pool.store_dtype != torch.bfloat16
        or full_pool.page_size != 1
    ):
        return (
            f"the full-attention KV pool has layout {full_pool.kv_cache_layout} and "
            f"dtype {full_pool.dtype}, not an unquantized bfloat16 pool of one-slot "
            f"pages in nhd layout"
        )
    elif (full_pool.head_num, full_pool.head_dim, full_pool.v_head_dim) != (
        kv_head_count,
        head_dim,
        head_dim,
    ):
        return (
            f"the KV pool holds {full_pool.head_num} heads of {full_pool.head_dim} / "
            f"{full_pool.v_head_dim}, not {kv_head_count} of {head_dim}"
        )
    else:
        pass
    for layer_id in tower_layer_ids:
        if layer_id not in kv_pool.full_attention_layer_id_mapping:
            return f"tower layer id {layer_id} has no full-attention KV buffer"
        else:
            pass
        for buffer in (
            kv_pool.get_key_buffer(layer_id),
            kv_pool.get_value_buffer(layer_id),
        ):
            if (
                buffer.dim() != 3
                or tuple(buffer.shape[1:])
                != (
                    kv_head_count,
                    head_dim,
                )
                or not buffer.is_contiguous()
            ):
                return (
                    f"the KV buffer of tower layer id {layer_id} is "
                    f"{tuple(buffer.shape)}, not contiguous slots of "
                    f"{kv_head_count} x {head_dim}"
                )
            else:
                pass
    return None
