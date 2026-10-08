# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import dataclasses

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    from sglang_omni.platforms import current_platform

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    libdevice = None
    current_platform = None
    HAS_TRITON = False

SELECTED_CANDIDATES = 32
TAIL_BLOCK_WIDTH = 32
MAX_BLOCK_WIDTH = 4096
SCORE_KEY_BITS = 32
SAMPLER_WARPS = 8
PHILOX_SEED_MODULUS = 1 << 63
FIRST_DRAW_STREAM = 0
RAS_DRAW_STREAM = 1
FIRST_CODEBOOK_HEAD_STREAM = 2
VALIDATED_COMPUTE_CAPABILITY = (9, 0)


@dataclasses.dataclass(frozen=True, kw_only=True)
class CodebookSampler:
    step: torch.Tensor
    seed: int


if HAS_TRITON:
    UNIFORM_MANTISSA_BITS = tl.constexpr(23)

    @triton.jit
    def ordered_keys(scores, candidate_indices, is_candidate, INDEX_BITS: tl.constexpr):
        score_bits = tl.where(scores == 0.0, 0.0, scores).to(tl.uint32, bitcast=True)
        sign_bit = tl.full(scores.shape, 1, tl.uint32) << 31
        all_bits = sign_bit | (sign_bit - 1)
        ordered_bits = score_bits ^ tl.where(
            (score_bits & sign_bit) != 0, all_bits, sign_bit
        )
        keys = (ordered_bits.to(tl.int64) << INDEX_BITS) | (
            (1 << INDEX_BITS) - 1 - candidate_indices
        ).to(tl.int64)
        return tl.where(is_candidate, keys, 0)

    @triton.jit
    def key_scores(keys, INDEX_BITS: tl.constexpr):
        ordered_bits = (keys >> INDEX_BITS).to(tl.uint32)
        sign_bit = tl.full(keys.shape, 1, tl.uint32) << 31
        all_bits = sign_bit | (sign_bit - 1)
        score_bits = ordered_bits ^ tl.where(
            (ordered_bits & sign_bit) != 0, sign_bit, all_bits
        )
        return score_bits.to(tl.float32, bitcast=True)

    @triton.jit
    def key_indices(keys, INDEX_BITS: tl.constexpr):
        return ((1 << INDEX_BITS) - 1 - (keys & ((1 << INDEX_BITS) - 1))).to(tl.int32)

    @triton.jit
    def exponential_noise(seed, step, row, stream_id, candidate_indices):
        random_bits, _, _, _ = tl.philox(
            seed ^ ((step >> 32) << 32),
            candidate_indices.to(tl.uint32),
            row,
            stream_id,
            (step & 0xFFFFFFFF).to(tl.uint32),
        )
        uniform = (
            (random_bits >> (32 - UNIFORM_MANTISSA_BITS)).to(tl.float32) + 0.5
        ) * (1.0 / (1 << UNIFORM_MANTISSA_BITS))
        return -libdevice.log(uniform)

    @triton.jit
    def candidate_selection(
        main_logits,
        main_is_candidate,
        tail_logits,
        tail_is_candidate,
        temperature,
        INDEX_BITS: tl.constexpr,
        SELECTED: tl.constexpr,
        HAS_TAIL: tl.constexpr,
    ):
        main_scores = tl.where(
            main_is_candidate,
            tl.math.div_rn(
                main_logits, tl.broadcast_to(temperature, main_logits.shape)
            ),
            float("-inf"),
        )
        tail_scores = tl.where(
            tail_is_candidate,
            tl.math.div_rn(
                tail_logits, tl.broadcast_to(temperature, tail_logits.shape)
            ),
            float("-inf"),
        )
        row_maximum = tl.maximum(
            tl.max(main_scores, axis=0), tl.max(tail_scores, axis=0)
        )
        main_weights = libdevice.exp(main_scores - row_maximum)
        tail_weights = libdevice.exp(tail_scores - row_maximum)
        weight_sum = tl.sum(main_weights, axis=0) + tl.sum(tail_weights, axis=0)
        main_width: tl.constexpr = main_logits.shape[0]
        main_keys = ordered_keys(
            main_scores, tl.arange(0, main_width), main_is_candidate, INDEX_BITS
        )
        tail_keys = ordered_keys(
            tail_scores,
            main_width + tl.arange(0, tail_logits.shape[0]),
            tail_is_candidate,
            INDEX_BITS,
        )
        top_keys = tl.topk(main_keys, SELECTED)
        if HAS_TAIL:
            top_keys = tl.topk(
                tl.reshape(
                    tl.join(top_keys, tl.topk(tail_keys, SELECTED)), [2 * SELECTED]
                ),
                SELECTED,
            )
        else:
            pass
        return (
            main_keys,
            main_weights,
            tail_keys,
            tail_weights,
            row_maximum,
            weight_sum,
            top_keys,
        )

    @triton.jit
    def selected_candidates(
        top_keys, row_maximum, weight_sum, top_k, top_p, INDEX_BITS: tl.constexpr
    ):
        selected_count: tl.constexpr = top_keys.shape[0]
        ranks = tl.arange(0, selected_count)
        selected_indices = key_indices(top_keys, INDEX_BITS)
        selected_weights = libdevice.exp(key_scores(top_keys, INDEX_BITS) - row_maximum)
        cumulative_probabilities = tl.cumsum(
            tl.math.div_rn(
                selected_weights, tl.broadcast_to(weight_sum, selected_weights.shape)
            ),
            axis=0,
        )
        is_kept = (ranks == 0) | ((ranks < top_k) & (cumulative_probabilities <= top_p))
        last_cumulative_probability = tl.sum(
            tl.where(ranks == selected_count - 1, cumulative_probabilities, 0.0), axis=0
        )
        holds_kept_set = (top_k <= selected_count) | (
            last_cumulative_probability > top_p
        )
        return selected_indices, selected_weights, is_kept, holds_kept_set

    @triton.jit
    def search_threshold(
        main_keys,
        main_probabilities,
        tail_keys,
        tail_probabilities,
        top_k,
        top_p,
        KEY_BITS: tl.constexpr,
        SEARCH_ROUNDS: tl.constexpr,
    ):
        lower_key = tl.full([], 0, tl.int64)
        upper_key = tl.full([], 1, tl.int64) << KEY_BITS
        for _ in range(SEARCH_ROUNDS):
            middle_key = (lower_key + upper_key) >> 1
            main_is_above = main_keys >= middle_key
            tail_is_above = tail_keys >= middle_key
            above_count = tl.sum(main_is_above.to(tl.int64), axis=0) + tl.sum(
                tail_is_above.to(tl.int64), axis=0
            )
            above_probability = tl.sum(
                tl.where(main_is_above, main_probabilities, 0.0), axis=0
            ) + tl.sum(tl.where(tail_is_above, tail_probabilities, 0.0), axis=0)
            fits = (above_count <= top_k) & (above_probability <= top_p)
            upper_key = tl.where(fits, middle_key, upper_key)
            lower_key = tl.where(fits, lower_key, middle_key + 1)
        return upper_key

    @triton.jit
    def race_selected(
        selected_indices, selected_weights, is_kept, seed, step, row, stream_id
    ):
        race = tl.where(
            is_kept,
            selected_weights
            / exponential_noise(seed, step, row, stream_id, selected_indices),
            -1.0,
        )
        chosen_rank = tl.argmax(race, axis=0)
        ranks = tl.arange(0, selected_indices.shape[0])
        return tl.sum(tl.where(ranks == chosen_rank, selected_indices, 0), axis=0)

    @triton.jit
    def race_above_threshold(
        main_keys,
        main_weights,
        tail_keys,
        tail_weights,
        threshold_key,
        top_key,
        seed,
        step,
        row,
        stream_id,
        INDEX_BITS: tl.constexpr,
    ):
        main_is_kept = (main_keys >= threshold_key) | (main_keys == top_key)
        tail_is_kept = (tail_keys >= threshold_key) | (tail_keys == top_key)
        main_race = tl.where(
            main_is_kept,
            main_weights
            / exponential_noise(
                seed, step, row, stream_id, key_indices(main_keys, INDEX_BITS)
            ),
            -1.0,
        )
        tail_race = tl.where(
            tail_is_kept,
            tail_weights
            / exponential_noise(
                seed, step, row, stream_id, key_indices(tail_keys, INDEX_BITS)
            ),
            -1.0,
        )
        best_race = tl.maximum(tl.max(main_race, axis=0), tl.max(tail_race, axis=0))
        chosen_key = tl.maximum(
            tl.max(tl.where(main_race == best_race, main_keys, 0), axis=0),
            tl.max(tl.where(tail_race == best_race, tail_keys, 0), axis=0),
        )
        return key_indices(tl.where(chosen_key > 0, chosen_key, top_key), INDEX_BITS)

    @triton.jit
    def draw_candidate(
        main_keys,
        main_weights,
        tail_keys,
        tail_weights,
        row_maximum,
        weight_sum,
        top_keys,
        top_k,
        top_p,
        seed,
        step,
        row,
        stream_id,
        INDEX_BITS: tl.constexpr,
        KEY_BITS: tl.constexpr,
    ):
        selected_indices, selected_weights, is_kept, holds_kept_set = (
            selected_candidates(
                top_keys, row_maximum, weight_sum, top_k, top_p, INDEX_BITS
            )
        )
        if holds_kept_set:
            chosen_index = race_selected(
                selected_indices, selected_weights, is_kept, seed, step, row, stream_id
            )
        else:
            threshold_key = search_threshold(
                main_keys,
                tl.math.div_rn(
                    main_weights, tl.broadcast_to(weight_sum, main_weights.shape)
                ),
                tail_keys,
                tl.math.div_rn(
                    tail_weights, tl.broadcast_to(weight_sum, tail_weights.shape)
                ),
                top_k,
                top_p,
                KEY_BITS,
                KEY_BITS,
            )
            chosen_index = race_above_threshold(
                main_keys,
                main_weights,
                tail_keys,
                tail_weights,
                threshold_key,
                tl.max(top_keys, axis=0),
                seed,
                step,
                row,
                stream_id,
                INDEX_BITS,
            )
        return chosen_index

    @triton.jit
    def codebook_head_kernel(
        logits_pointer,
        logits_row_stride,
        temperature_pointer,
        top_k_pointer,
        top_p_pointer,
        step_pointer,
        sampled_codes_pointer,
        seed,
        stream_id,
        CODEBOOK_SIZE: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        KEY_BITS: tl.constexpr,
        SELECTED: tl.constexpr,
        TAIL_WIDTH: tl.constexpr,
        MINIMUM_TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK_WIDTH)
        is_column = columns < CODEBOOK_SIZE
        logits = tl.load(
            logits_pointer + row * logits_row_stride + columns,
            mask=is_column,
            other=0.0,
        ).to(tl.float32)
        temperature = tl.maximum(
            tl.load(temperature_pointer + row), MINIMUM_TEMPERATURE
        )
        (
            main_keys,
            main_weights,
            tail_keys,
            tail_weights,
            row_maximum,
            weight_sum,
            top_keys,
        ) = candidate_selection(
            logits,
            is_column,
            tl.zeros([TAIL_WIDTH], tl.float32),
            tl.arange(0, TAIL_WIDTH) < 0,
            temperature,
            INDEX_BITS,
            SELECTED,
            False,
        )
        sampled_code = draw_candidate(
            main_keys,
            main_weights,
            tail_keys,
            tail_weights,
            row_maximum,
            weight_sum,
            top_keys,
            tl.load(top_k_pointer + row),
            tl.load(top_p_pointer + row),
            seed,
            tl.load(step_pointer),
            row,
            stream_id,
            INDEX_BITS,
            KEY_BITS,
        )
        tl.store(sampled_codes_pointer + row, sampled_code.to(tl.int64))

    @triton.jit
    def semantic_kernel(
        semantic_logits_pointer,
        semantic_logits_row_stride,
        candidate_token_ids_pointer,
        previous_main_tokens_pointer,
        previous_main_tokens_row_stride,
        temperature_pointer,
        top_k_pointer,
        top_p_pointer,
        step_pointer,
        main_tokens_pointer,
        codebook_zero_pointer,
        seed,
        semantic_begin_id,
        CANDIDATE_COUNT: tl.constexpr,
        MAIN_WIDTH: tl.constexpr,
        TAIL_WIDTH: tl.constexpr,
        INDEX_BITS: tl.constexpr,
        KEY_BITS: tl.constexpr,
        SELECTED: tl.constexpr,
        CODEBOOK_SIZE: tl.constexpr,
        WINDOW_FRAMES: tl.constexpr,
        MINIMUM_TEMPERATURE: tl.constexpr,
        RAS_TEMPERATURE: tl.constexpr,
        RAS_TOP_K: tl.constexpr,
        RAS_TOP_P: tl.constexpr,
        FIRST_STREAM: tl.constexpr,
        RAS_STREAM: tl.constexpr,
    ):
        row = tl.program_id(0)
        main_candidates = tl.arange(0, MAIN_WIDTH)
        tail_candidates = MAIN_WIDTH + tl.arange(0, TAIL_WIDTH)
        main_is_candidate = main_candidates < CANDIDATE_COUNT
        tail_is_candidate = tail_candidates < CANDIDATE_COUNT
        row_logits_pointer = semantic_logits_pointer + row * semantic_logits_row_stride
        main_logits = tl.load(
            row_logits_pointer + tl.load(candidate_token_ids_pointer + main_candidates)
        ).to(tl.float32)
        tail_logits = tl.load(
            row_logits_pointer
            + tl.load(
                candidate_token_ids_pointer + tail_candidates,
                mask=tail_is_candidate,
                other=0,
            ),
            mask=tail_is_candidate,
            other=0.0,
        ).to(tl.float32)
        temperature = tl.maximum(
            tl.load(temperature_pointer + row), MINIMUM_TEMPERATURE
        )
        step = tl.load(step_pointer)
        (
            first_main_keys,
            first_main_weights,
            first_tail_keys,
            first_tail_weights,
            first_row_maximum,
            first_weight_sum,
            first_top_keys,
        ) = candidate_selection(
            main_logits,
            main_is_candidate,
            tail_logits,
            tail_is_candidate,
            temperature,
            INDEX_BITS,
            SELECTED,
            True,
        )
        if temperature == RAS_TEMPERATURE:
            ras_main_keys = first_main_keys
            ras_main_weights = first_main_weights
            ras_tail_keys = first_tail_keys
            ras_tail_weights = first_tail_weights
            ras_row_maximum = first_row_maximum
            ras_weight_sum = first_weight_sum
            ras_top_keys = first_top_keys
        else:
            (
                ras_main_keys,
                ras_main_weights,
                ras_tail_keys,
                ras_tail_weights,
                ras_row_maximum,
                ras_weight_sum,
                ras_top_keys,
            ) = candidate_selection(
                main_logits,
                main_is_candidate,
                tail_logits,
                tail_is_candidate,
                tl.full([], RAS_TEMPERATURE, tl.float32),
                INDEX_BITS,
                SELECTED,
                True,
            )
        first_candidate = draw_candidate(
            first_main_keys,
            first_main_weights,
            first_tail_keys,
            first_tail_weights,
            first_row_maximum,
            first_weight_sum,
            first_top_keys,
            tl.load(top_k_pointer + row),
            tl.load(top_p_pointer + row),
            seed,
            step,
            row,
            FIRST_STREAM,
            INDEX_BITS,
            KEY_BITS,
        )
        ras_candidate = draw_candidate(
            ras_main_keys,
            ras_main_weights,
            ras_tail_keys,
            ras_tail_weights,
            ras_row_maximum,
            ras_weight_sum,
            ras_top_keys,
            tl.full([], RAS_TOP_K, tl.int64),
            tl.full([], RAS_TOP_P, tl.float32),
            seed,
            step,
            row,
            RAS_STREAM,
            INDEX_BITS,
            KEY_BITS,
        )
        first_token = tl.load(candidate_token_ids_pointer + first_candidate)
        ras_token = tl.load(candidate_token_ids_pointer + ras_candidate)
        window = tl.load(
            previous_main_tokens_pointer
            + row * previous_main_tokens_row_stride
            + tl.arange(0, WINDOW_FRAMES)
        )
        is_repeat = tl.max((window == first_token).to(tl.int32), axis=0) > 0
        main_token = tl.where(is_repeat, ras_token, first_token)
        tl.store(main_tokens_pointer + row, main_token)
        tl.store(
            codebook_zero_pointer + row,
            tl.minimum(
                tl.maximum(main_token - semantic_begin_id, 0), CODEBOOK_SIZE - 1
            ),
        )

else:
    pass


def main_block_width(candidate_count: int) -> int:
    return 1 << (candidate_count.bit_length() - 1)


def untested_gpu_reason(device: torch.device) -> str | None:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) == VALIDATED_COMPUTE_CAPABILITY:
        return None
    else:
        return f"untested GPU sm_{major}{minor}"


def triton_sampling_unsupported_reason(
    device: torch.device, codebook_size: int, candidate_count: int
) -> str | None:
    main_width = main_block_width(candidate_count)
    if not HAS_TRITON:
        return "Triton is not installed"
    elif device.type != "cuda" or not current_platform.is_cuda():
        return f"device {device} is not a CUDA device of the CUDA platform"
    elif (gpu_reason := untested_gpu_reason(device)) is not None:
        return gpu_reason
    elif not SELECTED_CANDIDATES <= codebook_size <= MAX_BLOCK_WIDTH:
        return (
            f"codebook size {codebook_size} is outside "
            f"[{SELECTED_CANDIDATES}, {MAX_BLOCK_WIDTH}]"
        )
    elif (
        main_width < SELECTED_CANDIDATES
        or main_width > MAX_BLOCK_WIDTH
        or candidate_count - main_width > TAIL_BLOCK_WIDTH
    ):
        return (
            f"{candidate_count} main-token candidates do not fit a block of "
            f"{SELECTED_CANDIDATES} to {MAX_BLOCK_WIDTH} plus {TAIL_BLOCK_WIDTH}"
        )
    elif SELECTED_CANDIDATES in (codebook_size, candidate_count):
        return (
            f"PyTorch orders tied logits in rows of {SELECTED_CANDIDATES} columns "
            f"unlike the kernels; codebook size {codebook_size}, "
            f"{candidate_count} main-token candidates"
        )
    else:
        return None


def sample_codebook_head(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    sampler: CodebookSampler,
    stream_id: int,
    minimum_temperature: float,
) -> torch.Tensor:
    row_count, codebook_size = logits.shape
    assert logits.stride(1) == 1
    index_bits = (codebook_size - 1).bit_length()
    sampled_codes = torch.empty(row_count, dtype=torch.int64, device=logits.device)
    codebook_head_kernel[(row_count,)](
        logits,
        logits.stride(0),
        temperature.float().contiguous(),
        top_k.long().contiguous(),
        top_p.float().contiguous(),
        sampler.step,
        sampled_codes,
        sampler.seed,
        stream_id,
        CODEBOOK_SIZE=codebook_size,
        BLOCK_WIDTH=triton.next_power_of_2(codebook_size),
        INDEX_BITS=index_bits,
        KEY_BITS=SCORE_KEY_BITS + index_bits,
        SELECTED=SELECTED_CANDIDATES,
        TAIL_WIDTH=TAIL_BLOCK_WIDTH,
        MINIMUM_TEMPERATURE=minimum_temperature,
        num_warps=SAMPLER_WARPS,
    )
    return sampled_codes


def sample_semantic_candidates(
    semantic_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    previous_main_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    semantic_begin_id: int,
    codebook_size: int,
    sampler: CodebookSampler,
    *,
    minimum_temperature: float,
    ras_temperature: float,
    ras_top_k: int,
    ras_top_p: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = semantic_logits.shape[0]
    candidate_count = candidate_token_ids.shape[0]
    assert semantic_logits.stride(1) == 1 and previous_main_tokens.stride(1) == 1
    index_bits = (candidate_count - 1).bit_length()
    main_tokens = torch.empty(
        row_count, dtype=torch.int64, device=semantic_logits.device
    )
    codebook_zero = torch.empty_like(main_tokens)
    semantic_kernel[(row_count,)](
        semantic_logits,
        semantic_logits.stride(0),
        candidate_token_ids.long().contiguous(),
        previous_main_tokens,
        previous_main_tokens.stride(0),
        temperature.float().contiguous(),
        top_k.long().contiguous(),
        top_p.float().contiguous(),
        sampler.step,
        main_tokens,
        codebook_zero,
        sampler.seed,
        semantic_begin_id,
        CANDIDATE_COUNT=candidate_count,
        MAIN_WIDTH=main_block_width(candidate_count),
        TAIL_WIDTH=TAIL_BLOCK_WIDTH,
        INDEX_BITS=index_bits,
        KEY_BITS=SCORE_KEY_BITS + index_bits,
        SELECTED=SELECTED_CANDIDATES,
        CODEBOOK_SIZE=codebook_size,
        WINDOW_FRAMES=previous_main_tokens.shape[1],
        MINIMUM_TEMPERATURE=minimum_temperature,
        RAS_TEMPERATURE=ras_temperature,
        RAS_TOP_K=ras_top_k,
        RAS_TOP_P=ras_top_p,
        FIRST_STREAM=FIRST_DRAW_STREAM,
        RAS_STREAM=RAS_DRAW_STREAM,
        num_warps=SAMPLER_WARPS,
    )
    return main_tokens, codebook_zero
