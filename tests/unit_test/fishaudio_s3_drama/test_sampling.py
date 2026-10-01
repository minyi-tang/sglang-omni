# SPDX-License-Identifier: Apache-2.0
"""S3-drama sampling: top-p cut, top-k, semantic bias, repetition-aware resampling."""

from __future__ import annotations

import math

import pytest
import torch

from sglang_omni.models.fishaudio_s3_drama.sampling import (
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
    sample_semantic_with_ras,
    sample_top_k_top_p,
)

DRAW_COUNT = 2000
TEXT_VOCAB_SIZE = 48
SEMANTIC_BEGIN_ID = 8
CODEBOOK_SIZE = 32
EOS_TOKEN_ID = 2
# No token id is negative, so this window never matches.
EMPTY_WINDOW_TOKEN = -1


def repeated(values: list[float], rows: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32).repeat(rows, 1)


def request_parameters(
    rows: int, temperature: float, top_k: int, top_p: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.full((rows,), temperature),
        torch.full((rows,), top_k, dtype=torch.int64),
        torch.full((rows,), top_p),
    )


def semantic_bias() -> torch.Tensor:
    bias = torch.full((TEXT_VOCAB_SIZE,), float("-inf"))
    bias[SEMANTIC_BEGIN_ID : SEMANTIC_BEGIN_ID + CODEBOOK_SIZE] = 0.0
    bias[EOS_TOKEN_ID] = 0.0
    return bias


def test_top_k_one_takes_the_argmax() -> None:
    torch.manual_seed(0)
    logits = torch.randn(64, 32)

    sampled = sample_top_k_top_p(logits, *request_parameters(64, 1.0, 1, 1.0))

    assert torch.equal(sampled, logits.argmax(dim=-1))


@pytest.mark.parametrize(("top_p", "kept_tokens"), [(0.8, {0, 1}), (0.79, {0})])
def test_top_p_drops_the_token_that_crosses_p(
    top_p: float, kept_tokens: set[int]
) -> None:
    """Cumulative probability counts the token itself; at 0.79 only the top stays."""
    torch.manual_seed(0)
    logits = repeated([math.log(0.5), math.log(0.3), math.log(0.2)], DRAW_COUNT)

    sampled = sample_top_k_top_p(logits, *request_parameters(DRAW_COUNT, 1.0, 3, top_p))

    assert set(sampled.tolist()) == kept_tokens


def test_top_k_and_temperature_limit_the_candidates() -> None:
    torch.manual_seed(0)
    logits = repeated([2.0, 1.9, 1.8, 1.7, 1.6], DRAW_COUNT)

    sampled = sample_top_k_top_p(logits, *request_parameters(DRAW_COUNT, 5.0, 2, 1.0))

    assert set(sampled.tolist()) == {0, 1}


def test_sampling_keeps_inputs_and_returns_int64_ids() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 16, dtype=torch.bfloat16)
    temperature, top_k, top_p = request_parameters(4, 0.7, 5, 0.9)
    originals = [tensor.clone() for tensor in (logits, temperature, top_k, top_p)]

    sampled = sample_top_k_top_p(logits, temperature, top_k, top_p)

    assert sampled.dtype == torch.int64
    assert sampled.device == logits.device
    assert sampled.shape == (4,)
    for original, tensor in zip(originals, (logits, temperature, top_k, top_p)):
        assert torch.equal(original, tensor)


def ras_logits(rows: int) -> torch.Tensor:
    """Near-flat semantic logits with a clear top token, plus a huge non-band logit."""
    logits = torch.zeros(rows, TEXT_VOCAB_SIZE)
    logits[:, SEMANTIC_BEGIN_ID : SEMANTIC_BEGIN_ID + CODEBOOK_SIZE] = torch.linspace(
        0.0, 0.3, CODEBOOK_SIZE
    )
    logits[:, SEMANTIC_BEGIN_ID + 5] = 3.0
    logits[:, 40] = 50.0
    return logits


def test_ras_takes_the_second_draw_exactly_when_the_first_repeats() -> None:
    """Both draws happen in order; the fixed-parameter one wins only on a window hit."""
    rows = 8
    logits = ras_logits(rows)
    temperature, top_k, top_p = request_parameters(rows, 1.0, 1, 1.0)
    biased_logits = logits + semantic_bias()
    torch.manual_seed(3)
    first_draw = sample_top_k_top_p(biased_logits, temperature, top_k, top_p)
    second_draw = sample_top_k_top_p(
        biased_logits, *request_parameters(rows, RAS_TEMPERATURE, RAS_TOP_K, RAS_TOP_P)
    )
    window = torch.full((rows, 16), EMPTY_WINDOW_TOKEN)
    window[::2, 7] = first_draw[::2]
    is_repeat = torch.tensor([True, False] * (rows // 2))

    torch.manual_seed(3)
    main_tokens, codebook_zero = sample_semantic_with_ras(
        logits,
        semantic_bias(),
        window,
        temperature,
        top_k,
        top_p,
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
    )

    assert torch.equal(first_draw, torch.full((rows,), SEMANTIC_BEGIN_ID + 5))
    assert torch.equal(main_tokens, torch.where(is_repeat, second_draw, first_draw))
    assert not torch.equal(second_draw[::2], first_draw[::2])
    assert torch.equal(codebook_zero, main_tokens - SEMANTIC_BEGIN_ID)


def test_ras_draws_the_same_random_numbers_with_or_without_a_hit() -> None:
    rows = 4
    logits = ras_logits(rows)
    parameters = request_parameters(rows, 1.0, 1, 1.0)
    hit_window = torch.full((rows, 16), SEMANTIC_BEGIN_ID + 5)
    miss_window = torch.full((rows, 16), EMPTY_WINDOW_TOKEN)
    next_numbers = []
    for window in (hit_window, miss_window):
        torch.manual_seed(5)
        sample_semantic_with_ras(
            logits,
            semantic_bias(),
            window,
            *parameters,
            SEMANTIC_BEGIN_ID,
            CODEBOOK_SIZE,
        )
        next_numbers.append(torch.rand(3))

    assert torch.equal(next_numbers[0], next_numbers[1])


def test_semantic_bias_allows_only_the_band_and_eos() -> None:
    torch.manual_seed(0)
    rows = DRAW_COUNT
    logits = torch.zeros(rows, TEXT_VOCAB_SIZE)
    logits[:, 40] = 50.0
    window = torch.full((rows, 16), EMPTY_WINDOW_TOKEN)

    main_tokens, codebook_zero = sample_semantic_with_ras(
        logits,
        semantic_bias(),
        window,
        *request_parameters(rows, 1.0, TEXT_VOCAB_SIZE, 1.0),
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
    )
    logits[:, EOS_TOKEN_ID] = 9.0
    eos_tokens, eos_codebook_zero = sample_semantic_with_ras(
        logits,
        semantic_bias(),
        window,
        *request_parameters(rows, 1.0, 1, 1.0),
        SEMANTIC_BEGIN_ID,
        CODEBOOK_SIZE,
    )

    allowed = set(range(SEMANTIC_BEGIN_ID, SEMANTIC_BEGIN_ID + CODEBOOK_SIZE)) | {
        EOS_TOKEN_ID
    }
    assert set(main_tokens.tolist()) <= allowed
    assert torch.equal(codebook_zero, (main_tokens - SEMANTIC_BEGIN_ID).clamp(min=0))
    assert set(eos_tokens.tolist()) == {EOS_TOKEN_ID}
    assert set(eos_codebook_zero.tolist()) == {0}


def test_ras_window_must_hold_sixteen_frames() -> None:
    logits = torch.zeros(1, TEXT_VOCAB_SIZE)

    with pytest.raises(ValueError, match="16"):
        sample_semantic_with_ras(
            logits,
            semantic_bias(),
            torch.full((1, 8), EMPTY_WINDOW_TOKEN),
            *request_parameters(1, 1.0, 1, 1.0),
            SEMANTIC_BEGIN_ID,
            CODEBOOK_SIZE,
        )
