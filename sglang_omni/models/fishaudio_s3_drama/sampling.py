# SPDX-License-Identifier: Apache-2.0
"""Token sampling for the S3-drama audio heads: top-k / top-p and semantic RAS."""

from __future__ import annotations

import torch

# note (SunskyXH): repetition-aware sampling is fixed by the training recipe:
# when the first draw repeats a main token from the last 16 frames, the frame
# takes a second draw made with these parameters instead.
RAS_WINDOW_FRAMES = 16
# Pads the windows of requests with fewer than 16 frames; matches no token id.
EMPTY_RAS_SLOT = -1
RAS_TEMPERATURE = 1.0
RAS_TOP_P = 0.95
RAS_TOP_K = 30
MINIMUM_TEMPERATURE = 1e-5


def sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    """One token id per row of [rows, vocab] logits, computed in fp32.

    Sorted descending, a candidate is dropped when its rank is >= top_k or when
    the cumulative probability up to and including it exceeds top_p; the most
    likely token always stays. top_k must be >= 1.
    """
    scaled_logits = logits.float() / temperature.float().clamp(
        min=MINIMUM_TEMPERATURE
    ).unsqueeze(-1)
    sorted_logits, sorted_token_ids = torch.sort(scaled_logits, dim=-1, descending=True)
    cumulative_probabilities = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )
    ranks = torch.arange(sorted_logits.shape[-1], device=logits.device).unsqueeze(0)
    is_dropped = (cumulative_probabilities > top_p.float().unsqueeze(-1)) | (
        ranks >= top_k.unsqueeze(-1)
    )
    is_dropped[:, 0] = False
    probabilities = torch.softmax(
        sorted_logits.masked_fill(is_dropped, float("-inf")), dim=-1
    )
    exponential_noise = torch.empty_like(probabilities).exponential_(1)
    sorted_choice = torch.argmax(
        probabilities / exponential_noise, dim=-1, keepdim=True
    )
    return torch.gather(sorted_token_ids, -1, sorted_choice).squeeze(-1)


def sample_semantic_with_ras(
    semantic_logits: torch.Tensor,
    candidate_token_ids: torch.Tensor,
    previous_main_tokens: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    semantic_begin_id: int,
    codebook_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample each frame's main token and its codebook-0 code.

    previous_main_tokens [rows, 16] are the main tokens of the last 16 frames.
    Both draws always happen, so the random stream does not depend on the window.
    """
    if previous_main_tokens.shape[-1] != RAS_WINDOW_FRAMES:
        raise ValueError(
            f"previous_main_tokens must hold the last {RAS_WINDOW_FRAMES} frames; "
            f"got shape {tuple(previous_main_tokens.shape)}"
        )
    else:
        pass
    candidate_logits = semantic_logits.index_select(-1, candidate_token_ids).float()
    first_draw = candidate_token_ids[
        sample_top_k_top_p(candidate_logits, temperature, top_k, top_p)
    ]
    row_count = candidate_logits.shape[0]
    device = candidate_logits.device
    ras_draw = candidate_token_ids[
        sample_top_k_top_p(
            candidate_logits,
            torch.full(
                (row_count,), RAS_TEMPERATURE, dtype=torch.float32, device=device
            ),
            torch.full((row_count,), RAS_TOP_K, dtype=torch.int64, device=device),
            torch.full((row_count,), RAS_TOP_P, dtype=torch.float32, device=device),
        )
    ]
    is_repeat = (previous_main_tokens == first_draw.unsqueeze(-1)).any(dim=-1)
    main_tokens = torch.where(is_repeat, ras_draw, first_draw)
    codebook_zero = (main_tokens - semantic_begin_id).clamp(0, codebook_size - 1)
    return main_tokens, codebook_zero
