# SPDX-License-Identifier: Apache-2.0
"""S3-drama generation requests: prompt frame rows in, sampled frames out."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.fishaudio_s3_drama.hf_config import FishOmniConfig
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

# The tower prefill pairs every row with the next one, so a prompt needs two.
MINIMUM_PROMPT_ROWS = 2


@dataclass(frozen=True, kw_only=True)
class S3DramaGenerationInput:
    """One request: prompt frame rows [rows, 1 + num_codebooks] and sampling knobs."""

    prompt_rows: torch.Tensor
    max_new_tokens: int
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True, kw_only=True)
class S3DramaGenerationResult:
    """Sampled frames [frames, 1 + num_codebooks], without the end-of-speech frame."""

    frames: torch.Tensor


@dataclass
class S3DramaRequestData(SGLangARRequestData):
    """frame_rows holds the prompt rows and then every sampled frame, eos included."""

    frame_rows: list[list[int]] = field(default_factory=list)
    prompt_row_count: int = 0


class S3DramaRequestBuilder:
    """Validates an S3DramaGenerationInput payload and builds its SGLang request."""

    def __init__(self, *, config: FishOmniConfig, context_length: int) -> None:
        self.vocab_size = config.text_config.vocab_size
        self.codebook_size = config.audio_decoder_config.vocab_size
        self.frame_width = 1 + config.audio_decoder_config.num_codebooks
        self.eos_token_id = config.eos_token_id
        self.context_length = context_length

    def __call__(self, payload: StagePayload) -> S3DramaRequestData:
        generation_input = payload.data
        if not isinstance(generation_input, S3DramaGenerationInput):
            raise TypeError(
                f"S3-drama requests carry S3DramaGenerationInput; got "
                f"{type(generation_input).__name__}"
            )
        else:
            pass
        prompt_rows = generation_input.prompt_rows
        if (
            prompt_rows.dtype != torch.int64
            or prompt_rows.dim() != 2
            or prompt_rows.shape[1] != self.frame_width
        ):
            raise ValueError(
                f"prompt_rows must be int64 [rows, {self.frame_width}]; got "
                f"{prompt_rows.dtype} {tuple(prompt_rows.shape)}"
            )
        elif prompt_rows.shape[0] < MINIMUM_PROMPT_ROWS:
            raise ValueError(
                f"a prompt needs at least {MINIMUM_PROMPT_ROWS} rows; got "
                f"{prompt_rows.shape[0]}"
            )
        elif not bool(
            ((prompt_rows[:, 0] >= 0) & (prompt_rows[:, 0] < self.vocab_size)).all()
        ):
            raise ValueError(f"prompt main tokens must lie in [0, {self.vocab_size})")
        elif not bool(
            (
                (prompt_rows[:, 1:] >= 0) & (prompt_rows[:, 1:] < self.codebook_size)
            ).all()
        ):
            raise ValueError(f"prompt codes must lie in [0, {self.codebook_size})")
        elif generation_input.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens must be >= 1; got {generation_input.max_new_tokens}"
            )
        elif prompt_rows.shape[0] + generation_input.max_new_tokens > (
            self.context_length
        ):
            raise ValueError(
                f"{prompt_rows.shape[0]} prompt rows plus max_new_tokens="
                f"{generation_input.max_new_tokens} exceed the context length "
                f"{self.context_length}"
            )
        elif not (
            math.isfinite(generation_input.temperature)
            and generation_input.temperature >= 0.0
        ):
            raise ValueError(
                f"temperature must be finite and >= 0; got {generation_input.temperature}"
            )
        elif generation_input.top_k < 1:
            raise ValueError(f"top_k must be >= 1; got {generation_input.top_k}")
        elif not 0.0 < generation_input.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1]; got {generation_input.top_p}")
        else:
            pass
        sampling_params = SamplingParams(
            max_new_tokens=generation_input.max_new_tokens,
            temperature=generation_input.temperature,
            top_p=generation_input.top_p,
            top_k=generation_input.top_k,
            stop_token_ids=[self.eos_token_id],
        )
        sampling_params.normalize(None)
        sampling_params.verify(self.vocab_size)
        frame_rows = prompt_rows.tolist()
        req = Req(
            rid=payload.request_id,
            origin_input_text="",
            origin_input_ids=array("q", [frame_row[0] for frame_row in frame_rows]),
            sampling_params=sampling_params,
            vocab_size=self.vocab_size,
            eos_token_ids={self.eos_token_id},
        )
        return S3DramaRequestData(
            req=req,
            input_ids=prompt_rows[:, 0].clone(),
            stage_payload=payload,
            max_new_tokens=generation_input.max_new_tokens,
            temperature=generation_input.temperature,
            top_k=generation_input.top_k,
            top_p=generation_input.top_p,
            frame_rows=frame_rows,
            prompt_row_count=len(frame_rows),
        )


class S3DramaResultAdapter:
    """Puts a finished request's sampled frames into its stage payload."""

    def __init__(self, *, eos_token_id: int) -> None:
        self.eos_token_id = eos_token_id

    def __call__(self, request_data: S3DramaRequestData) -> StagePayload:
        payload = request_data.stage_payload
        assert payload is not None
        sampled_rows = request_data.frame_rows[request_data.prompt_row_count :]
        if sampled_rows and sampled_rows[-1][0] == self.eos_token_id:
            sampled_rows = sampled_rows[:-1]
        else:
            pass
        frame_width = len(request_data.frame_rows[0])
        payload.data = S3DramaGenerationResult(
            frames=torch.tensor(sampled_rows, dtype=torch.int64).reshape(
                len(sampled_rows), frame_width
            )
        )
        return payload
