# SPDX-License-Identifier: Apache-2.0
"""The graph test's S3-drama model: copies what each forward read and sampled into
fixed buffers, inside any captured graph, and takes its eos logits from one."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Protocol

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from torch import nn

from sglang_omni.models.fishaudio_s3_drama.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_s3_drama.sampling import RAS_WINDOW_FRAMES
from sglang_omni.models.fishaudio_s3_drama.sglang_model import S3DramaForCausalLM


@dataclass(frozen=True, kw_only=True)
class RecordingBuffers:
    """One row per request: the eos logit the runner stages, then copies of what
    the forward read and sampled."""

    end_of_speech_logits: torch.Tensor
    frames: torch.Tensor
    ras_windows: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    # [rows, 2]: the first semantic draw, then the RAS draw.
    draws: torch.Tensor
    semantic_logits: torch.Tensor
    head_logits: torch.Tensor


class EndOfSpeechControlledLogits(nn.Module):
    """Wraps the logits processor: each row's eos logit comes from a fixed buffer."""

    def __init__(self, logits_processor: LogitsProcessor, eos_token_id: int) -> None:
        super().__init__()
        self.logits_processor = logits_processor
        self.eos_token_id = eos_token_id
        self.recording_buffers: RecordingBuffers | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        output = self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)
        recording_buffers = self.recording_buffers
        assert recording_buffers is not None
        logits = output.next_token_logits
        row_count = logits.shape[0]
        logits[:, self.eos_token_id] = recording_buffers.end_of_speech_logits[
            :row_count
        ]
        recording_buffers.semantic_logits[:row_count].copy_(logits)
        return output


class GraphRecordingModel(S3DramaForCausalLM):
    """S3DramaForCausalLM that copies its reads and draws into RecordingBuffers."""

    def __init__(
        self,
        config: FishOmniConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.end_of_speech_logits = EndOfSpeechControlledLogits(
            self.logits_processor, config.eos_token_id
        )
        self.logits_processor = self.end_of_speech_logits
        self.recording_buffers: RecordingBuffers | None = None
        self.draw_column = 0
        for head_index, head in enumerate(self.audio_decoder.codebook_heads):
            head.register_forward_hook(
                functools.partial(self.keep_head_logits, head_index)
            )

    def allocate_recording_buffers(self, row_count: int, device: torch.device) -> None:
        decoder_config = self.config.audio_decoder_config
        recording_buffers = RecordingBuffers(
            end_of_speech_logits=torch.zeros(row_count, device=device),
            frames=torch.zeros(
                (row_count, 1 + decoder_config.num_codebooks),
                dtype=torch.int64,
                device=device,
            ),
            ras_windows=torch.zeros(
                (row_count, RAS_WINDOW_FRAMES), dtype=torch.int64, device=device
            ),
            temperature=torch.zeros(row_count, device=device),
            top_k=torch.zeros(row_count, dtype=torch.int64, device=device),
            top_p=torch.zeros(row_count, device=device),
            draws=torch.zeros((row_count, 2), dtype=torch.int64, device=device),
            semantic_logits=torch.zeros(
                (row_count, self.config.text_config.vocab_size), device=device
            ),
            head_logits=torch.zeros(
                (
                    row_count,
                    decoder_config.num_codebooks - 1,
                    decoder_config.vocab_size,
                ),
                device=device,
            ),
        )
        self.recording_buffers = recording_buffers
        self.end_of_speech_logits.recording_buffers = recording_buffers

    def keep_head_logits(
        self,
        head_index: int,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        logits: torch.Tensor,
    ) -> None:
        """Forward-hook contract; the decode step calls the heads in order."""
        recording_buffers = self.recording_buffers
        assert recording_buffers is not None
        recording_buffers.head_logits[: logits.shape[0], head_index].copy_(logits)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        self.draw_column = 0
        output = super().forward(input_ids, positions, forward_batch)
        recording_buffers = self.recording_buffers
        decode_buffers = self.decode_buffers
        assert recording_buffers is not None and decode_buffers is not None
        if forward_batch.forward_mode == ForwardMode.DECODE:
            row_count = input_ids.shape[0]
            for staged, recorded in (
                (decode_buffers.frames, recording_buffers.frames),
                (decode_buffers.ras_windows, recording_buffers.ras_windows),
                (decode_buffers.temperature, recording_buffers.temperature),
                (decode_buffers.top_k, recording_buffers.top_k),
                (decode_buffers.top_p, recording_buffers.top_p),
            ):
                recorded[:row_count].copy_(staged[:row_count])
        else:
            pass
        return output


class TopKTopPSampler(Protocol):
    def __call__(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
    ) -> torch.Tensor: ...


class DrawRecorder:
    """Stands in for sampling.sample_top_k_top_p, which a decode step calls twice
    for the main token: samples, then keeps each draw in the model's buffers."""

    def __init__(self, model: GraphRecordingModel, sample: TopKTopPSampler) -> None:
        self.model = model
        self.sample = sample

    def __call__(
        self,
        logits: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
    ) -> torch.Tensor:
        candidate_columns = self.sample(logits, temperature, top_k, top_p)
        recording_buffers = self.model.recording_buffers
        column = self.model.draw_column
        assert recording_buffers is not None and column < 2
        recording_buffers.draws[: candidate_columns.shape[0], column].copy_(
            self.model.audio_decoder.main_token_candidate_ids[candidate_columns]
        )
        self.model.draw_column = column + 1
        return candidate_columns
