# SPDX-License-Identifier: Apache-2.0
"""S3-drama vocoder: Fish codec decoding of sampled frames, streamed in fixed windows."""

from __future__ import annotations

import bisect

import torch

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import DAC
from sglang_omni.models.fishaudio_s2_pro.streaming_vocoder import (
    S2ProFinalAudioData,
    S2ProVocoderScheduler,
    StreamVocoderState,
    WindowDecoder,
    build_stream_vocoder_chunk,
)
from sglang_omni.models.fishaudio_s3_drama.request_builders import (
    S3DramaGenerationResult,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.device_graph import ReplayableGraph
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage


def generation_result_of(payload: StagePayload) -> S3DramaGenerationResult:
    """The engine's result in a vocoder payload, which must hold an audio frame."""
    result = payload.data
    if not isinstance(result, S3DramaGenerationResult):
        raise TypeError(
            f"S3-drama vocoder payloads carry S3DramaGenerationResult; got "
            f"{type(result).__name__}"
        )
    elif result.frames.shape[0] == 0:
        raise ValueError(
            f"Request {payload.request_id}: S3-drama generated no audio frames"
        )
    else:
        return result


class WindowGraphDecoder:
    def __init__(
        self,
        decode: WindowDecoder,
        *,
        num_codebooks: int,
        frame_length: int,
        device: torch.device,
        window_lengths: tuple[int, ...],
    ) -> None:
        graph_backend = current_platform.get_device_graph_backend(device)
        if graph_backend is None:
            raise RuntimeError(
                f"no device graph backend for {device}; set "
                f"stages.vocoder.factory.window_cuda_graphs to false to decode eagerly"
            )
        else:
            pass
        self.decode = decode
        self.frame_length = frame_length
        self.window_lengths = tuple(sorted(window_lengths))
        self.static_indices: dict[int, torch.Tensor] = {}
        self.static_audio: dict[int, torch.Tensor] = {}
        self.graphs: dict[int, ReplayableGraph] = {}
        device_module = torch.get_device_module(device)
        current_stream = device_module.current_stream(device)
        capture_stream = device_module.Stream(device=device)
        capture_stream.wait_stream(current_stream)
        with torch.no_grad(), device_module.device(device):
            with device_module.stream(capture_stream):
                for window_length in reversed(self.window_lengths):
                    decode(
                        torch.zeros(
                            (1, num_codebooks, window_length),
                            dtype=torch.long,
                            device=device,
                        )
                    )
            capture_stream.synchronize()
            graph_pool = device_module.graph_pool_handle()
            for window_length in reversed(self.window_lengths):
                static_indices = torch.zeros(
                    (1, num_codebooks, window_length), dtype=torch.long, device=device
                )
                with graph_backend.capture(
                    pool=graph_pool, stream=capture_stream
                ) as graph:
                    self.static_audio[window_length] = decode(static_indices)
                self.static_indices[window_length] = static_indices
                self.graphs[window_length] = graph
        current_stream.wait_stream(capture_stream)
        device_module.synchronize(device)

    def __call__(self, indices: torch.Tensor) -> torch.Tensor:
        frame_count = int(indices.shape[-1])
        if indices.shape[0] != 1 or frame_count > self.window_lengths[-1]:
            return self.decode(indices)
        else:
            pass
        window_length = self.window_lengths[
            bisect.bisect_left(self.window_lengths, frame_count)
        ]
        static_indices = self.static_indices[window_length]
        static_indices[..., :frame_count].copy_(indices)
        static_indices[..., frame_count:].zero_()
        self.graphs[window_length].replay()
        return self.static_audio[window_length][
            ..., : frame_count * self.frame_length
        ].clone()


class S3DramaVocoderScheduler(S2ProVocoderScheduler):
    """Feeds S2-Pro's windows one frame at a time, so chunk boundaries do not
    depend on how many frames a stream chunk or the final payload carries."""

    def __init__(
        self,
        codec: DAC,
        *,
        device: str,
        stream_stride: int,
        stream_followup_stride_ramp: tuple[int, ...],
        stream_followup_stride: int,
        stream_overlap_tokens: int,
        stream_crossfade_samples: int,
        max_batch_size: int,
        max_batch_wait_ms: int,
        decode_window: WindowDecoder | None = None,
    ) -> None:
        if not all(stride > 0 for stride in stream_followup_stride_ramp):
            raise ValueError(
                f"stream_followup_stride_ramp entries must be > 0; got "
                f"{stream_followup_stride_ramp}"
            )
        else:
            pass
        super().__init__(
            codec,
            device=device,
            stream_stride=stream_stride,
            stream_followup_stride=stream_followup_stride,
            stream_overlap_tokens=stream_overlap_tokens,
            stream_crossfade_samples=stream_crossfade_samples,
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            decode_window=decode_window,
        )
        self.stream_followup_stride_ramp = tuple(stream_followup_stride_ramp)
        self.stream_window_counts: dict[str, int] = {}

    def followup_stride(self, request_id: str) -> int:
        window_count = self.stream_window_counts.get(request_id, 0)
        if window_count < len(self.stream_followup_stride_ramp):
            return self.stream_followup_stride_ramp[window_count]
        else:
            return self.stream_followup_stride

    def stream_frames(
        self, request_id: str, frame_codes: torch.Tensor
    ) -> list[OutgoingMessage]:
        """Audio chunks for frames [1 + num_codebooks, frames] given in order."""
        state = self.stream_states.setdefault(request_id, StreamVocoderState())
        messages: list[OutgoingMessage] = []
        for frame_index in range(frame_codes.shape[1]):
            is_window_end = state.total_tokens + 1 >= (
                state.next_vocode_tokens or self.stream_stride
            )
            audio_chunk = build_stream_vocoder_chunk(
                state,
                frame_codes[:, frame_index : frame_index + 1],
                codec=self.codec,
                device=self.device,
                stream_stride=self.stream_stride,
                stream_followup_stride=self.followup_stride(request_id),
                stream_overlap_tokens=self.stream_overlap_tokens,
                stream_crossfade_samples=self.stream_crossfade_samples,
                decode_window=self.decode_window,
            )
            if is_window_end:
                self.stream_window_counts[request_id] = (
                    self.stream_window_counts.get(request_id, 0) + 1
                )
            else:
                pass
            if audio_chunk is not None:
                messages.append(
                    OutgoingMessage(
                        request_id=request_id,
                        type="stream",
                        data=audio_chunk,
                        metadata={"modality": "audio"},
                    )
                )
            else:
                pass
        return messages

    def on_stream_chunk(
        self, request_id: str, chunk: StreamItem
    ) -> list[OutgoingMessage]:
        frame_codes = chunk.data
        if not isinstance(frame_codes, torch.Tensor):
            raise TypeError(
                f"S3-drama stream chunk for {request_id!r} must carry a torch.Tensor, "
                f"got {type(frame_codes).__name__}"
            )
        else:
            return self.stream_frames(request_id, frame_codes)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        """Streams the frames no chunk carried, counted from the streamed ones so none
        replays, then flushes the tail and the final result."""
        frames = generation_result_of(self.payloads[request_id]).frames
        streamed_frame_count = self.stream_states.setdefault(
            request_id, StreamVocoderState()
        ).total_tokens
        if streamed_frame_count > frames.shape[0]:
            raise RuntimeError(
                f"Request {request_id}: {streamed_frame_count} frames streamed but "
                f"the result holds {frames.shape[0]}"
            )
        else:
            pass
        return self.stream_frames(
            request_id, frames[streamed_frame_count:].T
        ) + super().on_stream_done(request_id)

    def validate_non_streaming_payload(self, payload: StagePayload) -> None:
        generation_result_of(payload)

    def clear_stream_state(self, request_id: str) -> None:
        super().clear_stream_state(request_id)
        self.stream_window_counts.pop(request_id, None)

    def final_result_data(self, payload: StagePayload) -> S2ProFinalAudioData:
        result = generation_result_of(payload)
        completion_tokens = int(result.frames.shape[0])
        final_data: S2ProFinalAudioData = {
            "modality": "audio",
            "sample_rate": int(self.codec.sample_rate),
            "usage": {
                "prompt_tokens": result.prompt_row_count,
                "completion_tokens": completion_tokens,
                "total_tokens": result.prompt_row_count + completion_tokens,
            },
        }
        if result.finish_reason is not None:
            final_data["finish_reason"] = result.finish_reason
        else:
            pass
        return final_data

    def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        audios = self.decode_codebook_codes(
            [generation_result_of(payload).frames[:, 1:].T for payload in payloads]
        )
        return [
            StagePayload(
                request_id=payload.request_id,
                request=payload.request,
                data={**self.final_result_data(payload), "audio_data": audio.tolist()},
            )
            for payload, audio in zip(payloads, audios, strict=True)
        ]
