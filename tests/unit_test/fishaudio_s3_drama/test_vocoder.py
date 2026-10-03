# SPDX-License-Identifier: Apache-2.0
"""S3-drama vocoder windows on CPU with a codec that decodes a frame to its number."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.fishaudio_s3_drama.config import S3DramaVocoderFactoryArgs
from sglang_omni.models.fishaudio_s3_drama.request_builders import (
    S3DramaGenerationResult,
)
from sglang_omni.models.fishaudio_s3_drama.vocoder import S3DramaVocoderScheduler
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import OutgoingMessage

SAMPLES_PER_FRAME = 4
SAMPLE_RATE = 44100
NUM_CODEBOOKS = 10
PROMPT_ROW_COUNT = 37
# [first, end) frame numbers, one pair per chunk or decode.
FrameRanges = list[tuple[int, int]]


class FrameNumberCodec:
    """Decodes every frame to SAMPLES_PER_FRAME samples equal to its first code, which
    the tests set to the frame number, and records the codes of each decode."""

    sample_rate = SAMPLE_RATE
    frame_length = SAMPLES_PER_FRAME

    def __init__(self) -> None:
        self.decoded_codes: list[torch.Tensor] = []

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.decoded_codes.append(indices.clone())
        return indices[:, :1, :].float().repeat_interleave(SAMPLES_PER_FRAME, dim=-1)

    def decoded_frame_ranges(self) -> FrameRanges:
        """The [first, end) frame numbers of each decode of one request."""
        return [
            (int(codes[0, 0, 0]), int(codes[0, 0, -1]) + 1)
            for codes in self.decoded_codes
        ]


def numbered_frames(frame_count: int) -> torch.Tensor:
    """[frames, 1 + num_codebooks] rows whose first code is the frame number."""
    frames = torch.zeros((frame_count, 1 + NUM_CODEBOOKS), dtype=torch.int64)
    frames[:, 1] = torch.arange(frame_count)
    frames[:, 0] = 5000 + frames[:, 1]
    return frames


def vocoder_with(codec: FrameNumberCodec) -> S3DramaVocoderScheduler:
    """The vocoder with the configured windows."""
    window_kwargs = S3DramaVocoderFactoryArgs().model_dump(
        include={
            "stream_stride",
            "stream_followup_stride_ramp",
            "stream_followup_stride",
            "stream_overlap_tokens",
            "stream_crossfade_samples",
            "max_batch_size",
            "max_batch_wait_ms",
        }
    )
    return S3DramaVocoderScheduler(codec, device="cpu", **window_kwargs)


def final_payload(
    frames: torch.Tensor, *, is_streaming: bool, finish_reason: str = "length"
) -> StagePayload:
    return StagePayload(
        request_id="request-a",
        request=OmniRequest(inputs="hello", params={"stream": is_streaming}),
        data=S3DramaGenerationResult(
            frames=frames,
            prompt_row_count=PROMPT_ROW_COUNT,
            finish_reason=finish_reason,
        ),
    )


def drained(vocoder: S3DramaVocoderScheduler) -> list[OutgoingMessage]:
    messages = []
    while not vocoder.outbox.empty():
        messages.append(vocoder.outbox.get_nowait())
    return messages


def chunk_frame_ranges(messages: list[OutgoingMessage]) -> FrameRanges:
    """The [first, end) frame numbers each streamed audio chunk covers, in order."""
    ranges = []
    for message in messages:
        if message.type == "stream":
            assert isinstance(message.data, dict)
            samples = np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            assert samples.size % SAMPLES_PER_FRAME == 0
            ranges.append((int(samples[0]), int(samples[-1]) + 1))
        else:
            pass
    return ranges


def streamed_audio(messages: list[OutgoingMessage]) -> np.ndarray:
    return np.concatenate(
        [
            np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
            for message in messages
            if message.type == "stream" and isinstance(message.data, dict)
        ]
    )


def streamed(
    total_frame_count: int, chunk_frame_counts: list[int], *, is_done_first: bool
) -> tuple[FrameNumberCodec, list[OutgoingMessage]]:
    """Stream chunks of the given sizes, then hand over the final payload with all
    total_frame_count frames and signal the end of the stream."""
    codec = FrameNumberCodec()
    vocoder = vocoder_with(codec)
    frames = numbered_frames(total_frame_count)
    start = 0
    for chunk_id, frame_count in enumerate(chunk_frame_counts):
        vocoder.handle_stream_chunk(
            "request-a",
            StreamItem(
                chunk_id=chunk_id,
                data=frames[start : start + frame_count].T,
                from_stage="tts_engine",
            ),
        )
        start += frame_count
    payload = final_payload(frames, is_streaming=True)
    if is_done_first:
        vocoder.handle_stream_done("request-a")
        vocoder.handle_streaming_new_request("request-a", payload)
    else:
        vocoder.handle_streaming_new_request("request-a", payload)
        vocoder.handle_stream_done("request-a")
    return codec, drained(vocoder)


WINDOWS_TO_214 = [(0, 10), (10, 20), (20, 32), (32, 64), (64, 128), (128, 192)]
WINDOWS_TO_300 = WINDOWS_TO_214 + [(192, 256)]
STREAMING_CASES = [
    pytest.param(
        214,
        [1] * 214,
        WINDOWS_TO_214 + [(192, 214)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128), (96, 192), (160, 214)],
        id="frame-by-frame",
    ),
    pytest.param(
        214,
        [1] * 5 + [150] + [1] * 59,
        WINDOWS_TO_214 + [(192, 214)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128), (96, 192), (160, 214)],
        id="a-150-frame-chunk-midstream",
    ),
    pytest.param(
        300,
        [1] * 5 + [295],
        WINDOWS_TO_300 + [(256, 300)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128), (96, 192), (160, 256)]
        + [(224, 300)],
        id="a-295-frame-last-chunk",
    ),
    pytest.param(
        300,
        [1] * 5,
        WINDOWS_TO_300 + [(256, 300)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128), (96, 192), (160, 256)]
        + [(224, 300)],
        id="295-frames-only-in-the-final-payload",
    ),
    pytest.param(
        128,
        [1] * 128,
        [(0, 10), (10, 20), (20, 32), (32, 64), (64, 128)],
        [(0, 10), (0, 20), (0, 32), (0, 64), (32, 128)],
        id="whole-windows-and-no-tail",
    ),
    pytest.param(7, [1] * 7, [(0, 7)], [(0, 7)], id="only-a-tail"),
]


@pytest.mark.parametrize(
    "is_done_first", [False, True], ids=["payload-first", "done-first"]
)
@pytest.mark.parametrize(
    ("total_frame_count", "chunk_frame_counts", "chunk_ranges", "decoded_ranges"),
    STREAMING_CASES,
)
def test_streaming_emits_the_fixed_windows_however_frames_arrive(
    total_frame_count: int,
    chunk_frame_counts: list[int],
    chunk_ranges: FrameRanges,
    decoded_ranges: FrameRanges,
    is_done_first: bool,
) -> None:
    codec, messages = streamed(
        total_frame_count, chunk_frame_counts, is_done_first=is_done_first
    )

    assert chunk_frame_ranges(messages) == chunk_ranges
    assert codec.decoded_frame_ranges() == decoded_ranges
    assert max(end - start for start, end in decoded_ranges) <= 96
    # No crossfade and no held-back tail: the chunks are each frame once, in order.
    np.testing.assert_array_equal(
        streamed_audio(messages),
        np.repeat(np.arange(total_frame_count, dtype=np.float32), SAMPLES_PER_FRAME),
    )
    assert [message.type for message in messages][-1] == "result"


def test_the_final_result_counts_prompt_rows_and_audio_frames() -> None:
    _, messages = streamed(20, [1] * 20, is_done_first=False)

    result = messages[-1].data
    assert isinstance(result, StagePayload)
    assert result.data == {
        "modality": "audio",
        "sample_rate": SAMPLE_RATE,
        "usage": {
            "prompt_tokens": PROMPT_ROW_COUNT,
            "completion_tokens": 20,
            "total_tokens": PROMPT_ROW_COUNT + 20,
        },
        "finish_reason": "length",
    }


def test_no_audio_frames_fail_without_a_codec_call() -> None:
    codec = FrameNumberCodec()
    vocoder = vocoder_with(codec)
    vocoder.handle_streaming_new_request(
        "request-a", final_payload(numbered_frames(0), is_streaming=True)
    )

    with pytest.raises(ValueError, match="generated no audio frames"):
        vocoder.handle_stream_done("request-a")
    with pytest.raises(ValueError, match="generated no audio frames"):
        vocoder.validate_non_streaming_payload(
            final_payload(numbered_frames(0), is_streaming=False)
        )
    assert codec.decoded_codes == []


def test_more_streamed_frames_than_the_result_holds_is_refused() -> None:
    vocoder = vocoder_with(FrameNumberCodec())
    vocoder.handle_stream_chunk(
        "request-a",
        StreamItem(chunk_id=0, data=numbered_frames(6).T, from_stage="tts_engine"),
    )
    vocoder.handle_streaming_new_request(
        "request-a", final_payload(numbered_frames(5), is_streaming=True)
    )

    with pytest.raises(RuntimeError, match="6 frames streamed but the result holds 5"):
        vocoder.handle_stream_done("request-a")


def test_a_non_streaming_batch_decodes_once() -> None:
    codec = FrameNumberCodec()
    vocoder = vocoder_with(codec)
    payloads = [
        final_payload(numbered_frames(5), is_streaming=False, finish_reason="stop"),
        final_payload(numbered_frames(7), is_streaming=False, finish_reason="length"),
    ]

    results = vocoder.vocode_payloads(payloads)

    assert len(codec.decoded_codes) == 1
    assert codec.decoded_codes[0].shape == (2, NUM_CODEBOOKS, 7)
    for result, frame_count, finish_reason in zip(
        results, [5, 7], ["stop", "length"], strict=True
    ):
        assert isinstance(result.data, dict)
        assert (
            result.data["audio_data"]
            == np.repeat(
                np.arange(frame_count, dtype=np.float32), SAMPLES_PER_FRAME
            ).tolist()
        )
        assert result.data["usage"] == {
            "prompt_tokens": PROMPT_ROW_COUNT,
            "completion_tokens": frame_count,
            "total_tokens": PROMPT_ROW_COUNT + frame_count,
        }
        assert result.data["finish_reason"] == finish_reason


def test_the_ramp_advances_when_a_crossfade_holds_back_a_whole_window() -> None:
    codec = FrameNumberCodec()
    vocoder = S3DramaVocoderScheduler(
        codec,
        device="cpu",
        stream_stride=10,
        stream_followup_stride_ramp=(10, 12, 32),
        stream_followup_stride=64,
        stream_overlap_tokens=32,
        stream_crossfade_samples=40,
        max_batch_size=8,
        max_batch_wait_ms=2,
    )
    frames = numbered_frames(128)
    for chunk_id in range(128):
        vocoder.handle_stream_chunk(
            "request-a",
            StreamItem(
                chunk_id=chunk_id,
                data=frames[chunk_id : chunk_id + 1].T,
                from_stage="tts_engine",
            ),
        )
    vocoder.handle_streaming_new_request(
        "request-a", final_payload(frames, is_streaming=True)
    )
    vocoder.handle_stream_done("request-a")

    assert [end for _, end in codec.decoded_frame_ranges()] == [10, 20, 32, 64, 128]
    np.testing.assert_allclose(
        streamed_audio(drained(vocoder)),
        np.repeat(np.arange(128, dtype=np.float32), SAMPLES_PER_FRAME),
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "ramp",
    [
        pytest.param((0,), id="zero"),
        pytest.param((-1,), id="negative"),
        pytest.param((2, 0), id="zero-after-positive"),
    ],
)
def test_a_ramp_with_a_non_positive_stride_is_refused(ramp: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="stream_followup_stride_ramp"):
        S3DramaVocoderScheduler(
            FrameNumberCodec(),
            device="cpu",
            stream_stride=10,
            stream_followup_stride_ramp=ramp,
            stream_followup_stride=64,
            stream_overlap_tokens=32,
            stream_crossfade_samples=0,
            max_batch_size=8,
            max_batch_wait_ms=2,
        )
