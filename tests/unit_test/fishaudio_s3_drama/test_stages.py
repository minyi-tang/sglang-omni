# SPDX-License-Identifier: Apache-2.0
"""S3-drama stage factories on CPU: codec path resolution and what each factory builds."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import torch

from sglang_omni.models.fishaudio_s3_drama import stages
from sglang_omni.models.fishaudio_s3_drama.config import (
    DEFAULT_CODEC_PATH,
    S3DramaVocoderFactoryArgs,
)
from sglang_omni.models.fishaudio_s3_drama.engine_builder import S3DramaEngineBuilder
from sglang_omni.models.fishaudio_s3_drama.speech_request import (
    S3DramaSpeechPreprocessor,
)
from sglang_omni.models.fishaudio_s3_drama.vocoder import S3DramaVocoderScheduler
from tests.unit_test.fishaudio_s3_drama.test_prompt import (
    DEFAULT_SYSTEM_TURN,
    END,
    USER,
    VOICE_TURN,
    build_toy_tokenizer,
    expected_rows,
)
from tests.unit_test.fishaudio_s3_drama.test_speech_request import (
    generation_input_of,
    lowered,
    refused,
    toy_config,
)
from tests.unit_test.fishaudio_s3_drama.test_vocoder import FrameNumberCodec

COMMIT = "1de9996b6be38b745688de084d87a5633f714e4e"


class HubDownloads:
    """Stands in for hf_hub_download and records each call."""

    def __init__(self, local_file: Path) -> None:
        self.local_file = local_file
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, repo_id: str, filename: str, *, revision: str) -> str:
        self.calls.append((repo_id, filename, revision))
        return str(self.local_file)


@pytest.fixture
def hub_downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> HubDownloads:
    downloads = HubDownloads(tmp_path / "snapshot" / "codec.pth")
    monkeypatch.setattr(stages, "hf_hub_download", downloads)
    return downloads


def test_a_local_codec_file_or_directory_resolves_without_the_hub(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")

    assert stages.resolve_codec_directory(str(codec_file)) == str(tmp_path)
    assert stages.resolve_codec_directory(str(tmp_path)) == str(tmp_path)
    assert hub_downloads.calls == []


def test_a_hub_codec_downloads_only_codec_pth_at_its_commit(
    hub_downloads: HubDownloads,
) -> None:
    assert stages.resolve_codec_directory(DEFAULT_CODEC_PATH) == str(
        hub_downloads.local_file.parent
    )
    assert hub_downloads.calls == [("fishaudio/s2-pro", "codec.pth", COMMIT)]


@pytest.mark.parametrize(
    "codec_path",
    [
        pytest.param("fishaudio/s2-pro", id="no-revision"),
        pytest.param("fishaudio/s2-pro@main", id="branch"),
        pytest.param("fishaudio/s2-pro@v1.0", id="tag"),
        pytest.param(f"fishaudio/s2-pro@{COMMIT[:7]}", id="short-commit"),
        pytest.param(f"fishaudio/s2-pro@{COMMIT.upper()}", id="upper-case-commit"),
        pytest.param("missing/codec.pth", id="missing-file"),
    ],
)
def test_a_codec_path_that_could_drift_or_does_not_exist_is_refused(
    hub_downloads: HubDownloads, codec_path: str
) -> None:
    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(codec_path)
    assert hub_downloads.calls == []


def test_a_codec_file_of_another_name_is_refused(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    renamed_codec_file = tmp_path / "renamed-codec.pth"
    renamed_codec_file.write_bytes(b"")

    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(str(renamed_codec_file))


def test_a_directory_without_codec_pth_is_refused(
    hub_downloads: HubDownloads, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="names no codec.pth file"):
        stages.resolve_codec_directory(str(tmp_path))


class CodecLoads:
    """Stands in for load_codec and records each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def __call__(self, checkpoint_dir: str, device: str) -> FrameNumberCodec:
        self.calls.append((checkpoint_dir, device))
        return FrameNumberCodec()


@pytest.fixture
def codec_loads(monkeypatch: pytest.MonkeyPatch) -> CodecLoads:
    loads = CodecLoads()
    monkeypatch.setattr(stages, "load_codec", loads)
    return loads


def test_preprocessing_reads_the_checkpoint_and_loads_the_codec_on_cpu(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    tokenizer = build_toy_tokenizer()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))
    toy_config(tokenizer).save_pretrained(str(checkpoint_dir))
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")
    rows = expected_rows(
        tokenizer, DEFAULT_SYSTEM_TURN + [USER, "<|speaker:0|> hello", END, VOICE_TURN]
    )

    scheduler = stages.create_preprocessing_executor(
        str(checkpoint_dir),
        max_concurrency=2,
        max_prompt_rows=rows.shape[0],
        codec_path=str(codec_file),
    )

    preprocessor = scheduler.fn
    assert isinstance(preprocessor, S3DramaSpeechPreprocessor)
    assert scheduler.max_concurrency == 2
    assert codec_loads.calls == [(str(tmp_path), "cpu")]
    assert torch.equal(
        generation_input_of(preprocessor(lowered({"input": "hello"}))).prompt_rows,
        rows,
    )
    assert "maximum allowed length" in refused(
        preprocessor, lowered({"input": "hello there"})
    )


def test_the_engine_factory_hands_its_knobs_to_the_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(
        builder: S3DramaEngineBuilder,
        model_path: str,
        *,
        device: str | None,
        gpu_id: int | None,
        server_args_overrides: Mapping[str, object] | None,
    ) -> str:
        assert (model_path, device, gpu_id, server_args_overrides) == (
            "checkpoint",
            None,
            1,
            {"mem_fraction_static": 0.5},
        )
        assert (
            builder.context_length,
            builder.max_running_requests,
            builder.max_prompt_rows,
            builder.max_new_tokens_cap,
        ) == (4096, 4, 1024, 512)
        return "engine"

    monkeypatch.setattr(S3DramaEngineBuilder, "build", build)

    engine = stages.create_tts_engine_executor(
        "checkpoint",
        gpu_id=1,
        context_length=4096,
        max_running_requests=4,
        max_new_tokens=512,
        max_prompt_rows=1024,
        server_args_overrides={"mem_fraction_static": 0.5},
    )

    assert engine == "engine"


def test_the_vocoder_factory_loads_the_codec_on_its_device_with_the_windows(
    codec_loads: CodecLoads, tmp_path: Path
) -> None:
    codec_file = tmp_path / "codec.pth"
    codec_file.write_bytes(b"")
    window_kwargs = S3DramaVocoderFactoryArgs(codec_path=str(codec_file)).model_dump(
        include={
            "codec_path",
            "stream_stride",
            "stream_followup_stride_ramp",
            "stream_followup_stride",
            "stream_overlap_tokens",
            "stream_crossfade_samples",
            "max_batch_size",
            "max_batch_wait_ms",
        }
    )

    vocoder = stages.create_vocoder_executor(device="cpu", **window_kwargs)

    assert isinstance(vocoder, S3DramaVocoderScheduler)
    assert codec_loads.calls == [(str(tmp_path), "cpu")]
    assert (
        vocoder.stream_stride,
        vocoder.stream_followup_stride_ramp,
        vocoder.stream_followup_stride,
        vocoder.stream_overlap_tokens,
        vocoder.stream_crossfade_samples,
        vocoder.max_batch_size,
    ) == (10, (10, 12, 32), 64, 32, 0, 8)
