# SPDX-License-Identifier: Apache-2.0
"""S3-drama pipeline configuration: topology, launch kwargs and the speech allowlist."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.fishaudio_s3_drama.config import (
    DEFAULT_CODEC_PATH,
    S3DramaEngineStageConfig,
    S3DramaPipelineConfig,
    S3DramaVocoderStageConfig,
)
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.serve.speech_service import SpeechRequestValidator

REPO_ROOT = Path(__file__).resolve().parents[3]
VOCODER_WINDOWS = {
    "stream_stride": 10,
    "stream_followup_stride_ramp": (10, 12, 32),
    "stream_followup_stride": 64,
    "stream_overlap_tokens": 32,
    "stream_crossfade_samples": 0,
    "max_batch_size": 8,
    "max_batch_wait_ms": 2,
}


def launch_kwargs(config: S3DramaPipelineConfig, stage_name: str) -> dict[str, object]:
    """What the launcher passes the stage's factory."""
    return resolve_stage_factory_args(config.stage_named(stage_name), config)


def test_frames_stream_from_the_engine_to_the_vocoder_on_one_gpu() -> None:
    config = S3DramaPipelineConfig(model_path="checkpoint")
    preprocessing, tts_engine, vocoder = config.stages

    assert (
        preprocessing.name,
        preprocessing.process,
        preprocessing.gpu,
        preprocessing.next,
    ) == ("preprocessing", "preprocessing", None, "tts_engine")
    assert isinstance(tts_engine, S3DramaEngineStageConfig)
    assert (
        tts_engine.name,
        tts_engine.process,
        tts_engine.gpu,
        tts_engine.next,
        tts_engine.stream_to,
    ) == ("tts_engine", "pipeline", 0, "vocoder", ["vocoder"])
    assert isinstance(vocoder, S3DramaVocoderStageConfig)
    assert (
        vocoder.name,
        vocoder.process,
        vocoder.gpu,
        vocoder.terminal,
        vocoder.can_accept_stream_before_payload,
    ) == ("vocoder", "pipeline", 0, True, True)
    assert config.supports_uploaded_voice_references()


def test_each_factory_gets_the_defaults_written_once() -> None:
    config = S3DramaPipelineConfig(model_path="checkpoint")

    assert launch_kwargs(config, "preprocessing") == {
        "model_path": "checkpoint",
        "max_concurrency": 8,
        "max_prompt_rows": 8192,
        "codec_path": DEFAULT_CODEC_PATH,
    }
    assert launch_kwargs(config, "tts_engine") == {
        "model_path": "checkpoint",
        "gpu_id": 0,
        "context_length": 16384,
        "max_running_requests": 16,
        "max_new_tokens": 2048,
        "max_prompt_rows": 8192,
    }
    assert launch_kwargs(config, "vocoder") == {
        "gpu_id": 0,
        "codec_path": DEFAULT_CODEC_PATH,
        **VOCODER_WINDOWS,
    }


def test_preprocessing_follows_the_engine_cap_and_the_vocoder_codec() -> None:
    config = ConfigManager(S3DramaPipelineConfig(model_path="checkpoint")).merge_config(
        [
            ("tts_engine.factory.max_prompt_rows", "4096"),
            ("vocoder.factory.codec_path", "local/codec.pth"),
        ]
    )
    assert isinstance(config, S3DramaPipelineConfig)

    assert launch_kwargs(config, "preprocessing") == {
        "model_path": "checkpoint",
        "max_concurrency": 8,
        "max_prompt_rows": 4096,
        "codec_path": "local/codec.pth",
    }
    assert launch_kwargs(config, "tts_engine")["max_prompt_rows"] == 4096
    assert launch_kwargs(config, "vocoder")["codec_path"] == "local/codec.pth"


@pytest.mark.parametrize(
    ("field_name", "value", "owner"),
    [
        pytest.param("max_prompt_rows", "4096", "tts_engine", id="max-prompt-rows"),
        pytest.param("codec_path", "local/codec.pth", "vocoder", id="codec-path"),
    ],
)
def test_preprocessing_cannot_set_what_it_derives(
    field_name: str, value: str, owner: str
) -> None:
    manager = ConfigManager(S3DramaPipelineConfig(model_path="checkpoint"))

    with pytest.raises(ValidationError) as rejection:
        manager.merge_config([(f"preprocessing.factory.{field_name}", value)])

    assert f"set stages.{owner}.factory.{field_name} instead" in str(rejection.value)


def test_the_example_config_loads_with_a_local_checkpoint_placeholder() -> None:
    config = ConfigManager.from_file(
        str(REPO_ROOT / "examples/configs/s3_drama_tts.yaml")
    ).config

    assert isinstance(config, S3DramaPipelineConfig)
    assert config.model_path == "s3-drama-checkpoint"
    assert launch_kwargs(config, "preprocessing")["codec_path"] == DEFAULT_CODEC_PATH


def speech_validator() -> SpeechRequestValidator:
    return SpeechRequestValidator(
        default_model="s3-drama",
        supported_speech_fields=S3DramaPipelineConfig.supported_speech_fields,
    )


def test_every_supported_speech_field_is_accepted() -> None:
    validator = speech_validator()

    validator.parse_raw_request(
        {
            "model": "s3-drama",
            "input": "hello",
            "voice": "default",
            "response_format": "pcm",
            "speed": 1.0,
            "stream": True,
            "stream_format": "sse",
            "references": [{"text": "take", "vq_codes": [[0]] * 10}],
            "ref_audio": "data:audio/wav;base64,AAAA",
            "ref_text": "take",
            "task_type": "Base",
            "max_new_tokens": 16,
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 30,
            "repetition_penalty": 1.0,
        }
    )
    validator.parse_raw_request({"input": "hello", "speaker": "default"})


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("instructions", "whisper"),
        ("language", "English"),
        ("seed", 0),
        ("stage_params", {}),
        ("x_vector_only_mode", False),
        ("stream_codec_output", False),
        ("suppress_bootstrap_silence", False),
        ("token_count", 1),
        ("duration_tokens", 1),
        ("initial_codec_chunk_frames", 0),
        ("stage_sampling", {}),
        ("infill", None),
    ],
)
def test_speech_fields_the_model_cannot_honor_are_refused(
    field_name: str, value: object
) -> None:
    with pytest.raises(SpeechAPIError) as rejection:
        speech_validator().parse_raw_request({"input": "hello", field_name: value})

    assert rejection.value.param == field_name
