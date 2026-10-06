# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration for FishAudio S3-drama TTS."""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import Field, PositiveInt

from sglang_omni.config.schema import (
    EngineStageConfig,
    FactoryArgs,
    PipelineConfig,
    StageConfig,
)

PACKAGE = "sglang_omni.models.fishaudio_s3_drama"
DEFAULT_CONTEXT_LENGTH = 16384
DEFAULT_MAX_RUNNING_REQUESTS = 16
# A request that sets no max_new_tokens fills the remaining context up to this.
DEFAULT_MAX_NEW_TOKENS_CAP = 2048
# A prompt with more rows is refused before admission (HTTP 400).
DEFAULT_MAX_PROMPT_ROWS = 8192
# The codec of the public S2-Pro release, pinned to an immutable commit.
DEFAULT_CODEC_PATH = "fishaudio/s2-pro@1de9996b6be38b745688de084d87a5633f714e4e"
PREPROCESSING_TOTAL_RESERVE_BYTES = "6GiB"
TTS_ENGINE_GPU_MEMORY_FRACTION = 0.70
VOCODER_GPU_MEMORY_FRACTION = 0.10


class S3DramaPreprocessingFactoryArgs(FactoryArgs):
    max_concurrency: int = Field(default=8, ge=1)
    reference_encode_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    max_reference_audio_seconds: float = Field(default=60.0, gt=0)


class S3DramaPreprocessingStageConfig(StageConfig):
    factory: S3DramaPreprocessingFactoryArgs = Field(
        default_factory=S3DramaPreprocessingFactoryArgs
    )


class S3DramaEngineFactoryArgs(FactoryArgs):
    """Engine knobs; max_new_tokens caps a request that sets none, and preprocessing
    refuses prompts by the same max_prompt_rows."""

    context_length: int = Field(default=DEFAULT_CONTEXT_LENGTH, gt=0)
    max_running_requests: int = Field(default=DEFAULT_MAX_RUNNING_REQUESTS, ge=1)
    max_new_tokens: int = Field(default=DEFAULT_MAX_NEW_TOKENS_CAP, gt=0)
    max_prompt_rows: int = Field(default=DEFAULT_MAX_PROMPT_ROWS, ge=2)


class S3DramaEngineStageConfig(EngineStageConfig):
    factory: S3DramaEngineFactoryArgs = Field(default_factory=S3DramaEngineFactoryArgs)


class S3DramaVocoderFactoryArgs(FactoryArgs):
    """The codec preprocessing also encodes with, and the streaming windows in frames:
    chunks of 10, 10, 12 and 32 new frames, then 64 each, decoded with up to 32
    emitted frames of left context and no crossfade."""

    codec_path: str = DEFAULT_CODEC_PATH
    stream_stride: int = Field(default=10, ge=1)
    stream_followup_stride_ramp: tuple[PositiveInt, ...] = (10, 12, 32)
    stream_followup_stride: int = Field(default=64, ge=1)
    stream_overlap_tokens: int = Field(default=32, ge=0)
    stream_crossfade_samples: int = Field(default=0, ge=0)
    max_batch_size: int = Field(default=8, ge=1)
    max_batch_wait_ms: int = Field(default=2, ge=0)
    window_cuda_graphs: bool = True
    decode_dtype: Literal["float32", "float16"] = "float16"


class S3DramaVocoderStageConfig(StageConfig):
    factory: S3DramaVocoderFactoryArgs = Field(
        default_factory=S3DramaVocoderFactoryArgs
    )


class S3DramaPipelineConfig(PipelineConfig):
    """preprocessing → tts_engine → vocoder, streaming audio frame by frame."""

    architecture: ClassVar[str] = "FishOmniModel"
    requires_model_capabilities: ClassVar[bool] = True
    supported_speech_fields: ClassVar[frozenset[str] | None] = frozenset(
        {
            "model",
            "input",
            "voice",
            "response_format",
            "speed",
            "stream",
            "stream_format",
            "references",
            "ref_audio",
            "ref_text",
            "task_type",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "stream_audio",
            "split_granularity",
        }
    )
    stage_config_types: ClassVar[dict[str, type[StageConfig]]] = {
        "preprocessing": S3DramaPreprocessingStageConfig,
        "tts_engine": S3DramaEngineStageConfig,
        "vocoder": S3DramaVocoderStageConfig,
    }

    model_path: str
    stages: list[StageConfig] = [
        S3DramaPreprocessingStageConfig(
            name="preprocessing",
            process="preprocessing",
            factory_path=f"{PACKAGE}.stages.create_preprocessing_executor",
            gpu=0,
            total_reserve_bytes=PREPROCESSING_TOTAL_RESERVE_BYTES,
            next="tts_engine",
        ),
        S3DramaEngineStageConfig(
            name="tts_engine",
            process="pipeline",
            factory_path=f"{PACKAGE}.stages.create_tts_engine_executor",
            gpu=0,
            gpu_memory_fraction=TTS_ENGINE_GPU_MEMORY_FRACTION,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        S3DramaVocoderStageConfig(
            name="vocoder",
            process="vocoder",
            factory_path=f"{PACKAGE}.stages.create_vocoder_executor",
            gpu=0,
            gpu_memory_fraction=VOCODER_GPU_MEMORY_FRACTION,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]

    def model_post_init(self, __context: object = None) -> None:
        super().model_post_init(__context)
        preprocessing_extra = (
            self.stage_named("preprocessing").factory.model_extra or {}
        )
        for field_name, owner in (
            ("max_prompt_rows", "tts_engine"),
            ("codec_path", "vocoder"),
        ):
            if field_name in preprocessing_extra:
                raise ValueError(
                    f"S3-drama preprocessing takes {field_name} from the {owner} "
                    f"stage; set stages.{owner}.factory.{field_name} instead"
                )
            else:
                pass

    def stage_factory_kwargs(self, stage_name: str) -> dict[str, int | str]:
        if stage_name == "preprocessing":
            engine_factory = self.stage_named("tts_engine").factory
            vocoder_factory = self.stage_named("vocoder").factory
            assert isinstance(engine_factory, S3DramaEngineFactoryArgs)
            assert isinstance(vocoder_factory, S3DramaVocoderFactoryArgs)
            return {
                "max_prompt_rows": engine_factory.max_prompt_rows,
                "codec_path": vocoder_factory.codec_path,
            }
        else:
            return {}

    def supports_uploaded_voice_references(self) -> bool:
        return True


EntryClass = S3DramaPipelineConfig
