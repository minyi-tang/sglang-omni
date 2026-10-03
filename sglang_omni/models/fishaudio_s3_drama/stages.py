# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the FishAudio S3-drama pipeline."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from sglang_omni.models.fishaudio_s2_pro.stages import (
    FishReferenceEncodeHook,
    load_codec,
)
from sglang_omni.models.fishaudio_s3_drama.config import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_NEW_TOKENS_CAP,
    DEFAULT_MAX_PROMPT_ROWS,
    DEFAULT_MAX_RUNNING_REQUESTS,
)
from sglang_omni.models.fishaudio_s3_drama.engine_builder import S3DramaEngineBuilder
from sglang_omni.models.fishaudio_s3_drama.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_s3_drama.request_builders import S3DramaRequestData
from sglang_omni.models.fishaudio_s3_drama.speech_request import (
    S3DramaSpeechPreprocessor,
)
from sglang_omni.models.fishaudio_s3_drama.vocoder import S3DramaVocoderScheduler
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

CODEC_FILE_NAME = "codec.pth"
# A hub codec names a full commit, so preprocessing and the vocoder load one file.
HUB_CODEC_PATTERN = re.compile(
    r"(?P<repo_id>[^@\s]+/[^@\s]+)@(?P<revision>[0-9a-f]{40})"
)


def resolve_codec_directory(codec_path: str) -> str:
    """The local directory holding the codec.pth codec_path names: that file, a
    directory holding it, or <repo_id>@<40-hex commit>, of which only codec.pth
    downloads."""
    hub_codec = HUB_CODEC_PATTERN.fullmatch(codec_path)
    if os.path.isfile(codec_path) and os.path.basename(codec_path) == CODEC_FILE_NAME:
        return os.path.dirname(os.path.abspath(codec_path))
    elif os.path.isfile(os.path.join(codec_path, CODEC_FILE_NAME)):
        return codec_path
    elif hub_codec is not None:
        return os.path.dirname(
            hf_hub_download(
                hub_codec["repo_id"], CODEC_FILE_NAME, revision=hub_codec["revision"]
            )
        )
    else:
        raise ValueError(
            f"codec_path {codec_path!r} names no {CODEC_FILE_NAME} file, no "
            f"directory holding one and no <repo_id>@<40-hex commit> on the Hub"
        )


def create_preprocessing_executor(
    model_path: str, *, max_concurrency: int, max_prompt_rows: int, codec_path: str
) -> ThreadedSimpleScheduler:
    """Speech requests to prompt rows; reference audio encodes with the vocoder's codec."""
    checkpoint_dir = resolve_checkpoint(model_path)
    codec_directory = resolve_codec_directory(codec_path)
    preprocessor = S3DramaSpeechPreprocessor(
        tokenizer=Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json")),
        config=FishOmniConfig.from_pretrained(checkpoint_dir),
        max_prompt_rows=max_prompt_rows,
        reference_encoder=ReferenceEncodeService(
            FishReferenceEncodeHook(
                codec=load_codec(codec_directory, "cpu"),
                checkpoint_id=os.path.join(codec_directory, CODEC_FILE_NAME),
            ),
            log_prefix="FishAudio S3-drama",
        ),
    )
    return ThreadedSimpleScheduler(preprocessor, max_concurrency=max_concurrency)


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_CAP,
    max_prompt_rows: int = DEFAULT_MAX_PROMPT_ROWS,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[S3DramaRequestData]:
    """The engine; max_new_tokens caps a request that sets none."""
    return S3DramaEngineBuilder(
        context_length=context_length,
        max_running_requests=max_running_requests,
        max_prompt_rows=max_prompt_rows,
        max_new_tokens_cap=max_new_tokens,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_path: str,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
    max_batch_size: int,
    max_batch_wait_ms: int,
) -> S3DramaVocoderScheduler:
    resolved_device = str(resolve_concrete_device(device, gpu_id))
    return S3DramaVocoderScheduler(
        load_codec(resolve_codec_directory(codec_path), resolved_device),
        device=resolved_device,
        stream_stride=stream_stride,
        stream_followup_stride_ramp=stream_followup_stride_ramp,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
