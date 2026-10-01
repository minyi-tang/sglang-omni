# SPDX-License-Identifier: Apache-2.0
"""S3-drama SGLang engine builder: the AR stage that samples frames."""

from __future__ import annotations

from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.models.registry import ModelRegistry
from sglang.srt.server_args import ServerArgs
from transformers import AutoConfig

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fishaudio_s3_drama.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_s3_drama.model_runner import S3DramaModelRunner
from sglang_omni.models.fishaudio_s3_drama.quantization import (
    QUANTIZATION_METHOD,
    ExpertFp8Config,
)
from sglang_omni.models.fishaudio_s3_drama.request_builders import (
    S3DramaRequestBuilder,
    S3DramaRequestData,
    S3DramaResultAdapter,
)
from sglang_omni.models.fishaudio_s3_drama.sglang_model import S3DramaForCausalLM
from sglang_omni.scheduling.engine_factory import (
    GenerationDefaults,
    SGLangGenerationEngineBuilder,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor

FISH_OMNI_ARCHITECTURE = "FishOmniModel"
# The tower re-plans this backend's metadata for its first frame.
ATTENTION_BACKEND = "triton"


class S3DramaEngineBuilder(SGLangGenerationEngineBuilder[S3DramaRequestData]):
    """Single-GPU eager engine: no radix, chunked prefill, overlap or CUDA graphs."""

    model_name = "FishAudio S3-drama"

    def __init__(self, *, context_length: int, max_running_requests: int) -> None:
        self.context_length = context_length
        self.max_running_requests = max_running_requests

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        """Register the classes the checkpoint names; the directory is the contract."""
        AutoConfig.register(FishOmniConfig.model_type, FishOmniConfig, exist_ok=True)
        ModelRegistry.models[FISH_OMNI_ARCHITECTURE] = S3DramaForCausalLM
        QUANTIZATION_METHODS[QUANTIZATION_METHOD] = ExpertFp8Config

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            "max_running_requests": self.max_running_requests,
            "disable_cuda_graph": True,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_mixed_chunk": False,
            "chunked_prefill_size": -1,
            "attention_backend": ATTENTION_BACKEND,
            "enable_torch_compile": False,
            "trust_remote_code": False,
            "dtype": dtype,
        }

    def adjust_overrides(self, overrides: dict[str, object]) -> None:
        overrides.setdefault("page_size", 1)

    def validate_before_infrastructure(self, server_args: ServerArgs) -> None:
        attention_backends = {
            server_args.attention_backend,
            server_args.prefill_attention_backend or server_args.attention_backend,
            server_args.decode_attention_backend or server_args.attention_backend,
        }
        parallel_sizes = (server_args.tp_size, server_args.pp_size, server_args.dp_size)
        chunked_prefill_size = server_args.chunked_prefill_size or 0
        if not server_args.disable_radix_cache:
            raise ValueError("S3-drama needs disable_radix_cache=True")
        elif chunked_prefill_size > 0 or server_args.enable_mixed_chunk:
            raise ValueError(
                "S3-drama needs chunked_prefill_size <= 0 and no mixed chunk: a "
                "prompt must reach the tower in one extend"
            )
        elif not (
            server_args.disable_cuda_graph and server_args.disable_overlap_schedule
        ):
            raise ValueError("S3-drama runs eager without overlap scheduling")
        elif server_args.enable_torch_compile:
            raise ValueError("S3-drama does not support torch.compile")
        elif attention_backends != {ATTENTION_BACKEND}:
            raise ValueError(
                f"S3-drama needs the {ATTENTION_BACKEND} attention backend; got "
                f"{sorted(map(str, attention_backends))}"
            )
        elif server_args.page_size != 1 or server_args.kv_cache_dtype != "auto":
            raise ValueError(
                "S3-drama keeps page_size=1 and the KV cache in the model dtype"
            )
        elif (
            parallel_sizes != (1, 1, 1) or server_args.speculative_algorithm is not None
        ):
            raise ValueError(
                f"S3-drama runs on one rank without speculative decoding; got "
                f"tp/pp/dp={parallel_sizes} and "
                f"speculative_algorithm={server_args.speculative_algorithm}"
            )
        else:
            pass

    def make_model_runner(
        self, model_worker: ModelWorker, output_proc: SGLangOutputProcessor
    ) -> S3DramaModelRunner:
        return S3DramaModelRunner(model_worker, output_proc)

    def make_adapters(
        self, model: S3DramaForCausalLM
    ) -> tuple[S3DramaRequestBuilder, S3DramaResultAdapter]:
        return (
            S3DramaRequestBuilder(
                config=model.config, context_length=self.context_length
            ),
            S3DramaResultAdapter(eos_token_id=model.config.eos_token_id),
        )
