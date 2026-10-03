# SPDX-License-Identifier: Apache-2.0
"""The S3-drama pipeline served over HTTP from a synthetic checkpoint; its engine never
samples end-of-speech, so every request generates exactly max_new_tokens frames."""

from __future__ import annotations

import argparse
from collections.abc import Mapping

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.registry import ModelRegistry
from torch import nn

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import StageConfig
from sglang_omni.models.fishaudio_s3_drama.config import (
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_NEW_TOKENS_CAP,
    DEFAULT_MAX_PROMPT_ROWS,
    DEFAULT_MAX_RUNNING_REQUESTS,
    S3DramaPipelineConfig,
)
from sglang_omni.models.fishaudio_s3_drama.engine_builder import (
    FISH_OMNI_ARCHITECTURE,
    S3DramaEngineBuilder,
)
from sglang_omni.models.fishaudio_s3_drama.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_s3_drama.request_builders import S3DramaRequestData
from sglang_omni.models.fishaudio_s3_drama.sglang_model import S3DramaForCausalLM
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.serve.launcher import launch_server

MODULE = "tests.unit_test.fishaudio_s3_drama.http_server"
MODEL_NAME = "s3-drama"
# The synthetic model needs small KV pools, and the vocoder shares the GPU.
ENGINE_MEMORY_FRACTION = "0.3"


class EosMaskedLogits(nn.Module):
    """Wraps the logits processor so end-of-speech never wins a sample."""

    def __init__(self, logits_processor: LogitsProcessor, *, eos_token_id: int) -> None:
        super().__init__()
        self.logits_processor = logits_processor
        self.eos_token_id = eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        output = self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)
        output.next_token_logits[:, self.eos_token_id] = float("-inf")
        return output


class EosMaskedS3DramaModel(S3DramaForCausalLM):
    def __init__(
        self,
        config: FishOmniConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.logits_processor = EosMaskedLogits(
            self.logits_processor, eos_token_id=config.eos_token_id
        )


class EosMaskedEngineBuilder(S3DramaEngineBuilder):
    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        super().pre_infra_setup(checkpoint_dir)
        ModelRegistry.models[FISH_OMNI_ARCHITECTURE] = EosMaskedS3DramaModel


def create_eos_masked_tts_engine_executor(
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
    return EosMaskedEngineBuilder(
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


class EosMaskedPipelineConfig(S3DramaPipelineConfig):
    """The S3-drama pipeline with its engine built by the factory above."""

    stages: list[StageConfig] = [
        (
            stage.model_copy(
                update={
                    "factory_path": f"{MODULE}.create_eos_masked_tts_engine_executor"
                }
            )
            if stage.name == "tts_engine"
            else stage
        )
        for stage in S3DramaPipelineConfig.model_fields["stages"].default
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    arguments = parser.parse_args()
    config = ConfigManager(
        EosMaskedPipelineConfig(model_path=arguments.checkpoint)
    ).merge_config([("tts_engine.engine.mem_fraction_static", ENGINE_MEMORY_FRACTION)])
    launch_server(config, host="127.0.0.1", port=arguments.port, model_name=MODEL_NAME)
