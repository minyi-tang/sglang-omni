# SPDX-License-Identifier: Apache-2.0
"""Engine for test_mtp_engine, one per process: S3-drama on OmniScheduler, recorded.

The recording model keeps the runner's RAS windows as records but samples with
every window empty, and pins the end-of-speech logit per request, so every
sampled token is a deterministic argmax.
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

import torch
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.registry import ModelRegistry
from torch import nn

from sglang_omni.models.fishaudio_s3_drama.engine_builder import (
    FISH_OMNI_ARCHITECTURE,
    S3DramaEngineBuilder,
)
from sglang_omni.models.fishaudio_s3_drama.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_s3_drama.model_runner import EMPTY_RAS_SLOT
from sglang_omni.models.fishaudio_s3_drama.request_builders import (
    S3DramaGenerationInput,
    S3DramaGenerationResult,
    S3DramaRequestData,
)
from sglang_omni.models.fishaudio_s3_drama.sglang_model import (
    S3DramaForCausalLM,
    S3DramaStepInputs,
)
from sglang_omni.proto.admin import ADMIN_CONTINUE_GENERATION, ADMIN_PAUSE_GENERATION
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.bootstrap import InfrastructureOptions
from sglang_omni.scheduling.engine_factory import GenerationDefaults
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.fishaudio_s3_drama.mtp_fixtures import (
    EngineFacts,
    EngineRun,
    EngineSpec,
    ForwardRecord,
    FreePoolCounts,
    RequestOutcome,
    RequestSpec,
    RequestStep,
    Scenario,
    ScenarioOutcome,
)

SCENARIO_TIMEOUT_SECONDS = 300.0
IDLE_TIMEOUT_SECONDS = 30.0
OUTBOX_POLL_SECONDS = 0.05
# Far above any logit the synthetic heads produce, so greedy sampling takes eos.
FORCED_EOS_LOGIT = 1.0e4
SAMPLING_TEMPERATURE = 1.0

logger = logging.getLogger(__name__)


class EosControlledLogits(nn.Module):
    """Wraps the logits processor: eos only at each request's chosen frame."""

    def __init__(
        self,
        logits_processor: LogitsProcessor,
        *,
        eos_token_id: int,
        request_specs: dict[str, RequestSpec],
        prompt_row_counts: dict[str, int],
    ) -> None:
        super().__init__()
        self.logits_processor = logits_processor
        self.eos_token_id = eos_token_id
        self.request_specs = request_specs
        self.prompt_row_counts = prompt_row_counts
        self.last_logits: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: ParallelLMHead,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        output = self.logits_processor(input_ids, hidden_states, lm_head, forward_batch)
        logits = output.next_token_logits
        sequence_lengths = forward_batch.seq_lens_cpu.tolist()
        for row, (request_id, sequence_length) in enumerate(
            zip(forward_batch.rids, sequence_lengths, strict=True)
        ):
            frame_index = sequence_length - self.prompt_row_counts[request_id]
            if frame_index == self.request_specs[request_id].end_at_frame:
                logits[row, self.eos_token_id] = FORCED_EOS_LOGIT
            else:
                logits[row, self.eos_token_id] = float("-inf")
        self.last_logits = logits.float().cpu()
        return output


class RecordingS3DramaModel(S3DramaForCausalLM):
    """S3DramaForCausalLM that records every forward for the dense reference."""

    def __init__(
        self,
        config: FishOmniConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.request_specs: dict[str, RequestSpec] = {}
        self.prompt_row_counts: dict[str, int] = {}
        self.sampled_frame_counts: dict[str, int] = {}
        self.forwards: list[ForwardRecord] = []
        self.eos_controlled_logits = EosControlledLogits(
            self.logits_processor,
            eos_token_id=config.eos_token_id,
            request_specs=self.request_specs,
            prompt_row_counts=self.prompt_row_counts,
        )
        self.logits_processor = self.eos_controlled_logits
        self.backbone_hidden_states: torch.Tensor | None = None
        self.model.register_forward_hook(self.keep_backbone_hidden_states)
        self.head_logits: list[torch.Tensor] = []
        for head in self.audio_decoder.codebook_heads:
            head.register_forward_hook(self.keep_head_logits)

    def keep_backbone_hidden_states(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        hidden_states: torch.Tensor,
    ) -> None:
        """Forward-hook contract; only the backbone output is kept."""
        self.backbone_hidden_states = hidden_states

    def keep_head_logits(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        logits: torch.Tensor,
    ) -> None:
        """Forward-hook contract; the decode step calls the heads in order."""
        self.head_logits.append(logits.float().cpu())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        staged_inputs = self.step_exchange.staged_inputs
        assert staged_inputs is not None
        # The staged frames carry the batch's tokens as their main tokens.
        assert torch.equal(staged_inputs.frames[:, 0], input_ids)
        self.head_logits.clear()
        self.step_exchange.stage(
            replace(
                staged_inputs,
                ras_windows=torch.full_like(staged_inputs.ras_windows, EMPTY_RAS_SLOT),
            )
        )
        output = super().forward(input_ids, positions, forward_batch)
        self.forwards.append(
            self.forward_record(forward_batch, positions, staged_inputs)
        )
        return output

    def forward_record(
        self,
        forward_batch: ForwardBatch,
        positions: torch.Tensor,
        staged_inputs: S3DramaStepInputs,
    ) -> ForwardRecord:
        attention_backend = get_attn_backend()
        kv_pool = attention_backend.token_to_kv_pool
        assert isinstance(kv_pool, HybridLinearKVPool)
        request_slots = attention_backend.req_to_token_pool.req_to_token
        tower_layer_ids = [layer.layer_id for layer in self.tower_attention]
        hidden_states = self.backbone_hidden_states
        sampled_frames = self.step_exchange.sampled_frames
        logits = self.eos_controlled_logits.last_logits
        assert hidden_states is not None
        assert sampled_frames is not None
        assert logits is not None
        is_extend = forward_batch.forward_mode == ForwardMode.EXTEND
        if is_extend:
            row_starts = forward_batch.extend_start_loc.tolist()
            row_counts = forward_batch.extend_seq_lens.tolist()
        else:
            row_starts = list(range(forward_batch.batch_size))
            row_counts = [1] * forward_batch.batch_size
        sequence_lengths = forward_batch.seq_lens_cpu.tolist()
        pool_indices = forward_batch.req_pool_indices.tolist()
        sampled_rows = sampled_frames.tolist()
        # [requests, heads, codebook size]
        head_logits = torch.stack(self.head_logits, dim=1)
        steps = []
        for request_index, request_id in enumerate(forward_batch.rids):
            rows = slice(
                row_starts[request_index],
                row_starts[request_index] + row_counts[request_index],
            )
            written_slots = forward_batch.out_cache_loc[rows]
            sequence_length = sequence_lengths[request_index]
            frame_index = sequence_length - self.prompt_row_counts[request_id]
            self.sampled_frame_counts[request_id] = frame_index + 1
            sampled_frame = sampled_rows[request_index]
            is_last_step = (
                sampled_frame[0] == self.config.eos_token_id
                or frame_index + 1 == self.request_specs[request_id].max_new_tokens
            )
            sequence_slots = request_slots[
                pool_indices[request_index], :sequence_length
            ]
            steps.append(
                RequestStep(
                    request_id=request_id,
                    is_extend=is_extend,
                    positions=positions[rows].tolist(),
                    hidden_states=hidden_states[rows].cpu(),
                    input_frames=staged_inputs.frames[rows].cpu(),
                    semantic_logits=logits[request_index],
                    head_logits=head_logits[request_index],
                    ras_window=staged_inputs.ras_windows[request_index].tolist(),
                    sampled_frame=sampled_frame,
                    request_row=pool_indices[request_index],
                    written_slots=written_slots.tolist(),
                    sequence_slots=sequence_slots.tolist(),
                    written_cache=tower_cache(kv_pool, tower_layer_ids, written_slots),
                    sequence_cache=(
                        tower_cache(kv_pool, tower_layer_ids, sequence_slots)
                        if is_last_step
                        else None
                    ),
                )
            )
        return ForwardRecord(
            forward_index=len(self.forwards), is_extend=is_extend, steps=steps
        )


def tower_cache(
    kv_pool: HybridLinearKVPool, tower_layer_ids: list[int], slots: torch.Tensor
) -> torch.Tensor:
    """[tower layers, slots, 2 (key, value), kv heads * head_dim] copied off the pool."""
    return torch.stack(
        [
            torch.stack(
                (
                    kv_pool.get_key_buffer(layer_id)[slots].flatten(1),
                    kv_pool.get_value_buffer(layer_id)[slots].flatten(1),
                ),
                dim=1,
            )
            for layer_id in tower_layer_ids
        ]
    ).cpu()


class RecordingEngineBuilder(S3DramaEngineBuilder):
    """The production builder serving RecordingS3DramaModel with the spec's knobs."""

    def __init__(self, *, spec: EngineSpec) -> None:
        super().__init__(
            context_length=spec.context_length,
            max_running_requests=spec.max_running_requests,
        )
        self.spec = spec

    def pre_infra_setup(self, checkpoint_dir: str) -> None:
        super().pre_infra_setup(checkpoint_dir)
        ModelRegistry.models[FISH_OMNI_ARCHITECTURE] = RecordingS3DramaModel

    def generation_defaults(self, *, dtype: str) -> GenerationDefaults:
        return {
            **super().generation_defaults(dtype=dtype),
            "max_prefill_tokens": self.spec.max_prefill_tokens,
            "max_total_tokens": self.spec.max_total_tokens,
        }

    def infra_kwargs(self) -> InfrastructureOptions:
        return {"total_gpu_memory_fraction": self.spec.total_gpu_memory_fraction}


class SchedulerDriver:
    """Feeds OmniScheduler's inbox and files its outbox per request."""

    def __init__(
        self,
        scheduler: OmniScheduler[S3DramaRequestData],
        model: RecordingS3DramaModel,
        prompts: dict[str, list[list[int]]],
    ) -> None:
        self.scheduler = scheduler
        self.model = model
        self.prompts = prompts
        self.outcomes: dict[str, RequestOutcome] = {}
        self.failures: list[BaseException] = []
        # Daemon so a wedged scheduler cannot keep the engine process alive.
        self.thread = threading.Thread(
            target=self.serve, name="omni-scheduler", daemon=True
        )

    def serve(self) -> None:
        try:
            self.scheduler.start()
        except BaseException as error:
            self.failures.append(error)
            raise

    def start(self) -> None:
        self.thread.start()

    def stop_and_drain(self) -> None:
        """Stop the scheduler, then file every message it left in the outbox."""
        self.scheduler.stop()
        self.thread.join(timeout=SCENARIO_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            raise RuntimeError("OmniScheduler thread did not stop")
        elif self.failures:
            raise RuntimeError("OmniScheduler thread failed") from self.failures[0]
        else:
            pass
        while True:
            try:
                message = self.scheduler.outbox.get_nowait()
            except queue.Empty:
                return
            self.record(message)

    def submit(self, request_id: str, request_spec: RequestSpec) -> str:
        prompt_rows = self.prompts[request_spec.prompt_name]
        self.model.request_specs[request_id] = request_spec
        self.model.prompt_row_counts[request_id] = len(prompt_rows)
        self.outcomes[request_id] = RequestOutcome(
            request_id=request_id, request_spec=request_spec
        )
        payload = StagePayload(
            request_id=request_id,
            request=OmniRequest(inputs=request_spec.prompt_name),
            data=S3DramaGenerationInput(
                prompt_rows=torch.tensor(prompt_rows, dtype=torch.int64),
                max_new_tokens=request_spec.max_new_tokens,
                temperature=SAMPLING_TEMPERATURE,
                top_k=1,
                top_p=1.0,
            ),
        )
        self.scheduler.inbox.put(
            IncomingMessage(request_id=request_id, type="new_request", data=payload)
        )
        return request_id

    def abort(self, request_id: str) -> None:
        self.outcomes[request_id].aborted = True
        self.scheduler.inbox.put(IncomingMessage(request_id=request_id, type="abort"))

    def record(self, message: OutgoingMessage) -> None:
        outcome = self.outcomes[message.request_id]
        if message.type == "result":
            payload = message.data
            assert isinstance(payload, StagePayload)
            result = payload.data
            assert isinstance(result, S3DramaGenerationResult)
            outcome.result_frames.append(result.frames.tolist())
        elif message.type == "error":
            outcome.error_messages.append(repr(message.data))
        elif message.type == "stream":
            outcome.stream_message_count += 1
        else:
            raise RuntimeError(f"unexpected outbox message {message.type!r}")

    def pump(self) -> None:
        """File the next outbox message, if one arrives within a poll interval."""
        if self.failures:
            raise RuntimeError("OmniScheduler thread failed") from self.failures[0]
        elif not self.thread.is_alive():
            raise RuntimeError("OmniScheduler thread exited")
        else:
            pass
        try:
            message = self.scheduler.outbox.get(timeout=OUTBOX_POLL_SECONDS)
        except queue.Empty:
            return
        self.record(message)

    def wait_for_results(self, request_ids: list[str]) -> None:
        deadline = time.monotonic() + SCENARIO_TIMEOUT_SECONDS
        while any(
            not self.outcomes[request_id].result_frames
            and not self.outcomes[request_id].error_messages
            for request_id in request_ids
        ):
            if time.monotonic() > deadline:
                raise TimeoutError(f"requests did not finish: {request_ids}")
            else:
                pass
            self.pump()

    def wait_for_frames(self, request_id: str, frame_count: int) -> None:
        deadline = time.monotonic() + SCENARIO_TIMEOUT_SECONDS
        outcome = self.outcomes[request_id]
        while self.model.sampled_frame_counts.get(request_id, 0) < frame_count:
            if outcome.result_frames or outcome.error_messages:
                raise RuntimeError(f"{request_id} ended before {frame_count} frames")
            elif time.monotonic() > deadline:
                raise TimeoutError(f"{request_id} never reached {frame_count} frames")
            else:
                pass
            self.pump()

    def retract_running_requests(self) -> int:
        """Pause with retract, then resume; returns how many requests were retracted."""
        paused = self.scheduler.admin(ADMIN_PAUSE_GENERATION, {"mode": "retract"})
        resumed = self.scheduler.admin(ADMIN_CONTINUE_GENERATION, {})
        paused_data = paused.get("data")
        if not (paused.get("success") and resumed.get("success")):
            raise RuntimeError(f"admin retract failed: {paused} {resumed}")
        elif not isinstance(paused_data, dict):
            raise RuntimeError(f"admin pause returned no data: {paused}")
        else:
            return int(paused_data["num_paused_requests"])

    def free_pool_counts(self) -> FreePoolCounts:
        pool = self.scheduler.req_to_token_pool
        assert isinstance(pool, HybridReqToTokenPool)
        return FreePoolCounts(
            request_row_count=pool.available_size(),
            mamba_slot_count=pool.mamba_allocator.available_size(),
            kv_token_count=self.scheduler.token_to_kv_pool_allocator.available_size(),
        )

    def wait_for_idle(self, baseline: FreePoolCounts) -> FreePoolCounts:
        """Free counts once every slot is back, or the last counts on timeout."""
        deadline = time.monotonic() + IDLE_TIMEOUT_SECONDS
        free_counts = self.free_pool_counts()
        while free_counts != baseline and time.monotonic() < deadline:
            self.pump()
            free_counts = self.free_pool_counts()
        return free_counts


def run_scenario(
    driver: SchedulerDriver, scenario: Scenario, baseline: FreePoolCounts
) -> ScenarioOutcome:
    first_forward_index = len(driver.model.forwards)
    request_ids: list[str] = []
    frames_at_abort = None
    results_at_abort = None
    retracted_request_count = None
    if scenario.abort_target is None:
        abort_id = None
    else:
        abort_id = driver.submit(f"{scenario.name}-target", scenario.abort_target)
        request_ids.append(abort_id)
    for wave_index, wave in enumerate(scenario.waves):
        wave_ids = []
        for position, request_spec in enumerate(wave):
            request_id = driver.submit(
                f"{scenario.name}-{wave_index}-{position}-{request_spec.prompt_name}",
                request_spec,
            )
            wave_ids.append(request_id)
            if scenario.is_sequential:
                driver.wait_for_results([request_id])
            else:
                pass
        request_ids.extend(wave_ids)
        if abort_id is not None and wave_index == 0:
            driver.wait_for_frames(abort_id, scenario.abort_after_frames)
            frames_at_abort = driver.model.sampled_frame_counts[abort_id]
            results_at_abort = len(driver.outcomes[abort_id].result_frames)
            driver.abort(abort_id)
        else:
            pass
        if scenario.retract_after_frames is not None and wave_index == 0:
            driver.wait_for_frames(wave_ids[0], scenario.retract_after_frames)
            retracted_request_count = driver.retract_running_requests()
        else:
            pass
        driver.wait_for_results(wave_ids)
    idle_free_counts = driver.wait_for_idle(baseline)
    logger.info(f"scenario {scenario.name}: free pools after drain {idle_free_counts}")
    return ScenarioOutcome(
        name=scenario.name,
        request_ids=request_ids,
        first_forward_index=first_forward_index,
        end_forward_index=len(driver.model.forwards),
        idle_free_counts=idle_free_counts,
        frames_at_abort=frames_at_abort,
        results_at_abort=results_at_abort,
        retracted_request_count=retracted_request_count,
    )


def run_engine(spec: EngineSpec) -> EngineRun:
    builder = RecordingEngineBuilder(spec=spec)
    scheduler = builder.build(spec.model_dir, dtype="bfloat16")
    sglang_runner = scheduler.tp_worker.model_runner
    model = sglang_runner.model
    assert isinstance(model, RecordingS3DramaModel)
    kv_pool = sglang_runner.token_to_kv_pool
    request_pool = sglang_runner.req_to_token_pool
    assert isinstance(kv_pool, HybridLinearKVPool)
    assert isinstance(request_pool, HybridReqToTokenPool)
    facts = EngineFacts(
        full_attention_layer_ids=sorted(kv_pool.full_attention_layer_id_mapping),
        mamba_layer_ids=sorted(request_pool.mamba_map),
        kv_layer_count=kv_pool.full_layer_nums,
        kv_head_count=kv_pool.head_num,
        kv_head_dim=kv_pool.head_dim,
        kv_element_bytes=kv_pool.dtype.itemsize,
        chunked_prefill_size=scheduler.chunked_prefill_size,
        max_prefill_tokens=scheduler.max_prefill_tokens,
        max_running_requests=scheduler.max_running_requests,
    )
    logger.info(f"S3-drama test engine: {facts}")
    driver = SchedulerDriver(scheduler, model, spec.prompts)
    baseline = driver.free_pool_counts()
    driver.start()
    try:
        scenarios = [
            run_scenario(driver, scenario, baseline) for scenario in spec.scenarios
        ]
    finally:
        driver.stop_and_drain()
    return EngineRun(
        facts=facts,
        baseline_free_counts=baseline,
        scenarios=scenarios,
        outcomes=driver.outcomes,
        forwards=model.forwards,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    spec = EngineSpec.model_validate_json(args.spec.read_text())
    torch.save(run_engine(spec), args.output)


if __name__ == "__main__":
    main()
