# SPDX-License-Identifier: Apache-2.0
"""S3-drama model runner: stages each forward's frames and collects sampled frames.

Everything runs inside the model's forward, behind SGLang's own forward path,
so the deferred clear of reused linear-attention state slots always happens.
Decode steps go through the model's fixed buffers, which a decode CUDA graph reads.
"""

from __future__ import annotations

import torch
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.fishaudio_s3_drama.request_builders import (
    MINIMUM_PROMPT_ROWS,
    S3DramaRequestData,
)
from sglang_omni.models.fishaudio_s3_drama.sampling import (
    EMPTY_RAS_SLOT,
    RAS_WINDOW_FRAMES,
)
from sglang_omni.models.fishaudio_s3_drama.sglang_model import (
    S3DramaDecodeBuffers,
    S3DramaForCausalLM,
    S3DramaStepInputs,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import (
    ForwardBatch,
    GenerationBatchResult,
    ScheduleBatch,
)


def ras_window(frame_rows: list[list[int]]) -> list[int]:
    """Main tokens of the last 16 frames, oldest first, left-padded with EMPTY_RAS_SLOT.

    Built from the whole history each step, so a retracted request re-prefilled
    from that history gets the same window it would have had.
    """
    main_tokens = [frame_row[0] for frame_row in frame_rows[-RAS_WINDOW_FRAMES:]]
    return [EMPTY_RAS_SLOT] * (RAS_WINDOW_FRAMES - len(main_tokens)) + main_tokens


def extend_frame_rows(data: S3DramaRequestData) -> list[list[int]]:
    """The rows of a prefill, which must cover the request's whole frame history."""
    req = data.req
    assert req is not None
    sequence_ids = list(req.full_untruncated_fill_ids)
    extend_range = req.extend_range
    if (
        extend_range is None
        or extend_range.start != 0
        or extend_range.end != len(sequence_ids)
    ):
        raise RuntimeError(
            f"{req.rid}: an S3-drama prefill must cover the whole sequence of "
            f"{len(sequence_ids)} rows in one extend; got {extend_range}"
        )
    elif len(sequence_ids) < MINIMUM_PROMPT_ROWS:
        raise RuntimeError(
            f"{req.rid}: an extend needs at least {MINIMUM_PROMPT_ROWS} rows"
        )
    elif [frame_row[0] for frame_row in data.frame_rows] != sequence_ids:
        raise RuntimeError(f"{req.rid}: the frame history is not the sequence")
    else:
        return data.frame_rows


def decode_frame_row(data: S3DramaRequestData) -> list[int]:
    """The newest frame, the input row of a decode step."""
    req = data.req
    assert req is not None
    if len(data.frame_rows) != req.seqlen:
        raise RuntimeError(
            f"{req.rid}: {len(data.frame_rows)} frames for a sequence of {req.seqlen}"
        )
    else:
        return data.frame_rows[-1]


def request_data_of(request: SchedulerRequest) -> S3DramaRequestData:
    data = request.data
    assert isinstance(data, S3DramaRequestData)
    return data


def request_ids_of(batch_request_data: list[S3DramaRequestData]) -> list[str]:
    request_ids = []
    for request_data in batch_request_data:
        request = request_data.req
        assert request is not None
        request_ids.append(request.rid)
    return request_ids


def stage_decode_rows(
    buffers: S3DramaDecodeBuffers, batch_request_data: list[S3DramaRequestData]
) -> None:
    """Copy each request's newest frame, RAS window and sampling knobs into its row,
    in place, since a captured decode graph reads these addresses."""
    row_count = len(batch_request_data)
    buffers.frames[:row_count].copy_(
        torch.tensor(
            [decode_frame_row(request_data) for request_data in batch_request_data],
            dtype=torch.int64,
        )
    )
    buffers.ras_windows[:row_count].copy_(
        torch.tensor(
            [
                ras_window(request_data.frame_rows)
                for request_data in batch_request_data
            ],
            dtype=torch.int64,
        )
    )
    buffers.temperature[:row_count].copy_(
        torch.tensor(
            [request_data.temperature for request_data in batch_request_data],
            dtype=torch.float32,
        )
    )
    buffers.top_k[:row_count].copy_(
        torch.tensor(
            [request_data.top_k for request_data in batch_request_data],
            dtype=torch.int64,
        )
    )
    buffers.top_p[:row_count].copy_(
        torch.tensor(
            [request_data.top_p for request_data in batch_request_data],
            dtype=torch.float32,
        )
    )


def sampled_decode_frames(
    buffers: S3DramaDecodeBuffers, row_count: int
) -> torch.Tensor:
    """The frames the last decode forward sampled, copied before the next overwrites them."""
    return buffers.sampled_frames[:row_count].clone()


class S3DramaModelRunner(ModelRunner[S3DramaRequestData]):
    """Hands each forward its frames, RAS windows and sampling knobs, then appends
    the frame the model sampled for every request."""

    model: S3DramaForCausalLM

    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        sglang_runner = tp_worker.model_runner
        model = sglang_runner.model
        assert isinstance(model, S3DramaForCausalLM)
        text_config = model.config.text_config
        attention_backend = sglang_runner.attn_backend
        kv_pool = sglang_runner.token_to_kv_pool
        request_pool = sglang_runner.req_to_token_pool
        prefill_graph_runner = sglang_runner.prefill_cuda_graph_runner
        decode_graph_runner = sglang_runner.decode_cuda_graph_runner
        decode_buffers = model.decode_buffers
        if not isinstance(attention_backend, HybridLinearAttnBackend):
            raise RuntimeError(
                f"S3-drama needs the hybrid linear-attention backend; got "
                f"{type(attention_backend).__name__}"
            )
        elif not isinstance(attention_backend.full_attn_backend, TritonAttnBackend):
            raise RuntimeError(
                f"S3-drama needs triton for full attention; got "
                f"{type(attention_backend.full_attn_backend).__name__}"
            )
        elif not (
            decode_graph_runner is None
            or isinstance(decode_graph_runner, DecodeCudaGraphRunner)
        ) or not (
            prefill_graph_runner is None
            or isinstance(prefill_graph_runner, EagerRunner)
        ):
            raise RuntimeError(
                "S3-drama replays decode CUDA graphs only; its prefill runs eager"
            )
        elif decode_buffers is None or decode_buffers.frames.shape[0] < max(
            request_pool.size,
            0 if decode_graph_runner is None else decode_graph_runner.max_bs,
        ):
            raise RuntimeError(
                "the S3-drama decode buffers need a row for every request slot and "
                "every captured decode batch size"
            )
        elif not (
            isinstance(kv_pool, HybridLinearKVPool)
            and set(kv_pool.full_attention_layer_id_mapping)
            == set(text_config.full_attention_layer_ids)
        ):
            raise RuntimeError(
                f"the KV pool does not hold exactly the backbone and tower layers "
                f"{text_config.full_attention_layer_ids}"
            )
        elif not (
            isinstance(request_pool, HybridReqToTokenPool)
            and set(request_pool.mamba_map) == set(text_config.linear_layer_ids)
        ):
            raise RuntimeError(
                f"the state pool does not hold exactly the linear-attention layers "
                f"{text_config.linear_layer_ids}"
            )
        else:
            pass
        self.decode_buffers: S3DramaDecodeBuffers = decode_buffers

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        """Interface contract; never eligible, since collection publishes the frame."""
        return False

    def stage_step(
        self, requests: list[SchedulerRequest], frame_rows: list[list[int]]
    ) -> None:
        request_data = [request_data_of(request) for request in requests]
        self.model.step_exchange.stage(
            S3DramaStepInputs(
                request_ids=request_ids_of(request_data),
                frames=torch.tensor(frame_rows, dtype=torch.int64, device=self.device),
                ras_windows=torch.tensor(
                    [ras_window(data.frame_rows) for data in request_data],
                    dtype=torch.int64,
                    device=self.device,
                ),
                temperature=torch.tensor(
                    [data.temperature for data in request_data],
                    dtype=torch.float32,
                    device=self.device,
                ),
                top_k=torch.tensor(
                    [data.top_k for data in request_data],
                    dtype=torch.int64,
                    device=self.device,
                ),
                top_p=torch.tensor(
                    [data.top_p for data in request_data],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        )

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Stage every request's whole frame history; the batch arguments are the
        interface contract."""
        frame_rows: list[list[int]] = []
        for request in requests:
            frame_rows.extend(extend_frame_rows(request_data_of(request)))
        self.stage_step(requests, frame_rows)

    def cleanup_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Drop inputs a failed prefill left staged; the arguments are the contract."""
        self.model.step_exchange.clear()

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        """Stage every request's newest frame into the decode buffers, in batch
        order; the other arguments are the interface contract."""
        batch_request_data = [request_data_of(request) for request in requests]
        request_ids = request_ids_of(batch_request_data)
        if forward_batch is None or forward_batch.rids != request_ids:
            raise RuntimeError(
                f"decode requests {request_ids} are not the batch's "
                f"{None if forward_batch is None else forward_batch.rids}"
            )
        else:
            stage_decode_rows(self.decode_buffers, batch_request_data)

    def collect_frames(
        self,
        result: GenerationBatchResult,
        requests: list[SchedulerRequest],
        frames: torch.Tensor,
    ) -> None:
        result.next_token_ids = frames[:, 0]
        for request, frame_row in zip(requests, frames.tolist(), strict=True):
            request_data_of(request).frame_rows.append(frame_row)

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Collect the first frames; the batch arguments are the interface contract."""
        self.collect_frames(result, requests, self.model.step_exchange.take_frames())

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Collect the next frames; the batch arguments are the interface contract."""
        self.collect_frames(
            result,
            requests,
            sampled_decode_frames(self.decode_buffers, len(requests)),
        )
