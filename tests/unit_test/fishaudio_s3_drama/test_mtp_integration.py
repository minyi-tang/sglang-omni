# SPDX-License-Identifier: Apache-2.0
"""S3-drama tower integration on CPU: requests, staging, frame labels, rejections."""

from __future__ import annotations

import json
import math
from array import array
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch
from sglang.srt.server_args import ServerArgs

from sglang_omni.models.fishaudio_s3_drama.engine_builder import S3DramaEngineBuilder
from sglang_omni.models.fishaudio_s3_drama.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_s3_drama.model_runner import (
    EMPTY_RAS_SLOT,
    decode_frame_row,
    extend_frame_rows,
    ras_window,
)
from sglang_omni.models.fishaudio_s3_drama.request_builders import (
    S3DramaGenerationInput,
    S3DramaGenerationResult,
    S3DramaRequestBuilder,
    S3DramaRequestData,
    S3DramaResultAdapter,
)
from sglang_omni.models.fishaudio_s3_drama.sglang_model import (
    S3DramaForCausalLM,
    S3DramaStepInputs,
    StepExchange,
    next_frame_labels,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling import sglang_backend
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
)
from sglang_omni.serve.openai_errors import is_bad_request_error
from tests.unit_test.fishaudio_s3_drama.dense_attention import DenseAttention
from tests.unit_test.fishaudio_s3_drama.test_audio_decoder import (
    CODEBOOK_SIZE,
    DIM,
    EOS_TOKEN_ID,
    HEAD_COUNT,
    HEAD_DIM,
    KV_HEAD_COUNT,
    SEMANTIC_BEGIN_ID,
    TEXT_VOCAB_SIZE,
    loaded_decoder,
    random_checkpoint,
)
from tests.unit_test.fishaudio_s3_drama.test_hf_config import fish_config_dict

FRAME_WIDTH = 5
CONTEXT_LENGTH = 32
MAX_PROMPT_ROWS = 6
# The prompt cap the serving contract asks for by default.
SERVING_PROMPT_CAP_ROWS = 8192
SamplingField = Literal["max_new_tokens", "temperature", "top_k", "top_p"]


class GenerationInputChange(Protocol):
    def __call__(
        self, generation_input: S3DramaGenerationInput
    ) -> S3DramaGenerationInput: ...


def frame_rows_of(lengths: list[int], seed: int) -> list[list[list[int]]]:
    """Per request, alternating text rows (codes 0) and semantic rows."""
    generator = torch.Generator().manual_seed(seed)
    requests = []
    for length in lengths:
        rows = []
        for row_index in range(length):
            codes = torch.randint(0, CODEBOOK_SIZE, (4,), generator=generator).tolist()
            if row_index % 2 == 0:
                rows.append(
                    [int(torch.randint(10, 79, (1,), generator=generator))] + [0] * 4
                )
            else:
                rows.append([SEMANTIC_BEGIN_ID + codes[0]] + codes)
        requests.append(rows)
    return requests


def generation_input(prompt_rows: torch.Tensor) -> S3DramaGenerationInput:
    return S3DramaGenerationInput(
        prompt_rows=prompt_rows,
        max_new_tokens=4,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
    )


def built_request(prompt_rows: list[list[int]]) -> S3DramaRequestData:
    builder = S3DramaRequestBuilder(
        config=FishOmniConfig(**fish_config_dict()),
        context_length=CONTEXT_LENGTH,
        max_prompt_rows=MAX_PROMPT_ROWS,
    )
    return builder(
        StagePayload(
            request_id="request-a",
            request=OmniRequest(inputs="prompt"),
            data=generation_input(torch.tensor(prompt_rows, dtype=torch.int64)),
        )
    )


def test_ras_window_is_the_last_sixteen_main_tokens_of_the_history() -> None:
    history = [[token] + [0] * 4 for token in range(10, 13)]
    assert ras_window(history) == [EMPTY_RAS_SLOT] * 13 + [10, 11, 12]

    history = [[token] + [0] * 4 for token in range(30)]
    window = ras_window(history)
    assert window == list(range(14, 30))
    history.append([81, 1, 2, 3, 4])
    assert ras_window(history) == window[1:] + [81]


def test_ras_window_of_a_prompt_keeps_its_end_of_turn_token() -> None:
    """The first frame's window is the prompt tail, eos included, never all empty."""
    prompt_tail = [21, 22, EOS_TOKEN_ID, 23, 24, 25, 26]
    window = ras_window([[token] + [0] * 4 for token in prompt_tail])

    assert EOS_TOKEN_ID in window
    assert window[-len(prompt_tail) :] == prompt_tail


def test_request_builder_keeps_the_rows_and_stops_on_eos() -> None:
    prompt_rows = frame_rows_of([5], seed=0)[0]

    data = built_request(prompt_rows)

    req = data.req
    assert req is not None
    assert req.rid == "request-a"
    assert list(req.origin_input_ids) == [row[0] for row in prompt_rows]
    assert req.eos_token_ids == {EOS_TOKEN_ID}
    assert EOS_TOKEN_ID in req.sampling_params.stop_token_ids
    assert data.frame_rows == prompt_rows
    assert data.prompt_row_count == 5
    assert data.input_ids is not None
    assert data.input_ids.tolist() == [row[0] for row in prompt_rows]
    assert (data.temperature, data.top_k, data.top_p) == (0.8, 5, 0.9)


def changed_rows(rows: torch.Tensor) -> GenerationInputChange:
    def change(generation_input: S3DramaGenerationInput) -> S3DramaGenerationInput:
        return replace(generation_input, prompt_rows=rows)

    return change


def changed_field(name: SamplingField, value: int | float) -> GenerationInputChange:
    def change(generation_input: S3DramaGenerationInput) -> S3DramaGenerationInput:
        return replace(generation_input, **{name: value})

    return change


GOOD_ROWS = torch.tensor(frame_rows_of([4], seed=1)[0], dtype=torch.int64)


def with_entry(row: int, column: int, value: int) -> torch.Tensor:
    rows = GOOD_ROWS.clone()
    rows[row, column] = value
    return rows


REJECTED_INPUTS = [
    pytest.param(changed_rows(GOOD_ROWS.float()), "int64", id="float-rows"),
    pytest.param(changed_rows(GOOD_ROWS[:, :4]), "int64", id="narrow-rows"),
    pytest.param(changed_rows(GOOD_ROWS[:1]), "at least 2 rows", id="one-row"),
    pytest.param(
        changed_rows(with_entry(0, 0, TEXT_VOCAB_SIZE)),
        "main tokens",
        id="token-past-vocab",
    ),
    pytest.param(
        changed_rows(with_entry(0, 0, -1)), "main tokens", id="negative-token"
    ),
    pytest.param(
        changed_rows(with_entry(1, 2, CODEBOOK_SIZE)), "codes", id="code-past-codebook"
    ),
    pytest.param(
        changed_field("max_new_tokens", 0), "max_new_tokens", id="no-new-tokens"
    ),
    pytest.param(
        changed_field("max_new_tokens", CONTEXT_LENGTH),
        "context length",
        id="past-context",
    ),
    pytest.param(
        changed_field("temperature", -0.5), "temperature", id="negative-temperature"
    ),
    pytest.param(
        changed_field("temperature", math.nan), "temperature", id="nan-temperature"
    ),
    pytest.param(changed_field("top_k", 0), "top_k", id="zero-top-k"),
    pytest.param(changed_field("top_k", -1), "top_k", id="disabled-top-k"),
    pytest.param(changed_field("top_p", 0.0), "top_p", id="zero-top-p"),
    pytest.param(changed_field("top_p", 1.5), "top_p", id="top-p-above-one"),
]


@pytest.mark.parametrize(("change", "error"), REJECTED_INPUTS)
def test_request_builder_rejects_bad_inputs(
    change: GenerationInputChange, error: str
) -> None:
    builder = S3DramaRequestBuilder(
        config=FishOmniConfig(**fish_config_dict()),
        context_length=CONTEXT_LENGTH,
        max_prompt_rows=MAX_PROMPT_ROWS,
    )
    payload = StagePayload(
        request_id="request-a",
        request=OmniRequest(inputs="prompt"),
        data=change(generation_input(GOOD_ROWS)),
    )

    with pytest.raises(ValueError, match=error):
        builder(payload)


def test_request_builder_rejects_a_foreign_payload() -> None:
    builder = S3DramaRequestBuilder(
        config=FishOmniConfig(**fish_config_dict()),
        context_length=CONTEXT_LENGTH,
        max_prompt_rows=MAX_PROMPT_ROWS,
    )
    payload = StagePayload(
        request_id="request-a", request=OmniRequest(inputs="prompt"), data=[1, 2, 3]
    )

    with pytest.raises(TypeError, match="S3DramaGenerationInput"):
        builder(payload)


def test_prompt_cap_is_inclusive_and_refused_as_a_bad_request() -> None:
    built_request(frame_rows_of([MAX_PROMPT_ROWS], seed=4)[0])

    with pytest.raises(ValueError) as rejection:
        built_request(frame_rows_of([MAX_PROMPT_ROWS + 1], seed=4)[0])

    assert is_bad_request_error(rejection.value)


class ConfigOnlyModel(S3DramaForCausalLM):
    """S3DramaForCausalLM for make_adapters, which reads only the config."""

    def __init__(self, config: FishOmniConfig) -> None:
        torch.nn.Module.__init__(self)
        self.config = config


def test_default_prompt_cap_reaches_the_request_builder() -> None:
    """8192 rows pass and 8193 are refused; the context budget is still checked."""
    request_builder, _ = S3DramaEngineBuilder(
        context_length=SERVING_PROMPT_CAP_ROWS + 5, max_running_requests=1
    ).make_adapters(ConfigOnlyModel(FishOmniConfig(**fish_config_dict())))
    capped_rows, past_cap_rows = (
        torch.tensor(rows, dtype=torch.int64)
        for rows in frame_rows_of(
            [SERVING_PROMPT_CAP_ROWS, SERVING_PROMPT_CAP_ROWS + 1], seed=5
        )
    )

    def payload(prompt_rows: torch.Tensor, max_new_tokens: int) -> StagePayload:
        return StagePayload(
            request_id="request-a",
            request=OmniRequest(inputs="prompt"),
            data=replace(generation_input(prompt_rows), max_new_tokens=max_new_tokens),
        )

    request_data = request_builder(payload(capped_rows, max_new_tokens=5))
    assert request_data.prompt_row_count == SERVING_PROMPT_CAP_ROWS
    with pytest.raises(ValueError, match="exceeds the maximum allowed length"):
        request_builder(payload(past_cap_rows, max_new_tokens=4))
    with pytest.raises(ValueError, match="context length"):
        request_builder(payload(capped_rows, max_new_tokens=6))


@pytest.mark.parametrize(
    "max_prompt_rows",
    [
        True,
        1,
        -SERVING_PROMPT_CAP_ROWS,
        math.inf,
        math.nan,
        float(SERVING_PROMPT_CAP_ROWS),
    ],
)
def test_prompt_cap_must_be_an_integer_of_at_least_two_rows(
    max_prompt_rows: int | float,
) -> None:
    with pytest.raises(ValueError, match="max_prompt_rows"):
        S3DramaRequestBuilder(
            config=FishOmniConfig(**fish_config_dict()),
            context_length=CONTEXT_LENGTH,
            max_prompt_rows=max_prompt_rows,
        )


@pytest.mark.parametrize(
    ("sampled_main_tokens", "kept_count"),
    [([81, 84], 2), ([81, EOS_TOKEN_ID], 1), ([EOS_TOKEN_ID], 0)],
    ids=["no-eos", "eos-last", "only-eos"],
)
def test_result_holds_the_sampled_frames_without_eos(
    sampled_main_tokens: list[int], kept_count: int
) -> None:
    data = built_request(frame_rows_of([3], seed=2)[0])
    sampled = [[token, 1, 2, 3, 0] for token in sampled_main_tokens]
    data.frame_rows.extend(sampled)

    payload = S3DramaResultAdapter(eos_token_id=EOS_TOKEN_ID)(data)

    result = payload.data
    assert isinstance(result, S3DramaGenerationResult)
    assert result.frames.shape == (kept_count, FRAME_WIDTH)
    assert result.frames.tolist() == sampled[:kept_count]


def admitted_for_prefill(data: S3DramaRequestData, start: int, end: int) -> None:
    """Leave the request as prefill admission does: fill ids rebuilt, extend range set."""
    req = data.req
    assert req is not None
    # Without a tree cache this only rebuilds fill ids; no prefix is matched.
    req.init_next_round_input()
    req.set_extend_range(start, end)


def test_prefill_rows_cover_the_whole_history_after_a_retract() -> None:
    prompt_rows = frame_rows_of([4], seed=3)[0]
    data = built_request(prompt_rows)
    req = data.req
    assert req is not None
    admitted_for_prefill(data, 0, 4)
    assert extend_frame_rows(data) == prompt_rows

    sampled = [[82, 2, 0, 1, 3], [85, 5, 1, 1, 1]]
    req.output_ids.extend(array("q", [row[0] for row in sampled]))
    data.frame_rows.extend(sampled)
    assert decode_frame_row(data) == sampled[-1]
    admitted_for_prefill(data, 0, 6)
    assert extend_frame_rows(data) == prompt_rows + sampled


@pytest.mark.parametrize(
    "extend_range", [(0, 3), (1, 4)], ids=["partial-chunk", "cached-prefix"]
)
def test_prefill_rows_reject_a_partial_extend(extend_range: tuple[int, int]) -> None:
    data = built_request(frame_rows_of([4], seed=4)[0])
    admitted_for_prefill(data, *extend_range)

    with pytest.raises(RuntimeError, match="whole sequence"):
        extend_frame_rows(data)


def test_prefill_rows_reject_a_history_that_is_not_the_sequence() -> None:
    data = built_request(frame_rows_of([4], seed=4)[0])
    admitted_for_prefill(data, 0, 4)
    data.frame_rows[0] = [data.frame_rows[0][0] + 1] + data.frame_rows[0][1:]

    with pytest.raises(RuntimeError, match="history"):
        extend_frame_rows(data)


def test_decode_row_rejects_a_history_behind_the_sequence() -> None:
    data = built_request(frame_rows_of([3], seed=5)[0])
    req = data.req
    assert req is not None
    req.output_ids.append(84)

    with pytest.raises(RuntimeError, match="frames for a sequence of 4"):
        decode_frame_row(data)


def step_inputs(
    request_ids: list[str], row_count: int, marker: int
) -> S3DramaStepInputs:
    request_count = len(request_ids)
    return S3DramaStepInputs(
        request_ids=request_ids,
        frames=torch.full((row_count, FRAME_WIDTH), marker),
        ras_windows=torch.full((request_count, 16), EMPTY_RAS_SLOT),
        temperature=torch.ones(request_count),
        top_k=torch.ones(request_count, dtype=torch.int64),
        top_p=torch.ones(request_count),
    )


def test_step_exchange_hands_inputs_and_frames_over_once() -> None:
    exchange = StepExchange()
    with pytest.raises(RuntimeError, match="staged no inputs"):
        exchange.take_inputs(["a"], 1)

    exchange.stage(step_inputs(["a", "b"], 2, marker=1))
    assert exchange.take_inputs(["a", "b"], 2).frames[0, 0] == 1
    with pytest.raises(RuntimeError, match="staged no inputs"):
        exchange.take_inputs(["a", "b"], 2)

    exchange.stage(step_inputs(["a", "b"], 2, marker=1))
    with pytest.raises(RuntimeError, match="not the batch"):
        exchange.take_inputs(["b", "a"], 2)
    exchange.stage(step_inputs(["a"], 3, marker=1))
    with pytest.raises(RuntimeError, match="3 frames for 1 rows"):
        exchange.take_inputs(["a"], 1)

    with pytest.raises(RuntimeError, match="no forward has sampled"):
        exchange.take_frames()
    exchange.publish(torch.zeros(1, FRAME_WIDTH))
    assert exchange.take_frames().shape == (1, FRAME_WIDTH)
    with pytest.raises(RuntimeError, match="no forward has sampled"):
        exchange.take_frames()


def test_a_failed_step_never_feeds_the_next_step_of_the_same_requests() -> None:
    """Request ids repeat across steps; each step's own staging decides its input."""
    exchange = StepExchange()

    # A forward that took its inputs and then failed publishes no frames.
    exchange.stage(step_inputs(["a"], 1, marker=1))
    exchange.publish(torch.zeros(1, FRAME_WIDTH))
    exchange.take_inputs(["a"], 1)
    with pytest.raises(RuntimeError, match="no forward has sampled"):
        exchange.take_frames()
    exchange.stage(step_inputs(["a"], 1, marker=2))
    assert exchange.take_inputs(["a"], 1).frames[0, 0] == 2

    # A step that failed before its forward is replaced by the next step's staging.
    exchange.stage(step_inputs(["a"], 1, marker=3))
    exchange.stage(step_inputs(["a"], 1, marker=4))
    assert exchange.take_inputs(["a"], 1).frames[0, 0] == 4

    # A failed prefill clears what it staged.
    exchange.stage(step_inputs(["a"], 1, marker=5))
    exchange.clear()
    with pytest.raises(RuntimeError, match="staged no inputs"):
        exchange.take_inputs(["a"], 1)


def test_next_frame_labels_pair_rows_within_each_request() -> None:
    frames = torch.tensor(sum(frame_rows_of([3, 2], seed=6), []))
    original = frames.clone()

    labels = next_frame_labels(frames, torch.tensor([2, 4]), EOS_TOKEN_ID)

    placeholder = [EOS_TOKEN_ID, 0, 0, 0, 0]
    assert labels.tolist() == [
        original[1].tolist(),
        original[2].tolist(),
        placeholder,
        original[4].tolist(),
        placeholder,
    ]
    assert torch.equal(frames, original)


def tower_cache(attention: DenseAttention) -> dict[tuple[int, str, int], torch.Tensor]:
    return {
        (layer_index, request_id, position): torch.cat((key, value))
        for (layer_index, request_id), entries in attention.entries.items()
        for position, key, value in entries
    }


def test_placeholder_rows_end_like_first_writes_and_a_dense_recompute() -> None:
    """The integration's labels, then the decode step's rewrite of each last slot,
    leave the cache of N-1-row prefills and of a whole dense recompute, in fp32."""
    decoder = loaded_decoder(random_checkpoint(seed=0), torch.float32, "cpu")
    lengths = [6, 2, 3]
    request_ids = ["a", "b", "c"]
    requests = frame_rows_of(lengths, seed=7)
    frames = torch.tensor(sum(requests, []))
    row_request_ids = [
        request_id
        for request_id, length in zip(request_ids, lengths, strict=True)
        for _ in range(length)
    ]
    positions = torch.tensor([row for length in lengths for row in range(length)])
    last_rows = torch.cumsum(torch.tensor(lengths), dim=0) - 1
    generator = torch.Generator().manual_seed(8)
    hidden_states = torch.randn(frames.shape[0], DIM, generator=generator)
    semantic_logits = torch.randn(len(lengths), TEXT_VOCAB_SIZE, generator=generator)
    last_positions = positions[last_rows].tolist()

    def sample(attention: DenseAttention) -> torch.Tensor:
        attention.bind(request_ids, last_positions)
        return decoder.decode_step(
            semantic_logits,
            hidden_states[last_rows],
            positions[last_rows],
            frames[last_rows],
            torch.full((len(lengths), 16), EMPTY_RAS_SLOT),
            torch.ones(len(lengths)),
            torch.ones(len(lengths), dtype=torch.int64),
            torch.ones(len(lengths)),
            attention,
        )

    def new_attention() -> DenseAttention:
        return DenseAttention(
            head_count=HEAD_COUNT, kv_head_count=KV_HEAD_COUNT, head_dim=HEAD_DIM
        )

    placeholder_attention = new_attention()
    placeholder_attention.bind(row_request_ids, positions.tolist())
    decoder.prefill(
        hidden_states,
        positions,
        frames,
        next_frame_labels(frames, last_rows, EOS_TOKEN_ID),
        placeholder_attention,
    )
    placeholder_frames = sample(placeholder_attention)

    first_write_attention = new_attention()
    kept_rows = torch.tensor(
        [row for row in range(frames.shape[0]) if row not in last_rows.tolist()]
    )
    first_write_attention.bind(
        [row_request_ids[row] for row in kept_rows.tolist()],
        positions[kept_rows].tolist(),
    )
    decoder.prefill(
        hidden_states[kept_rows],
        positions[kept_rows],
        frames[kept_rows],
        frames[kept_rows + 1],
        first_write_attention,
    )
    first_write_frames = sample(first_write_attention)

    dense_attention = new_attention()
    dense_labels = frames.roll(-1, dims=0)
    dense_labels[last_rows] = placeholder_frames
    dense_attention.bind(row_request_ids, positions.tolist())
    decoder.prefill(hidden_states, positions, frames, dense_labels, dense_attention)

    assert torch.equal(placeholder_frames, first_write_frames)
    placeholder_cache = tower_cache(placeholder_attention)
    assert placeholder_cache.keys() == tower_cache(first_write_attention).keys()
    assert placeholder_cache.keys() == tower_cache(dense_attention).keys()
    for slot, cached in placeholder_cache.items():
        torch.testing.assert_close(
            cached, tower_cache(first_write_attention)[slot], rtol=0, atol=1e-4
        )
        torch.testing.assert_close(
            cached, tower_cache(dense_attention)[slot], rtol=0, atol=1e-4
        )


def test_model_rejects_a_tower_that_cannot_share_the_backbone_pool() -> None:
    # fish_config_dict() gives the tower head_dim 12 and the backbone head_dim 8.
    with pytest.raises(ValueError, match="must equal the backbone's"):
        S3DramaForCausalLM(FishOmniConfig(**fish_config_dict()))

    config_dict = fish_config_dict()
    decoder_config = config_dict["audio_decoder_config"]
    assert isinstance(decoder_config, dict)
    decoder_config["head_dim"] = 8
    config = FishOmniConfig(**config_dict)
    config.text_config.tower_layer_count = 4
    with pytest.raises(ValueError, match="tower layers"):
        S3DramaForCausalLM(config)


def engine_server_args(
    checkpoint_dir: Path, overrides: dict[str, object]
) -> tuple[S3DramaEngineBuilder, ServerArgs]:
    """The server args the builder resolves for a checkpoint and caller overrides."""
    (checkpoint_dir / "config.json").write_text(json.dumps(fish_config_dict()))
    builder = S3DramaEngineBuilder(
        context_length=CONTEXT_LENGTH, max_running_requests=2
    )
    builder.pre_infra_setup(str(checkpoint_dir))
    generation_overrides = build_generation_batch_overrides(
        server_args_overrides=overrides,
        **builder.generation_defaults(dtype="bfloat16"),
    )
    builder.adjust_overrides(generation_overrides)
    return builder, sglang_backend.build_sglang_server_args(
        str(checkpoint_dir), context_length=CONTEXT_LENGTH, **generation_overrides
    )


def test_engine_defaults_pass_its_own_validation(tmp_path: Path) -> None:
    builder, server_args = engine_server_args(tmp_path, {})

    builder.validate_before_infrastructure(server_args)
    assert server_args.page_size == 1
    assert server_args.chunked_prefill_size == -1


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"disable_radix_cache": False}, "disable_radix_cache"),
        ({"chunked_prefill_size": 64}, "chunked_prefill_size"),
        ({"disable_cuda_graph": False}, "eager"),
        ({"disable_overlap_schedule": False}, "eager"),
        ({"attention_backend": "flashinfer"}, "attention backend"),
        ({"page_size": 16}, "page_size"),
        ({"kv_cache_dtype": "fp8_e4m3"}, "model dtype"),
        ({"enable_torch_compile": True}, "torch.compile"),
    ],
    ids=[
        "radix",
        "chunked-prefill",
        "cuda-graph",
        "overlap",
        "flashinfer",
        "paged",
        "fp8-kv",
        "torch-compile",
    ],
)
def test_engine_rejects_settings_outside_the_supported_runtime(
    tmp_path: Path, overrides: dict[str, object], error: str
) -> None:
    builder, server_args = engine_server_args(tmp_path, overrides)

    with pytest.raises(ValueError, match=error):
        builder.validate_before_infrastructure(server_args)
