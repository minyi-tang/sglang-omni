# SPDX-License-Identifier: Apache-2.0
"""S3-drama decode CUDA graphs on a GPU: eager and graph engines, each in its own
process (graph_engines.py), must agree bit for bit on the synthetic checkpoint."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from sglang_omni.models.fishaudio_s3_drama.sampling import EMPTY_RAS_SLOT
from tests.unit_test.fishaudio_s3_drama.graph_checks import (
    EnginePair,
    bit_problems,
    draw_problems,
    finite_problems,
    history_problems,
    outcome_problems,
    prefill_path_problems,
    prompt_row_counts,
    replay_problems,
    resampled_choice_count,
    schedule_problems,
    stray_write_problems,
    stream_problems,
    wave_admission_problems,
)
from tests.unit_test.fishaudio_s3_drama.graph_fixtures import (
    CUDA_GRAPH_BATCH_SIZES,
    FP8_SCENARIOS,
    GREEDY_ABORT_REUSE,
    MAIN_SCENARIOS,
    PREFILL_CUDA_GRAPH_TOKEN_COUNTS,
    RANDOM_WAVES,
    RETRACT_INTERVAL,
    RETRACT_SCENARIOS,
    EngineKind,
    GraphEngineRun,
    GraphEngineSpec,
    GraphForwardRecord,
    GraphRequestStep,
    GraphScenario,
    bits,
)
from tests.unit_test.fishaudio_s3_drama.mtp_fixtures import (
    EOS_TOKEN_ID,
    TOTAL_GPU_MEMORY_FRACTION,
    write_expert_variant,
    write_synthetic_checkpoint,
)
from tests.unit_test.fixtures.accelerator import require_cuda
from tests.unit_test.scheduling.test_hybrid_gdn import (
    kill_process_group,
    require_free_gpu_memory,
)

pytestmark = [pytest.mark.accelerator, pytest.mark.benchmark]

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_MODULE = "tests.unit_test.fishaudio_s3_drama.graph_engines"
ENGINE_TIMEOUT_SECONDS = 1200
RETRACT_ENVIRONMENT = {
    "SGLANG_TEST_RETRACT": "1",
    "SGLANG_TEST_RETRACT_INTERVAL": str(RETRACT_INTERVAL),
}
# Requests whose last step never comes: the abort target.
PADDED_BATCH_SIZES = {3, 5, 6, 7}
# One line per problem is enough to locate the first divergence.
REPORTED_PROBLEM_COUNT = 20


@dataclass(frozen=True, kw_only=True)
class GraphEngineRuns:
    main: EnginePair
    fp8: EnginePair
    retract: EnginePair
    prefill_main: EnginePair
    prefill_fp8: EnginePair
    prefill_retract: EnginePair
    workdir: Path


def run_engine(
    name: str, spec: GraphEngineSpec, workdir: Path, extra_environment: dict[str, str]
) -> GraphEngineRun:
    """Run one engine process to completion and load what it recorded."""
    require_free_gpu_memory(spec.total_gpu_memory_fraction)
    spec_path = workdir / f"{name}-spec.json"
    output_path = workdir / f"{name}-run.pt"
    log_path = workdir / f"{name}.log"
    spec_path.write_text(spec.model_dump_json())
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                ENGINE_MODULE,
                "--spec",
                str(spec_path),
                "--output",
                str(output_path),
            ],
            cwd=REPO_ROOT,
            env={**os.environ, **extra_environment},
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return_code = process.wait(timeout=ENGINE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise AssertionError(
                f"{name} engine timed out after {ENGINE_TIMEOUT_SECONDS}s; "
                f"log: {log_path}"
            ) from error
        finally:
            kill_process_group(process)
    if return_code != 0:
        log_tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-60:])
        raise AssertionError(
            f"{name} engine exited with {return_code}; log: {log_path}\n{log_tail}"
        )
    else:
        pass
    run = torch.load(output_path, weights_only=False)
    assert isinstance(run, GraphEngineRun)
    return run


def engine_pairs(
    name: str,
    model_dir: Path,
    scenarios: list[GraphScenario],
    workdir: Path,
    extra_environment: dict[str, str],
) -> tuple[EnginePair, EnginePair]:
    runs: dict[EngineKind, GraphEngineRun] = {}
    for kind in ("eager", "graph", "prefill_graph"):
        spec = GraphEngineSpec(model_dir=str(model_dir), kind=kind, scenarios=scenarios)
        runs[kind] = run_engine(f"{name}-{kind}", spec, workdir, extra_environment)
    return (
        EnginePair(eager=runs["eager"], graph=runs["graph"], scenarios=scenarios),
        EnginePair(
            eager=runs["eager"], graph=runs["prefill_graph"], scenarios=scenarios
        ),
    )


@pytest.fixture(scope="module")
def engine_runs(tmp_path_factory: pytest.TempPathFactory) -> GraphEngineRuns:
    require_cuda()
    require_free_gpu_memory(TOTAL_GPU_MEMORY_FRACTION)
    workdir = tmp_path_factory.mktemp("s3_drama_graph")
    checkpoint_dir = workdir / "checkpoint"
    write_synthetic_checkpoint(checkpoint_dir)
    fp8_checkpoint_dir = workdir / "checkpoint-fp8"
    write_expert_variant(checkpoint_dir, fp8_checkpoint_dir, "fp8")
    # Serial on purpose: each engine owns its process and its share of the GPU.
    main, prefill_main = engine_pairs(
        "main", checkpoint_dir, MAIN_SCENARIOS, workdir, {}
    )
    fp8, prefill_fp8 = engine_pairs(
        "fp8", fp8_checkpoint_dir, FP8_SCENARIOS, workdir, {}
    )
    retract, prefill_retract = engine_pairs(
        "retract", checkpoint_dir, RETRACT_SCENARIOS, workdir, RETRACT_ENVIRONMENT
    )
    runs = GraphEngineRuns(
        main=main,
        fp8=fp8,
        retract=retract,
        prefill_main=prefill_main,
        prefill_fp8=prefill_fp8,
        prefill_retract=prefill_retract,
        workdir=workdir,
    )
    print(f"Artifacts: {workdir}")
    return runs


def all_pairs(engine_runs: GraphEngineRuns) -> dict[str, EnginePair]:
    return {
        "main": engine_runs.main,
        "fp8": engine_runs.fp8,
        "retract": engine_runs.retract,
        "prefill main": engine_runs.prefill_main,
        "prefill fp8": engine_runs.prefill_fp8,
        "prefill retract": engine_runs.prefill_retract,
    }


def first_problems(problems: list[str]) -> str:
    return "\n".join(problems[:REPORTED_PROBLEM_COUNT])


def test_engines_start_as_specified(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        eager_facts = pair.eager.facts
        graph_facts = pair.graph.facts
        assert eager_facts.decode_graph_batch_sizes is None, name
        assert graph_facts.decode_graph_batch_sizes == CUDA_GRAPH_BATCH_SIZES, name
        assert eager_facts.prefill_runner == "EagerRunner", name
        assert eager_facts.prefill_capture_token_counts is None, name
        if pair.graph.kind == "prefill_graph":
            assert graph_facts.prefill_runner == "RequestIdPrefillCudaGraphRunner", name
            assert (
                graph_facts.prefill_capture_token_counts
                == PREFILL_CUDA_GRAPH_TOKEN_COUNTS
            ), name
        else:
            assert graph_facts.prefill_runner == "EagerRunner", name
            assert graph_facts.prefill_capture_token_counts is None, name
        for facts in (eager_facts, graph_facts):
            assert set(facts.tower_layer_ids) <= set(facts.recorded_kv_layer_ids), name
            assert facts.decode_buffer_rows >= facts.request_pool_size, name
            assert facts.decode_buffer_rows >= max(CUDA_GRAPH_BATCH_SIZES), name
        # The split count saturating at its cap keeps padding out of the split.
        assert eager_facts.max_kv_splits == graph_facts.max_kv_splits == 8, name
        assert eager_facts.kv_pool_size == graph_facts.kv_pool_size, name
        assert eager_facts.expert_paths == graph_facts.expert_paths, name
    for expert_path in engine_runs.fp8.graph.facts.expert_paths:
        assert expert_path.quant_method == "Fp8MoEMethod"
        assert expert_path.runner_backend == "TRITON"
        assert expert_path.uses_fp8_w8a8
        assert expert_path.block_shape == [128, 128]


def test_both_engines_run_the_same_batches(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        problems = (
            schedule_problems(pair)
            + wave_admission_problems(pair.eager, pair.scenarios)
            + wave_admission_problems(pair.graph, pair.scenarios)
        )
        assert not problems, f"{name}:\n{first_problems(problems)}"


def test_graph_engine_matches_the_eager_engine_bit_for_bit(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        assert not schedule_problems(pair), name
        problems = bit_problems(pair)
        assert not problems, f"{name}:\n{first_problems(problems)}"


def test_graph_engine_replays_every_decode_padded_and_unpadded(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        problems = replay_problems(pair)
        assert not problems, f"{name}:\n{first_problems(problems)}"
    replayed_sizes = {
        len(record.steps)
        for record in engine_runs.main.graph.forwards
        if record.is_graph_replay and not record.is_extend
    }
    assert replayed_sizes == set(range(1, 9))
    # Random sampling is compared only where both engines draw the same shapes.
    assert all(
        record.graph_batch_size == len(record.steps)
        for record in engine_runs.main.graph.forwards
        if record.scenario_name == RANDOM_WAVES.name
        and record.is_graph_replay
        and not record.is_extend
    )
    assert PADDED_BATCH_SIZES <= {
        len(record.steps)
        for record in engine_runs.fp8.graph.forwards
        if record.is_graph_replay and not record.is_extend
    }


def test_each_prefill_replays_its_bucket_or_runs_eager(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = prefill_path_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_prefill_graph_replays_every_admissible_shape(
    engine_runs: GraphEngineRuns,
) -> None:
    run = engine_runs.prefill_main.graph
    extends = [record for record in run.forwards if record.is_extend]
    replays = [record for record in extends if record.is_graph_replay]
    replayed_buckets = [
        bucket for record in replays for bucket in record.prefill_replay_buckets
    ]
    assert extends[0].is_graph_replay
    assert any(
        record.token_count in PREFILL_CUDA_GRAPH_TOKEN_COUNTS for record in replays
    )
    assert any(
        record.token_count not in PREFILL_CUDA_GRAPH_TOKEN_COUNTS for record in replays
    )
    assert any(len(record.steps) > 1 for record in replays)
    assert any(
        later < earlier
        for earlier, later in zip(replayed_buckets, replayed_buckets[1:])
    )
    assert any(
        record.token_count > max(PREFILL_CUDA_GRAPH_TOKEN_COUNTS)
        for record in extends
        if not record.is_graph_replay
    )
    assert any(
        record.token_count <= max(PREFILL_CUDA_GRAPH_TOKEN_COUNTS)
        for record in extends
        if not record.is_graph_replay
    )


def test_prefill_graph_replays_beside_live_requests(
    engine_runs: GraphEngineRuns,
) -> None:
    run = engine_runs.prefill_retract.graph
    decode_forwards: dict[str, list[int]] = {}
    for record in run.forwards:
        for step in record.steps:
            if not record.is_extend:
                decode_forwards.setdefault(step.request_id, []).append(
                    record.forward_index
                )
            else:
                pass
    replays_beside_live_requests = [
        record
        for record in run.forwards
        if record.is_extend
        and record.is_graph_replay
        and any(
            min(forward_indices) < record.forward_index < max(forward_indices)
            for request_id, forward_indices in decode_forwards.items()
            if request_id not in {step.request_id for step in record.steps}
        )
    ]
    assert replays_beside_live_requests


def with_record(run: GraphEngineRun, record: GraphForwardRecord) -> GraphEngineRun:
    forwards = list(run.forwards)
    forwards[record.forward_index] = record
    return replace(run, forwards=forwards)


def test_prefill_checks_catch_a_changed_record(engine_runs: GraphEngineRuns) -> None:
    run = engine_runs.prefill_main.graph
    replayed = next(
        record for record in run.forwards if record.is_extend and record.is_graph_replay
    )
    fallen_back = next(
        record
        for record in run.forwards
        if record.is_extend and not record.is_graph_replay
    )
    for changed in (
        replace(
            replayed,
            prefill_replay_buckets={
                2 * bucket: count
                for bucket, count in replayed.prefill_replay_buckets.items()
            },
        ),
        replace(
            replayed,
            prefill_replay_count=0,
            prefill_eager_count=1,
            prefill_replay_buckets={},
        ),
        replace(replayed, is_graph_replay=False),
        replace(fallen_back, prefill_replay_count=1, prefill_eager_count=0),
    ):
        problems = prefill_path_problems(with_record(run, changed))
        assert len(problems) == 1, problems
    for changed in (
        replace(replayed, stray_kv_slots=[0]),
        replace(replayed, stray_mamba_slots=[replayed.steps[0].mamba_slot + 1]),
    ):
        problems = stray_write_problems(with_record(run, changed))
        assert len(problems) == 1, problems


def test_inputs_the_model_read_follow_each_history(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = history_problems(run, pair.scenarios)
            assert not problems, f"{name} {run.kind}:\n{first_problems(problems)}"


def test_ras_resampling_happens_and_picks_the_main_token(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = draw_problems(run)
            assert not problems, f"{name} {run.kind}:\n{first_problems(problems)}"
    for run in (engine_runs.main.eager, engine_runs.main.graph):
        assert resampled_choice_count(run, RANDOM_WAVES.name) > 0, run.kind


def test_every_recorded_value_is_finite(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = finite_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_decode_steps_write_only_their_own_slots(engine_runs: GraphEngineRuns) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = stray_write_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_streaming_requests_send_each_frame_once_a_whole_window_at_a_time(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = stream_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


def test_every_request_finishes_once_and_pools_return(
    engine_runs: GraphEngineRuns,
) -> None:
    for name, pair in all_pairs(engine_runs).items():
        for run in (pair.eager, pair.graph):
            problems = outcome_problems(run)
            assert not problems, f"{name}:\n{first_problems(problems)}"


@pytest.mark.parametrize("pair_name", ["main", "prefill main"])
def test_rows_and_states_an_abort_freed_serve_the_next_wave(
    engine_runs: GraphEngineRuns, pair_name: str
) -> None:
    """The aborted request's row and state slot are reused by the next wave, whose
    key/value slots were written before; freed KV slots queue behind free ones."""
    run = all_pairs(engine_runs)[pair_name].graph
    aborted = [
        outcome for outcome in run.outcomes.values() if outcome.aborted_after_frames
    ]
    assert [outcome.aborted_after_frames for outcome in aborted] == [15]
    aborted_id = aborted[0].request_id
    scenario_records = [
        record
        for record in run.forwards
        if record.scenario_name == GREEDY_ABORT_REUSE.name
    ]
    aborted_steps = [
        step
        for record in scenario_records
        for step in record.steps
        if step.request_id == aborted_id
    ]
    later_steps = [
        step
        for record in scenario_records
        for step in record.steps
        if step.request_id.startswith(f"{GREEDY_ABORT_REUSE.name}-1-")
    ]
    assert aborted_steps and later_steps
    assert aborted_steps[0].request_row in {step.request_row for step in later_steps}
    assert aborted_steps[0].mamba_slot in {step.mamba_slot for step in later_steps}
    first_later_forward = min(
        record.forward_index
        for record in scenario_records
        if any(step in later_steps for step in record.steps)
    )
    earlier_slots = {
        slot
        for record in run.forwards[:first_later_forward]
        for step in record.steps
        for slot in step.written_slots
    }
    later_slots = {slot for step in later_steps for slot in step.written_slots}
    assert earlier_slots & later_slots
    assert run.forwards[first_later_forward].is_extend
    assert run.forwards[first_later_forward].is_graph_replay == (
        run.kind == "prefill_graph"
    )


def test_retracted_requests_reprefill_their_history_between_replays(
    engine_runs: GraphEngineRuns,
) -> None:
    run = engine_runs.retract.graph
    row_counts = prompt_row_counts(run)
    reprefills = [
        (record.forward_index, step.request_id)
        for record in run.forwards
        if record.is_extend
        for step in record.steps
        if len(step.positions) > row_counts[step.request_id]
    ]
    assert reprefills
    for forward_index, request_id in reprefills:
        steps_around = [
            record.forward_index < forward_index
            for record in run.forwards
            if record.is_graph_replay
            and not record.is_extend
            and any(step.request_id == request_id for step in record.steps)
        ]
        assert True in steps_around and False in steps_around, request_id


def changed_step(
    run: GraphEngineRun, forward_index: int, step: GraphRequestStep
) -> GraphEngineRun:
    """The run with one step of one forward replaced."""
    forwards = list(run.forwards)
    record = forwards[forward_index]
    forwards[forward_index] = replace(
        record,
        steps=[
            step if current.request_id == step.request_id else current
            for current in record.steps
        ],
    )
    return replace(run, forwards=forwards)


def first_padded_decode(run: GraphEngineRun) -> GraphForwardRecord:
    return next(
        record
        for record in run.forwards
        if record.is_graph_replay
        and not record.is_extend
        and len(record.steps) in PADDED_BATCH_SIZES
    )


def test_bit_check_catches_one_changed_bit(engine_runs: GraphEngineRuns) -> None:
    pair = engine_runs.main
    record = first_padded_decode(pair.graph)
    step = record.steps[0]
    flipped_cache = step.written_cache.clone()
    bits(flipped_cache).view(-1)[0] ^= 1
    nudged_logits = step.semantic_logits.clone()
    bits(nudged_logits).view(-1)[0] += 1
    for changed, field in (
        (replace(step, written_cache=flipped_cache), "written key/value cache"),
        (replace(step, semantic_logits=nudged_logits), "semantic logits"),
        (replace(step, state_digest="0" * 64), "linear-attention state"),
        (replace(step, sampled_frame=[step.sampled_frame[0] + 1]), "sampled frame"),
    ):
        changed_pair = replace(
            pair, graph=changed_step(pair.graph, record.forward_index, changed)
        )
        problems = bit_problems(changed_pair)
        assert len(problems) == 1 and field in problems[0], problems


def test_history_check_catches_stale_inputs(engine_runs: GraphEngineRuns) -> None:
    pair = engine_runs.main
    record = first_padded_decode(pair.graph)
    step = record.steps[0]
    stale_window = [EMPTY_RAS_SLOT] + step.staged_ras_window[:-1]
    for changed, field in (
        (replace(step, input_frames=[[0] * len(step.sampled_frame)]), "input frames"),
        (replace(step, staged_ras_window=stale_window), "staged RAS window"),
        (replace(step, model_ras_window=stale_window), "model RAS window"),
        (replace(step, top_k=step.top_k + 1), "sampling knobs"),
    ):
        run = changed_step(pair.graph, record.forward_index, changed)
        problems = history_problems(run, pair.scenarios)
        assert len(problems) == 1 and field in problems[0], problems


def test_schedule_and_draw_checks_catch_a_changed_record(
    engine_runs: GraphEngineRuns,
) -> None:
    pair = engine_runs.main
    record = first_padded_decode(pair.graph)
    step = record.steps[0]
    moved = replace(step, written_slots=[slot + 1 for slot in step.written_slots])
    changed_pair = replace(
        pair, graph=changed_step(pair.graph, record.forward_index, moved)
    )
    assert schedule_problems(changed_pair)

    wrong_choice = replace(
        step, sampled_frame=[step.first_draw + 1, *step.sampled_frame[1:]]
    )
    run = changed_step(pair.graph, record.forward_index, wrong_choice)
    assert len(draw_problems(run)) == 1


def with_both_engines_changed(
    pair: EnginePair, forward_index: int, step: GraphRequestStep
) -> EnginePair:
    """Both engines given the same changed step, so bit comparison cannot tell."""
    return replace(
        pair,
        eager=changed_step(pair.eager, forward_index, step),
        graph=changed_step(pair.graph, forward_index, step),
    )


def test_finite_check_rejects_the_same_bad_value_in_both_engines(
    engine_runs: GraphEngineRuns,
) -> None:
    pair = engine_runs.main
    record = first_padded_decode(pair.graph)
    step = record.steps[0]
    final_record, final_step = next(
        (record, step)
        for record in pair.graph.forwards
        for step in record.steps
        if step.final_state is not None
    )
    bad_cache = step.written_cache.clone()
    bad_cache.view(-1)[0] = float("nan")
    bad_head_logits = step.head_logits.clone()
    bad_head_logits[0, 0] = float("inf")
    bad_semantic_logits = step.semantic_logits.clone()
    bad_semantic_logits[EOS_TOKEN_ID + 1] = float("nan")
    bad_final_state = [tensor.clone() for tensor in final_step.final_state or []]
    bad_final_state[0].view(-1)[0] = float("nan")
    for forward_index, changed, field in (
        (
            record.forward_index,
            replace(step, written_cache=bad_cache),
            "written key/value cache",
        ),
        (
            record.forward_index,
            replace(step, head_logits=bad_head_logits),
            "head logits",
        ),
        (
            record.forward_index,
            replace(step, semantic_logits=bad_semantic_logits),
            "semantic logits",
        ),
        (
            record.forward_index,
            replace(step, is_state_finite=False),
            "linear-attention state",
        ),
        (
            final_record.forward_index,
            replace(final_step, final_state=bad_final_state),
            "final state",
        ),
    ):
        changed_pair = with_both_engines_changed(pair, forward_index, changed)
        assert not bit_problems(changed_pair)
        for run in (changed_pair.eager, changed_pair.graph):
            problems = finite_problems(run)
            assert len(problems) == 1 and field in problems[0], problems


@pytest.mark.parametrize(
    ("end_of_speech_logit", "is_accepted"),
    [
        (float("-inf"), True),
        (1.0e4, True),
        (float("nan"), False),
        (float("inf"), False),
    ],
    ids=["masked", "forced", "nan", "positive-inf"],
)
def test_finite_check_allows_only_a_masked_end_of_speech(
    engine_runs: GraphEngineRuns, end_of_speech_logit: float, is_accepted: bool
) -> None:
    run = engine_runs.main.graph
    record = first_padded_decode(run)
    step = record.steps[0]
    semantic_logits = step.semantic_logits.clone()
    semantic_logits[EOS_TOKEN_ID] = end_of_speech_logit
    changed = replace(step, semantic_logits=semantic_logits)
    problems = finite_problems(changed_step(run, record.forward_index, changed))
    assert (not problems) == is_accepted, problems

    elsewhere = semantic_logits.clone()
    elsewhere[EOS_TOKEN_ID] = float("-inf")
    elsewhere[EOS_TOKEN_ID + 1] = float("-inf")
    changed = replace(step, semantic_logits=elsewhere)
    assert finite_problems(changed_step(run, record.forward_index, changed))
