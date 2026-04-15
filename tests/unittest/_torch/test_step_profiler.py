# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import json
import os
import tempfile

import pytest

from tensorrt_llm._torch.step_profiler import StepProfiler


class MockDist:
    """Mock Distributed object for testing."""

    def __init__(self, rank=0, tp_rank=0):
        self.rank = rank
        self.tp_rank = tp_rank

    def broadcast(self, obj, root=0):
        return obj  # single-process: just return the value


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the StepProfiler singleton between tests."""
    StepProfiler._instance = None
    yield
    StepProfiler._instance = None


@pytest.fixture(autouse=True)
def clean_env():
    """Clean up env vars after each test."""
    yield
    for key in [
            'TRTLLM_STEP_PROFILER', 'TRTLLM_STEP_PROFILER_EVENTS',
            'TRTLLM_STEP_PROFILER_PATH'
    ]:
        os.environ.pop(key, None)


@pytest.fixture
def tmp_output(tmp_path):
    """Provide a temp output directory and set the env var."""
    output_dir = str(tmp_path / "profiler_output")
    os.environ['TRTLLM_STEP_PROFILER_PATH'] = output_dir
    yield output_dir


def _enable(span='0-10', events='*'):
    """Helper to set both required env vars."""
    os.environ['TRTLLM_STEP_PROFILER'] = span
    os.environ['TRTLLM_STEP_PROFILER_EVENTS'] = events


def _find_output(output_dir, rank=0):
    """Find the output file for a given rank (instance ID is random)."""
    matches = glob.glob(os.path.join(output_dir, f"*_rank{rank}.jsonl"))
    assert len(matches) == 1, f"Expected 1 file for rank {rank}, got {matches}"
    return matches[0]


class TestStepProfilerDisabled:

    def test_create_noop_without_env_var(self):
        StepProfiler.create(MockDist())
        assert StepProfiler._instance is None

    def test_create_noop_without_events_var(self):
        os.environ['TRTLLM_STEP_PROFILER'] = '0-10'
        StepProfiler.create(MockDist())
        assert StepProfiler._instance is None

    def test_record_noop_when_disabled(self):
        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=512)

    def test_set_step_noop_when_disabled(self):
        StepProfiler.set_step(100)


class TestStepProfilerEnabled:

    def test_create_with_env_vars(self):
        _enable('10-20')
        StepProfiler.create(MockDist(rank=0))
        assert StepProfiler._instance is not None
        assert StepProfiler._instance.start == 10
        assert StepProfiler._instance.stop == 20
        assert StepProfiler._instance.rank_id == 0

    def test_create_invalid_format(self):
        _enable('bad')
        with pytest.raises(ValueError, match="TRTLLM_STEP_PROFILER"):
            StepProfiler.create(MockDist())

    def test_record_within_range(self, tmp_output):
        _enable('0-3')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128, can_run_graph=True)
        StepProfiler.set_step(1)
        StepProfiler.record("forward", gen_batch_size=256, can_run_graph=True)
        StepProfiler.set_step(2)
        StepProfiler.record("forward", gen_batch_size=512, can_run_graph=False)

        assert len(StepProfiler._instance._events) == 3
        assert StepProfiler._instance._events[0] == {
            "step": 0,
            "event": "forward",
            "gen_batch_size": 128,
            "can_run_graph": True,
        }

    def test_record_outside_range_ignored(self, tmp_output):
        _enable('5-10')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(3)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.set_step(10)
        StepProfiler.record("forward", gen_batch_size=256)

        assert len(StepProfiler._instance._events) == 0

    def test_record_before_set_step_has_null_step(self, tmp_output):
        _enable('0-10')
        StepProfiler.create(MockDist())

        StepProfiler.record("warm_up", phase="start")
        assert len(StepProfiler._instance._events) == 1
        assert StepProfiler._instance._events[0]["step"] is None
        assert StepProfiler._instance._events[0]["event"] == "warm_up"

    def test_save_triggers_at_stop(self, tmp_output):
        _enable('0-2')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.set_step(1)
        StepProfiler.record("forward", gen_batch_size=256)
        StepProfiler.set_step(2)

        output_file = _find_output(tmp_output)
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 2

        record0 = json.loads(lines[0])
        assert record0["step"] == 0
        assert record0["event"] == "forward"
        assert record0["gen_batch_size"] == 128

        record1 = json.loads(lines[1])
        assert record1["step"] == 1
        assert record1["gen_batch_size"] == 256

    def test_multiple_events_per_step(self, tmp_output):
        _enable('0-2')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=512)
        StepProfiler.record("ctx_schedule",
                            trigger="token_ratio",
                            tokens=16689)
        StepProfiler.set_step(1)
        StepProfiler.record("forward", gen_batch_size=256)

        assert len(StepProfiler._instance._events) == 3
        assert StepProfiler._instance._events[1]["event"] == "ctx_schedule"
        assert StepProfiler._instance._events[1]["trigger"] == "token_ratio"

    def test_multi_rank_output(self, tmp_output):
        _enable('0-1')
        StepProfiler.create(MockDist(rank=3, tp_rank=0))

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=64)
        StepProfiler.set_step(1)

        output_file = _find_output(tmp_output, rank=3)
        assert os.path.exists(output_file)

    def test_save_only_once(self, tmp_output):
        _enable('0-2')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.set_step(1)
        StepProfiler.record("forward", gen_batch_size=256)
        StepProfiler.set_step(2)  # triggers save

        output_file = _find_output(tmp_output)
        with open(output_file) as f:
            first_content = f.read()

        StepProfiler.set_step(2)
        with open(output_file) as f:
            second_content = f.read()
        assert first_content == second_content

    def test_list_values_in_events(self, tmp_output):
        _enable('0-1')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward",
                            gen_batch_size=512,
                            graph_key=[512, 0, False, False])
        StepProfiler.set_step(1)

        output_file = _find_output(tmp_output)
        with open(output_file) as f:
            record = json.loads(f.readline())
        assert record["graph_key"] == [512, 0, False, False]

    def test_save_at_exit_if_stop_not_reached(self, tmp_output):
        _enable('0-99999')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.set_step(1)
        StepProfiler.record("forward", gen_batch_size=256)

        assert not StepProfiler._instance._saved
        StepProfiler._instance._save_if_needed()

        output_file = _find_output(tmp_output)
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 2

    def test_instance_id_in_filename(self, tmp_output):
        _enable('0-1')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.set_step(1)

        output_file = _find_output(tmp_output)
        filename = os.path.basename(output_file)
        # Format: {instance_id}_rank{rank}.jsonl
        assert filename.endswith("_rank0.jsonl")
        instance_id = filename.split("_rank")[0]
        assert len(instance_id) == 8  # 8-char hex


class TestEventFilter:

    def test_specific_events_only(self, tmp_output):
        _enable('0-2', 'forward')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.record("ctx_schedule", trigger="token_ratio")

        assert len(StepProfiler._instance._events) == 1
        assert StepProfiler._instance._events[0]["event"] == "forward"

    def test_multiple_events_in_filter(self, tmp_output):
        _enable('0-2', 'forward,ctx_schedule')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.record("ctx_schedule", trigger="token_ratio")
        StepProfiler.record("other_event", value=1)

        assert len(StepProfiler._instance._events) == 2

    def test_wildcard_records_all(self, tmp_output):
        _enable('0-2', '*')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.record("ctx_schedule", trigger="token_ratio")
        StepProfiler.record("anything", value=1)

        assert len(StepProfiler._instance._events) == 3

    def test_whitespace_in_events_trimmed(self, tmp_output):
        _enable('0-2', ' forward , ctx_schedule ')
        StepProfiler.create(MockDist())

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)
        StepProfiler.record("ctx_schedule", trigger="token_ratio")

        assert len(StepProfiler._instance._events) == 2


class TestTpRankFilter:

    def test_tp_rank0_records_by_default(self, tmp_output):
        _enable('0-2')
        StepProfiler.create(MockDist(rank=0, tp_rank=0))

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)

        assert len(StepProfiler._instance._events) == 1

    def test_non_tp_rank0_skipped_by_default(self, tmp_output):
        _enable('0-2')
        StepProfiler.create(MockDist(rank=1, tp_rank=1))

        StepProfiler.set_step(0)
        StepProfiler.record("forward", gen_batch_size=128)

        assert len(StepProfiler._instance._events) == 0

    def test_non_tp_rank0_records_with_all_ranks(self, tmp_output):
        _enable('0-2')
        StepProfiler.create(MockDist(rank=1, tp_rank=1))

        StepProfiler.set_step(0)
        StepProfiler.record("forward", all_ranks=True, gen_batch_size=128)

        assert len(StepProfiler._instance._events) == 1


class TestInstanceId:

    def test_different_instances_different_ids(self, tmp_output):
        _enable('0-1')

        StepProfiler.create(MockDist())
        id1 = StepProfiler._instance.instance_id

        StepProfiler._instance = None
        StepProfiler.create(MockDist())
        id2 = StepProfiler._instance.instance_id

        assert id1 != id2

    def test_broadcast_sets_same_id_across_ranks(self):
        """Simulate broadcast: rank 0 generates, others receive."""
        _enable('0-10')
        shared_id = None

        class BroadcastDist:
            def __init__(self, rank):
                self.rank = rank
                self.tp_rank = rank

            def broadcast(self, obj, root=0):
                nonlocal shared_id
                if self.rank == 0:
                    shared_id = obj
                return shared_id

        StepProfiler.create(BroadcastDist(0))
        id_rank0 = StepProfiler._instance.instance_id

        StepProfiler._instance = None
        StepProfiler.create(BroadcastDist(1))
        id_rank1 = StepProfiler._instance.instance_id

        assert id_rank0 == id_rank1
