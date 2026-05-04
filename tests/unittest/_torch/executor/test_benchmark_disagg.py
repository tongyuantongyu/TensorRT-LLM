# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for benchmark disaggregated serving gating.

In benchmark disagg mode the GEN executor must defer the forward pass
until all benchmark requests have completed KV transfer.  These tests
cover:
- ``is_benchmark_disagg`` initialisation (on :class:`Configuration`)
- ``Ops.is_benchmark_disagg_fill_complete`` (non-ADP and ADP paths)
- ``can_forward`` gating initialisation and transitions
- Incremental fill convergence when CTX has limited KV cache capacity

Post-refactor notes
-------------------
``_is_benchmark_disagg_fill_complete`` and ``_check_benchmark_disagg_gate``
used to be private methods on :class:`PyExecutor`. They've been migrated
to :class:`tensorrt_llm._torch.pyexecutor.executor_loop_ops.Ops` and now
read from ``self._ctx.{service, config, state, port}`` instead of
``self``. The tests build a lightweight :class:`MockOpsCtx` that
populates just enough of the bucket tree for these two methods.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor.executor_loop_context import (
    Configuration,
    PersistentState,
    Service,
)
from tensorrt_llm._torch.pyexecutor.executor_loop_tasks import BenchmarkGateTask, IngestionTask
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gen_request(is_dummy: bool = False) -> Mock:
    """Create a generation request stub with the ``is_attention_dp_dummy`` flag."""
    req = Mock()
    req.is_attention_dp_dummy = is_dummy
    return req


def _make_scheduled_batch(num_gen_requests: int, num_dummy_requests: int = 0) -> ScheduledRequests:
    """Create a ScheduledRequests with generation stubs.

    Args:
        num_gen_requests: Number of real (non-dummy) generation requests.
        num_dummy_requests: Number of ADP dummy generation requests.
    """
    batch = ScheduledRequests()
    batch.generation_requests = [
        _make_gen_request(is_dummy=False) for _ in range(num_gen_requests)
    ] + [_make_gen_request(is_dummy=True) for _ in range(num_dummy_requests)]
    return batch


class _BenchmarkTaskMock(BenchmarkGateTask):
    """BenchmarkGateTask subclass that exposes legacy ``ex.X`` attribute
    names as read-write proxy properties over the ``_ctx.{state,config}``
    bucket tree.

    The old tests treated PyExecutor as a flat object with direct
    ``ex.num_fetch_requests = ...`` writes; after the refactor those
    fields live on ``_ctx.state`` / ``_ctx.config``. These properties
    let the existing test bodies remain untouched.
    """

    # Read/write proxies to state.
    _benchmark_fill_phase_active = property(
        lambda self: self._ctx.state.benchmark_fill_phase_active,
        lambda self, v: setattr(self._ctx.state, "benchmark_fill_phase_active", v),
    )
    num_fetch_requests = property(
        lambda self: self._ctx.state.num_fetch_requests,
        lambda self, v: setattr(self._ctx.state, "num_fetch_requests", v),
    )

    # Aliases for the method names the old tests call through.
    @property
    def _is_benchmark_disagg_fill_complete(self):
        return self.is_benchmark_disagg_fill_complete

    @property
    def _check_benchmark_disagg_gate(self):
        return self.check_benchmark_disagg_gate


def MockBenchmarkExecutor(
    benchmark_req_queues_size: int = 0,
    kv_cache_transceiver=None,
    enable_attention_dp: bool = False,
    tp_size: int = 1,
    rank: int = 0,
    num_fetch_requests: int = 0,
    is_warmup: bool = False,
):
    """Build a ``_BenchmarkOpsMock`` wired to a minimal Ctx that
    populates just the buckets / fields the benchmark-disagg helpers
    touch.
    """
    dist = Mock()
    dist.rank = rank
    dist.tp_size = tp_size

    svc = Service(dist=dist)
    svc.kv_cache_transceiver = kv_cache_transceiver
    state = PersistentState()
    state.is_warmup = is_warmup
    state.num_fetch_requests = num_fetch_requests
    state.benchmark_fill_phase_active = (
        benchmark_req_queues_size > 0 and kv_cache_transceiver is not None
    )
    cfg = Configuration()
    cfg.enable_attention_dp = enable_attention_dp
    cfg.benchmark_req_queues_size = benchmark_req_queues_size
    cfg.is_benchmark_disagg = benchmark_req_queues_size > 0 and kv_cache_transceiver is not None

    ctx = SimpleNamespace(service=svc, config=cfg, state=state)
    task = _BenchmarkTaskMock(ctx)
    task.dist = dist
    task.is_benchmark_disagg = cfg.is_benchmark_disagg
    return task


# ---------------------------------------------------------------------------
# _is_benchmark_disagg_fill_complete  (non-ADP)
# ---------------------------------------------------------------------------


class TestFillCompleteNonADP:
    @pytest.mark.parametrize(
        "num_gen_requests, expected",
        [
            pytest.param(4, True, id="meets_threshold"),
            pytest.param(6, True, id="exceeds_threshold"),
            pytest.param(2, False, id="below_threshold"),
            pytest.param(0, False, id="zero_requests"),
        ],
    )
    def test_threshold(self, num_gen_requests, expected):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        batch = _make_scheduled_batch(num_gen_requests=num_gen_requests)

        assert ex._is_benchmark_disagg_fill_complete(batch) is expected

    def test_no_allgather_called_without_adp(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4, kv_cache_transceiver=Mock(), enable_attention_dp=False
        )
        batch = _make_scheduled_batch(num_gen_requests=4)

        ex._is_benchmark_disagg_fill_complete(batch)
        ex.dist.tp_allgather.assert_not_called()

    def test_logs_progress_on_rank_zero(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4, kv_cache_transceiver=Mock(), rank=0, num_fetch_requests=2
        )
        batch = _make_scheduled_batch(num_gen_requests=1)

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.logger") as mock_logger:
            ex._is_benchmark_disagg_fill_complete(batch)
            mock_logger.debug.assert_called_once()
            msg = mock_logger.debug.call_args[0][0]
            assert "fill in progress" in msg
            assert "num_fetched=2" in msg

    def test_no_log_on_non_zero_rank(self):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock(), rank=1)
        batch = _make_scheduled_batch(num_gen_requests=1)

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.logger") as mock_logger:
            ex._is_benchmark_disagg_fill_complete(batch)
            mock_logger.debug.assert_not_called()


# ---------------------------------------------------------------------------
# _is_benchmark_disagg_fill_complete  (ADP)
# ---------------------------------------------------------------------------


class TestFillCompleteADP:
    @pytest.mark.parametrize(
        "num_gen_requests, allgather_result, expected",
        [
            pytest.param(2, [2, 2, 2, 2], True, id="meets_threshold"),
            pytest.param(3, [3, 3, 3, 3], True, id="exceeds_threshold"),
            pytest.param(1, [1, 1, 1, 0], False, id="below_threshold"),
            pytest.param(5, [5, 1, 1, 1], True, id="uneven_distribution"),
        ],
    )
    def test_threshold(self, num_gen_requests, allgather_result, expected):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=4,
        )
        batch = _make_scheduled_batch(num_gen_requests=num_gen_requests)
        ex.dist.tp_allgather.return_value = allgather_result

        assert ex._is_benchmark_disagg_fill_complete(batch) is expected

    def test_allgather_receives_local_count(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=3)
        ex.dist.tp_allgather.return_value = [3, 1]

        ex._is_benchmark_disagg_fill_complete(batch)
        ex.dist.tp_allgather.assert_called_once_with(3)

    def test_allgather_excludes_dummy_requests(self):
        """Dummy requests must not inflate the local count sent via allgather."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=2, num_dummy_requests=3)
        ex.dist.tp_allgather.return_value = [2, 2]

        ex._is_benchmark_disagg_fill_complete(batch)
        ex.dist.tp_allgather.assert_called_once_with(2)

    def test_logs_progress_on_rank_zero(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
            num_fetch_requests=3,
        )
        batch = _make_scheduled_batch(num_gen_requests=2)
        ex.dist.tp_allgather.return_value = [2, 1]

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.logger") as mock_logger:
            ex._is_benchmark_disagg_fill_complete(batch)
            mock_logger.debug.assert_called_once()
            msg = mock_logger.debug.call_args[0][0]
            assert "total_gen_count=3" in msg
            assert "local=2" in msg

    def test_no_log_on_non_zero_rank(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=8,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
            rank=1,
        )
        batch = _make_scheduled_batch(num_gen_requests=1)
        ex.dist.tp_allgather.return_value = [1, 0]

        with patch("tensorrt_llm._torch.pyexecutor.py_executor.logger") as mock_logger:
            ex._is_benchmark_disagg_fill_complete(batch)
            mock_logger.debug.assert_not_called()


# ---------------------------------------------------------------------------
# ADP regression: mixed real + dummy generation requests
# ---------------------------------------------------------------------------


class TestFillCompleteADPDummyExclusion:
    """Verify that ADP dummy requests do not inflate the fill threshold.

    In ADP, ``_pad_attention_dp_dummy_request`` injects dummy generation
    requests on ranks with no active work.  These dummies must be excluded
    from the ``total_gen_count`` so the ``can_forward`` gate only opens
    after the required number of *real* requests complete KV transfer.
    """

    def test_dummies_do_not_trigger_threshold(self):
        """8 dummies + 0 real must not satisfy a threshold of 4."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=0, num_dummy_requests=8)
        ex.dist.tp_allgather.return_value = [0, 0]

        assert ex._is_benchmark_disagg_fill_complete(batch) is False

    def test_mixed_real_and_dummy_only_counts_real(self):
        """2 real + 3 dummies on each rank: total real = 4, threshold = 4."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=2, num_dummy_requests=3)
        ex.dist.tp_allgather.return_value = [2, 2]

        assert ex._is_benchmark_disagg_fill_complete(batch) is True

    def test_mixed_below_threshold(self):
        """1 real + 5 dummies on each rank: total real = 2, threshold = 4."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=True,
            tp_size=2,
        )
        batch = _make_scheduled_batch(num_gen_requests=1, num_dummy_requests=5)
        ex.dist.tp_allgather.return_value = [1, 1]

        assert ex._is_benchmark_disagg_fill_complete(batch) is False

    def test_non_adp_dummies_excluded(self):
        """Without ADP, dummies should also be excluded from the local count."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=False,
        )
        batch = _make_scheduled_batch(num_gen_requests=2, num_dummy_requests=5)

        assert ex._is_benchmark_disagg_fill_complete(batch) is False

    def test_non_adp_real_only_meets_threshold(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            enable_attention_dp=False,
        )
        batch = _make_scheduled_batch(num_gen_requests=4, num_dummy_requests=3)

        assert ex._is_benchmark_disagg_fill_complete(batch) is True


# ---------------------------------------------------------------------------
# can_forward gating  (unit-level, no real executor loop)
# ---------------------------------------------------------------------------


class TestCanForwardGating:
    """Verify can_forward initialisation and state transitions.

    The can_forward gate is shared by _executor_loop and
    _executor_loop_overlap to defer the forward pass in benchmark
    disagg mode until all requests are generation-ready.
    """

    @pytest.mark.parametrize(
        "benchmark_size, transceiver, is_disagg, can_forward",
        [
            pytest.param(0, None, False, True, id="no_benchmark"),
            pytest.param(8, None, False, True, id="benchmark_without_disagg"),
            pytest.param(0, "mock", False, True, id="disagg_without_benchmark"),
            pytest.param(8, "mock", True, False, id="benchmark_and_disagg"),
        ],
    )
    def test_initial_value(self, benchmark_size, transceiver, is_disagg, can_forward):
        kv = Mock() if transceiver == "mock" else None
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=benchmark_size, kv_cache_transceiver=kv
        )
        assert ex.is_benchmark_disagg is is_disagg
        assert (not ex.is_benchmark_disagg) is can_forward

    @pytest.mark.parametrize(
        "num_gen_requests, expected_after_fill",
        [
            pytest.param(4, True, id="complete_fill"),
            pytest.param(2, False, id="incomplete_fill"),
        ],
    )
    def test_transition(self, num_gen_requests, expected_after_fill):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        can_forward = not ex.is_benchmark_disagg
        assert can_forward is False

        batch = _make_scheduled_batch(num_gen_requests=num_gen_requests)
        can_forward = ex._is_benchmark_disagg_fill_complete(batch)
        assert can_forward is expected_after_fill

    def test_can_forward_stays_true_once_set(self):
        """can_forward is latching: once True it must not revert."""
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        can_forward = not ex.is_benchmark_disagg

        batch_partial = _make_scheduled_batch(num_gen_requests=4)
        can_forward = ex._is_benchmark_disagg_fill_complete(batch_partial)
        assert can_forward is True

        batch_empty = _make_scheduled_batch(num_gen_requests=0)
        # After can_forward is True, the gate is never re-entered in the
        # real loop (guarded by `if not can_forward`).  Verify that calling
        # the helper again with an empty batch would return False, but
        # can_forward itself is not mutated.
        result = ex._is_benchmark_disagg_fill_complete(batch_empty)
        assert result is False
        assert can_forward is True  # local variable unchanged


# ---------------------------------------------------------------------------
# _check_benchmark_disagg_gate  (consolidated gate helper)
# ---------------------------------------------------------------------------


class TestCheckBenchmarkDisaggGate:
    """Verify the consolidated gate helper used by both executor loops."""

    @patch("tensorrt_llm._torch.pyexecutor.executor_loop_tasks.time")
    def test_gate_opens_when_fill_complete(self, mock_time):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        batch = _make_scheduled_batch(num_gen_requests=4)
        assert ex._benchmark_fill_phase_active is True

        can_forward, should_retry = ex._check_benchmark_disagg_gate(batch, False)
        assert can_forward is True
        assert should_retry is False
        assert ex._benchmark_fill_phase_active is False
        mock_time.sleep.assert_not_called()

    @patch("tensorrt_llm._torch.pyexecutor.executor_loop_tasks.time")
    def test_gate_retries_with_short_sleep_when_incomplete(self, mock_time):
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        batch = _make_scheduled_batch(num_gen_requests=1)

        can_forward, should_retry = ex._check_benchmark_disagg_gate(batch, False)
        assert can_forward is False
        assert should_retry is True
        mock_time.sleep.assert_called_once_with(0.1)

    @patch("tensorrt_llm._torch.pyexecutor.executor_loop_tasks.time")
    def test_warmup_bypasses_gate(self, mock_time):
        """During warmup, the gate must not block even in benchmark disagg mode."""
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=4,
            kv_cache_transceiver=Mock(),
            is_warmup=True,
        )
        batch = _make_scheduled_batch(num_gen_requests=0)

        can_forward, should_retry = ex._check_benchmark_disagg_gate(batch, False)
        assert can_forward is False
        assert should_retry is False
        mock_time.sleep.assert_not_called()

    @patch("tensorrt_llm._torch.pyexecutor.executor_loop_tasks.time")
    def test_already_forwarding_skips_check(self, mock_time):
        """Once can_forward is True, the gate is a no-op."""
        ex = MockBenchmarkExecutor(benchmark_req_queues_size=4, kv_cache_transceiver=Mock())
        batch = _make_scheduled_batch(num_gen_requests=0)

        can_forward, should_retry = ex._check_benchmark_disagg_gate(batch, True)
        assert can_forward is True
        assert should_retry is False
        mock_time.sleep.assert_not_called()


# ---------------------------------------------------------------------------
# _pad_attention_dp_dummy_request  (benchmark disagg condition)
# ---------------------------------------------------------------------------


def _make_active_request(in_init: bool = False, in_transfer: bool = False) -> Mock:
    """Create an active request stub for _pad_attention_dp_dummy_request."""
    req = Mock()
    req.is_disagg_generation_init_state = in_init
    req.is_disagg_generation_transmission_in_progress = in_transfer
    return req


class _PadDummyTaskMock(IngestionTask):
    """IngestionTask subclass exposing the legacy
    ``ex._pad_attention_dp_dummy_request()`` names as properties that
    forward to the task-method counterparts.
    """

    @property
    def _pad_attention_dp_dummy_request(self):
        return self.pad_attention_dp_dummy_request

    @property
    def _count_schedulable_active_requests(self):
        return self.count_schedulable_active_requests

    @property
    def _should_skip_dummy_for_benchmark_disagg(self):
        return self.should_skip_dummy_for_benchmark_disagg


def MockPadDummyExecutor(
    *,
    is_benchmark_disagg: bool = False,
    benchmark_fill_phase_active: bool | None = None,
    is_warmup: bool = False,
    enable_attention_dp: bool = True,
    kv_cache_transceiver=None,
    active_requests=None,
    expected_num_active_requests: int = 1,
    num_fetch_requests: int = 0,
    benchmark_req_queues_size: int = 8,
    tp_size: int = 1,
):
    """Build a ``_PadDummyOpsMock`` wired to a lightweight Ctx that
    populates just the buckets / fields
    ``pad_attention_dp_dummy_request`` /
    ``count_schedulable_active_requests`` /
    ``should_skip_dummy_for_benchmark_disagg`` touch.
    """
    dist = Mock()
    dist.tp_size = tp_size

    kv_cache_manager = Mock()
    dummy_req = Mock()
    dummy_req.is_attention_dp_dummy = True
    kv_cache_manager.add_dummy_requests.return_value = [dummy_req]

    resource_manager = Mock()
    resource_manager.get_resource_manager.return_value = None

    svc = Service(dist=dist, resource_manager=resource_manager)
    svc.kv_cache_transceiver = kv_cache_transceiver
    svc.kv_cache_manager = kv_cache_manager

    state = PersistentState()
    state.is_warmup = is_warmup
    state.num_fetch_requests = num_fetch_requests
    state.active_requests = active_requests if active_requests is not None else []
    state.expected_num_active_requests = expected_num_active_requests
    state.max_total_draft_tokens = 0
    state.benchmark_fill_phase_active = (
        benchmark_fill_phase_active
        if benchmark_fill_phase_active is not None
        else is_benchmark_disagg
    )

    cfg = Configuration()
    cfg.enable_attention_dp = enable_attention_dp
    cfg.is_benchmark_disagg = is_benchmark_disagg
    cfg.benchmark_req_queues_size = benchmark_req_queues_size

    ctx = SimpleNamespace(service=svc, config=cfg, state=state)
    task = _PadDummyTaskMock(ctx)
    task.kv_cache_manager = kv_cache_manager
    task.dist = dist
    task.is_benchmark_disagg = is_benchmark_disagg
    return task


class TestPadAttentionDpDummyBenchmarkDisagg:
    """Verify _pad_attention_dp_dummy_request skips dummy insertion correctly.

    During the fill phase (_benchmark_fill_phase_active=True), dummies
    are skipped because the can_forward gate prevents forward-pass
    collectives and stuck dummies would permanently waste KV cache slots.

    Once the fill phase ends (_benchmark_fill_phase_active=False), the
    normal dummy add-forward-terminate lifecycle resumes to handle
    taper-down (ranks emptying at different rates due to e.g. speculative
    decoding acceptance variance).
    """

    def test_skips_during_fill_phase(self):
        """During fill phase, skip dummies."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=[],
            expected_num_active_requests=1,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_skips_during_fill_even_with_requests_in_transfer(self):
        """Fill phase: requests in INIT/transfer, still skip dummies."""
        reqs = [_make_active_request(in_init=True), _make_active_request(in_transfer=True)]
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            kv_cache_transceiver=Mock(),
            active_requests=reqs,
            expected_num_active_requests=3,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_allows_dummy_after_fill_phase_taper_down(self):
        """After fill phase, rank empties due to taper-down: allow dummy."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            benchmark_fill_phase_active=False,
            kv_cache_transceiver=Mock(),
            active_requests=[],
            expected_num_active_requests=1,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_called_once()

    def test_allows_dummy_during_warmup(self):
        """Warmup must bypass the benchmark disagg guard."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            is_warmup=True,
            kv_cache_transceiver=Mock(),
            active_requests=[],
            expected_num_active_requests=1,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_called_once()

    def test_allows_dummy_when_not_benchmark_disagg(self):
        """Non-benchmark or non-disagg mode: normal dummy insertion."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=False,
            active_requests=[],
            expected_num_active_requests=1,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_called_once()

    def test_no_dummy_needed_when_active_requests_ready(self):
        """Rank has ready requests: needs_dummy condition is False."""
        ready_req = _make_active_request(in_init=False, in_transfer=False)
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            benchmark_fill_phase_active=False,
            kv_cache_transceiver=Mock(),
            active_requests=[ready_req],
            expected_num_active_requests=2,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()

    def test_skips_when_adp_disabled(self):
        """_pad_attention_dp_dummy_request early-returns when ADP is off."""
        ex = MockPadDummyExecutor(
            is_benchmark_disagg=True,
            enable_attention_dp=False,
        )
        ex._pad_attention_dp_dummy_request()
        ex.kv_cache_manager.add_dummy_requests.assert_not_called()


# ---------------------------------------------------------------------------
# _prepare_and_schedule_batch is non-blocking
# ---------------------------------------------------------------------------


class TestIncrementalFillScenario:
    """Simulate incremental request arrival when CTX has limited KV cache.

    Setup: the CTX server can only send a few requests per iteration
    (limited KV cache), while the GEN server has enough capacity for all
    requests.  The GEN executor must cycle its main loop — fetching a
    batch, servicing KV transfers, checking readiness — so the CTX
    server can free KV cache and make progress incrementally.

    These tests replay the outer-loop logic over multiple iterations
    with controlled request arrival and KV-transfer completion to verify
    the system converges without blocking.
    """

    TOTAL_REQUESTS = 8
    CTX_CAPACITY = 2  # CTX can only release this many per iteration

    def test_gen_side_processes_requests_incrementally(self):
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=self.TOTAL_REQUESTS,
            kv_cache_transceiver=Mock(),
        )

        fetched = 0
        gen_ready = 0

        def simulate_fetch():
            nonlocal fetched
            new = min(self.CTX_CAPACITY, self.TOTAL_REQUESTS - fetched)
            fetched += new
            ex.num_fetch_requests = fetched

        def simulate_kv_transfer():
            nonlocal gen_ready
            gen_ready = fetched

        can_forward = not ex.is_benchmark_disagg
        assert can_forward is False

        iterations = 0
        MAX_ITER = 50

        while not can_forward and iterations < MAX_ITER:
            simulate_fetch()
            simulate_kv_transfer()
            batch = _make_scheduled_batch(num_gen_requests=gen_ready)
            can_forward = ex._is_benchmark_disagg_fill_complete(batch)
            iterations += 1

        assert can_forward is True
        assert fetched == self.TOTAL_REQUESTS
        assert gen_ready == self.TOTAL_REQUESTS
        assert iterations > 1, "Should take multiple iterations (not a single blocking call)"

    def test_kv_transfer_lag_still_converges(self):
        """KV transfers complete one iteration behind request fetching.

        Iteration 1: fetch 2, gen_ready 0
        Iteration 2: fetch 4, gen_ready 2  (transfers from iter 1 complete)
        ...
        This models the realistic case where KV transfer takes time.
        """
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=self.TOTAL_REQUESTS,
            kv_cache_transceiver=Mock(),
        )

        fetched = 0
        gen_ready = 0
        prev_fetched = 0

        can_forward = not ex.is_benchmark_disagg
        iterations = 0
        MAX_ITER = 50

        while not can_forward and iterations < MAX_ITER:
            gen_ready = prev_fetched
            prev_fetched = fetched

            new = min(self.CTX_CAPACITY, self.TOTAL_REQUESTS - fetched)
            fetched += new
            ex.num_fetch_requests = fetched

            batch = _make_scheduled_batch(num_gen_requests=gen_ready)
            can_forward = ex._is_benchmark_disagg_fill_complete(batch)
            iterations += 1

        assert can_forward is True
        assert gen_ready >= self.TOTAL_REQUESTS
        assert iterations > 2, "With transfer lag, should take more iterations than without"

    def test_single_request_at_a_time(self):
        """Worst case: CTX releases exactly 1 request per iteration."""
        total = 4
        ex = MockBenchmarkExecutor(
            benchmark_req_queues_size=total,
            kv_cache_transceiver=Mock(),
        )

        gen_ready = 0
        can_forward = not ex.is_benchmark_disagg
        iterations = 0

        while not can_forward and iterations < 50:
            gen_ready = min(gen_ready + 1, total)
            ex.num_fetch_requests = gen_ready
            batch = _make_scheduled_batch(num_gen_requests=gen_ready)
            can_forward = ex._is_benchmark_disagg_fill_complete(batch)
            iterations += 1

        assert can_forward is True
        assert iterations == total


@pytest.mark.skip(
    reason="`_prepare_and_schedule_batch` was deleted in the PyExecutor "
    "final-cleanup refactor: its responsibilities are now split across "
    "the `make_request_ingestion`, `make_scheduling`, and `make_batch_gating` "
    "coroutines (plus `Ops.fetch_and_activate_new_requests`). The 'fetch "
    "called once per iteration' invariant is now a property of the "
    "single `driver.resume(request_ingestion, ...)` call at AFTER_FETCH "
    "in the driver body of `PyExecutor._executor_loop`, which is verified "
    "by the coroutine-runtime tests in test_coroutines.py rather "
    "than by a monolithic _prepare_and_schedule_batch test."
)
class TestPrepareAndScheduleBatchNoBlock:
    """Preserved as a documentation placeholder; see skip reason."""

    def test_fetch_called_once_even_in_benchmark_disagg(self):
        pass
