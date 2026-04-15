# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""General-purpose per-step event recorder for offline analysis.

Records named events with structured data during model execution.
Events are saved to JSONL files, designed for agent-driven tuning
and workload characterization.

Enable via environment variables:
    TRTLLM_STEP_PROFILER=start-end          # iteration range [start, stop)
    TRTLLM_STEP_PROFILER_EVENTS=ev1,ev2     # which events to record (required)
    TRTLLM_STEP_PROFILER_EVENTS=*           # explicitly record all events
    TRTLLM_STEP_PROFILER_PATH=dir           # output directory (default: step_profiler)

Only tp_rank 0 records by default. Individual record() calls can opt in to
all ranks via all_ranks=True.

Example instrumentation:
    StepProfiler.record("forward", gen_batch_size=512, can_run_graph=True)
    StepProfiler.record("ctx_schedule", trigger="token_ratio", tokens=16689)

Output (step_profiler/a1b2c3d4_rank0.jsonl):
    {"step": 100, "event": "forward", "gen_batch_size": 512, "can_run_graph": true}
    {"step": 101, "event": "ctx_schedule", "trigger": "token_ratio", "tokens": 16689}

Each LLM instance gets a random ID generated on rank 0 and broadcast to all
workers, so files from the same instance share the same prefix.
"""

from __future__ import annotations

import atexit
import functools
import json
import os
import time
import uuid
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed.communicator import Distributed

from tensorrt_llm.logger import logger

EventValue = Union[int, float, str, bool, list, None]


class StepProfiler:
    """Singleton event recorder, gated by TRTLLM_STEP_PROFILER env var."""

    _instance: Optional[StepProfiler] = None

    @staticmethod
    def create(dist: Distributed) -> None:
        """Initialize from TRTLLM_STEP_PROFILER env var. No-op if unset.

        Generates a random instance ID on rank 0 and broadcasts it to all
        workers so that files from the same LLM instance share a prefix.
        """
        span = os.environ.get('TRTLLM_STEP_PROFILER', None)
        if span is None:
            return
        try:
            start, stop = span.strip().split('-')
            start, stop = int(start), int(stop)
        except ValueError as e:
            raise ValueError(
                f"TRTLLM_STEP_PROFILER must be 'start-end', got '{span}': {e}"
            )

        events_str = os.environ.get('TRTLLM_STEP_PROFILER_EVENTS', None)
        if events_str is None:
            logger.warning(
                '[StepProfiler] TRTLLM_STEP_PROFILER is set but '
                'TRTLLM_STEP_PROFILER_EVENTS is not. Set it to a '
                'comma-separated list of event names, or * for all events.')
            return
        if events_str.strip() == '*':
            events_filter = None  # accept all
        else:
            events_filter = set(
                e.strip() for e in events_str.split(',') if e.strip())

        # Generate instance ID on rank 0, broadcast to all workers
        instance_id = uuid.uuid4().hex[:8] if dist.rank == 0 else None
        instance_id = dist.broadcast(instance_id, root=0)

        instance = StepProfiler(dist.rank, dist.tp_rank, instance_id, start,
                                stop, events_filter)
        StepProfiler._instance = instance
        atexit.register(instance._save_if_needed)

    @staticmethod
    def set_step(step: int) -> None:
        """Set current step. Triggers save when step == stop."""
        if StepProfiler._instance is not None:
            StepProfiler._instance._set_step(step)

    @staticmethod
    def record(event: str, *, all_ranks: bool = False,
               **data: EventValue) -> None:
        """Record a named event with key-value data. No-op if disabled.

        By default only records on tp_rank 0. Pass all_ranks=True to
        record on every TP rank.
        """
        if StepProfiler._instance is not None:
            StepProfiler._instance._record(event, all_ranks=all_ranks, **data)

    @staticmethod
    def span(
        event: str,
        *,
        all_ranks: bool = False,
        on_start: Optional[Callable[..., dict]] = None,
        on_end: Optional[Callable[..., dict]] = None,
    ) -> Callable:
        """Decorator that records paired ``phase='start'`` / ``phase='end'``
        events around the wrapped callable. No-op unless the profiler is
        enabled.

        Duration is deliberately **not** emitted; analyzers compute it by
        subtracting the ``ts`` fields of matched records (see the
        ``step_profiler_analyzer`` module).

        Usage::

            @StepProfiler.span("sample")
            def _sample_async(self, ...):
                ...

            @StepProfiler.span(
                "dequeue",
                on_end=lambda rv, self, timeout: {"batch_size": len(rv)},
            )
            def get_from_request_queue(self, timeout):
                ...

        To quickly disable a high-overhead span, comment out the decorator
        line — the underlying function is unaffected.

        Args:
            event: Event name used for both start and end records.
            all_ranks: If True, record on all TP ranks instead of rank 0 only.
            on_start: Optional callable invoked as ``on_start(*args, **kwargs)``
                that returns a dict of extra fields for the ``start`` record.
            on_end: Optional callable invoked as
                ``on_end(return_value, *args, **kwargs)`` that returns a dict
                of extra fields for the ``end`` record.
        """

        def decorator(fn: Callable) -> Callable:

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                if StepProfiler._instance is None:
                    return fn(*args, **kwargs)
                try:
                    start_extra = on_start(
                        *args, **kwargs) if on_start is not None else {}
                except Exception:
                    start_extra = {}
                StepProfiler.record(event,
                                    phase="start",
                                    all_ranks=all_ranks,
                                    **start_extra)
                rv: Any = None
                try:
                    rv = fn(*args, **kwargs)
                    return rv
                finally:
                    try:
                        end_extra = on_end(
                            rv, *args, **
                            kwargs) if on_end is not None else {}
                    except Exception:
                        end_extra = {}
                    StepProfiler.record(event,
                                        phase="end",
                                        all_ranks=all_ranks,
                                        **end_extra)

            return wrapper

        return decorator

    def __init__(self, rank_id: int, tp_rank: int, instance_id: str,
                 start: int, stop: int,
                 events_filter: Optional[set[str]]) -> None:
        self.rank_id = rank_id
        self.tp_rank = tp_rank
        self.instance_id = instance_id
        self.start = start
        self.stop = stop
        self.events_filter = events_filter  # None means accept all
        self.current_step: Optional[int] = None
        self._events: list[dict] = []
        self._saved = False

    def _set_step(self, step: int) -> None:
        self.current_step = step
        if step == self.stop and not self._saved:
            self._save()

    def _record(self, event: str, *, all_ranks: bool = False,
                **data: EventValue) -> None:
        if self.tp_rank != 0 and not all_ranks:
            return
        if self.events_filter is not None and event not in self.events_filter:
            return
        if self.current_step is not None and not (
                self.start <= self.current_step < self.stop):
            return
        record = {"step": self.current_step, "event": event, "ts": time.perf_counter_ns(), **data}
        self._events.append(record)

    def _save_if_needed(self) -> None:
        """Save if there are unsaved events. Called at exit or at stop."""
        if not self._saved and self._events:
            self._save()

    def _save(self) -> None:
        self._saved = True
        path = os.path.expanduser(
            os.environ.get('TRTLLM_STEP_PROFILER_PATH', 'step_profiler'))
        os.makedirs(path, exist_ok=True)
        filepath = os.path.join(
            path, f"{self.instance_id}_rank{self.rank_id}.jsonl")
        with open(filepath, "w") as f:
            for event in self._events:
                f.write(json.dumps(event, separators=(',', ':')) + '\n')
        logger.info(
            f'[StepProfiler] Rank={self.rank_id}, saved {len(self._events)} '
            f'events (iter {self.start}-{self.stop}) to {filepath}')
