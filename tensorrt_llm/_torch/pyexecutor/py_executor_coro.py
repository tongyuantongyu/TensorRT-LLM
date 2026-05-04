"""Coroutine-based executor (replacement for the legacy ``PyExecutor``).

This module is the loop-thread + main-thread skeleton for the new
coroutine architecture. It deliberately does NOT import from
:mod:`py_executor`; the two modules are independent so the legacy
implementation can be removed once the new one covers all features.
Anything previously living inside ``py_executor.py`` that the new
design still needs should be extracted into a standalone module
(see e.g. ``concerns/`` for the new concern + service classes).

Layout
======

* :class:`PyExecutorCoro` -- main-thread surface (constructor,
  public API, lifecycle). Spawns the loop thread; its
  :meth:`PyExecutorCoro._run_loop` is the loop-thread entry
  point that builds the :class:`Driver` + scheduler coroutine,
  runs it, and tears down concern refs (so concern ``__del__``
  hooks get to drain MPI handles on the loop thread before
  shutdown).
* :func:`scheduler_iter*` -- the SCHEDULER iter (top-level
  coroutine). Decides when to stop and constructs each batch.
* :func:`batch_body` -- per-batch coroutine. Drives concerns through
  every active phase via ``await resume(handle)``.

Bring-up scope
==============

* Single rank only -- no PP, no TP collective fan-out, no ADP.
* Plain (``disable_overlap_scheduler=True``) AND overlap
  (``disable_overlap_scheduler=False``) variants are both wired.
  The scheduler-iter shape differs (``scheduler_iter_plain`` /
  ``scheduler_iter_overlap``) but the per-batch :func:`batch_body`
  is shared: same yield set across the lifecycle, the scheduler
  decides how to interleave one or two batches per iter.
* No optional features (disagg, kv_connector, spec_decode,
  guided_decoder, perf_metric, iter_stats, profile, control,
  benchmark_disagg_gate, dwdp, kv_cache_events, save_hidden_states,
  ring_broadcast_sample). Optional concerns will be added as
  ``Concerns`` fields default to ``None``; their batch_body
  participation is gated by ``if crn.X is not None:`` -- the
  bag is passed as a separate orchestrator-only argument (see
  :class:`Context` for why it is NOT on ``ctx``).

Threading
---------

The legacy ``PyExecutor`` mixed main-thread and loop-thread access
freely. The new design enforces a strict cut:

* Main thread: holds ``self._port`` + ``self._client`` references
  (built during ``__init__``); calls only their thread-safe methods
  (``enqueue_request``, ``await_responses``, ``cancel_request``,
  ``shutdown``).
* Loop thread: receives ``ctx`` via the ``run_loop(ctx)`` thread
  target. NEVER touches ``PyExecutorCoro`` instance attributes.

If you find a code path where main thread reaches anything other
than ``self._port`` / ``self._client`` / ``self._loop_thread``, the
boundary has slipped -- promote the dependency onto MessagePort
or onto a service that exposes a thread-safe write surface.
"""

from __future__ import annotations

import collections
import datetime
import functools
import itertools
import os
import threading
from typing import (TYPE_CHECKING, Any, List, Optional, Union)

import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator

from .batch_storage import BatchPhase, BatchStorage, batch_phase, step, try_step
from .concerns import (ClientChannel, Concerns, ForwardConcern,
                       PpScheduleConcern, RecvOffload, RequestPool,
                       ResourceConcern, ResponseConcern,
                       RingBroadcastSampleConcern, SampleConcern,
                       ScheduleConcern, StateAdvanceConcern,
                       TerminationService, fail_requests)
from .context import (Configuration, Context, MessagePort, PersistentState,
                      Service)
from .coroutines import Batch, Concern, Driver, again, resume, try_resume
from .executor_request_queue import ExecutorRequestQueue
from .pp_helpers import PendingIsend, ring_broadcast_executed_batch_num
from .resource_manager import KVCacheManagerV2, ResourceManagerType

if TYPE_CHECKING:
    from ..distributed import Distributed
    from .llm_request import ExecutorRequest, LlmResponse
    from .model_engine import ModelEngine
    from .resource_manager import ResourceManager
    from .sampler import Sampler
    from .scheduler import RequestScheduler


# --------------------------------------------------------------------------- #
# Profile driver (per-iter; SCHEDULER-layer, not a concern)
# --------------------------------------------------------------------------- #

# Iteration ranges for profiling start/stop. Format:
# ``"start1-stop1,start2-stop2,..."`` or single iters ``"iter1,iter2,..."``.
PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"

# Path to save the torch profiler chrome trace. The rank is appended
# to the filename (``<base>-rank-N<ext>``) so multi-rank runs do not
# overwrite each other.
PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"


@functools.cache
def _load_iteration_indexes(env_var: str):
    """Parse comma-separated iter spans from ``env_var``.

    Returns ``(starts, stops)`` -- two ``frozenset``s of iter
    indexes. Single iters become equal start/stop values; ``"a-b"``
    ranges become matched start/stop pairs. Empty / unset env var
    returns empty sets.

    Mirrors ``py_executor._load_iteration_indexes`` so the new
    coroutine executor honors the same ``TLLM_PROFILE_START_STOP``
    contract as the legacy executor.
    """
    spans = os.environ.get(env_var, None)
    starts, stops = [], []
    if spans:
        for span in spans.split(','):
            try:
                if '-' in span:
                    start, stop = span.strip().split('-')
                    starts.append(int(start))
                    stops.append(int(stop))
                else:
                    it = int(span.strip())
                    starts.append(it)
                    stops.append(it)
            except ValueError as e:
                raise ValueError(
                    f"Cannot parse span in environment variable "
                    f"`{env_var}`: {e}") from None
    return frozenset(starts), frozenset(stops)


def profiler(ctx: Context):
    """Per-iter profiler driver. Generator; yields once per scheduler iter.

    Replaces the legacy ``PyExecutor._profiler`` context manager +
    ``profile_step`` callback. The SCHEDULER iter drives this with a
    plain ``for _ in profiler(ctx):`` loop. The legacy callback ran
    "post(prev iter) -> advance counter -> pre(curr iter)" on each
    invocation because it was a single function called from the iter
    top; the generator splits naturally around the yield -- per-iter
    pre / post bracket the body, with ``it`` advanced by
    ``itertools.count`` starting at 0 for the first post-warmup iter
    (so ``TLLM_PROFILE_START_STOP`` indexes are relative to the
    post-warmup run, not the worker thread's lifetime).

    "Iter" is a SCHEDULER-layer concept (concerns think in batches),
    so the profile driver lives here as a module-level generator
    rather than as a concern -- see :class:`Context` and
    :mod:`concerns` for the layering rationale.

    Drives three independent profilers, all gated on env vars / iter
    ranges (no-op when nothing is configured):

    * **CUDA profiler** (``cudaProfilerStart`` / ``cudaProfilerStop``)
      -- toggles around iter ranges in
      ``TLLM_PROFILE_START_STOP``.
    * **Torch profiler** (``torch.profiler.profile``) -- enabled
      when ``TLLM_TORCH_PROFILE_TRACE`` is set AND
      ``TLLM_PROFILE_START_STOP`` is set. Exports a per-rank chrome
      trace at the stop iter.
    * **Layer-wise benchmark calibrator** -- per-iter
      ``pre_step`` / ``post_step`` callbacks plus start / stop
      bracket calls.

    Warmup pass: a ``while ctx.port.is_warmup: yield`` block at the
    top of the try-body drains warmup iters without touching any
    profile state. Relies on the one-shot ``is_warmup`` invariant
    (``False -> True -> False`` at most once over the executor
    lifetime; see :attr:`MessagePort.is_warmup`) so warmup is
    never re-entered once cleared, and the post-warmup
    ``itertools.count`` block runs at most once.

    Cleanup: on early exit (``return`` from / exception in the
    for-loop body), the suspended generator's ``GeneratorExit``
    runs the ``finally`` clause, which stops the profilers if they
    are still enabled.
    """
    profile_start_iters, profile_stop_iters = _load_iteration_indexes(
        PROFILE_START_STOP_ENV_VAR_NAME)

    enabled = False

    # Append the rank so each rank writes to its own file. Without
    # this, TP/PP/DP > 1 runs have every rank calling
    # ``torch_profiler.export_chrome_trace`` on the same path
    # concurrently, producing interleaved output that fails to parse
    # in Chrome tracing / Perfetto.
    torch_trace_path = os.environ.get(PROFILE_TRACE_ENV_VAR_NAME, None)
    if torch_trace_path is not None:
        trace_base, trace_ext = os.path.splitext(torch_trace_path)
        torch_trace_path = (
            f"{trace_base}-rank-{ctx.svc.dist.rank}{trace_ext}")
    profile_start_stop = os.environ.get(PROFILE_START_STOP_ENV_VAR_NAME,
                                        None)
    enable_torch_trace = bool(torch_trace_path and profile_start_stop)
    if torch_trace_path and profile_start_stop is None:
        logger.warning(
            f"{PROFILE_START_STOP_ENV_VAR_NAME} environment variable "
            "needs to be set to enable the torch trace. Example to "
            f"profile iteration 10-20: export "
            f"{PROFILE_START_STOP_ENV_VAR_NAME}=10-20")

    if enable_torch_trace:
        torch_profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.XPU,
            ],
            record_shapes=True,
            with_modules=True,
        )

    calibrator = get_calibrator()

    try:
        # Warmup drain: tick the SCHEDULER body without touching the
        # profile state. ``is_warmup`` is one-shot (see the property
        # / port docstrings), so this loop runs at most once over
        # the generator's lifetime; the ``itertools.count`` block
        # below then takes over with a fresh ``it = 0``.
        while ctx.port.is_warmup:
            yield -1

        for it in itertools.count():
            assert not ctx.port.is_warmup, "Cannot go back to warmup"

            # Pre-iter: maybe start profiling; calibrator pre-step.
            if it in profile_start_iters:
                assert not enabled, "Inconsistent CUDA profiling state"
                calibrator.start()
                torch.cuda.cudart().cudaProfilerStart()
                if enable_torch_trace:
                    torch_profiler.start()
                logger.info(f"Profiling started at iteration {it}.")
                enabled = True
            calibrator.pre_step(it)

            yield it

            # Post-iter: calibrator post-step; maybe stop profiling.
            calibrator.post_step(it)
            if it in profile_stop_iters:
                assert enabled, "Inconsistent CUDA profiling state"
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(f"Profiling stopped at iteration {it}, "
                                f"trace saved to {torch_trace_path}")
                torch.cuda.cudart().cudaProfilerStop()
                calibrator.stop()
                enabled = False
    finally:
        # Early exit (caller ``return`` / exception, or stop iter
        # never reached): close the profilers if still on. ``it`` is
        # always bound here when ``enabled`` is True -- enabling
        # only happens inside the for-loop body, after ``it`` is
        # set.
        if enabled:
            if enable_torch_trace:
                torch_profiler.stop()
                torch_profiler.export_chrome_trace(torch_trace_path)
                logger.info(f"Profiling stopped at iteration {it}, "
                            f"trace saved to {torch_trace_path}")
            torch.cuda.cudart().cudaProfilerStop()
            calibrator.stop()


# --------------------------------------------------------------------------- #
# Loop layer: scheduler_iter, batch_body, run_loop
# --------------------------------------------------------------------------- #


async def batch_body(ctx: Context, crn: Concerns) -> None:
    """One per-batch coroutine. Drives all participating concerns.

    Loop-agnostic body: the same per-batch lifecycle works for all
    three scheduler-iter variants -- ``scheduler_iter_plain`` (one
    batch per iter, every phase), ``scheduler_iter_overlap``
    (``current`` driven through STATE_UPD_4 interleaved with
    ``previous`` driven through FINALIZE_9 each iter), and
    ``scheduler_iter_pp`` (``in_flight`` deque of ``pp_size - 1``
    parked batches, polling pass + retire). The body's yield set
    is the same across all three; only the scheduler's interleave
    changes.

    Phases yielded:

    * SCHEDULE_0: schedule
    * RESOURCE_PREP_1: resource (prep)
    * FORWARD_2: forward (method, called inline, write batch_outputs).
      Reads ``previous_tensors_device`` /
      ``num_accepted_tokens_device`` from the SCHEDULE_0 view -- the
      OVERLAP scheduler bridges these in from the prior batch's
      ``sample_state``; plain leaves them ``None``.
    * SAMPLE_3: sample (queue sampling kernel + record event).
      Yielded separately from STATE_UPD_4 so the OVERLAP scheduler
      can pause ``current`` between SAMPLE_3 and STATE_UPD_4 to
      run ``previous``'s RESPOND_8 + FINALIZE_9 first -- required
      for the cross-batch ``exclude_last_generation_logits``
      ordering documented on ``StateAdvanceConcern``.
    * STATE_UPD_4: state_advance for both ctx and gen requests
      (chunk-position + GENERATION_* transitions, including the
      overlap-mode ``GENERATION_TO_COMPLETE`` mark). Yielded
      separately from APPLY_7 so the OVERLAP scheduler can pause
      ``current`` here and let ``previous`` resume into APPLY_7.
    * APPLY_7: sample (block on this batch's ``sampler_event``,
      apply sampled tokens to the requests).
    * RESPOND_8: response (build + enqueue + terminate) + resource
      (update_resources / free finished)
    * FINALIZE_9: PP-only multi-phase concerns wait on their
      per-batch isend handles here (``schedule`` waits on the
      schedule-broadcast / items-broadcast isends;
      ``ring_broadcast`` waits on its sample-state isend).
      Becomes the home for ``iter_stats`` / ``perf_metric`` once
      those land.

    The intermediate phases SYNC_EVT_5 and HANDOFF_6 are PP-only.
    :class:`RingBroadcastSampleConcern` runs its body across
    both, with a clean "post" / "wait + send" split:

    * SYNC_EVT_5 -- "post the async HC10 op", one-shot per rank.
      Source: ``synchronize()`` (typically no-op since the GPU
      has had a full iter of runway since admission) +
      ``pp_source_isend`` (non-blocking). Non-source: submit
      the recv future via the offload pool. Driven uniformly
      by the SCHEDULER at step 1b.
    * HANDOFF_6 -- "wait for completion + cross-rank forward
      send". Source: skipped (its full HC10 work was done at
      SYNC_EVT_5). rk0: bounded polling on ``recv_future.done()``
      via SCHEDULER's step 3. Other intermediate ranks: blocking
      ``recv_future.result()`` driven by SCHEDULER's step 2/5
      force-retire / extras-retire.

    The ``async with batch_phase(P): ...`` blocks let the runtime
    interleave concerns at each phase. ``await resume(handle)`` on
    a concern handle drives that concern through the current phase
    and short-circuits if the concern has no work there or has
    already finished.

    ``crn`` is passed as a SEPARATE argument (not via ``ctx``)
    because per the design rule only the orchestrators
    (``run_loop`` / ``scheduler_iter*`` / this body) may reach for
    peer concerns; concerns themselves receive ``ctx`` only. See
    "Why no ``crn`` field" on :class:`Context` for the four-homes
    routing rule for cross-concern data (within-batch ->
    BatchStorage; batch-to-batch handoff -> SCHEDULER bridge;
    cross-batch shared state -> ``ctx.svc.*``; private latch ->
    concern instance attribute).
    """
    schedule = Concern(crn.schedule.handle_batch(ctx))
    resource = Concern(crn.resource.handle_batch(ctx))
    forward = Concern(crn.forward.handle_batch(ctx))
    sample = Concern(crn.sample.handle_batch(ctx))
    state_advance = Concern(crn.state_advance.handle_batch(ctx))
    response = Concern(crn.response.handle_batch(ctx))
    ring_broadcast = (Concern(crn.ring_broadcast.handle_batch(ctx))
                      if crn.ring_broadcast is not None else None)

    async with batch_phase(BatchPhase.SCHEDULE_0):
        await resume(schedule)

    async with batch_phase(BatchPhase.RESOURCE_PREP_1):
        await resume(resource)

    async with batch_phase(BatchPhase.FORWARD_2):
        await resume(forward)

    async with batch_phase(BatchPhase.SAMPLE_3):
        await resume(sample)

    async with batch_phase(BatchPhase.STATE_UPD_4):
        await resume(state_advance)

    # SYNC_EVT_5: PP-only "post the async HC10 op" phase.
    # Source: sync + isend. Non-source: submit recv future. Single
    # ``await resume(ring_broadcast)`` -- the concern body is
    # one-shot per rank, no polling loop here. Single-rank /
    # plain / overlap have ``ring_broadcast=None``; the
    # ``resume(None)`` no-op covers them.
    async with batch_phase(BatchPhase.SYNC_EVT_5):
        await resume(ring_broadcast)

    # HANDOFF_6: PP-only HC10 ring-broadcast hop. Mirrors the
    # bounded polling loop inside ``RingBroadcastSampleConcern``:
    # ``pp_size - 2`` ``try_resume`` rounds with cascade-``again()``
    # if the concern asked to retry, then a final ``resume`` (no
    # retry allowed) which falls into the concern's ``else``-arm
    # blocking wait. Single-rank / plain / overlap have
    # ``ring_broadcast=None``; the concern call short-circuits via
    # the ``resume(None)`` no-op.
    async with batch_phase(BatchPhase.HANDOFF_6):
        for _ in range(ctx.svc.dist.pp_size - 2):
            if await try_resume(ring_broadcast):
                break
            else:
                await again()
        else:
            await resume(ring_broadcast)

    async with batch_phase(BatchPhase.APPLY_7):
        await resume(sample)

    async with batch_phase(BatchPhase.RESPOND_8):
        await resume(response)
        await resume(resource)

    async with batch_phase(BatchPhase.FINALIZE_9):
        await resume(schedule)
        await resume(ring_broadcast)


async def scheduler_iter_plain(ctx: Context, crn: Concerns) -> None:
    """Top-level loop coroutine -- PLAIN variant. One batch per iter.

    A single batch is in flight at a time, no batch-to-batch
    SCHEDULER bookkeeping (no HC1 bridge, ``previous_batch``
    promotion, slot ring, etc.). Two top-level concerns:

    * Shutdown gate: break when shutdown was signalled AND the pool
      / waiting queue are drained.
    * Catastrophic-error handler: if any concern propagates an
      exception, fail every active request via ``fail_requests``,
      flag shutdown on the port, and re-raise so :func:`run_loop`'s
      teardown sees the failure.

    Per-iter SCHEDULER state (iter_counter, etc.) is not needed
    for the plain bring-up; it lands as locals here when
    iter_stats / perf_metric are wired.

    ``crn`` is the concern bag, passed as a SEPARATE arg (not via
    ``ctx``). The orchestrator legitimately needs it to query the
    drain check and to construct the per-batch coroutine.

    The outer ``for _ in profiler(ctx):`` drives the
    SCHEDULER-layer profiler tick (env-var-gated torch / CUDA /
    layer-wise calibrator drivers; warmup iters drained at the top
    of :func:`profiler` before any profile state is touched). See
    :func:`profiler`.
    """
    logger.warning("PLAIN Coro executor")
    for it in profiler(ctx):
        if (ctx.port.is_shutdown
                and ctx.svc.pool.is_drained()
                and crn.schedule.waiting_queue_empty()):
            return

        storage = BatchStorage()
        handle = Batch(batch_body(ctx, crn), storage, idx=it)
        try:
            await step(handle, through=BatchPhase.FINALIZE_9)
        except Exception as exc:
            # Catastrophic failure (Mode A in the failure-handling
            # design). Fail every active request so each client
            # gets a response, flag shutdown, propagate.
            try:
                fail_requests(ctx, list(ctx.svc.pool), str(exc))
            finally:
                ctx.port.is_shutdown = True
                ctx.port.shutdown_event.set()
            raise


async def scheduler_iter_overlap(ctx: Context, crn: Concerns) -> None:
    """Top-level loop coroutine -- OVERLAP variant. Two batches in flight.

    Each iter splits ``current``'s phases at three barrier points
    so ``previous``'s remaining work can interleave (= the overlap)
    AND the cross-batch ordering invariants below are honored:

    Invariants
    ----------

    * **Fresh ``seq_lens`` at curr's SAMPLE_3.** The torch sampler
      kernel reads each request's current ``num_tokens`` to compute
      ``seq_lens`` and decide LENGTH-finish. Without prev's APPLY
      first, curr's sampler sees stale lengths and fails to tag the
      iter's last-eligible token, so ``add_token`` later runs but
      the state never transitions to ``GENERATION_COMPLETE``.
    * **State advance after prev's RESPOND.**
      ``StateAdvanceConcern.handle_batch`` flips
      ``set_exclude_last_generation_logits(False)`` on requests
      transitioning to ``GENERATION_TO_COMPLETE`` (both ctx->gen
      and gen halves). Streaming-mode responses for shared-by-id
      requests in prev use that flag to slice generation logits;
      flipping before prev's RESPOND corrupts the indices.

    Steps (per iter)
    ----------------

    1. SCHEDULE_0 of current.
    2. **HC1 batch-to-batch bridge**: read prev's
       ``sample_state.device`` (populated at prev's SAMPLE_3 in the
       previous iter) and write it as current's
       ``previous_tensors_device`` through the SCHEDULE_0 write
       view. FORWARD_2 reads it back through the read view. Only
       the SCHEDULER does batch-to-batch handoff; concerns do not.
    3. Drive current through FORWARD_2 -- queues forward kernel on
       the execution stream; suspends at SAMPLE_3.
    4. Drive previous through APPLY_7 -- prev's
       ``sampler.update_requests`` blocks on prev's sampler_event
       (host stall, hidden under the GPU running curr's forward
       from step 3 = the overlap), then writes the just-sampled
       tokens onto requests. This is what makes invariant 1 hold.
    5. Drive current through SAMPLE_3 -- queues sampler kernel
       reading the fresh ``num_tokens`` from step 4. Suspends at
       STATE_UPD_4.
    6. Drive previous through FINALIZE_9 -- RESPOND_8 + FINALIZE_9
       (build / enqueue responses, terminate finished, free
       resources). Satisfies invariant 2 wrt curr's STATE_UPD_4
       in step 7.
    7. Drive current through STATE_UPD_4 -- runs
       ``StateAdvanceConcern`` (ctx state advance + gen TO_COMPLETE
       marking). The TO_COMPLETE flag plus the LENGTH finish-
       reason from step 5's sampler kernel together drive the
       request to ``GENERATION_COMPLETE`` during NEXT iter's prev-
       APPLY in step 4 -> prev RESPOND in step 6 sequence.

    Shutdown drain: when no more work to admit AND pool is empty AND
    waiting queue is empty AND no ``previous`` is parked, return.
    Until then the loop keeps iterating; once shutdown is signalled
    but ``previous`` is still parked, ``current`` is built as
    ``None`` and the per-step ``step(None, ...)`` no-ops carry the
    body through unchanged so the only useful work that iter is
    draining ``previous`` to FINALIZE_9. After that iter ``previous``
    is promoted to ``None`` (= the new ``current``) and the next
    iter's termination check returns.

    Catastrophic-error handler: fails the active pool and re-raises.
    """
    previous: Optional[Batch] = None
    # Read view from previous's SAMPLE_3 step -- exposes
    # ``sample_state`` (produced at SAMPLE_3) so step 2's HC1 bridge
    # can read it. Held across iters.
    previous_view = None

    logger.warning("OVERLAP Coro executor")
    for it in profiler(ctx):
        # Termination: nothing to admit AND nothing parked. The loop
        # may still take one extra iter past the shutdown signal --
        # see the docstring's "Shutdown drain" note: when ``previous``
        # is still parked we run one more iter with ``current=None``
        # to drain it, after which the next iter's check returns.
        more_to_admit = (not ctx.port.is_shutdown
                         or not ctx.svc.pool.is_drained()
                         or not crn.schedule.waiting_queue_empty())
        if not more_to_admit and previous is None:
            return

        # Build ``current`` only if we're still admitting work.
        # When draining (``current=None``) the step(None, ...) no-op
        # convention turns every per-step call below into a no-op,
        # so the only effective work this iter is draining
        # ``previous`` to FINALIZE_9.
        current: Optional[Batch] = None
        if more_to_admit:
            storage = BatchStorage()
            current = Batch(batch_body(ctx, crn), storage, idx=it)

        # Body: 7 phase-driving steps + promote, per the docstring's
        # "Steps (per iter)". Steps 1, 3, 5, 7 act on ``current``
        # (no-ops when ``current`` is None during drain); steps 4
        # and 6 advance ``previous``. Step 2 (HC1 bridge) is the
        # only place we write into ``current``'s storage directly
        # rather than via ``step`` -- guard on ``w_sched is not
        # None`` so the no-op path stays no-op.
        try:
            _, w_sched = await step(current, through=BatchPhase.SCHEDULE_0)
            if (w_sched is not None
                    and previous is not None
                    and previous_view.sample_state is not None):
                w_sched.previous_tensors_device = previous_view.sample_state.device
            await step(current, through=BatchPhase.FORWARD_2)
            await step(previous, through=BatchPhase.APPLY_7)
            r_sample, _ = await step(current, through=BatchPhase.SAMPLE_3)
            await step(previous, through=BatchPhase.FINALIZE_9)
            await step(current, through=BatchPhase.STATE_UPD_4)
            previous, previous_view = current, r_sample
        except Exception as exc:
            try:
                fail_requests(ctx, list(ctx.svc.pool), str(exc))
            finally:
                ctx.port.is_shutdown = True
                ctx.port.shutdown_event.set()
            raise


async def scheduler_iter_pp(ctx: Context, crn: Concerns) -> None:
    """Top-level loop coroutine -- PIPELINE PARALLEL variant.

    Works for any ``pp_size >= 2``. Each rank runs an identical
    copy of this scheduler-iter; cross-rank coordination happens
    inside the per-batch concerns and inside two scheduler-direct
    calls -- HC9 (retire-count vote rk0 -> rk(N-1)) and HC10
    (sample-state ring rk(N-1) -> rk0 -> rk1 -> ... -> rk(N-2),
    driven by the per-batch :class:`RingBroadcastSampleConcern`).

    Authority model: rk0 votes opportunism, forced is local
    -------------------------------------------------------

    Per-iter retirement on each rank is split into two parts:

    * **``forced``** -- whether the iter must retire the deadline
      (oldest) batch. Each rank evaluates the SAME predicate
      ``len(in_flight) >= in_flight_max or not more_to_admit``
      locally, NO cross-rank communication. The predicate is in
      lockstep across ranks (see "Lockstep invariant" below), so
      every rank reaches the same answer.
    * **``opp``** -- the opportunistic count, i.e. how many
      additional oldest deque entries (after the deadline) have
      already completed HANDOFF_6 this iter. ONLY rk0 polls
      (counts contiguous head ``try_step`` successes) -- it sees
      HC10 receives FIRST in the ring (rk(N-1) -> rk0 is the
      only ring hop with the source on one side), so rk0 is the
      natural "queue authority" for retire eligibility.
      Downstream ranks always lag in wall-clock for the same
      batch's HC10 progress, so polling them locally would never
      opportunistically retire ahead of rk0. rk0 broadcasts
      ``opp`` via HC9; every non-rk0 rank receives it.

    Total retired this iter on every rank = ``forced + opp``.
    All ranks retire the same set of batches in the same iter,
    keeping the per-rank active-request pool in sync (request
    admission feeds it from the same broadcast items, request
    retirement removes the same ones via this combined drive).

    Lockstep invariant
    ------------------

    The forced-predicate ``len(in_flight) >= in_flight_max or
    not more_to_admit`` is computed locally on every rank and
    must agree across ranks. It does, by induction:

    * ``in_flight_max = pp_size - 1`` is constant.
    * ``len(in_flight)`` mutates by ``+1`` on admit (gated by
      ``more_to_admit``) and ``-K`` on retire (``K = forced +
      opp`` -- ``forced`` lockstep by IH; ``opp`` from broadcast).
      Starting from an empty deque (lockstep base case), each
      iter preserves lockstep.
    * ``more_to_admit = not is_shutdown or not pool.is_drained()
      or not waiting_queue_empty()`` -- each clause is lockstep:
      shutdown markers and request items reach every rank via the
      :func:`pp_broadcast_request_items` chain at SCHEDULE_0;
      pool mutations (admit / retire / unmark_inflight) take
      lockstep inputs; waiting queue is fed by the same broadcast.

    Future code changes that introduce a per-rank source of state
    (e.g. asymmetric request admission, async retirement) would
    break this invariant and require revisiting the vote payload
    to include ``forced`` as well.

    Per-rank in-flight model
    ------------------------

    The SCHEDULER holds a :class:`collections.deque` of in-flight
    :class:`Batch` handles, age order **left=oldest, right=newest**:

    * ``deque[-1]``  (= ``k=1`` -- the most-recently-admitted
      batch, one iter old).
    * ``deque[0]``   (= ``k_max``, the oldest in-flight batch).
      Steady-state ``k_max <= n-1``, equal to n-1 only when no
      opportunistic retirement happened this iter.
    * Intermediate slots: parked. Get rk0-side polling rounds
      via the per-iter ``try_step`` pass below.

    For ``pp_size == 2``: deque max size = 1, so ``deque[-1] ==
    deque[0]``; the same batch's HANDOFF_6 work and FINALIZE_9
    work happen in the same iter. Functionally collapses cleanly
    out of the same code path used for n>2.

    Per-iter steps
    --------------

    Step 1 is split into three sub-steps (1a/1b/1c) so source's
    HC10 isend and rk0's recv submission both happen between
    curr's FORWARD and SAMPLE. This puts wall-clock between
    each rank's SYNC_EVT_5 "post" (1b) and HANDOFF_6's wait
    (step 3 polling on rk0; step 2/5 force-finalize on
    intermediates), giving the recv worker thread time to
    actually run before we check ``future.done()`` -- which is
    what converts source's HC10-send-earlier into actual iter
    savings.

    1a. **Build current; drive through FORWARD_2.** Runs
        SCHEDULE_0 (PP-broadcast + propagated schedule),
        RESOURCE_PREP_1, FORWARD_2 (NCCL p2p hands activations
        to the next PP rank). ``current`` is now suspended at
        SAMPLE_3 entry; the forward kernel is queued on the
        execution stream.
    1b. **Post async ops (uniform across all ranks).** A single
        ``await step(in_flight[-1], through=SYNC_EVT_5)`` drives
        the newest parked batch's SYNC_EVT_5 phase. The phase
        body is rank-specific:

        * **Source**: ``synchronize()`` (typically no-op since
          the GPU has had a full iter of runway) +
          ``pp_source_isend`` (non-blocking).
        * **Non-source**: ``pp_post_recv_sample_state`` (submits
          the blocking ``recv_object`` to the offload thread
          pool; returns immediately with a future).

        Older parked batches were driven at their own
        admit_iter+1's step 1b -- inductive invariant: each
        batch's SYNC_EVT_5 post happens exactly once at the iter
        after admission. ``step`` is idempotent past ``through``
        so calls on already-past-SYNC_EVT_5 batches are no-ops.
    1c. **Drive ``current`` through STATE_UPD_4.** Runs SAMPLE_3
        (real on last rank -- ``sample_async`` queues kernel +
        D2H, records ``sampler_event``; placeholder elsewhere)
        and STATE_UPD_4. The CPU wall-clock here (sample_async
        is multi-ms in production) is the window in which rk0's
        recv worker thread runs and marks the future done, so
        the polling at step 3 can detect completion at iter T+1
        instead of iter T+2.
    2.  **Forced retire (deadline; symmetric across ranks).** If
        the deque is full or we're draining, pop the oldest and
        ``step(.., FINALIZE_9)`` it. On rk0 this is the polling
        concern's deadline ``else``-arm blocking ``result()``; on
        non-rk0 intermediate ranks it's the concern's direct
        blocking-wait branch; on the source rank it's a fast
        pass-through (concern already past HANDOFF_6 from step
        1b's SYNC_EVT_5 sync+isend). ``in_flight`` non-empty is
        implied: ``in_flight_max >= 1`` rules out the ``len() >=
        in_flight_max`` arm, and the early ``return`` above rules
        out the ``not more_to_admit`` arm with empty deque.
    3.  **Polling pass (rk0 only).** rk0 calls
        ``opportunistic_polling``, which iterates parked batches
        and ``try_step``s each through HANDOFF_6. Each parked's
        first ``try_step`` here runs the polling concern body's
        ``for`` loop iter 0: check ``recv_future.done()``; if
        True, break + apply + intermediate isend, past HANDOFF_6.
        Count contiguous head successes as ``opp``; stop counting
        at the first ``try_step`` retry but keep iterating so
        later batches' polling counters advance toward future
        deadlines. Source doesn't poll here -- its full HC10
        work (sync + isend) is done at step 1b. Other
        intermediate ranks (rk1..rk(N-3)) skip this pass too --
        their HC10 branch is the direct blocking-wait variant
        and would stall the SCHEDULER if try_step'd.
    4.  **HC9 vote.** Send ``opp`` -- only rk0's opportunistic
        count is on the wire; ``forced`` is recomputed locally
        on every rank (lockstep). Sent along the PP forward chain
        (rk0 -> rk1 -> ... -> rk(N-1)) via non-blocking
        ``isend_object`` with synchronous ``recv_object`` on every
        non-rk0 rank. Each rank assigns the received value back
        to ``opp`` -- on rk0 this is identity (its own value), on
        non-rk0 it's rk0's authoritative count. The forward isend
        handle is parked in a 1-element :class:`PendingIsend`;
        next iter's ``vote.set`` waits the previous handle and
        stores the new one (one iter of slack -> non-blocking).
    5.  **Retire ``opp`` more.** Pop ``opp`` more batches from
        deque head, ``step(.., FINALIZE_9)`` each. On rk0 these
        are the opportunistic ones (already past HANDOFF_6 from
        step 3, so ``step`` just runs APPLY_7 + RESPOND_8 +
        FINALIZE_9 -- releases KV resources, sends responses,
        hands isend handle to lingering queue). On non-rk0
        intermediate ranks each ``step`` enters the concern's
        ``recv_future.result()`` blocking branch -- which
        completes promptly because rk0 wouldn't have voted
        ``opp`` unless its matching ``opp`` isends had been
        posted. On the source rank each ``step`` is a fast
        pass-through. Total retired this iter on every rank:
        ``forced + opp``.
    6.  **Push ``current`` to deque right** (now k=0 this iter,
        becomes k=1 next iter). Skip when draining.

    Polling deadline (rk0)
    ----------------------

    rk0's polling concern body has a ``for _ in range(pp_size -
    2): if recv_future.done(): break; await again()`` loop with a
    post-loop ``recv_future.result()`` deadline (or the batch_body
    HANDOFF_6 ``else`` arm at force-retire). One pass per iter
    (step 3 only): the deadline lands at iter ``T + (pp_size - 1)``
    -- the legacy worst case. For fast-MPI cases (sub-iter
    latency, the typical case in production) the ``break`` fires
    at iter T+1, retiring 1 iter after admission. The semantic
    cleanup here -- SYNC_EVT_5 "post" decoupled from HANDOFF_6
    "wait + send" -- prioritizes uniformity over the previous
    2-passes/iter design's halved worst-case deadline; if perf
    data ever shows the worst case matters for slow-MPI on high
    pp_size, a second polling pass (e.g. at step 5) can be
    re-introduced.

    Last-rank fast path (special behavior)
    --------------------------------------

    The last PP rank is the HC10 source -- its ``handle_batch``
    HANDOFF_6 body just issues a non-blocking ``pp_source_isend``
    and returns. So every batch on the last rank is trivially
    past HANDOFF_6 the moment its STATE_UPD_4 completes; both
    the deadline step (2) and the extra-retire step (5) are
    fast pass-throughs through the concern body, never blocking
    on a recv (since there isn't one). The HC9 vote chain still
    propagates through the last rank as a follower (it receives
    K and uses it to drive its own deque), to keep the active
    pool in lockstep with rk0's retirement decisions.

    Pending-isend reap
    ------------------

    Two pending-isend reservoirs ride along with the loop:

    * **HC9 vote** -- one ``PendingIsend`` slot. ``vote.set(new)``
      at step (4) waits the previous iter's handle (one iter of
      slack matches the next-PP-rank's recv latency, so
      non-blocking) and stores the new one. The trailing handle
      is waited by ``PendingIsend.__del__`` when this function
      returns and the local goes out of scope.
    * **HC10 ring-broadcast isends** -- a per-rank deque on
      :class:`RingBroadcastSampleConcern`, self-bounded to
      ``pp_size`` handles on push (waits + pops the oldest when
      a new push arrives on a full deque). Trailing handles are
      waited by :meth:`RingBroadcastSampleConcern.__del__`.

    Both destructors fire on the loop thread before it exits,
    courtesy of the ref-nulling in
    :meth:`PyExecutorCoro._run_loop`. The SCHEDULER body itself
    therefore needs no explicit final-drain calls.

    Shutdown drain
    --------------

    When ``more_to_admit`` flips False, the loop keeps running
    iters with ``current=None`` (the per-step ``step(None, ...)``
    no-ops carry the body through unchanged) until ``in_flight``
    drains. Each drain iter still does steps 2-5 so the PP
    forward chain stays in lockstep across ranks. Step (2)'s
    forced-retire condition includes ``not more_to_admit``, so
    the deque is drained at least one batch per iter --
    ``pp_size - 1`` iters total in the worst case (less if
    opportunistic retirement kicks in during drain too). After
    the deque is empty AND no new batch is admitted, the loop
    returns; pending-isend cleanup rides on the GC chain
    documented above.
    """
    if ctx.svc.dist.pp_size < 2:
        raise NotImplementedError(
            "scheduler_iter_pp: requires pp_size >= 2; "
            f"got pp_size={ctx.svc.dist.pp_size}.")
    if crn.ring_broadcast is None:
        raise RuntimeError(
            "scheduler_iter_pp: requires Concerns.ring_broadcast to be "
            "set (PyExecutorCoro.__init__ wires this when pp_size > 1).")

    logger.warning("PP Coro executor")
    pp_size = ctx.svc.dist.pp_size
    # Per-rank in-flight batch ring. Left=oldest, right=newest.
    # Max size = n-1; current iter's opportunistic retirement may
    # leave it smaller (the next iter's admit refills it back up
    # to the max).
    in_flight: collections.deque = collections.deque()
    in_flight_max = pp_size - 1
    # HC9 vote isend slot: ``vote.set(new_handle)`` waits the
    # previous iter's handle (one iter of slack matches the
    # next-PP-rank's recv latency, so non-blocking) and stores
    # the new one. Trailing handle is waited by ``__del__`` when
    # this function returns and the local goes out of scope.
    vote = PendingIsend()

    async def opportunistic_polling():
        finished_count = 0
        counting = True
        for parked in in_flight:
            result = await try_step(parked,
                                    through=BatchPhase.HANDOFF_6)
            if result is None:
                counting = False
            elif counting:
                finished_count += 1
        return finished_count

    for it in profiler(ctx):
        more_to_admit = (not ctx.port.is_shutdown
                         or not ctx.svc.pool.is_drained()
                         or not crn.schedule.waiting_queue_empty())
        if not more_to_admit and not in_flight:
            return

        # Build ``current`` only if we're still admitting work.
        # When draining (``current=None``) the step(None, ...)
        # no-op convention turns the curr-side calls below into
        # no-ops, leaving the polling pass + retire as the only
        # effective work this iter.
        current: Optional[Batch] = None
        if more_to_admit:
            storage = BatchStorage()
            current = Batch(batch_body(ctx, crn), storage, idx=it)

        # Per-iter steps (numbered to match the docstring's
        # "Per-iter steps" section). Step 1 is split into
        # 1a/1b/1c so each rank's SYNC_EVT_5 "post async op"
        # happens between curr's FORWARD and SAMPLE; HANDOFF_6's
        # "wait + forward send" happens later (step 3 polling on
        # rk0; step 2/5 force-finalize on intermediates):
        #
        # (1a) Drive ``current`` through FORWARD_2 (queue forward
        #      kernel; no GPU sync).
        # (1b) Drive newest parked through SYNC_EVT_5 (uniform
        #      across ranks). Source: sync + isend. Non-source:
        #      submit recv future. No polling, no count -- just
        #      one-shot "post the async op". Older parked were
        #      driven at their own admit_iter+1's step 1b
        #      (idempotent past ``through``).
        # (1c) Drive ``current`` through STATE_UPD_4 (sample_async
        #      + state advance). Wall-clock here gives rk0's recv
        #      worker thread time to run so step 3's polling sees
        #      ``future.done() == True`` for recvs that landed.
        # (2)  Forced retire of the deadline batch. Each rank
        #      evaluates the same forced-predicate locally; the
        #      predicate is in lockstep across ranks because
        #      ``len(in_flight)`` and ``more_to_admit`` are both
        #      lockstep (see "Lockstep invariant" in the docstring).
        #      ``in_flight`` non-empty is implied:
        #      ``in_flight_max >= 1`` rules out the ``len() >=
        #      in_flight_max`` arm, and the early ``return`` above
        #      rules out the ``not more_to_admit`` arm with empty
        #      deque.
        # (3)  Polling pass on rk0 only. Each parked batch's
        #      ``try_step(parked, HANDOFF_6)`` runs the polling
        #      concern body's ``for`` loop iter 0: check
        #      ``recv_future.done()``; if True, break + apply +
        #      intermediate isend. Count contiguous head successes
        #      as ``opp``; stop counting at first retry but keep
        #      iterating so later batches' for-loop counters
        #      advance. Source doesn't poll here -- its full HC10
        #      work was done at step 1b. Other intermediates
        #      (rk1..rk(N-3)) skip too -- direct blocking-wait
        #      branch would stall the SCHEDULER if try_step'd.
        # (4)  HC9 vote. rk0 broadcasts only ``opp`` -- ``forced``
        #      is computed identically on every rank from the
        #      lockstep state, so it doesn't need to be on the
        #      wire. Non-rk0 ranks' local ``opp`` (always 0 since
        #      they didn't poll for opp) is overwritten by the recv.
        # (5)  Retire ``opp`` more batches. On rk0 these are the
        #      opportunistic ones already past HANDOFF_6 from
        #      step 3 (``step`` just runs FINALIZE_9). On non-rk0
        #      intermediate ranks ``step`` drives the concern's
        #      blocking-wait branch for each. On source ``step``
        #      is a fast pass-through. Total retired this iter on
        #      every rank: ``forced + opp``.
        # (6)  Push current to deque right (skip when draining).
        try:
            # (1a)
            await step(current, through=BatchPhase.FORWARD_2)

            # (1b)
            if in_flight:
                await step(in_flight[-1], through=BatchPhase.SYNC_EVT_5)

            # (1c)
            await step(current, through=BatchPhase.STATE_UPD_4)

            # (2)
            if len(in_flight) >= in_flight_max or not more_to_admit:
                deadline = in_flight.popleft()
                await step(deadline, through=BatchPhase.FINALIZE_9)

            # (3)
            opp = 0
            if ctx.svc.dist.is_first_pp_rank:
                opp = await opportunistic_polling()

            # (4)
            opp, vote_handle = ring_broadcast_executed_batch_num(
                dist=ctx.svc.dist,
                executed_batch_num=opp,
            )
            vote.set(vote_handle)

            # (5)
            for _ in range(opp):
                batch = in_flight.popleft()
                await step(batch, through=BatchPhase.FINALIZE_9)

            # (6)
            if current is not None:
                in_flight.append(current)
        except Exception as exc:
            try:
                fail_requests(ctx, list(ctx.svc.pool), str(exc))
            finally:
                ctx.port.is_shutdown = True
                ctx.port.shutdown_event.set()
            raise


# Type alias for the scheduler-iter shape: ``async def fn(ctx, crn) -> None``.
# Used by ``PyExecutorCoro`` to pick the variant.
SchedulerIterFn = Any  # Callable[[Context, Concerns], Coroutine] but kept
# loose so we don't have to import Coroutine here.


# --------------------------------------------------------------------------- #
# Main-thread surface
# --------------------------------------------------------------------------- #


class _LlmRequestWaitingQueue(collections.deque):
    """Plain-loop waiting queue: a deque of LlmRequests.

    Differs from the legacy ``WaitingQueue`` ABC (which holds
    ``RequestQueueItem``s and is converted to ``LlmRequest`` lazily
    on pop). Here the conversion happens at admit time
    (``ScheduleConcern._admit_new_requests``), so the queue holds
    ready-to-schedule ``LlmRequest`` objects.

    The legacy ABC's ``add_requests`` / ``pop_request`` /
    ``peek_request`` / ``remove_by_ids`` are NOT implemented because
    the plain-loop bring-up doesn't need them. PP / cancellation
    paths can re-add the missing methods (or swap to the legacy ABC)
    as those features land.
    """


class PyExecutorCoro:
    """Coroutine-based executor.

    Same external contract as the legacy ``PyExecutor`` for the
    plain-loop scope: enqueue requests via :meth:`enqueue_request` /
    :meth:`enqueue_requests`, await responses via
    :meth:`await_responses`, cancel via :meth:`cancel_request`, shut
    down via :meth:`shutdown`. Internal layout is reorganised around
    the new architecture (concerns + services + Context + Driver).

    Construction
    ------------

    The constructor takes the same heavy dependencies the legacy
    ``PyExecutor`` does (``model_engine``, ``sampler``, ``scheduler``,
    ``resource_manager``, ``dist``, ...). Instead of stashing them
    as ``self.X`` attributes used freely from anywhere, this class:

    1. Builds the loop-thread services (``RequestPool``,
       ``ClientChannel``, ``TerminationService``).
    2. Constructs each concern with its owned dependencies via plain
       kwargs.
    3. Bundles services + concerns into a :class:`Context` and
       ships the context to the loop thread.
    4. Keeps only ``self._port`` (cross-thread channel) and
       ``self._client`` (response surface) for the public API.

    The constructor runs WARMUP synchronously (model_engine.warmup)
    BEFORE the loop thread starts, mirroring the legacy ordering.
    """

    def __init__(
        self,
        # Positional to match the legacy ``PyExecutor`` signature
        # so call sites in ``_util.create_py_executor_instance`` can
        # construct ``PyExecutorCoro`` without keyword churn.
        resource_manager: "ResourceManager",
        scheduler: "RequestScheduler",
        model_engine: "ModelEngine",
        sampler: "Sampler",
        dist: "Distributed",
        max_num_sequences: int,
        # Optional config knobs (signature parity with the legacy
        # ``PyExecutor.__init__``). Anything we don't yet support is
        # rejected by guard checks below; signature-only kwargs are
        # accepted and ignored so the legacy call site can be reused
        # unchanged.
        drafter: Optional[Any] = None,
        disable_overlap_scheduler: bool = False,
        max_input_len: int = 0x7FFFFFFF,
        max_batch_size: int = 8,
        max_beam_width: int = 1,
        max_draft_len: int = 0,
        max_total_draft_tokens: int = 0,
        kv_cache_transceiver: Optional[Any] = None,
        guided_decoder: Optional[Any] = None,
        garbage_collection_gen0_threshold: Optional[int] = None,
        start_worker: bool = True,
        kv_connector_manager: Optional[Any] = None,
        max_seq_len: Optional[int] = None,
        peft_cache_config: Optional[Any] = None,
        virtual_memory_pools: Optional[Any] = None,
        hang_detection_timeout: Optional[int] = None,
        execution_stream: Optional[torch.cuda.Stream] = None,
        waiting_queue_policy: Optional[Any] = None,
        adp_router: Optional[Any] = None,
        dwdp_manager: Optional[Any] = None,
    ) -> None:
        # ----- Reject features the bring-up doesn't handle yet.
        # Plain (``disable_overlap_scheduler=True``) and overlap
        # (``disable_overlap_scheduler=False``) are both supported;
        # the difference is which scheduler-iter variant runs (see
        # ``scheduler_iter_plain`` / ``scheduler_iter_overlap``).
        if drafter is not None:
            raise NotImplementedError(
                "PyExecutorCoro: spec-decode (drafter) is not yet "
                "implemented; SpecDecodeConcern is in the planned set.")
        if kv_cache_transceiver is not None:
            raise NotImplementedError(
                "PyExecutorCoro: disagg KV transceiver is not yet "
                "implemented; DisaggConcern is in the planned set.")
        if guided_decoder is not None:
            raise NotImplementedError(
                "PyExecutorCoro: guided decoder is not yet "
                "implemented; GuidedDecoderConcern is in the planned set.")
        if kv_connector_manager is not None:
            raise NotImplementedError(
                "PyExecutorCoro: KV connector is not yet "
                "implemented; KvConnectorConcern is in the planned set.")
        if dwdp_manager is not None:
            raise NotImplementedError(
                "PyExecutorCoro: DWDP is not yet implemented; "
                "DwdpConcern is in the planned set.")
        if max_beam_width != 1:
            raise NotImplementedError(
                f"PyExecutorCoro: only beam_width=1 is supported in the "
                f"plain-loop bring-up (got max_beam_width={max_beam_width})."
            )
        if dist.pp_size > 1 and not disable_overlap_scheduler:
            # Bring-up: PP loop only with non-overlap mode (the
            # SCHEDULER's HC10 hop already gives the ring its
            # cross-iter overlap; combining with the in-rank
            # overlap path needs the spec_decode HC1 in-bridge,
            # which isn't wired yet).
            raise NotImplementedError(
                "PyExecutorCoro: PP + overlap_scheduler is not yet "
                "implemented; use disable_overlap_scheduler=True with "
                "pp_size>1.")

        logger.warning("Using prototype Coro Executor!")

        # Signature-only kwargs (parity but not used by the plain-
        # loop bring-up): max_draft_len, max_total_draft_tokens
        # (only meaningful with a drafter, already rejected above);
        # garbage_collection_gen0_threshold (was a wrap-around the
        # legacy ``_event_loop_wrapper``); virtual_memory_pools
        # (used by other concerns); waiting_queue_policy (FCFS
        # hardcoded); hang_detection_timeout (Driver watchdog not
        # enabled); adp_router (ADP not supported). Silenced via
        # underscore.
        del max_draft_len, max_total_draft_tokens
        del garbage_collection_gen0_threshold
        del virtual_memory_pools
        del waiting_queue_policy, hang_detection_timeout, adp_router
        self._dist = dist
        self._device_id = torch.cuda.current_device()
        self._execution_stream = (
            execution_stream
            if execution_stream is not None else torch.cuda.Stream())

        # ----- Warmup (main-thread, before loop start). -----
        # Legacy ordering: warmup runs on the execution stream and
        # the default stream waits for it before subsequent ops.
        # The ``model_engine.is_warmup`` flag gates several model-
        # internal code paths (torch.compile bootstrapping, MoE
        # load-balancer stat skip, etc.); set / clear it around the
        # warmup call.
        model_engine.is_warmup = True
        self._execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._execution_stream):
            model_engine.warmup(resource_manager)
        torch.cuda.current_stream().wait_stream(self._execution_stream)
        model_engine.is_warmup = False

        # ----- Cross-thread channels (MessagePort). -----
        executor_request_queue = ExecutorRequestQueue(
            dist=dist,
            max_batch_size=max_batch_size,
            enable_iter_perf_stats=False,
            batch_wait_timeout_ms=0,
        )
        port = MessagePort(executor_request_queue=executor_request_queue)

        # ----- Services. -----
        pool = RequestPool()
        client = ClientChannel()
        termination = TerminationService(
            resource_manager=resource_manager,
            client=client,
            rank=dist.rank,
            gather_all_responses=False,
        )
        # Single-worker thread pool that runs blocking ``recv_object``
        # so coroutines on the loop thread can poll the resulting
        # future. PP-only; on single-rank executors no concern needs
        # async recv. Shut down explicitly in ``_run_loop``'s
        # teardown ``finally`` -- see "Lifecycle" on
        # :class:`RecvOffload`.
        recv_offload = RecvOffload() if dist.pp_size > 1 else None
        svc = Service(
            dist=dist,
            pool=pool,
            client=client,
            termination=termination,
            recv_offload=recv_offload,
        )

        # ----- Concerns. -----
        waiting_queue = _LlmRequestWaitingQueue()
        # V2 KV cache manager owns KV suspend internally, so V1
        # paths still go through the legacy ``terminate / pause``
        # of paused requests in ScheduleConcern.
        kv_cache_manager = resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        scheduler_manages_kv_suspend = isinstance(kv_cache_manager,
                                                  KVCacheManagerV2)

        # In overlap mode (single rank), gen-logits are excluded
        # because the most recent token in ``generation_logits`` is
        # the "next" token speculatively decoded by the overlap
        # path and not yet applied -- mirrors the legacy
        # ``should_exclude_last_generation_logits``.
        exclude_last_generation_logits = (not disable_overlap_scheduler
                                          and dist.pp_size == 1)

        # Schedule concern: PP variant when pp_size>1 (does
        # rk0-side schedule + ring propagate + inflight tracking),
        # plain otherwise. Same constructor surface (the PP variant
        # subclasses the plain one and adds no ctor args -- per-
        # batch isend handles live as locals on the per-batch
        # coroutine).
        schedule_cls = PpScheduleConcern if dist.pp_size > 1 else ScheduleConcern
        schedule_concern = schedule_cls(
            scheduler=scheduler,
            sampler=sampler,
            waiting_queue=waiting_queue,
            max_num_active_requests=model_engine.get_max_num_sequences(),
            exclude_last_generation_logits=exclude_last_generation_logits,
            scheduler_manages_kv_suspend=scheduler_manages_kv_suspend,
            max_input_len=max_input_len,
        )
        # Per-batch HC10 ring-broadcast concern. Single instance,
        # invoked once per :class:`Batch` via
        # :meth:`RingBroadcastSampleConcern.handle_batch`. No ctor
        # state -- each per-batch coroutine holds its own isend
        # handle as a local and waits at FINALIZE_9.
        ring_broadcast_concern = (
            RingBroadcastSampleConcern() if dist.pp_size > 1 else None)

        crn = Concerns(
            schedule=schedule_concern,
            resource=ResourceConcern(
                resource_manager=resource_manager,
                kv_cache_dtype_byte_size=model_engine.kv_cache_dtype_byte_size,
            ),
            forward=ForwardConcern(
                model_engine=model_engine,
                resource_manager=resource_manager,
                sampler=sampler,
                execution_stream=self._execution_stream,
            ),
            sample=SampleConcern(
                sampler=sampler,
                resource_manager=resource_manager,
            ),
            state_advance=StateAdvanceConcern(
                disable_overlap_scheduler=disable_overlap_scheduler,
            ),
            response=ResponseConcern(
                rank=dist.rank,
                stream_interval=getattr(model_engine.llm_args,
                                         "stream_interval", 1),
            ),
            ring_broadcast=ring_broadcast_concern,
        )

        # ----- Configuration + PersistentState. -----
        conf = Configuration(device_id=self._device_id)
        state = PersistentState()

        # ----- Final Context. -----
        # ``crn`` is intentionally NOT a Context field -- only the
        # orchestrators (run_loop / scheduler_iter / batch_body)
        # legitimately need it. Concerns receive ``ctx`` only and
        # cannot reach for peer concerns.
        ctx = Context(svc=svc, conf=conf, state=state, port=port)

        # ----- Main-thread retains only the boundary refs. -----
        self._port = port
        self._client = client
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ctx = ctx
        # Concern bag is loop-thread state; stash it here so
        # ``start_worker`` can hand it to ``run_loop`` as a
        # separate arg.
        self._loop_crn = crn
        # SCHEDULER iter variant -- plain / overlap / PP.
        if dist.pp_size > 1:
            self._scheduler_iter_fn = scheduler_iter_pp
        elif disable_overlap_scheduler:
            self._scheduler_iter_fn = scheduler_iter_plain
        else:
            self._scheduler_iter_fn = scheduler_iter_overlap

        # ----- Legacy-API surface stash. -----
        # The LLM-API layer (in ``executor.base_worker`` and
        # ``py_executor_creator``) reads several attributes off the
        # constructed engine to wire up LoRA / disagg / stats
        # plumbing AND to drive the KV-cache memory-estimation pass
        # (see ``_util.KvCacheCreator.configure_kv_cache_capacity``).
        # Re-expose them here so a monkey-patched
        # ``PyExecutor -> PyExecutorCoro`` swap stays drop-in
        # compatible; they are NOT touched by concerns (the loop
        # thread reaches everything via ``ctx``).
        self.peft_cache_config = peft_cache_config
        self.kv_cache_transceiver = None  # plain loop rejects non-None.
        self.resource_manager = resource_manager
        self.model_engine = model_engine
        # ``base_worker._create_py_executor`` reads ``max_seq_len``
        # off the constructed engine to feed back into the LLM args.
        self.max_seq_len = max_seq_len
        # ``configure_kv_cache_capacity`` reads ``dist.mapping.rank``
        # / ``dist.broadcast``. The legacy executor stored ``dist``
        # as a self attribute for the same reason.
        self.dist = dist
        # Mirrors legacy flag the executor exposes to ``_util``.
        self.gather_all_responses = False
        # ``configure_kv_cache_capacity`` toggles
        # ``enable_iter_perf_stats`` around its dummy run. We store
        # whatever it sets and otherwise default to False -- the
        # ``iter_stats`` concern isn't wired in the bring-up so the
        # flag's value has no other effect for now.
        self.enable_iter_perf_stats = False

        if start_worker:
            self.start_worker()

    # ------------------------------------------------------------------ #
    # Public API (main thread only)
    # ------------------------------------------------------------------ #

    def start_worker(self) -> None:
        """Spawn the loop thread. Idempotent."""
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="trtllm-executor-loop",
            daemon=True,
        )
        self._loop_thread.start()

    def _run_loop(self) -> None:
        """Loop-thread entry point.

        Sets the device, runs :class:`Driver` against the chosen
        SCHEDULER iter until it completes (clean shutdown) OR
        raises (catastrophic). On the way out:

        * Always set ``ctx.port.is_shutdown = True`` so any blocked
          ``ClientChannel.await_*`` waiters return.
        * Always notify the response cv (via a no-op enqueue) so
          blocked waiters get woken up after the final drain.
        * Always set ``ctx.port.shutdown_event`` so the main thread
          ``shutdown()`` join can return.
        * **Null ``self._loop_ctx`` and ``self._loop_crn``** so
          the executor's refs to the :class:`Context` and
          :class:`Concerns` bags drop. Together with the local
          ``ctx`` / ``driver`` going out of scope when this
          function returns, that releases the last ref to every
          loop-thread-only object reachable through them --
          ``ctx.svc.pool`` / ``ctx.svc.termination`` /
          ``ctx.svc.recv_offload`` (PP) and every concern. The
          executor keeps direct refs to the cross-thread
          surfaces (``self._port``, ``self._client``,
          ``self._dist``) so the main thread retains its public
          API. Refcount GC then fires each loop-thread-only
          object's ``__del__`` synchronously here -- before the
          thread terminates and well before mpi4py's atexit
          finalizer runs. This is what lets:

          * concerns hold MPI handles past their last
            ``handle_batch`` call (e.g.
            :class:`RingBroadcastSampleConcern`'s lingering
            isend deque, drained in its ``__del__``);
          * :class:`RecvOffload` (PP) shut down its worker
            thread pool in its ``__del__`` rather than the
            scheduler having to remember an explicit
            ``shutdown()`` call.
        """
        ctx = self._loop_ctx
        torch.cuda.set_device(ctx.conf.device_id)
        driver = Driver(self._scheduler_iter_fn(ctx, self._loop_crn))
        try:
            driver.run()
        finally:
            ctx.port.is_shutdown = True
            # Wake any blocked ``await_responses`` callers. The cv
            # lives on ``ctx.svc.client``; calling enqueue with an
            # empty list does the notify_all under the lock without
            # producing spurious responses.
            ctx.svc.client.enqueue([])
            ctx.port.shutdown_event.set()
            # Drop the executor-side refs to the loop-thread bags.
            # See the docstring above for the GC-driven teardown
            # chain (RecvOffload + RingBroadcastSampleConcern
            # ``__del__`` both ride on this).
            self._loop_ctx = None
            self._loop_crn = None

    def __enter__(self) -> "PyExecutorCoro":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    @property
    def is_warmup(self) -> bool:
        """Mirror of the legacy ``is_warmup`` flag.

        Single source of truth is ``self._port.is_warmup`` -- the
        flag is a cross-thread signal between the main-thread
        public API and the loop-thread consumers. The loop thread
        picks it up via two paths:

        * :meth:`ForwardConcern.handle_batch` reflects
          ``ctx.port.is_warmup`` onto ``model_engine.is_warmup``
          once per batch (immediately before invoking
          ``forward()``) so model-internal gating
          (``torch.compile`` bootstrap, MoE load-balancer skip,
          etc.) sees the same value.
        * :func:`profiler` (SCHEDULER-layer iter generator) drains
          a warmup pass at the top of its try-body
          (``while ctx.port.is_warmup: yield``) so the loop body
          runs through the warmup pass without enabling the
          profiler; the post-warmup ``itertools.count`` block then
          takes over with ``it = 0``.

        Used e.g. by ``_util.py``'s KV-cache memory estimation
        pass to flag the dummy-request run.

        One-shot contract (caller-side; not runtime-enforced)
        -----------------------------------------------------

        Callers must transition ``is_warmup`` ``False -> True ->
        False`` at most once over the executor's lifetime: once
        cleared (``True -> False``), do not flip it back to True.
        :func:`profiler`'s warmup-drain block (a plain
        ``while ctx.port.is_warmup: yield``) relies on this so it
        exits exactly once and the post-warmup
        ``itertools.count`` block runs at most once. Matches the
        legacy ``PyExecutor.is_warmup`` (plain attribute, no
        runtime check).
        """
        return self._port.is_warmup

    @is_warmup.setter
    def is_warmup(self, value: bool) -> None:
        self._port.is_warmup = value

    def can_enqueue_requests(self) -> bool:
        """Indicates whether the current process can enqueue requests."""
        return self._port.executor_request_queue.can_enqueue_request()

    def get_latest_iteration_stats(self) -> List:
        """Per-iter stats. Returns an empty list in the plain-loop bring-up.

        The legacy method drains an internally-maintained list of
        ``IterationStats`` records produced by ``_process_iter_stats``
        (the ``iter_stats`` concern). That concern is not yet wired
        in this loop; the LLM API just observes "no stats produced".
        """
        return []

    def get_latest_kv_cache_events(self) -> List:
        """KV-cache events. Returns an empty list in the plain-loop bring-up.

        Same shape as :meth:`get_latest_iteration_stats` -- the
        ``kv_cache_events`` concern is not wired yet.
        """
        return []

    def set_gather_responses(self, gather_all_responses: bool) -> None:
        """Mirror of the legacy ``set_gather_responses``.

        The plain-loop bring-up is single-rank, so the gather-all
        path is a no-op anyway. Stored on the instance for legacy
        callers that read it back.
        """
        self.gather_all_responses = gather_all_responses

    def enqueue_request(
        self,
        request: "ExecutorRequest",
        query: Optional[List] = None,
        result_wait_queue: Optional[Any] = None,
    ) -> int:
        """Enqueue a single request. Returns the assigned request id."""
        req_id = self._port.executor_request_queue.enqueue_request(
            request, query)
        if result_wait_queue is not None:
            self._client.register_wait_queue(req_id, result_wait_queue)
        return req_id

    def enqueue_requests(
        self,
        requests: List["ExecutorRequest"],
        result_wait_queue: Optional[Any] = None,
    ) -> List[int]:
        """Enqueue a batch of requests. Returns the list of assigned ids."""
        req_ids = self._port.executor_request_queue.enqueue_requests(requests)
        if result_wait_queue is not None:
            for req_id in req_ids:
                self._client.register_wait_queue(req_id, result_wait_queue)
        return req_ids

    def cancel_request(self, req_id: int) -> None:
        """Cancel a previously-enqueued request."""
        self._port.executor_request_queue.enqueue_cancel_request(req_id)

    def await_responses(
        self,
        id: Optional[Union[List[int], int]] = None,
        timeout: Optional[datetime.timedelta] = None,
    ) -> Union[List[List["LlmResponse"]], List["LlmResponse"]]:
        """Block until responses are available.

        Mirrors the legacy ``await_responses`` shape:
        * ``id=None`` -> return any responses ready (drain).
        * ``id=int`` -> return responses for that specific request.
        * ``id=List[int]`` -> return one inner-list per id.
        """
        timeout_secs = (timeout.total_seconds()
                        if timeout is not None else None)
        if id is None:
            return self._client.await_any_response(
                timeout_secs,
                is_shutdown=lambda: self._port.is_shutdown,
            )
        if isinstance(id, int):
            return self._client.await_single_response(id, timeout_secs)
        return [
            self._client.await_single_response(req_id, timeout_secs)
            for req_id in id
        ]

    def wait_shutdown(self) -> None:
        """Block until the loop thread finishes shutting down."""
        self._port.shutdown_event.wait()

    def shutdown(self) -> None:
        """Drain the loop and stop the worker thread.

        Mirrors the legacy lifecycle: enqueue a shutdown marker so
        the loop sees it on its next fetch, then wait for the loop
        to exit, then join the thread.

        After this returns the executor is unusable.
        """
        if self._loop_thread is None:
            return
        # Enqueue the shutdown marker. ScheduleConcern picks this up
        # at the next SCHEDULE_0 fetch and flips ``port.is_shutdown``;
        # the SCHEDULER iter then exits once the pool drains.
        self._port.executor_request_queue.enqueue_shutdown_request()
        self._port.shutdown_event.wait()
        self._loop_thread.join()
        self._loop_thread = None
