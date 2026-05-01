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
  public API, lifecycle). Spawns the loop thread.
* :func:`run_loop` -- loop-thread entry point. Builds the
  :class:`Driver` + scheduler coroutine, runs it.
* :func:`scheduler_iter` -- the SCHEDULER iter (top-level coroutine).
  Decides when to stop and constructs each batch.
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
import threading
from typing import (TYPE_CHECKING, Any, List, Optional, Union)

import torch

from .batch_storage import BatchPhase, BatchStorage, batch_phase, step
from .concerns import (ClientChannel, Concerns, ForwardConcern,
                       RequestPool, ResourceConcern, ResponseConcern,
                       SampleConcern, ScheduleConcern, StateAdvanceConcern,
                       TerminationService, fail_requests)
from .context import (Configuration, Context, MessagePort, PersistentState,
                      Service)
from .coroutines import Batch, Concern, Driver, resume
from .executor_request_queue import ExecutorRequestQueue
from .resource_manager import KVCacheManagerV2, ResourceManagerType

if TYPE_CHECKING:
    from ..distributed import Distributed
    from .llm_request import ExecutorRequest, LlmResponse
    from .model_engine import ModelEngine
    from .resource_manager import ResourceManager
    from .sampler import Sampler
    from .scheduler import RequestScheduler


# --------------------------------------------------------------------------- #
# Loop layer: scheduler_iter, batch_body, run_loop
# --------------------------------------------------------------------------- #


async def batch_body(ctx: Context, crn: Concerns) -> None:
    """One per-batch coroutine. Drives all participating concerns.

    Loop-agnostic body: the same per-batch lifecycle works for both
    the plain loop (``scheduler_iter_plain`` drives one batch through
    every phase per iter) and the overlap loop
    (``scheduler_iter_overlap`` drives ``current`` through STATE_UPD_4
    and ``previous`` through FINALIZE_9 each iter -- the body's
    yield set is the same; only the scheduler's interleave changes).

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
    * FINALIZE_9: no concerns yet; iter-stats / perf-metric
      concerns will land here.

    The intermediate phases SYNC_EVT_5 and HANDOFF_6 are PP-only and
    not entered by the plain / overlap variants.

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
    sample = Concern(crn.sample.handle_batch(ctx))
    response = Concern(crn.response.handle_batch(ctx))

    async with batch_phase(BatchPhase.SCHEDULE_0):
        await resume(schedule)

    async with batch_phase(BatchPhase.RESOURCE_PREP_1):
        await resume(resource)

    async with batch_phase(BatchPhase.FORWARD_2) as (r, w):
        if r.can_queue:
            # ForwardConcern is a single-phase method (per the
            # planned-concerns table). The BATCH body invokes it
            # directly and writes the produced ``batch_outputs``
            # into the FORWARD_2 write view -- so SAMPLE_3
            # consumers see it through the read view. The
            # ``previous_tensors_device`` / ``num_accepted_tokens_device``
            # inputs are SCHEDULE_0 fields (the OVERLAP scheduler's
            # batch-to-batch bridge writes them; plain leaves None).
            w.batch_outputs = crn.forward.run(
                ctx,
                r.scheduled_batch,
                new_tensors_device=r.previous_tensors_device,
                num_accepted_tokens_device=r.num_accepted_tokens_device,
            )

    async with batch_phase(BatchPhase.SAMPLE_3):
        await resume(sample)

    async with batch_phase(BatchPhase.STATE_UPD_4) as (r, _):
        if r.can_queue:
            # ``advance`` does both ctx state advance (chunk
            # position, GENERATION_* transitions, optional
            # TO_COMPLETE jump on ctx->gen) AND gen TO_COMPLETE
            # marking (overlap only). Both halves flip
            # ``set_exclude_last_generation_logits(False)`` and so
            # MUST run after prev's RESPOND -- the OVERLAP scheduler
            # encodes this by suspending curr between SAMPLE_3 and
            # STATE_UPD_4 and driving prev through FINALIZE_9 first.
            # See ``StateAdvanceConcern`` for the full rationale.
            crn.state_advance.advance(ctx, r.scheduled_batch)

    async with batch_phase(BatchPhase.APPLY_7):
        # sample.update_requests blocks on THIS batch's
        # sampler_event then writes sampled tokens onto requests.
        # In overlap, this runs one iter AFTER the sample_async --
        # giving the GPU time to complete the sample kernel while
        # the next batch's forward is queued.
        await resume(sample)

    async with batch_phase(BatchPhase.RESPOND_8):
        await resume(response)
        await resume(resource)

    async with batch_phase(BatchPhase.FINALIZE_9):
        # No concerns participate at FINALIZE_9 yet. The phase
        # exists so iter_stats can later read RESPOND_8's
        # ``finished_requests`` here without colliding with its
        # producer.
        pass


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
    """
    while True:
        if (ctx.port.is_shutdown
                and ctx.svc.pool.is_drained()
                and crn.schedule.waiting_queue_empty()):
            return

        storage = BatchStorage()
        handle = Batch(batch_body(ctx, crn), storage)
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
      ``StateAdvanceConcern.advance`` flips
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
    waiting queue is empty, drain the final ``previous`` and return.

    Catastrophic-error handler: fails the active pool and re-raises.
    """
    previous: Optional[Batch] = None
    # Read view from previous's SAMPLE_3 step -- exposes
    # ``sample_state`` (produced at SAMPLE_3) so step 2's HC1 bridge
    # can read it. Held across iters.
    previous_view: Optional[object] = None

    while True:
        # Drain check. We can stop only if there's nothing left to
        # admit, the pool is empty, AND no leftover ``previous``.
        more_to_admit = (not ctx.port.is_shutdown
                         or not ctx.svc.pool.is_drained()
                         or not crn.schedule.waiting_queue_empty())
        if not more_to_admit:
            try:
                # No-op when ``previous is None``; otherwise drain
                # the final batch (apply tokens, respond, finalize)
                # so its requests get their last response and its
                # resources are freed.
                await step(previous, through=BatchPhase.FINALIZE_9)
            except Exception as exc:
                try:
                    fail_requests(ctx, list(ctx.svc.pool), str(exc))
                finally:
                    ctx.port.is_shutdown = True
                    ctx.port.shutdown_event.set()
                raise
            return

        storage = BatchStorage()
        current = Batch(batch_body(ctx, crn), storage)

        try:
            # 1. SCHEDULE_0 of current.
            _, w_sched = await step(current, through=BatchPhase.SCHEDULE_0)

            # 2. HC1 bridge: prev's sample_state.device ->
            #    current's previous_tensors_device. Skip when no
            #    previous (first iter), or previous had can_queue=
            #    False (no sample produced).
            if previous is not None:
                prev_sample_state = previous_view.sample_state
                if prev_sample_state is not None:
                    w_sched.previous_tensors_device = prev_sample_state.device

            # 3. Drive current through FORWARD_2 (RESOURCE_PREP_1
            #    + FORWARD_2). Suspends at SAMPLE_3.
            await step(current, through=BatchPhase.FORWARD_2)

            # 4. Drive previous through APPLY_7 -- before curr's
            #    SAMPLE_3 so curr's sampler reads fresh num_tokens
            #    (invariant 1). No-op when previous is None.
            await step(previous, through=BatchPhase.APPLY_7)

            # 5. Drive current through SAMPLE_3 -- queues sampler
            #    kernel; suspends at STATE_UPD_4.
            r_sample, _ = await step(current, through=BatchPhase.SAMPLE_3)

            # 6. Drive previous through FINALIZE_9 -- RESPOND_8
            #    + FINALIZE_9. No-op when previous is None.
            await step(previous, through=BatchPhase.FINALIZE_9)

            # 7. Drive current through STATE_UPD_4 -- runs state
            #    advance (ctx + gen) AFTER prev's RESPOND
            #    (invariant 2). Current is now suspended at APPLY_7,
            #    held until next iter's step 4.
            await step(current, through=BatchPhase.STATE_UPD_4)

            # 8. Promote.
            previous = current
            previous_view = r_sample

        except Exception as exc:
            try:
                fail_requests(ctx, list(ctx.svc.pool), str(exc))
            finally:
                ctx.port.is_shutdown = True
                ctx.port.shutdown_event.set()
            raise


# Type alias for the scheduler-iter shape: ``async def fn(ctx, crn) -> None``.
# Used by ``run_loop`` / ``PyExecutorCoro`` to pick the variant.
SchedulerIterFn = Any  # Callable[[Context, Concerns], Coroutine] but kept
# loose so we don't have to import Coroutine here.


def run_loop(
    ctx: Context,
    crn: Concerns,
    *,
    scheduler_iter_fn: SchedulerIterFn = scheduler_iter_plain,
) -> None:
    """Loop-thread entry point.

    Sets the device, runs :class:`Driver` against the chosen
    SCHEDULER iter (plain or overlap) until it completes (clean
    shutdown) OR raises (catastrophic). On the way out:

    * Always set ``ctx.port.is_shutdown = True`` so any blocked
      ``ClientChannel.await_*`` waiters return.
    * Always notify the response cv (via a no-op enqueue) so
      blocked waiters get woken up after the final drain.
    * Always set ``ctx.port.shutdown_event`` so the main thread
      ``shutdown()`` join can return.

    ``crn`` is forwarded into the scheduler iter -- the bag of
    concern instances is loop-thread state stashed on
    :class:`PyExecutorCoro` and passed in here, NOT on ``ctx``.
    """
    torch.cuda.set_device(ctx.conf.device_id)
    driver = Driver(scheduler_iter_fn(ctx, crn))
    try:
        driver.run()
    finally:
        ctx.port.is_shutdown = True
        # Wake any blocked ``await_responses`` callers. The cv
        # lives on ``ctx.svc.client``; calling enqueue with an empty
        # list does the notify_all under the lock without producing
        # spurious responses.
        ctx.svc.client.enqueue([])
        ctx.port.shutdown_event.set()


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
        if dist.pp_size > 1:
            raise NotImplementedError(
                "PyExecutorCoro: pipeline parallelism is not yet "
                "implemented; the plain-loop bring-up is single-rank.")

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
        svc = Service(
            dist=dist,
            pool=pool,
            client=client,
            termination=termination,
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

        crn = Concerns(
            schedule=ScheduleConcern(
                scheduler=scheduler,
                sampler=sampler,
                waiting_queue=waiting_queue,
                max_num_active_requests=model_engine.get_max_num_sequences(),
                exclude_last_generation_logits=exclude_last_generation_logits,
                scheduler_manages_kv_suspend=scheduler_manages_kv_suspend,
                max_input_len=max_input_len,
            ),
            resource=ResourceConcern(
                resource_manager=resource_manager,
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
        # SCHEDULER iter variant -- plain or overlap.
        self._scheduler_iter_fn = (
            scheduler_iter_plain
            if disable_overlap_scheduler else scheduler_iter_overlap)

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
            target=run_loop,
            args=(self._loop_ctx, self._loop_crn),
            kwargs={"scheduler_iter_fn": self._scheduler_iter_fn},
            name="trtllm-executor-loop",
            daemon=True,
        )
        self._loop_thread.start()

    def __enter__(self) -> "PyExecutorCoro":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    @property
    def is_warmup(self) -> bool:
        """Mirror of the legacy ``is_warmup`` flag.

        The setter propagates to ``model_engine.is_warmup`` so any
        in-model gating (torch.compile bootstrap path, MoE load-
        balancer skip, etc.) sees the same value. The bool also
        feeds the legacy ``configure_kv_cache_capacity`` toggle.
        """
        return getattr(self, "_is_warmup", False)

    @is_warmup.setter
    def is_warmup(self, value: bool) -> None:
        self._is_warmup = value
        # Reach into the model_engine the same way the legacy
        # executor's setter does.
        self.model_engine.is_warmup = value

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
