"""Schedule concern: fetch new requests + run scheduler + can_queue gate.

Plain-loop scope (single rank, no PP / TP collective fan-out, no
ADP, no spec-decode draft seeding, no benchmark-disagg gate):

* SCHEDULE_0
  - Fetch from ``ctx.port.executor_request_queue``. Blocks (with
    :func:`disable_hang_detect`) when idle so the watchdog doesn't
    fire on an empty server.
  - Process special queue items: shutdown markers flip
    ``ctx.port.is_shutdown``; cancellation marker IDs are
    collected locally and PUBLISHED via the SCHEDULE_0 write view
    (``w0.canceled_req_ids``). ``ResponseConcern`` reads them at
    RESPOND_8 -- per the design rule that batch-local
    concern->concern data flows through ``BatchStorage`` (not via
    peer method calls).
  - Convert remaining ``RequestQueueItem``s to ``LlmRequest`` and
    validate. Validation failures take the ``fail_requests``
    fail-fast path; the rest land in the waiting queue.
  - Pop from the waiting queue up to capacity, admit to
    ``ctx.svc.pool`` via ``add_active``.
  - Run ``scheduler.schedule_request(active, set())`` to produce
    the ``ScheduledRequests`` bundle.
  - Publish ``scheduled_batch``, ``can_queue``,
    ``fitting_disagg_gen_init_requests`` (always empty here),
    ``num_fitting_reqs`` to the SCHEDULE_0 write view.
  - Terminate any ``paused_requests`` from the scheduler output
    (V1 scheduler path; V2 manages KV suspend internally).

PP variant: see :class:`PpScheduleConcern` below.

The other concerns at SCHEDULE_0 in the planned design (``disagg``
probes, ``spec_decode`` gating, ``iter_stats`` init, ...) are not
yet implemented; they will land at this phase as separate concerns
with the BATCH body picking the order via ``await resume(...)``
sequence.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from tensorrt_llm.bindings.internal.batch_manager import ReqIdsSet

from ..batch_storage import BatchPhase, enter_phase
from ..coroutines import disable_hang_detect
from ..executor_request_queue import RequestQueueItem
from ..llm_request import LlmRequest
from ..pp_helpers import (pp_broadcast_request_items,
                           pp_schedule_and_propagate)
from ..request_utils import merge_requests_to_llm_requests
from ..scheduler import ScheduledRequests
from .shared import fail_requests

if TYPE_CHECKING:
    from ..context import Context
    from ..scheduler import RequestScheduler, WaitingQueue
    from ..sampler import Sampler


class ScheduleConcern:
    """Plain-loop request fetch + scheduler driver."""

    def __init__(
        self,
        *,
        scheduler: "RequestScheduler",
        sampler: "Sampler",
        waiting_queue: "WaitingQueue",
        max_num_active_requests: int,
        exclude_last_generation_logits: bool,
        scheduler_manages_kv_suspend: bool,
        max_input_len: int,
    ) -> None:
        self._scheduler = scheduler
        self._sampler = sampler
        # Loop-side staging area between fetch and schedule. Populated
        # in this concern, drained by ``_pop_to_pool``.
        self._waiting_queue = waiting_queue
        self._max_num_active_requests = max_num_active_requests
        self._exclude_last_generation_logits = exclude_last_generation_logits
        # ``True`` when the V2 KV cache manager owns suspend / resume
        # of paused requests. In that case we skip the legacy
        # ``_terminate_requests(scheduled_batch.paused_requests)`` step.
        self._scheduler_manages_kv_suspend = scheduler_manages_kv_suspend
        # Used by ``LlmRequest.pause(max_input_len)`` for V1 paused
        # requests.
        self._max_input_len = max_input_len

    # ------------------------------------------------------------------ #
    # Plain-loop helpers used from ``handle_batch`` (kept as methods to
    # let the coroutine body read top-to-bottom).
    # ------------------------------------------------------------------ #

    def waiting_queue_empty(self) -> bool:
        """Used by SCHEDULER's shutdown gate."""
        return not self._waiting_queue

    def _fetch_request_items(self, ctx: "Context") -> List[RequestQueueItem]:
        """Drain the executor_request_queue for one iter.

        Block (with :func:`disable_hang_detect`) only when the
        executor is idle (pool empty AND waiting queue empty).
        """
        if len(ctx.svc.pool) == 0 and not self._waiting_queue:
            # Idle -- block until something arrives. The hang
            # detector pause is GLOBAL on the Driver, but plain loop
            # has only this one coroutine in flight at this moment,
            # so no spurious "watchdog still paused" leakage.
            timeout = None
        else:
            timeout = datetime.timedelta(0)
        if timeout is None:
            # Note: ``async with disable_hang_detect()`` cannot wrap
            # a sync block from a non-async caller; the surrounding
            # ``handle_batch`` coroutine must wrap this method. See
            # :meth:`handle_batch`.
            return ctx.port.executor_request_queue.get_from_request_queue(None)
        return ctx.port.executor_request_queue.get_from_request_queue(timeout)

    def _classify_special_items(
        self,
        ctx: "Context",
        items: Iterable[RequestQueueItem],
    ) -> Tuple[List[RequestQueueItem], List[int]]:
        """Apply the markers and return ``(normal_items, canceled_ids)``.

        Mirrors the legacy ``_handle_special_queue_items``:

        * Shutdown markers flip ``ctx.port.is_shutdown`` (truly
          cross-thread state -- the SCHEDULER iter's drain check
          and ``ClientChannel.await_*`` early-return both read it).
        * Cancellation markers go into a returned local list,
          which the caller publishes via the SCHEDULE_0 write
          view (``w0.canceled_req_ids``). ``ResponseConcern``
          reads them at RESPOND_8.
        * Control requests are NOT supported in the plain-loop
          bring-up; if encountered they would need
          ``ControlConcern`` and the ``control_action`` CM wired
          up.
        """
        normal: List[RequestQueueItem] = []
        canceled: List[int] = []
        for item in items:
            if item.is_shutdown_request:
                ctx.port.is_shutdown = True
                # Drop everything after the shutdown marker -- the
                # legacy code does the same (``break``).
                break
            if item.is_canceled_request:
                canceled.append(item.id)
                continue
            if item.is_control_request:
                # Plain-loop minimum: no ControlConcern yet. Treat
                # as no-op and drop. A real control request would
                # require ``ctx.port.control_request_*`` slots and
                # a ControlConcern at SCHEDULE_0.
                continue
            normal.append(item)
        return normal, canceled

    def _admit_new_requests(
        self,
        ctx: "Context",
        items: List[RequestQueueItem],
    ) -> None:
        """Convert items to LlmRequests, validate, push into waiting queue."""
        if not items:
            return
        llm_requests = merge_requests_to_llm_requests(
            items,
            exclude_last_generation_logits=self._exclude_last_generation_logits,
        )
        # Validation: per-request fail-fast on bad ones (Mode B).
        survivors: List[LlmRequest] = []
        for req in llm_requests:
            try:
                self._sampler.validate_request(req)
            except Exception as exc:
                fail_requests(ctx, [req], str(exc))
                continue
            survivors.append(req)
        # The legacy code stores these as RequestQueueItems in the
        # waiting queue and re-converts on pop. We've already done
        # the conversion -- so we keep them as LlmRequests but adapt
        # the WaitingQueue interface by stuffing them into a small
        # holder. To keep the bring-up simple, we bypass the
        # ``WaitingQueue`` abstraction here and use a plain deque
        # of LlmRequests.
        #
        # NOTE: this means ScheduleConcern's `waiting_queue` is NOT
        # the legacy ``WaitingQueue`` ABC -- it's a deque-like
        # holder of LlmRequests. The pop_request / cancel-by-id
        # methods are still available because we use a deque
        # subclass (see PyExecutorCoro construction).
        self._waiting_queue.extend(survivors)

    def _pop_to_pool(self, ctx: "Context") -> None:
        """Move ready requests from the waiting queue into the active pool."""
        capacity = self._max_num_active_requests - len(ctx.svc.pool)
        if capacity <= 0 or not self._waiting_queue:
            return
        admitted: List[LlmRequest] = []
        while self._waiting_queue and len(admitted) < capacity:
            admitted.append(self._waiting_queue.popleft())
        if admitted:
            ctx.svc.pool.add_active(admitted)

    # ------------------------------------------------------------------ #
    # Per-batch coroutine
    # ------------------------------------------------------------------ #

    async def handle_batch(self, ctx: "Context") -> None:
        # SCHEDULE_0 is the only phase ScheduleConcern participates
        # in for the plain-loop bring-up. (PP / disagg variants will
        # add a RESPOND_8 inflight-id cleanup phase per the
        # planned-concerns table; not present here.)
        r0, w0 = await enter_phase(BatchPhase.SCHEDULE_0)

        # 1. Fetch new requests. Wrap the (potentially-blocking)
        # sync ``get_from_request_queue`` call so the Driver pauses
        # the hang watchdog -- otherwise idle servers would trigger
        # a false-positive hang report after the timeout.
        idle = len(ctx.svc.pool) == 0 and not self._waiting_queue
        if idle:
            async with disable_hang_detect():
                fetched = self._fetch_request_items(ctx)
        else:
            fetched = self._fetch_request_items(ctx)

        # 2. Special items (shutdown / cancel / control). Sets
        # ``ctx.port.is_shutdown`` for shutdown; collects cancel
        # IDs locally for publishing via ``w0.canceled_req_ids``
        # below.
        normal, canceled_ids = self._classify_special_items(ctx, fetched)

        # 3. Validate + admit to the waiting queue (or fail-fast).
        self._admit_new_requests(ctx, normal)

        # 4. Pop from waiting queue into the active pool, up to
        # capacity. The pool is what the scheduler reads from.
        self._pop_to_pool(ctx)

        # 5. Schedule. The plain-loop scheduler doesn't track inflight
        # ids (PP only); pass an empty ``ReqIdsSet`` -- the C++
        # ``MicroBatchScheduler`` binding rejects a Python set.
        scheduler_output = self._scheduler.schedule_request(
            list(ctx.svc.pool), ReqIdsSet())

        scheduled_batch = ScheduledRequests()
        # ``ScheduledRequests`` keeps context requests split into
        # ``..._chunking`` / ``..._last_chunk``; ``reset_context_requests``
        # distributes a flat list into those buckets via the
        # ``LlmRequest.is_last_context_chunk`` property.
        scheduled_batch.reset_context_requests(scheduler_output.context_requests)
        scheduled_batch.generation_requests = scheduler_output.generation_requests
        scheduled_batch.paused_requests = scheduler_output.paused_requests

        # 6. Paused-request lifecycle (V1 scheduler only). V2 owns
        # KV suspend internally and we leave its paused list alone.
        if not self._scheduler_manages_kv_suspend:
            for req in scheduled_batch.paused_requests:
                ctx.svc.termination.terminate(req)
            for req in scheduled_batch.paused_requests:
                req.pause(self._max_input_len)

        # 7. can_queue gate. Plain-loop scope is single-rank, so the
        # legacy TP-allgather / attention-DP branches in
        # ``_can_queue`` collapse to "non-empty here = can_queue".
        can_queue = scheduled_batch.batch_size > 0

        # 8. Publish to the SCHEDULE_0 write view. Loop-specific
        # disagg-gen-init / num_fitting_reqs / etc. fields stay
        # ``None`` (the BatchStorage default) -- they're for disagg
        # which isn't enabled in the plain loop.
        w0.scheduled_batch = scheduled_batch
        w0.can_queue = can_queue
        w0.canceled_req_ids = canceled_ids


class PpScheduleConcern(ScheduleConcern):
    """Pipeline-parallel SCHEDULE_0 driver.

    Two-phase coroutine: SCHEDULE_0 does the work (fetch /
    broadcast / schedule / propagate / mark inflight); FINALIZE_9
    waits on this batch's outstanding PP isend handles so they
    complete before the batch is retired.

    Differences vs the plain :class:`ScheduleConcern`:

    1. **rk0 -> all-PP-rank request-item broadcast** -- only rk0's
       ``MessagePort`` is wired to the cross-thread executor
       request queue, so non-rk0 ranks would never see new
       requests / shutdown markers / cancel markers without this
       broadcast. :func:`pp_broadcast_request_items` ring-
       propagates the freshly-fetched item list along the PP
       forward chain so every rank's local classification +
       admission step sees the same items. Without this step
       non-rk0 ranks would ``KeyError`` on the very first request
       in :func:`pp_schedule_and_propagate`'s deserialize path
       because their pool would be empty.
    2. **Schedule call uses ``pp_schedule_and_propagate``** -- rk0
       runs the local scheduler and serializes the decision, the
       PP forward chain ring-propagates it (rk0 -> rk1 -> ... ->
       rk(n-1)), and non-rk0 ranks deserialize against their own
       active-request pool. Identical request set ends up scheduled
       on every rank -- that is what makes the rest of the
       per-batch coroutine intra-batch.
    3. **Inflight ID tracking** -- the legacy PP scheduler reads
       ``inflight_req_ids`` to skip requests already in-flight
       through the pipeline (added at SCHEDULE_0, removed at
       RESPOND_8 inside ``handle_executed_batch``). Ported here
       as ``ctx.svc.pool.mark_inflight(...)`` after schedule;
       the pair-half ``unmark_inflight(...)`` lives at
       :class:`ResponseConcern` RESPOND_8 (PP only).
    4. **Per-batch isend handles** -- the items-broadcast and
       schedule-propagate isends are held as locals in this
       coroutine; the FINALIZE_9 phase waits on them. No
       per-microbatch slot ring (that was the legacy's
       ``send_schedule_handles[mid]``).

    NOT yet ported:

    * disagg ``_pp_retry_until_can_schedule`` retry loop (no
      disagg in the bring-up).
    * ADP ``enable_attention_dp`` broadcast (no ADP in the
      bring-up).
    """

    async def handle_batch(self, ctx: "Context") -> None:
        # ---- SCHEDULE_0 ----
        r0, w0 = await enter_phase(BatchPhase.SCHEDULE_0)

        # 1. rk0 fetches; broadcast the item list to every PP rank
        # so the rest of the rank-local pipeline (classify -> admit
        # -> deserialize-schedule) sees the same input.
        if ctx.svc.dist.rank == 0:
            idle = (len(ctx.svc.pool) == 0 and not self._waiting_queue)
            if idle:
                async with disable_hang_detect():
                    fetched = self._fetch_request_items(ctx)
            else:
                fetched = self._fetch_request_items(ctx)
        else:
            fetched = []
        fetched, items_isend = pp_broadcast_request_items(
            ctx.svc.dist, fetched)

        # 2. Each rank classifies its (broadcast) copy. Shutdown
        # and cancel markers fire locally on every rank as a
        # result.
        normal, canceled_ids = self._classify_special_items(ctx, fetched)

        # 3. Validate + admit + pop into pool.
        self._admit_new_requests(ctx, normal)
        self._pop_to_pool(ctx)

        # 4. PP schedule + propagate.
        (scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs,
         schedule_isend) = pp_schedule_and_propagate(
            dist=ctx.svc.dist,
            scheduler=self._scheduler,
            active_requests=list(ctx.svc.pool),
            inflight_req_ids=ctx.svc.pool.inflight_req_ids,
        )

        # 5. Paused-request lifecycle (V1 scheduler only).
        if not self._scheduler_manages_kv_suspend:
            for req in scheduled_batch.paused_requests:
                ctx.svc.termination.terminate(req)
            for req in scheduled_batch.paused_requests:
                req.pause(self._max_input_len)

        # 6. can_queue gate. Single-rank shortcut applies on each
        # rank independently here -- attention-DP / TP-allgather
        # variants will need the full ``_can_queue`` body.
        can_queue = scheduled_batch.batch_size > 0

        # 7. Inflight tracking (PP-only): mark the freshly-scheduled
        # request set so future iters skip it until tokens land.
        # Pair-half ``unmark_inflight`` runs at RESPOND_8 in
        # :class:`ResponseConcern`'s PP path.
        if can_queue:
            ctx.svc.pool.mark_inflight(scheduled_batch)

        # 8. Publish.
        w0.scheduled_batch = scheduled_batch
        w0.can_queue = can_queue
        w0.canceled_req_ids = canceled_ids

        # ---- FINALIZE_9 ----
        # Wait on this batch's outstanding PP isends. Held as locals
        # on this coroutine -- no slot ring, no cross-batch state.
        await enter_phase(BatchPhase.FINALIZE_9)
        if items_isend is not None:
            items_isend.wait()
        if schedule_isend is not None:
            schedule_isend.wait()


__all__ = ["ScheduleConcern", "PpScheduleConcern"]
