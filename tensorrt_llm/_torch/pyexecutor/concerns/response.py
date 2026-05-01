"""Response concern: build / enqueue / terminate at RESPOND_8.

Plain-loop scope (no spec_decode rolling acceptance gate, no
perf_metric per-step record, no disagg ctx-state forks, no first-
token response for disagg-promoted requests, no ADP dummy):

* RESPOND_8
  - Iterate ``ctx.svc.pool``. For each request:
      * Skip already-failed (``GENERATION_COMPLETE`` from Mode B
        fail-fast).
      * Build a response if it's the first iter, the request just
        finished, or we hit a streaming interval.
      * If finished, queue for termination.
  - ``ctx.svc.client.enqueue(responses)`` -- delivers to the
    main-thread waiters (and per-request streaming sinks).
  - ``ctx.svc.pool.remove_active(finished)`` -- evict from the
    pool so the next iter's schedule doesn't see them.
  - ``ctx.svc.termination.terminate(req)`` per finished request.
  - Read ``r8.canceled_req_ids`` (published by
    ``ScheduleConcern`` at SCHEDULE_0 from the cross-thread queue
    markers) and ``finish_by_reason(CANCELLED)`` matching pool
    entries. The response-build loop then picks them up as
    finished.
  - Publish ``finished_requests`` to the RESPOND_8 write view so a
    future ``IterStatsConcern`` can read it at FINALIZE_9.

Cancellation flow
-----------------

The cross-thread channel is :class:`ExecutorRequestQueue` (carries
both regular requests and cancel / shutdown markers).
``ScheduleConcern`` dispatches the markers at SCHEDULE_0 and
publishes the cancel IDs via the SCHEDULE_0 write view; this
concern reads them at RESPOND_8. There is NO peer method call
between concerns -- per the design rule, batch-local
concern-to-concern data flows through ``BatchStorage``.

The legacy ``self.canceled_req_ids`` list also re-kept IDs that
couldn't be cancelled this iter (disagg in-progress); plain loop
has no such state. Cross-iter retention for disagg lands as a
service or a private response-concern attribute when that path
is wired -- the plumbing change is local; the per-iter delivery
mechanism stays the BatchStorage field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple

from tensorrt_llm.bindings.executor import FinishReason

from ..batch_storage import BatchPhase, enter_phase
from ..llm_request import LlmRequest, LlmResponse

if TYPE_CHECKING:
    from ..context import Context


class ResponseConcern:
    """Plain-loop response builder + enqueue + terminate driver."""

    def __init__(
        self,
        *,
        rank: int,
        stream_interval: int = 1,
    ) -> None:
        self._rank = rank
        # Mirrors the legacy ``self.stream_interval`` -- emit a
        # response every Nth decoding iter (in addition to the
        # always-emit-on-iter-1 and always-emit-on-finish paths).
        self._stream_interval = max(1, stream_interval)

    def _handle_canceled(
        self,
        ctx: "Context",
        canceled_req_ids: Optional[Iterable[int]],
    ) -> None:
        """Mark pool entries with matching IDs as cancelled.

        Plain-loop scope: skips the disagg ``cancel_request`` path.
        For each ID in ``canceled_req_ids`` that has a matching
        active request, ``finish_by_reason(CANCELLED)`` flips the
        request's state so the subsequent response-build loop
        picks it up as finished.
        """
        if not canceled_req_ids:
            return
        ids_set = set(canceled_req_ids)
        for req in ctx.svc.pool:
            req_id = req.py_request_id if not req.is_child else req.parent_request_id
            if req_id not in ids_set:
                continue
            req.finish_by_reason(FinishReason.CANCELLED)
            req.decoding_iter = req.py_decoding_iter

    async def handle_batch(self, ctx: "Context") -> None:
        r8, w8 = await enter_phase(BatchPhase.RESPOND_8)

        # Always run cancel ingestion (cancellation may arrive even
        # for an empty / skipped iter; the SCHEDULE_0 write of
        # ``canceled_req_ids`` is unconditional).
        self._handle_canceled(ctx, r8.canceled_req_ids)

        new_responses: List[Tuple[int, LlmResponse]] = []
        finished: List[LlmRequest] = []

        # Iterate the pool. Note the discipline (per the FAILURE
        # HANDLING design): fail-fast'd requests are NOT in the pool
        # (``fail_requests`` calls ``pool.remove_active`` before this
        # concern runs), so we don't need an explicit
        # ``state == GENERATION_COMPLETE`` early-skip -- normal
        # completions ALSO transition to GENERATION_COMPLETE via
        # ``sample.update_requests`` and they MUST go through the
        # build / enqueue / terminate path here.
        for req in ctx.svc.pool:
            req_id = req.py_request_id

            # Build response on iter 1, on finish, or on the streaming
            # interval. Mirrors the legacy decision rule.
            do_emit = (req.py_decoding_iter == 1
                       or req.is_finished
                       or req.py_decoding_iter % self._stream_interval == 0)
            if do_emit:
                response = req.create_response(False, self._rank)
                if response is not None:
                    response.result.cached_tokens = req.cached_tokens
                    new_responses.append((req_id, response))

            if req.is_finished:
                finished.append(req)

        # Publish responses BEFORE terminating: the legacy code
        # comment notes that ``_terminate_request`` may free
        # per-request state needed by the response build path; doing
        # the enqueue first avoids that race.
        ctx.svc.client.enqueue(new_responses)
        ctx.svc.pool.remove_active(finished)
        for req in finished:
            ctx.svc.termination.terminate(req)

        # Publish the finished list to the RESPOND_8 write view so
        # FINALIZE_9-phase concerns (iter_stats, ...) can read it.
        w8.finished_requests = finished


__all__ = ["ResponseConcern"]
