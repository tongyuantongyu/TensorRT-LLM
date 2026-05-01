"""Cross-concern utilities (formerly the shared bottom of ``_handle_errors``).

Currently exports just :func:`fail_requests`, the per-request fail-
fast path. The other helpers planned for this module (e.g.
``iter_live(reqs)`` -- "iterate scheduled_batch but skip
already-terminated requests" -- to make state-checking less
boilerplatey) land here as the consumers appear.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, List, Tuple

from ..llm_request import LlmRequest, LlmRequestState, LlmResponse

if TYPE_CHECKING:
    from ..context import Context


def fail_requests(ctx: "Context", reqs: Iterable[LlmRequest], msg: str) -> None:
    """Per-request fail-fast utility (Mode B in the failure-handling design).

    Used when a specific subset of requests is bad but the rest of
    the batch (and the loop) should keep going. Composes the three
    services:

    1. ``ctx.svc.pool.remove_active(reqs)`` -- evict from the pool
       so subsequent iters don't see them.
    2. ``ctx.svc.client.enqueue(error_responses)`` -- deliver an
       error response so the client unblocks.
    3. ``ctx.svc.termination.terminate(req)`` per request -- free
       resources and unregister the streaming sink.

    Each request is also marked ``GENERATION_COMPLETE`` so any
    concern that subsequently iterates ``scheduled_batch`` and
    inspects ``req.state`` skips it.

    Catastrophic failures (Mode A) DO NOT call this directly. The
    concern just ``raise``s; the SCHEDULER catches at the top of
    its loop, calls ``fail_requests(ctx, list(ctx.svc.pool), msg)``
    on the active set, and sets the shutdown latch. So this utility
    serves both modes: Mode B explicitly, Mode A via the SCHEDULER's
    handler.
    """
    # Materialise once so we can iterate twice (for the response
    # build and the per-request terminate sweep) without exhausting
    # a generator passed in by the caller.
    reqs = list(reqs)
    if not reqs:
        return

    error_responses: List[Tuple[int, LlmResponse]] = []
    for req in reqs:
        req.state = LlmRequestState.GENERATION_COMPLETE
        error_responses.append((
            req.py_request_id,
            LlmResponse(
                request_id=req.py_request_id,
                error_msg=msg,
                client_id=req.py_client_id,
            ),
        ))

    ctx.svc.pool.remove_active(reqs)
    ctx.svc.client.enqueue(error_responses)
    for req in reqs:
        ctx.svc.termination.terminate(req)
