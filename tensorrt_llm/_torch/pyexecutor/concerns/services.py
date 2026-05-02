"""Loop-thread services exposed via ``ctx.svc.*``.

Four objects own state previously smeared across many ``PyExecutor``
methods OR work that the legacy executor offloaded to its own
dedicated thread:

* :class:`RequestPool` -- ``active_requests`` and (on PP)
  ``inflight_req_ids``. Replaces direct
  ``self.active_requests.<...>`` accesses scattered across the
  legacy executor.
* :class:`ClientChannel` -- the loop-side write surface of the
  response side of :class:`MessagePort`. Owns the body of the legacy
  ``_enqueue_responses`` (TP gather + cross-thread put on
  ``MessagePort.responses`` + ``response_cv.notify_all`` + per-
  request fan-out to ``MessagePort.result_wait_queues``).
* :class:`TerminationService` -- the body of the legacy
  ``_terminate_request`` + ``_do_terminate_request`` (resource free
  + ``result_wait_queues`` cleanup; PP-aware dispatch via
  ``DisaggPPTerminationHandler`` injected by the constructor).
* :class:`RecvOffload` -- single-worker thread pool that runs
  blocking ``recv_object`` calls so coroutines on the loop thread
  can poll the returned futures without blocking the SCHEDULER.
  Required because mpi4py's ``pkl5`` communicator (used by TRT-
  LLM) does not implement non-blocking ``irecv`` for pickled
  objects.

Every concern reaches them via ``ctx.svc.pool`` /
``ctx.svc.client`` / ``ctx.svc.termination`` /
``ctx.svc.recv_offload``. The ``fail_requests(ctx, reqs, msg)``
utility in :mod:`shared` composes the first three.

Plain-loop scope notes
======================

* ``RequestPool`` does not yet own a waiting queue / new-request
  fetcher. ``ScheduleConcern`` directly drives an
  :class:`ExecutorRequestQueue` consumer; the pool just receives the
  per-iter slice of validated, ready-to-schedule requests via
  ``add_active(reqs)``. PP / ADP variants will revisit this once
  those features land.
* ``ClientChannel`` skips the multi-rank TP gather / allgather
  branches in the legacy ``_enqueue_responses`` because the plain
  loop's bring-up scope is single-rank. The cross-thread put +
  per-request fan-out + condition-variable notify happens
  unconditionally.
* ``TerminationService`` skips the PP termination handler dispatch
  for the same reason; its body is just the legacy
  ``_do_terminate_request``.

These limitations are spelled out in code comments so that adding PP
/ ADP support (or moving the waiting queue onto the pool) is an
obvious local edit.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import threading
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from ..llm_request import LlmRequest, LlmResponse

if TYPE_CHECKING:
    from ..resource_manager import ResourceManager


# --------------------------------------------------------------------------- #
# RequestPool
# --------------------------------------------------------------------------- #


class RequestPool:
    """Single owner of the executor-wide ``active_requests`` list.

    Plain-loop scope: just a thin wrapper around a Python list with
    add / remove / iterate / count.

    PP additions:

    * ``inflight_req_ids`` -- a ``ReqIdsSet`` (C++ binding) the
      ``MicroBatchScheduler`` consults to skip requests already in
      flight through the pipeline. Mutated via :meth:`mark_inflight`
      / :meth:`unmark_inflight`. Initialized lazily so single-rank
      paths don't pay the binding cost.

    Future ADP additions (attention-DP dummy padding, ADP routing
    bookkeeping) will land as those features are wired up.
    """

    def __init__(self) -> None:
        self._active: List[LlmRequest] = []
        self._inflight_req_ids = None  # lazy ``ReqIdsSet``

    @property
    def inflight_req_ids(self):
        """Bound C++ ``ReqIdsSet`` of in-flight request IDs.

        Lazily initialised on first access so single-rank paths
        avoid the binding import. Read by ``schedule_request`` to
        skip requests that are mid-pipeline; written by
        :meth:`mark_inflight` (PP SCHEDULE_0) /
        :meth:`unmark_inflight` (PP RESPOND_8).
        """
        if self._inflight_req_ids is None:
            from tensorrt_llm.bindings.internal.batch_manager import \
                ReqIdsSet
            self._inflight_req_ids = ReqIdsSet()
        return self._inflight_req_ids

    # -- mutation -------------------------------------------------------- #

    def add_active(self, requests: Iterable[LlmRequest]) -> None:
        """Admit new (validated) requests to the active pool.

        Caller (``ScheduleConcern``) is responsible for validation and
        for filtering out invalid requests via ``fail_requests``
        first. By the time something reaches ``add_active`` it is
        intended to be scheduled.
        """
        self._active.extend(requests)

    def remove_active(self, requests: Iterable[LlmRequest]) -> None:
        """Evict the given requests from the active pool.

        Idempotent on requests already removed (silently skips them).
        Used by ``ResponseConcern`` after building responses for
        finished requests, by ``fail_requests`` for the per-request
        fail-fast path, and by the SCHEDULER's catastrophic handler
        for the active set.
        """
        # Use a set for O(1) membership; ``LlmRequest`` is identity-
        # hashed so this works.
        gone = set(requests)
        if not gone:
            return
        self._active = [req for req in self._active if req not in gone]

    def mark_inflight(self, scheduled_batch) -> None:
        """Mark every request in ``scheduled_batch`` as in-flight.

        Mirrors the legacy ``_add_inflight_ids``. Only context
        requests on their LAST chunk + generation requests are
        added: non-final context chunks should stay schedulable so
        the scheduler can keep feeding chunks of the same prefill
        through the pipeline without starving.
        """
        ids = self.inflight_req_ids
        for req in scheduled_batch.context_requests_last_chunk:
            ids.insert(req.request_id)
        for req in scheduled_batch.generation_requests:
            ids.insert(req.request_id)

    def unmark_inflight(self, scheduled_batch) -> None:
        """Inverse of :meth:`mark_inflight`. Pair-half at PP RESPOND_8."""
        ids = self.inflight_req_ids
        for req in scheduled_batch.context_requests_last_chunk:
            ids.erase(req.request_id)
        for req in scheduled_batch.generation_requests:
            ids.erase(req.request_id)

    # -- query ----------------------------------------------------------- #

    def __iter__(self) -> Iterator[LlmRequest]:
        # Yield from a SNAPSHOT so callers can mutate the pool mid-
        # iteration (the legacy ``_handle_responses`` does exactly
        # this -- iterates active_requests then rewrites the list).
        # The snapshot cost is acceptable -- batch sizes are bounded.
        return iter(list(self._active))

    def __len__(self) -> int:
        return len(self._active)

    def is_drained(self) -> bool:
        """True iff the pool is empty.

        Used by the SCHEDULER's shutdown gate, paired with
        ``MessagePort.is_shutdown`` and an empty waiting queue.
        """
        return not self._active


# --------------------------------------------------------------------------- #
# ClientChannel
# --------------------------------------------------------------------------- #


# Type alias for the per-request streaming queue handle. The legacy
# code accepted any object with a ``put_response.remote(client_id,
# resp)`` shape (Ray actor handle); we don't constrain it further
# here -- the channel just calls into it.
_ResultWaitQueue = object


@dataclasses.dataclass
class _ClientState:
    """Cross-thread response state mirrored from the legacy executor.

    Lives on :class:`ClientChannel` and is read by main-thread
    public-API methods (``await_responses``) via the holder helpers
    on :class:`ClientChannel`. Concerns / utilities only call
    ``enqueue``.

    The shape replicates the legacy ``self.responses`` /
    ``self.response_cv`` / ``self.result_wait_queues`` triple so
    main-thread waiters can keep their existing block-on-CV pattern.
    """

    # ``req_id -> [LlmResponse, ...]`` accumulated per request. Read
    # by main thread under ``cv``; written by loop thread under the
    # same lock.
    responses: Dict[int, List[LlmResponse]] = dataclasses.field(
        default_factory=dict)
    # ``req_id -> wait queue handle``. Populated on enqueue (main
    # thread) when a per-request streaming sink is provided; popped
    # on terminate (loop thread, via ``TerminationService``).
    result_wait_queues: Dict[int, _ResultWaitQueue] = dataclasses.field(
        default_factory=dict)
    # Underlying lock + condition variable for the cross-thread
    # rendezvous. ``cv`` wraps ``lock``; both threads use ``cv``.
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    cv: threading.Condition = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self.cv = threading.Condition(self.lock)


class ClientChannel:
    """Loop-side write surface of the response side of MessagePort.

    Concerns / utilities call :meth:`enqueue` to publish responses;
    main-thread API methods call :meth:`await_any_response` /
    :meth:`await_single_response` to block until results arrive.

    The state (``responses`` dict, condition variable, per-request
    streaming queues) is owned here and re-exposed to the main thread
    via the helper methods below -- so a main-thread caller never
    touches ``ctx`` directly; it goes through
    ``PyExecutorCoro``-held references that wrap this object.

    Plain-loop scope: skips the legacy multi-rank TP gather /
    allgather branches. Single-rank executors get the same
    cross-thread put + per-request fan-out + ``cv.notify_all`` the
    legacy code did at rank 0.
    """

    def __init__(self) -> None:
        self._state = _ClientState()

    # -- loop-side write surface ---------------------------------------- #

    def enqueue(self, items: Iterable[Tuple[int, LlmResponse]]) -> None:
        """Publish responses to main-thread waiters.

        ``items`` is an iterable of ``(request_id, LlmResponse)``
        pairs. The channel:

        1. Appends each response under its req_id in
           ``self._state.responses``.
        2. If a per-request streaming queue is registered for that
           req_id (and the response is an :class:`LlmResponse`), it
           is also put-forwarded to that queue (the legacy
           ``put_response.remote(client_id, resp)`` Ray-actor path).
        3. Notifies all waiters on the condition variable.

        Called from any concern that produces responses -- principal
        users today are ``ResponseConcern`` (the bulk of the work)
        and the ``fail_requests`` utility (error responses).

        An empty ``items`` iterable is a valid no-payload notify
        (used during shutdown teardown to wake every blocked
        ``await_*`` caller). The notify happens unconditionally.
        """
        # Drain to a list once so the lock window is small AND so
        # the condition-variable notify happens after all puts.
        items = list(items)
        with self._state.cv:
            for req_id, resp in items:
                bucket = self._state.responses.get(req_id)
                if bucket is None:
                    self._state.responses[req_id] = [resp]
                else:
                    bucket.append(resp)
                queue = self._state.result_wait_queues.get(req_id)
                # The legacy isinstance gate matched the most common
                # streaming case; preserve it so hooks that put
                # other types into ``responses`` (control replies,
                # cancellations) don't accidentally get re-published
                # via the streaming sink.
                if (queue is not None
                        and isinstance(resp, LlmResponse)
                        and hasattr(queue, "put_response")):
                    queue.put_response.remote(resp.client_id, resp)
            self._state.cv.notify_all()

    # -- main-thread read surface (called from PyExecutorCoro public API) #

    def register_wait_queue(self, req_id: int, queue: _ResultWaitQueue) -> None:
        """Attach a per-request streaming sink for ``req_id``.

        Called from main-thread ``enqueue_request*`` paths when the
        caller passes a ``result_wait_queue=`` arg. The mapping is
        cleared when ``TerminationService.terminate(req)`` runs (or
        when an explicit ``unregister_wait_queue`` is called, e.g.
        from cancellation paths).
        """
        with self._state.cv:
            self._state.result_wait_queues[req_id] = queue

    def unregister_wait_queue(self, req_id: int) -> None:
        """Remove the streaming sink for ``req_id``. Idempotent."""
        with self._state.cv:
            self._state.result_wait_queues.pop(req_id, None)

    def await_any_response(
        self,
        timeout: Optional[float],
        *,
        is_shutdown: Callable[[], bool],
    ) -> List[LlmResponse]:
        """Block until ANY response is available, then drain ``self.responses``.

        ``is_shutdown`` is a callable returning ``True`` once the
        loop thread has set its shutdown flag -- used to break out of
        the wait without a response when the executor is going down.
        Threading the callable in (rather than reaching into the
        port from here) keeps this class single-purpose: it owns the
        response state, not the shutdown signal.
        """

        def ready() -> bool:
            return bool(self._state.responses) or is_shutdown()

        out: List[LlmResponse] = []
        with self._state.cv:
            self._state.cv.wait_for(ready, timeout=timeout)
            for resps in self._state.responses.values():
                out.extend(resps)
            self._state.responses.clear()
        return out

    def await_single_response(
        self,
        req_id: int,
        timeout: Optional[float],
    ) -> List[LlmResponse]:
        """Block until ``req_id`` has at least one response."""

        def ready() -> bool:
            return req_id in self._state.responses

        with self._state.cv:
            self._state.cv.wait_for(ready, timeout=timeout)
            return self._state.responses.pop(req_id)


# --------------------------------------------------------------------------- #
# TerminationService
# --------------------------------------------------------------------------- #


class TerminationService:
    """Owns request termination: resource free + wait-queue cleanup.

    Plain-loop scope: only the direct path (legacy
    ``_do_terminate_request``). The PP-disagg dispatch (legacy
    ``_terminate_request``'s ``DisaggPPTerminationHandler`` branch)
    will land via a ``handler=`` kwarg once PP / disagg concerns
    arrive -- the public method is :meth:`terminate` so the dispatch
    point is centralised.
    """

    def __init__(
        self,
        *,
        resource_manager: "ResourceManager",
        client: ClientChannel,
        rank: int = 0,
        gather_all_responses: bool = False,
    ) -> None:
        self._resource_manager = resource_manager
        self._client = client
        self._rank = rank
        # Mirrors the legacy ``self.gather_all_responses`` flag:
        # if True, every rank pops its own wait-queue entry; if
        # False, only rank 0 does. Single-rank stays at False.
        self._gather_all_responses = gather_all_responses

    def terminate(self, request: LlmRequest) -> None:
        """Terminate one request.

        1. Free per-request resources (KV blocks, sampler / spec
           bookkeeping, ...).
        2. Clear the per-request streaming sink registration so
           garbage doesn't accumulate after the request ends.

        Called from normal-path concern code (``ResponseConcern``
        retire / cancellation, ``ScheduleConcern`` paused-request
        cleanup), from the SCHEDULER's catastrophic-error handler,
        and from the ``fail_requests`` utility.
        """
        self._resource_manager.free_resources(request)
        if self._gather_all_responses or self._rank == 0:
            self._client.unregister_wait_queue(request.py_request_id)


# --------------------------------------------------------------------------- #
# RecvOffload
# --------------------------------------------------------------------------- #


class RecvOffload:
    """Single-worker offload for blocking MPI ``recv_object``.

    Why this exists
    ---------------

    TRT-LLM uses ``mpi4py.util.pkl5.Intracomm`` so large pickled
    objects can be exchanged without blowing the default MPI buffer
    limits. The pkl5 communicator implements blocking ``recv`` /
    ``send`` but explicitly **NOT** ``irecv`` (it raises
    ``RuntimeError("unsupported")``). Concerns that need a
    non-blocking interface to pickle-recv -- e.g.
    :class:`RingBroadcastSampleConcern`'s HC10 polling loop -- can't
    use ``irecv`` and would block the SCHEDULER if they called
    ``recv`` directly.

    The legacy :class:`PyExecutor` works around this by running the
    blocking ``recv`` on a dedicated bcast thread. The coroutine
    runtime has only the loop thread, so we offload to a tiny
    thread pool here and expose the result as
    :class:`concurrent.futures.Future`. Coroutines poll
    ``future.done()`` (cheap, thread-state check; the worker
    thread's blocked ``recv`` drives MPI progress on its side)
    between ``await again()`` yields, then call ``future.result()``
    once the future is ready (or to block as the deadline if the
    polling budget is exhausted).

    Single worker = FIFO ordering
    -----------------------------

    The pool has ``max_workers=1``. Tasks run in submission order,
    one at a time. This matches MPI's in-order delivery for the
    same source-tag pair, so the futures resolve in the same
    order callers submit them. HC10 (single source rank, single
    ``PPCommTag.SAMPLE_STATE`` tag) relies on this.

    If a future use case multiplexes recvs across different
    source-tag pairs and wants concurrent worker progress, raise
    ``max_workers`` -- but be careful about MPI's thread-support
    level (TRT-LLM relies on at least ``MPI_THREAD_SERIALIZED``,
    which the legacy ``broadcast_sample_state_handler`` thread
    already exercises).

    Lifecycle
    ---------

    Instantiated by :meth:`PyExecutorCoro.__init__` only when
    ``pp_size > 1`` (single-rank executors don't need offloaded
    recvs). Shut down via :meth:`__del__` -- the executor's
    ``self._loop_ctx = None`` in :meth:`PyExecutorCoro._run_loop`'s
    teardown ``finally`` block drops the last reference, refcount
    GC fires :meth:`__del__` synchronously on the loop thread.
    Same teardown protocol as
    :class:`RingBroadcastSampleConcern.__del__`'s lingering-isend
    drain (loop-thread-side, MPI-finalize-safe).

    Shutdown semantics
    ------------------

    :meth:`__del__` calls
    ``ThreadPoolExecutor.shutdown(wait=False, cancel_futures=True)``:

    * Queued (not-yet-started) futures are cancelled --
      callers' ``result()`` gets ``CancelledError``.
    * The currently-running future (a blocked ``recv``) is **not**
      cancelled: Python can't interrupt a blocked C call. The
      worker is a daemon thread (the ``ThreadPoolExecutor``
      default), so it dies on process exit.

    For a clean shutdown the SCHEDULER drains all in-flight
    batches before exiting, which resolves every outstanding
    HC10 future. The cancel-pending behavior is the safety net
    for catastrophic exits.
    """

    def __init__(self) -> None:
        # Local import: ``mpi_recv_object`` is a no-op stub when
        # ``ENABLE_MULTI_DEVICE`` is False. The caller (PP path)
        # guarantees ENABLE_MULTI_DEVICE here.
        from tensorrt_llm._utils import mpi_recv_object
        self._mpi_recv_object = mpi_recv_object
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="trtllm-mpi-recv",
        )

    def __del__(self) -> None:
        # Refcount GC fires this on the loop thread when the
        # executor's ``self._loop_ctx = None`` drops the last ref.
        # See "Lifecycle" / "Shutdown semantics" in the class
        # docstring. Exceptions go through ``sys.unraisablehook``
        # -- acceptable, the executor is already winding down.
        self._pool.shutdown(wait=False, cancel_futures=True)

    def submit(self, source: int,
               tag: int) -> "concurrent.futures.Future":
        """Submit a blocking ``recv_object(source, tag)`` to the worker.

        Returns a future that the caller polls via
        :meth:`concurrent.futures.Future.done` (non-blocking) or
        :meth:`concurrent.futures.Future.result` (blocking until done,
        immediate if already done).
        """
        return self._pool.submit(self._mpi_recv_object, source, tag)
