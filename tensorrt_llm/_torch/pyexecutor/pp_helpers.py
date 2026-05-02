# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline-parallel helpers shared by the legacy and coroutine executors.

The legacy ``PyExecutor`` reaches its pipeline-parallel helpers
through ``self`` (``self.dist``, ``self.scheduler``, etc.). The new
``PyExecutorCoro`` cannot import :mod:`py_executor`, so the helpers
that both implementations need live here as plain module-level
functions taking the dependencies explicitly.

Handle ownership
----------------

The legacy executor parks each non-blocking send (``isend_object``)
in a per-microbatch slot list (``self.send_handles[mid]`` /
``self.send_schedule_handles[mid]`` / etc.) and waits on the slot
the next time the same microbatch ID is reused -- one slot per
microbatch, sized to ``num_micro_batches`` on the rank.

The coroutine refactor REMOVES the slot ring entirely. Every
``isend_object`` returns its handle to the caller, who holds it
as a per-batch local in the per-batch coroutine and waits on it
during the same batch's FINALIZE_9 phase. The deque of in-flight
:class:`Batch` handles in the SCHEDULER iter IS the ring; the
batch's own coroutine owns its sends; concerns and batches never
speak in terms of "microbatch slot indices".

Functions in this module therefore return ``Optional[Handle]``
where the legacy parked the handle in a slot. Callers that don't
care about the handle (single-rank degenerate paths) can ignore
the return.

What lives here:

* :class:`PPCommTag` -- MPI tag namespace for the four PP message
  kinds (mirrors the legacy enum verbatim plus a new
  ``REQUEST_ITEMS`` tag for rk0 -> all PP-rank request-list
  broadcast).
* :func:`pp_broadcast_request_items` -- rk0 fetches from the
  cross-thread request queue, then ring-propagates the list along
  the PP forward chain so every rank's local
  ``executor_request_queue`` view is consistent (so shutdown /
  cancel markers fire on every rank, and so non-rk0 ranks have
  the new requests in their pool when they deserialize the
  scheduler decision below).
* :func:`pp_schedule_and_propagate` -- rk0 schedules; the result
  is ring-propagated rk0 -> rk(n-1) along the PP forward chain so
  all ranks operate on the same request set.
* :func:`forward_step_inter_pp` -- non-last rank's
  ``forward + placeholder sample-event + state advance`` triple,
  packaged into a single :class:`SampleState` so the slot-ring
  shape is uniform across ranks.
* :func:`ring_broadcast_sample_state_hop` -- one HC10 hop on this
  rank: receive (non-last) + isend (non-second-last). Returns
  the new isend handle (or ``None``).
* :func:`ring_broadcast_executed_batch_num` -- HC9 retire-vote
  chain. Uses BLOCKING ``send_object`` because the payload is a
  single int and the chain has no bidirectional dependency that
  could deadlock; the bring-up favors simplicity over the
  legacy isend+slot-ring pattern.
"""

from __future__ import annotations

import concurrent.futures  # noqa: F401  (used in type annotation strings)
from enum import IntEnum
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from .scheduler.scheduler import SerializableSchedulerOutput

if TYPE_CHECKING:
    from ..distributed import Distributed
    from .concerns.services import RecvOffload
    from .executor_request_queue import RequestQueueItem
    from .model_engine import ModelEngine
    from .resource_manager import ResourceManager
    from .sampler import Sampler, SampleState
    from .scheduler import RequestList, RequestScheduler, ScheduledRequests


# --------------------------------------------------------------------------- #
# Wire format
# --------------------------------------------------------------------------- #


class PPCommTag(IntEnum):
    """MPI tag namespace for pipeline-parallel comms.

    Numerically aligned with the legacy ``py_executor.PPCommTag`` so
    that ``PyExecutor`` and ``PyExecutorCoro`` running under the
    same MPI job (or in mixed-mode CI) speak the same wire format.
    ``REQUEST_ITEMS`` is new -- the legacy uses ``RequestBroadcaster``
    which picks ``pp_size`` as the tag; the coroutine helper here
    uses an explicit named tag instead.
    """

    TERMINATION = 20000
    SCHEDULE_RESULT = 20001
    EXECUTED_BATCH_NUM = 20002
    SAMPLE_STATE = 20003
    REQUEST_ITEMS = 20004


# --------------------------------------------------------------------------- #
# Pending-isend slot
# --------------------------------------------------------------------------- #


class PendingIsend:
    """1-slot pending isend-handle holder.

    Replaces the boilerplate ``if pending is not None: pending.wait();
    pending = None`` pattern at SCHEDULER / concern call sites with
    a single ``slot.set(new_handle)``: any old handle is waited on
    first, then the new one (which may be ``None``) is stored.

    On destruction (refcount GC), waits the still-held handle if
    any -- so a holder that goes out of scope at loop-thread exit
    flushes its trailing isend without the call site needing an
    explicit final ``wait()``. Same teardown protocol as
    :class:`RingBroadcastSampleConcern`'s lingering deque: the
    SCHEDULER's ``_run_loop`` nulls its concern bag on the loop
    thread before MPI is finalized, so this destructor fires
    safely on the loop thread.

    Use the slot as the SCHEDULER's per-iter park-and-reap
    discipline -- e.g. for the HC9 vote isend in
    ``scheduler_iter_pp``::

        vote = PendingIsend()
        while True:
            ...
            _, handle = ring_broadcast_executed_batch_num(...)
            vote.set(handle)  # waits previous iter's handle if any
    """

    __slots__ = ("_handle", )

    def __init__(self) -> None:
        self._handle: Optional[object] = None

    def set(self, handle: Optional[object]) -> None:
        """Wait the previously-held handle (if any), then store ``handle``."""
        if self._handle is not None:
            self._handle.wait()
        self._handle = handle

    def __del__(self) -> None:
        if self._handle is not None:
            self._handle.wait()


# --------------------------------------------------------------------------- #
# Request-item broadcast (rk0 -> all PP ranks)
# --------------------------------------------------------------------------- #


def pp_broadcast_request_items(
    dist: "Distributed",
    items: "List[RequestQueueItem]",
) -> Tuple["List[RequestQueueItem]", Optional[object]]:
    """Propagate fetched request items from rk0 to every PP rank.

    rk0 calls this with whatever it just drained from
    ``executor_request_queue.get_from_request_queue(...)``; non-rk0
    ranks call it with an empty list and receive the broadcast in
    place. Returns ``(items_for_this_rank, isend_handle)`` -- the
    caller is responsible for waiting on the isend handle (PP
    non-last ranks) before the per-batch coroutine ends, which is
    why the handle is returned rather than fire-and-forget'd.

    Wire shape: PP forward chain only (rk0 -> rk1 -> ... ->
    rk(n-1)). TP / CP fan-out within the first-PP-rank stage is
    out-of-scope for the bring-up (single client into rk0 of a
    pure-PP job); when wired it lands as a ``tp_cp_broadcast``
    inside this function, the same way
    :func:`pp_schedule_and_propagate` does it.

    The bring-up uses an explicit ``REQUEST_ITEMS`` tag (instead of
    the legacy ``RequestBroadcaster`` which keys on ``pp_size``)
    because the named tag stays stable across pp_size changes.

    Args:
        dist: distributed handle (provides ``rank`` /
            ``is_first_pp_rank`` / ``is_last_pp_rank`` /
            ``prev_pp_rank`` / ``next_pp_rank`` /
            ``recv_object`` / ``isend_object``).
        items: rk0 passes the freshly fetched
            ``RequestQueueItem`` list; non-rk0 ranks pass ``[]``
            (the value is ignored on the chain's recv side).
    """
    if dist.pp_size == 1:
        return items, None

    if not dist.is_first_pp_rank:
        items = dist.recv_object(
            src=dist.prev_pp_rank,
            tag=PPCommTag.REQUEST_ITEMS,
        )

    handle: Optional[object] = None
    if not dist.is_last_pp_rank:
        handle = dist.isend_object(
            items,
            dest=dist.next_pp_rank,
            tag=PPCommTag.REQUEST_ITEMS,
        )
    return items, handle


# --------------------------------------------------------------------------- #
# Schedule + propagate (rk0 -> rk(n-1))
# --------------------------------------------------------------------------- #


def pp_schedule_and_propagate(
    dist: "Distributed",
    scheduler: "RequestScheduler",
    active_requests: "RequestList",
    inflight_req_ids,
    *,
    enable_attention_dp: bool = False,
) -> Tuple["ScheduledRequests", "RequestList", int, Optional[object]]:
    """rk0 (or first-PP rank with DP broadcast) schedules and propagates.

    Mirror of the legacy ``_pp_schedule_and_propagate`` body, with
    the slot-ring removed: the new isend handle is returned so the
    caller (a per-batch coroutine) can hold it and wait at its
    FINALIZE_9 phase. The legacy used to park the handle in
    ``send_schedule_handles[microbatch_id]`` and wait on it the
    next time the same slot was reused; in the coroutine model the
    batch's own lifetime guarantees the wait happens before any
    buffer is reused.

    Args:
        dist: distributed handle (provides ``rank`` /
            ``is_first_pp_rank`` / ``is_last_pp_rank`` /
            ``prev_pp_rank`` / ``next_pp_rank`` /
            ``recv_object`` / ``isend_object`` /
            ``tp_broadcast`` / ``cp_broadcast`` /
            ``tp_size`` / ``cp_size``).
        scheduler: the same ``schedule_request``-bearing scheduler
            the rank uses for its local schedule decision.
        active_requests: rank-local active-request pool (LlmRequest
            objects); used by both the rk0 schedule call and the
            non-rk0 deserialize path to look up requests by ID.
        inflight_req_ids: bound C++ ``ReqIdsSet`` of in-flight IDs
            consulted by the scheduler.
        enable_attention_dp: True when the rank uses attention DP;
            in that case the first PP rank does the schedule call
            (DP-rank-local) before the chain instead of delegating
            to global rk0.

    Returns: ``(scheduled_batch, fitting_disagg_gen_init_requests,
    num_fitting_reqs, isend_handle)``. The handle is ``None`` on the
    last PP rank (terminus) and on single-rank jobs; otherwise it's
    the new ``isend_object`` handle the caller MUST wait on before
    the per-batch coroutine retires.
    """
    scheduled_batch = None
    serializable_schedule = None
    fitting_disagg_gen_init_requests = None
    num_fitting_reqs = 0
    is_dp_broadcast = dist.tp_size > 1 and enable_attention_dp

    # Schedule on the first authoritative rank: rk0 globally, or the
    # first PP rank when DP broadcast is active (DP needs each DP
    # group's first-PP rank to schedule its own slice).
    if dist.rank == 0 or (dist.is_first_pp_rank and is_dp_broadcast):
        scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = \
            scheduler.schedule_request(active_requests, inflight_req_ids)
        serializable_schedule = SerializableSchedulerOutput.from_scheduler_result(
            scheduled_batch, fitting_disagg_gen_init_requests,
            num_fitting_reqs)

    # First-PP-rank intra-DP-group broadcast (TP / CP fanout).
    if dist.is_first_pp_rank:
        if dist.tp_size > 1 and not enable_attention_dp:
            serializable_schedule = dist.tp_broadcast(serializable_schedule,
                                                     root=0)
        if dist.cp_size > 1:
            serializable_schedule = dist.cp_broadcast(serializable_schedule,
                                                     root=0)

    # Non-first-PP ranks receive the decision from the previous
    # PP rank along the forward chain.
    if not dist.is_first_pp_rank:
        serializable_schedule = dist.recv_object(
            dist.prev_pp_rank, PPCommTag.SCHEDULE_RESULT)

    # Forward-propagate the decision to the next PP rank. Last
    # rank is the chain's terminus -- no further send.
    handle: Optional[object] = None
    if not dist.is_last_pp_rank:
        handle = dist.isend_object(
            serializable_schedule,
            dist.next_pp_rank,
            PPCommTag.SCHEDULE_RESULT,
        )

    if scheduled_batch is None:
        # Non-authoritative ranks deserialize back into the same
        # ``ScheduledRequests`` shape the local scheduler would have
        # produced. Request lookup uses ``active_requests``.
        scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = \
            serializable_schedule.to_scheduler_result(active_requests)
    return (scheduled_batch, fitting_disagg_gen_init_requests,
            num_fitting_reqs, handle)


# --------------------------------------------------------------------------- #
# Forward step (non-last rank)
# --------------------------------------------------------------------------- #


def forward_step_inter_pp(
    model_engine: "ModelEngine",
    sampler: "Sampler",
    resource_manager: "ResourceManager",
    scheduled_batch: "ScheduledRequests",
    *,
    new_tensors_device: Optional[object] = None,
    num_accepted_tokens_device: Optional[torch.Tensor] = None,
) -> "SampleState":
    """Non-last PP rank's ``forward + placeholder sample`` triple.

    The non-last rank has no logits to sample from, but every rank
    needs a ``SampleState`` shape so the slot ring + ring-broadcast
    coroutine see uniform per-batch storage. Mirrors the legacy
    ``_forward_step_inter_pp``: run the model forward (NCCL p2p
    inside ``forward`` propagates activations to the next rank),
    record a placeholder ``cuda.Event`` so the next iter's
    ``sampler_event.synchronize()`` has a wait target, and stash
    ctx-final + gen requests in the SampleState's ``requests`` list
    so the ring-broadcast hop at retire time knows which requests
    own the broadcast tokens.

    The ``_update_request_states`` call the legacy does inside
    ``_forward_step_inter_pp`` is intentionally NOT replicated --
    the :class:`StateAdvanceConcern` runs at STATE_UPD_4 in the
    batch body for every rank in the coroutine implementation, so
    non-last ranks get the same ctx-position advance there.
    """
    from .sampler import SamplerEvent

    gather_context_logits = any(req.py_return_context_logits
                                for req in scheduled_batch.context_requests)
    cache_indirection_buffer = sampler.get_cache_indirection()
    model_engine.forward(
        scheduled_batch,
        resource_manager,
        new_tensors_device,
        gather_context_logits=gather_context_logits,
        cache_indirection_buffer=cache_indirection_buffer,
        num_accepted_tokens_device=num_accepted_tokens_device,
    )
    sampler_event = torch.cuda.Event()
    sampler_event.record()
    sampling_requests = (scheduled_batch.context_requests_last_chunk +
                         scheduled_batch.generation_requests)
    return sampler.SampleState(
        requests=sampling_requests,
        sampler_event=SamplerEvent(cuda_event=sampler_event),
        runtime_draft_len=getattr(model_engine, "runtime_draft_len", 0),
    )


# --------------------------------------------------------------------------- #
# Ring broadcast (split into post-recv, process-and-forward, source-isend)
# --------------------------------------------------------------------------- #
#
# The legacy implementation does the recv as a BLOCKING ``recv_object``
# call inside a dedicated bcast thread; the recv blocks that thread,
# the main thread keeps doing other work, and by the time the main
# thread needs the data the bcast thread has produced it. The coroutine
# runtime has only one (loop) thread, so a blocking ``recv_object``
# would block the SCHEDULER -- not just the per-batch coroutine that
# owns the recv. mpi4py's ``pkl5.Intracomm`` (used by TRT-LLM)
# explicitly does NOT support non-blocking ``irecv`` for pickled
# objects (it raises "unsupported"), so we can't get a true MPI Request
# either. Instead, we offload the blocking ``recv`` to a single-worker
# thread pool (:class:`RecvOffload`, exposed as ``ctx.svc.recv_offload``)
# and the coroutine polls the resulting :class:`concurrent.futures.Future`:
# post the recv on first entry to HANDOFF_6, poll ``future.done()`` on
# subsequent scheduler iters with ``await again()`` between polls so
# the scheduler can interleave other work, then call ``future.result()``
# at the batch's deadline iter to block until the worker has finished
# (immediate if the future is already done from earlier polling).
#
# Three split functions because each rank-role uses a different
# subset:
#
# * Source rank (last PP rank): :func:`pp_source_isend` -- isend the
#   freshly-sampled host tokens out (no recv side, no forward).
# * Intermediate rank (rk0 .. rk(n-3)): :func:`pp_post_recv_sample_state`
#   submits the recv to the offload pool; once the future completes the
#   caller invokes :func:`pp_intermediate_isend` to forward to the
#   next ring neighbor.
# * Second-to-last rank (rk(n-2)): :func:`pp_post_recv_sample_state`
#   submits the recv; no forward isend (terminus).
#
# All ``isend`` functions return the new MPI Request handle which the
# caller MUST hold alive and ``.wait()`` on before the batch's
# ``sample_state`` storage is freed. The per-batch coroutine in
# :class:`RingBroadcastSampleConcern` does this at its FINALIZE_9
# phase.


def pp_source_isend(
    dist: "Distributed",
    sample_state: "SampleState",
) -> object:
    """Source rank's HC10 isend: send the freshly-sampled host tokens.

    Caller MUST sync ``sample_state.sampler_event`` before invoking
    this -- the host fields are populated by the sampler kernel's D2H
    copy fenced by that event, and the isend pickles them
    immediately.
    """
    py_result_diffs = []
    for request in sample_state.requests:
        diff = request.py_result.get_diff()
        py_result_diffs.append(diff)
        request.py_result.reset_diff()
    return dist.isend_object(
        (sample_state.host, py_result_diffs),
        dest=dist.next_pp_rank,
        tag=PPCommTag.SAMPLE_STATE,
    )


def pp_post_recv_sample_state(
    dist: "Distributed",
    recv_offload: "RecvOffload",
) -> "concurrent.futures.Future":
    """Submit the (offloaded blocking) recv for this rank's HC10 hop.

    Returns a :class:`concurrent.futures.Future`; the caller polls
    via ``future.done()`` (cheap thread-state check; the worker
    thread's blocking ``recv`` drives MPI progress on its side) and
    fetches the payload with ``future.result()``. Hand the payload
    to :func:`pp_apply_recv_sample_state` to install host tokens
    onto ``sample_state``.

    Tag uses :data:`PPCommTag.SAMPLE_STATE`; the matching ``isend``
    is issued by the previous ring neighbor (the last rank for
    rk0; rk(i-1) for intermediate ranks).

    Why a future instead of an MPI Request: mpi4py's pkl5
    communicator doesn't implement non-blocking ``irecv`` for
    pickled objects. See the :class:`RecvOffload` docstring and
    the module-level comment block above.
    """
    return recv_offload.submit(
        source=dist.prev_pp_rank,
        tag=PPCommTag.SAMPLE_STATE,
    )


def pp_apply_recv_sample_state(
    sample_state: "SampleState",
    payload: tuple,
) -> None:
    """Install received host tokens + per-request response diffs.

    Called once the polling loop in :class:`RingBroadcastSampleConcern`
    sees ``request.test()`` return done. Mutates ``sample_state.host``
    and applies per-request ``py_result`` diffs in place.
    """
    sample_state.host, py_result_diffs = payload
    for request, diff in zip(sample_state.requests, py_result_diffs):
        request.py_result.apply_diff(diff)


def pp_intermediate_isend(
    dist: "Distributed",
    sample_state: "SampleState",
) -> Optional[object]:
    """Intermediate rank's forward isend in the ring.

    Called AFTER :func:`pp_apply_recv_sample_state` has installed the
    received host data onto ``sample_state``. Returns the new isend
    handle which the caller MUST wait on before the batch retires.

    On the second-to-last rank (the ring terminus) no further forward
    is needed; this function returns ``None`` so callers can use the
    same call site on every non-source rank.
    """
    if dist.is_second_last_pp_rank:
        return None
    py_result_diffs = []
    for request in sample_state.requests:
        diff = request.py_result.get_diff()
        py_result_diffs.append(diff)
        request.py_result.reset_diff()
    return dist.isend_object(
        (sample_state.host, py_result_diffs),
        dest=dist.next_pp_rank,
        tag=PPCommTag.SAMPLE_STATE,
    )


# --------------------------------------------------------------------------- #
# Retire-vote chain (HC9)
# --------------------------------------------------------------------------- #


def ring_broadcast_executed_batch_num(
    dist: "Distributed",
    executed_batch_num: int,
    *,
    enable_attention_dp: bool = False,
) -> Tuple[int, Optional[object]]:
    """rk0 votes; the count propagates rk0 -> rk(n-1) along the PP forward chain.

    Returns ``(executed_batch_num, isend_handle)``. The handle is
    ``None`` on the last PP rank (terminus -- no forward send) and
    on single-rank jobs; otherwise it's the new ``isend_object``
    handle the caller MUST hold and ``.wait()`` on later. The
    SCHEDULER parks the handle in a 1-element pending slot and
    waits on it at the start of the next iter (before issuing
    the next vote) -- one iter of slack matches the legacy
    isend + slot-ring behavior and stays non-blocking because the
    matching recv on the next PP rank completes within ~one iter
    of wall-clock (its own SCHEDULER iter top).

    The blocking ``recv_object`` on non-first-PP ranks is left as
    is: the recv has to actually have a value to forward; there's
    no useful work to interleave with it inside the vote chain.
    """
    if dist.pp_size == 1:
        return executed_batch_num, None

    # First-PP-rank intra-DP-group broadcast.
    if dist.is_first_pp_rank and dist.tp_size * dist.cp_size > 1:
        executed_batch_num = dist.tp_cp_broadcast(executed_batch_num, root=0)

    if not dist.is_first_pp_rank:
        executed_batch_num = dist.recv_object(
            src=dist.prev_pp_rank,
            tag=PPCommTag.EXECUTED_BATCH_NUM,
        )

    handle: Optional[object] = None
    if not dist.is_last_pp_rank:
        handle = dist.isend_object(
            executed_batch_num,
            dest=dist.next_pp_rank,
            tag=PPCommTag.EXECUTED_BATCH_NUM,
        )
    return executed_batch_num, handle


__all__ = [
    "PPCommTag",
    "PendingIsend",
    "forward_step_inter_pp",
    "pp_apply_recv_sample_state",
    "pp_broadcast_request_items",
    "pp_intermediate_isend",
    "pp_post_recv_sample_state",
    "pp_schedule_and_propagate",
    "pp_source_isend",
    "ring_broadcast_executed_batch_num",
]
