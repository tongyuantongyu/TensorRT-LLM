# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HC10 sample-state ring broadcast (PP only).

Per-batch coroutine. The legacy implementation runs the ring on a
dedicated bcast thread that uses BLOCKING ``recv_object`` calls --
the thread parallelism hides the recv latency behind the main
thread's other work. The coroutine implementation has only the
loop thread, but mpi4py's pkl5 communicator (used by TRT-LLM)
explicitly does NOT support non-blocking ``irecv`` for pickled
objects. So we offload the blocking recv to a single-worker
thread pool (:class:`RecvOffload`, exposed as
``ctx.svc.recv_offload``) and the coroutine polls the resulting
:class:`concurrent.futures.Future`. Polling work is shifted to
rk0 ONLY (matching legacy's "rk0 is the queue authority"
pattern); the other ranks do a single blocking ``future.result()``
wait at deadline:

* HANDOFF_6 entry: sync ``sampler_event`` (correctness on the
  source -- host data must be valid before it is isend'd; backpressure
  on every other rank -- the placeholder event represents "this
  batch's forward complete on this rank", so syncing on it gives the
  same depth-2 1F1B backpressure the legacy STEP 2 did via
  ``previous_batch.sample_state.sampler_event.synchronize()``).
* HANDOFF_6 body splits three ways:

  - **Source rank (last PP rank)**: ``pp_source_isend``. No recv,
    no polling, no wait. The batch is trivially past HANDOFF_6 the
    moment the isend returns -- the SCHEDULER's force-finalize on
    this rank is therefore always a no-op pass-through.
  - **rk0 (the polling rank)**: submit the recv to
    ``ctx.svc.recv_offload`` (returns a future), then
    ``for _ in range(pp_size - 2): future.done() | again()``
    followed by a final ``future.result()`` -- ``pp_size - 1``
    rounds total. The first ``pp_size - 2`` come from the
    SCHEDULER's per-iter polling pass (``try_step(parked,
    HANDOFF_6)`` lets the ``await again()`` cascade back to the
    SCHEDULER so other batches in the deque get their polling
    rounds too). The last round is the ``future.result()`` call
    AFTER the for loop, which blocks if the future isn't already
    done -- only happens for the deadline batch when rk0 didn't
    manage to opportunistically retire it earlier.
  - **Other intermediate ranks (rk1 .. rk(N-2))**: submit the
    recv, then a single direct ``future.result()`` (blocks until
    the worker thread's recv completes). NO polling loop, NO
    ``await again()``. These ranks don't independently decide
    retirement; they follow rk0's HC9 vote (see
    ``scheduler_iter_pp``) and force-finalize K batches per iter.
    Each force-finalize blocks on this single ``result()`` --
    which completes promptly because rk0 wouldn't have voted K
    unless its K isends to rk1 had already been posted, and the
    chain propagates.

  After ``recv``, intermediate ranks ``pp_intermediate_isend`` to
  the next ring neighbor (returns ``None`` for the second-to-last
  rank, which is the ring terminus).

* FINALIZE_9: hand off this batch's own isend handle (or skip on
  second-to-last / source-only paths) to the lingering queue
  documented in "Lingering isend queue" below.

Lingering isend queue
---------------------

Naive design: the per-batch coroutine ``request.wait()``s on its
own isend handle at FINALIZE_9 before retiring. That works but
violates the rule "don't block where the legacy doesn't" in two
cases:

1. **HC10 source isend at pp_size=2**: the wait is in the same
   ``step(retired, FINALIZE_9)`` call as the isend itself, with
   only APPLY_7 + RESPOND_8 host work in between (sub-iter
   budget). The matching recv on rk0 might not have completed
   yet (rk0 posts its irecv in its own concurrent ``step(retired,
   FINALIZE_9)`` call), so the wait could briefly block.
2. **HC10 intermediate isend on the deadline iter**: same shape
   -- the intermediate isend is issued during the deadline drive
   (``step(retired, FINALIZE_9)`` -> concern's blocking ``else``
   arm -> recv done -> intermediate isend), and the wait happens
   in the same ``step()`` call moments later. The matching peer
   rk_{i+1}'s recv completion needs at least one MPI round-trip.

Legacy avoids both by parking each isend handle in a slot ring
of size ``pp_size`` and waiting on it only when the slot is
reused -- ``pp_size`` iters later, by which time the recv has
long since completed. We replicate the same "park for ``pp_size``
batches, then wait" semantics via a single instance-level deque
on this concern (one queue per rank). The bound is enforced
*on push* (i.e., at the per-batch coroutine's FINALIZE_9): if
the deque already holds ``pp_size`` handles, the oldest is
waited + popped before the new one goes in. The trailing
``pp_size`` handles left over when the SCHEDULER stops pushing
are drained by :meth:`__del__` -- safe because
``PyExecutorCoro._run_loop`` nulls its ``self._loop_crn`` ref
after ``driver.run()`` returns, and CPython's refcount GC
fires the destructor synchronously on the loop thread, before
the loop thread exits and well before ``mpi4py``'s atexit
finalizer runs.

How the SCHEDULER drives retirement (and therefore the polling)
---------------------------------------------------------------

The SCHEDULER (``scheduler_iter_pp``) collaborates with this
concern via three channels (per-iter, in body order):

1. **Forced-retire of the deadline batch (symmetric across
   ranks)**: at the top of the iter, after driving ``current``
   through STATE_UPD_4, the SCHEDULER pops the oldest in-flight
   batch (when the deque is full or we're draining) and calls
   ``step(deadline, FINALIZE_9)``. On rk0 this drives the polling
   for-loop's ``else`` arm blocking ``wait()`` -- the last
   polling round, only reached when prior iters' rk0 polling
   passes burned all ``range(pp_size - 2)`` body iterations
   without breaking early. On non-rk0 intermediate ranks the
   same ``step`` enters the concern's direct blocking-wait
   branch. On the last rank it's a no-op pass-through (source
   isend already issued at HANDOFF_6 entry).
2. **rk0 opportunistic polling pass**: after the deadline retire,
   rk0 walks the remaining in-flight deque from head and calls
   ``try_step(parked, through=HANDOFF_6)`` on each. Counts
   contiguous head successes as ``opp``: each succeeds when the
   batch's recv landed early enough that ``recv_handle.test()``
   returned done and the body's ``break`` short-circuited the
   for loop without consuming the deadline ``else`` arm.
3. **HC9 vote chain**: rk0 sends ``opp``; every non-rk0 rank
   receives it (``forced`` is computed locally on every rank
   from lockstep state, so it doesn't need to be on the wire --
   see ``scheduler_iter_pp``'s "Lockstep invariant" section).
   Then every rank pops and ``step(.., FINALIZE_9)`` ``opp``
   more batches. On rk0 these are exactly the batches polled past
   HANDOFF_6 in step (2) above. On non-rk0 intermediate ranks
   each ``step`` enters this concern's ``recv_handle.wait()``
   blocking branch. On the last rank it's again a no-op
   pass-through.

The number of polling rounds rk0 hands a batch is therefore not a
fixed ``pp_size - 1`` — it can be anywhere from 1 (rk0 retires
opportunistically the very first iter the batch is parked) up to
``pp_size - 1`` (rk0 never opportunistically retires; the deadline
``step(deadline, FINALIZE_9)`` triggers the ``else`` arm's
blocking ``wait()``). The ``range(pp_size - 2)`` polling cap +
``else`` arm guarantees the deadline iter always has a budget
of one final round, which the ``else`` arm consumes via
blocking wait.

Ring topology summary
---------------------

Forward ring rk(n-1) -> rk0 -> rk1 -> ... -> rk(n-2):

* **Last rank (rk(n-1))**: source. ``pp_source_isend`` (to rk0
  via PP wrap). No recv. No polling. No wait. Trivially past
  HANDOFF_6 the moment the isend call returns.
* **rk0**: receive from rk(n-1) (= ``prev_pp_rank`` since PP
  wraps), poll loop with ``await again()`` (lets SCHEDULER
  interleave other batches). Forward isend to rk1 via
  ``pp_intermediate_isend``.
* **Intermediate ranks rk1 .. rk(n-3)**: receive from prev,
  blocking wait, forward isend to next via
  ``pp_intermediate_isend``.
* **Second-to-last rank (rk(n-2))**: receive from prev, blocking
  wait. Terminus -- ``pp_intermediate_isend`` returns ``None``.

For ``pp_size == 2`` the ring degenerates to one hop rk1 -> rk0:
rk1 source-isends, rk0 receives only (rk0 is also second-to-last
for n=2). rk0 is also the polling rank, so all of rk0's HANDOFF_6
work happens via the polling loop / deadline blocking wait. The
``range(pp_size - 2) = range(0)`` polling body never runs and the
``else`` arm fires the blocking wait directly.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Optional

from ..batch_storage import BatchPhase, enter_phase
from ..coroutines import again
from ..pp_helpers import (pp_apply_recv_sample_state, pp_intermediate_isend,
                           pp_post_recv_sample_state, pp_source_isend)

if TYPE_CHECKING:
    from ..context import Context


class RingBroadcastSampleConcern:
    """Per-batch HC10 ring-broadcast hop driver."""

    def __init__(self) -> None:
        # Lingering isend handle queue. See "Lingering isend queue"
        # in the module docstring. ``handle_batch`` appends each
        # batch's outgoing isend handle at FINALIZE_9 with auto-
        # bound = ``pp_size`` (waits + pops the oldest when the
        # deque is full). The trailing ``pp_size`` handles after
        # the loop ends are drained by :meth:`__del__` -- the
        # SCHEDULER nukes ``self._loop_crn`` on the loop thread
        # before ``run_loop`` returns, so refcount-based GC
        # fires this destructor synchronously on the loop
        # thread, before MPI is finalized. See
        # ``PyExecutorCoro._run_loop`` for the teardown protocol.
        self._pending_isends: "collections.deque[object]" = (
            collections.deque())

    def __del__(self) -> None:
        # Drain trailing pending isends on the loop thread before
        # MPI is finalized. Safe because the SCHEDULER's
        # ``_run_loop`` teardown nulls ``self._loop_crn`` after
        # ``driver.run()`` returns -- this drops the last refs,
        # CPython's refcount GC fires this synchronously on the
        # loop thread, well before any module-level / mpi4py
        # atexit teardown. Exceptions here go through
        # ``sys.unraisablehook`` -- acceptable, since by this
        # point the loop is already winding down.
        while self._pending_isends:
            self._pending_isends.popleft().wait()

    async def handle_batch(self, ctx: "Context") -> None:
        # ---- HANDOFF_6: sync + (source-isend OR irecv + recv) ----
        r6, _ = await enter_phase(BatchPhase.HANDOFF_6)
        sample_state = r6.sample_state
        send_handle: Optional[object] = None

        if sample_state is None:
            # Empty batch (can_queue=False at SCHEDULE_0). No ring
            # work needed; just skip to FINALIZE_9.
            await enter_phase(BatchPhase.FINALIZE_9)
            return

        # Sampler-event sync. Two purposes:
        #   * SOURCE rank: correctness -- the source isend below
        #     reads ``sample_state.host`` which the sampler kernel
        #     populates via a D2H copy fenced by ``sampler_event``.
        #   * NON-SOURCE ranks: backpressure -- the placeholder
        #     ``cuda.Event()`` recorded at SAMPLE_3 represents
        #     "this batch's forward complete on this rank"; sync
        #     gives the same depth-2 1F1B backpressure the legacy
        #     STEP 2's ``previous_batch.sample_state.sampler_event.
        #     synchronize()`` does.
        if sample_state.sampler_event is not None:
            sample_state.sampler_event.synchronize()

        dist = ctx.svc.dist

        if dist.is_last_pp_rank:
            # Source: isend immediately. Non-blocking so we return
            # right away; FINALIZE_9 below hands the handle off to
            # the lingering queue. No recv, no polling, no wait --
            # the batch is trivially past HANDOFF_6 from this
            # point. The SCHEDULER's force-finalize on this rank
            # is a no-op pass-through.
            send_handle = pp_source_isend(dist, sample_state)
        else:
            # Non-source: submit a blocking recv to the offload
            # pool (returns a future). The recv-strategy splits on
            # whether this rank is the polling authority (rk0) or
            # a follower.
            recv_future = pp_post_recv_sample_state(
                dist, ctx.svc.recv_offload)

            if dist.rank == 0:
                # POLLING (rk0 only). The SCHEDULER's per-iter
                # polling pass calls ``try_step(.., HANDOFF_6)``
                # on each parked batch; the i-th call advances the
                # for-loop body by one round (``future.done()`` +
                # ``await again()`` if not done). Up to
                # ``pp_size - 2`` opportunistic rounds. The final
                # ``future.result()`` below is the deadline -- it
                # blocks if the future isn't already done, fired
                # when the SCHEDULER drives this batch with
                # ``step`` (no retry) and rk0 didn't manage to
                # opportunistically retire it earlier.
                for _ in range(dist.pp_size - 2):
                    if recv_future.done():
                        break
                    await again()
            # Non-rk0 intermediates skip the polling loop entirely
            # (no ``await again()``); they fall through to the
            # blocking ``result()`` below directly.
            recv_payload = recv_future.result()
            pp_apply_recv_sample_state(sample_state, recv_payload)
            # Forward isend (returns None on second-to-last).
            send_handle = pp_intermediate_isend(dist, sample_state)

        # ---- FINALIZE_9: hand off isend handle to lingering queue ----
        # We deliberately DO NOT ``send_handle.wait()`` here -- see
        # "Lingering isend queue" in the module docstring. Auto-
        # bound: if the queue already holds ``pp_size`` handles,
        # wait + pop the oldest before pushing the new one. The
        # ``pp_size`` cap matches the legacy slot-ring's
        # wait-deadline; by the time we wait on the oldest its
        # matching recv has long since completed (=> non-blocking).
        await enter_phase(BatchPhase.FINALIZE_9)
        if send_handle is not None:
            while len(self._pending_isends) >= dist.pp_size:
                self._pending_isends.popleft().wait()
            self._pending_isends.append(send_handle)


__all__ = ["RingBroadcastSampleConcern"]
