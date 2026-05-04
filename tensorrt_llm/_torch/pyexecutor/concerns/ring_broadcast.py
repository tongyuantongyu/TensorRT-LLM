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
wait at deadline.

The phase split (SYNC_EVT_5 + HANDOFF_6):

* **SYNC_EVT_5** = "post the async op" (one-shot per rank, no
  polling). Driven uniformly by the SCHEDULER at step 1b
  (between curr's FORWARD_2 and SAMPLE_3):

  - **Source**: ``sampler_event.synchronize()`` (typically no-op
    since the GPU has had a full iter of runway since admission)
    + ``pp_source_isend`` (non-blocking). Source's full HC10
    work is done here -- it skips HANDOFF_6 entirely.
  - **Non-source**: ``pp_post_recv_sample_state`` (submits the
    blocking ``recv_object`` to the offload thread pool; returns
    immediately with a future). Polling/waiting on the future
    happens at HANDOFF_6.

* **HANDOFF_6** = "wait for completion + cross-rank forward send"
  (non-source only; source skips):

  - **rk0 (the polling rank)**: ``for _ in range(pp_size - 2):
    if recv_future.done(): break; await again()`` followed by a
    post-loop ``recv_future.result()``. Driven by the SCHEDULER's
    step 3 ``opportunistic_polling`` -- one ``try_step`` per iter
    burns one polling round. The deadline is the post-loop
    ``result()`` at iter ``T + (pp_size - 1)`` (force retire) --
    or the ``break`` fires earlier when the recv lands.
  - **Intermediate (rk1..rk(N-3))**: no polling loop, just a
    direct ``future.result()`` (blocks until the worker's recv
    completes). The SCHEDULER doesn't poll these ranks; their
    batches reach HANDOFF_6's ``result()`` only at force-retire
    (step 2) or extras-retire (step 5), where the blocking wait
    is acceptable. Block is brief because rk0 wouldn't have
    voted ``opp`` unless its matching ``opp`` isends had been
    posted, and the chain propagates.

  After ``result()``, non-source ranks ``pp_intermediate_isend``
  to the next ring neighbor (returns ``None`` for the
  second-to-last rank, which is the ring terminus).

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
concern via four channels (per-iter, in body order):

1. **Step 1b SYNC_EVT_5 post (uniform across ranks)**: between
   curr's FORWARD_2 and SAMPLE_3, the SCHEDULER drives the
   newest parked batch through SYNC_EVT_5 with a single
   ``await step(in_flight[-1], through=SYNC_EVT_5)``. Source's
   body does sync + isend; non-source's body submits the recv
   future. One-shot per rank -- no polling, no count.
2. **Step 2 forced retire of the deadline batch (symmetric)**:
   pop oldest if the deque is full or draining, ``step(deadline,
   FINALIZE_9)``. On rk0 this drives the polling for-loop's
   post-loop ``result()`` -- the deadline ``result()``, only
   reached when prior iters' polling passes burned all
   ``range(pp_size - 2)`` body iterations without breaking
   early. On non-rk0 intermediate ranks the same ``step`` enters
   the direct ``recv_future.result()`` branch. On the source
   rank it's a fast pass-through (concern past HANDOFF_6 from
   step 1b's SYNC_EVT_5 sync+isend).
3. **Step 3 polling pass (rk0 only)**: rk0 calls
   ``opportunistic_polling`` -- iterates parked batches and
   ``try_step``s each through HANDOFF_6. Each parked's first
   ``try_step`` here runs the polling concern body's ``for``
   loop iter 0: check ``recv_future.done()``; if True, break +
   apply + intermediate isend, past HANDOFF_6. Count contiguous
   head successes as ``opp`` (the HC9 vote payload).
4. **Step 4 HC9 vote chain**: rk0 sends ``opp``; every non-rk0
   rank receives it (``forced`` is computed locally on every rank
   from lockstep state, so it doesn't need to be on the wire --
   see ``scheduler_iter_pp``'s "Lockstep invariant" section).
   Then every rank pops and ``step(.., FINALIZE_9)`` ``opp``
   more batches. On rk0 these are exactly the batches polled past
   HANDOFF_6 in step (3) above. On non-rk0 intermediate ranks
   each ``step`` enters this concern's ``recv_future.result()``
   blocking branch. On the source rank it's a fast pass-through.

The number of polling rounds rk0 hands a batch is therefore at
most ``pp_size - 1`` (one polling pass per iter at step 3, plus
the post-loop ``result()`` at force-retire). The deadline lands
at iter ``T + (pp_size - 1)`` (legacy worst case). For sub-iter
MPI latency (typical) the ``break`` fires at iter ``T + 1``.

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
import concurrent.futures  # noqa: F401  (used in type annotation string)
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
        # ---- SYNC_EVT_5: post the async op (one-shot per rank) ----
        # * Source: blocking ``synchronize()`` (typically a no-op since
        #   the SCHEDULER drives this phase at iter T+1's step 1b for
        #   a batch admitted at iter T -- the GPU has had a full iter
        #   of runway) + ``pp_source_isend`` (non-blocking). Mirrors
        #   the legacy depth-2 1F1B pattern -- legacy syncs prev's
        #   sampler_event at iter T+1's "Stage 1.2"
        #   (``py_executor.py:_executor_loop_pp`` line ~2317).
        # * Non-source: ``pp_post_recv_sample_state`` -- submits the
        #   blocking ``recv_object`` to the offload thread pool;
        #   returns immediately with a future that step 3's polling
        #   (rk0) or HANDOFF_6's ``result()`` (intermediate) consumes.
        # No polling loop, no ``await again()``: SYNC_EVT_5 is purely
        # for posting async ops. Completion / cross-rank forward send
        # is HANDOFF_6's job.
        r5, _ = await enter_phase(BatchPhase.SYNC_EVT_5)
        sample_state = r5.sample_state

        if sample_state is None:
            # Empty batch (can_queue=False at SCHEDULE_0). Skip every
            # downstream phase; jump straight to FINALIZE_9 to satisfy
            # the batch_body's resume contract.
            await enter_phase(BatchPhase.FINALIZE_9)
            return

        dist = ctx.svc.dist
        send_handle: Optional[object] = None
        recv_future: Optional["concurrent.futures.Future"] = None

        if not dist.is_last_pp_rank:
            recv_future = pp_post_recv_sample_state(
                dist, ctx.svc.recv_offload)

        if sample_state.sampler_event is not None:
            sample_state.sampler_event.synchronize()

        if dist.is_last_pp_rank:
            send_handle = pp_source_isend(dist, sample_state)

        # ---- HANDOFF_6: wait for the async op + cross-rank forward ----
        # Source skips this phase entirely (its isend was already done
        # at SYNC_EVT_5). Non-source ranks wait for the recv future
        # and forward via ``pp_intermediate_isend``:
        #
        # * rk0: bounded polling on ``recv_future.done()``. The
        #   SCHEDULER's step 3 ``opportunistic_polling`` call drives
        #   this -- one ``try_step`` per parked per iter advances the
        #   for-loop body by one round (check + ``await again()`` if
        #   not done). Up to ``pp_size - 2`` opportunistic rounds;
        #   the post-loop ``recv_future.result()`` is the deadline,
        #   fired when the SCHEDULER drives this batch with ``step``
        #   at the force-retire deadline iter.
        # * Intermediate (rk1..rk(N-3)): no polling loop, just direct
        #   ``recv_future.result()`` (blocks until worker's recv
        #   completes). The SCHEDULER doesn't poll intermediate ranks
        #   at step 3 -- their batches are driven through HANDOFF_6
        #   only at force-retire / extras-retire (step 2 / step 5),
        #   where the blocking wait is acceptable.
        if not dist.is_last_pp_rank:
            await enter_phase(BatchPhase.HANDOFF_6)
            if dist.rank == 0:
                for _ in range(dist.pp_size - 2):
                    if recv_future.done():
                        break
                    await again()
            recv_payload = recv_future.result()
            pp_apply_recv_sample_state(sample_state, recv_payload)
            send_handle = pp_intermediate_isend(dist, sample_state)

        # ---- FINALIZE_9: hand off isend handle to lingering queue ----
        # We deliberately DO NOT ``send_handle.wait()`` here -- see
        # "Lingering isend queue" in the module docstring. Auto-bound:
        # if the queue already holds ``pp_size`` handles, wait + pop
        # the oldest before pushing the new one. The ``pp_size`` cap
        # matches the legacy slot-ring's wait-deadline; by the time
        # we wait on the oldest its matching recv has long since
        # completed (=> non-blocking).
        await enter_phase(BatchPhase.FINALIZE_9)
        if send_handle is not None:
            while len(self._pending_isends) >= dist.pp_size:
                self._pending_isends.popleft().wait()
            self._pending_isends.append(send_handle)


__all__ = ["RingBroadcastSampleConcern"]
