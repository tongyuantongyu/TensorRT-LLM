# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HC10 sample-state reverse ring broadcast (PP only).

Per-batch coroutine. The legacy implementation runs the ring on a
dedicated bcast thread that uses BLOCKING ``recv_object`` calls --
the thread parallelism hides the recv latency behind the main
thread's other work. The coroutine implementation has only the
loop thread, so a blocking ``recv_object`` would block the
SCHEDULER. The replacement pattern:

* HANDOFF_6 entry: sync ``sampler_event`` (correctness on the
  source -- host data must be valid before it is isend'd; backpressure
  on every other rank -- the placeholder event represents "this
  batch's forward complete on this rank", so syncing on it gives the
  same depth-2 1F1B backpressure the legacy STEP 2 did via
  ``previous_batch.sample_state.sampler_event.synchronize()``).
* HANDOFF_6 body:

  - **Source rank (last PP rank)**: issue the isend immediately. No
    polling -- isend is non-blocking so the call returns at once.
  - **Non-source ranks**: post a non-blocking ``irecv``, then
    ``for _ in range(pp_size - 2): test() | again()`` followed by a
    ``for ... else: wait()`` -- ``pp_size - 1`` rounds total,
    each round one ``request.test()`` (which itself drives MPI's
    progress engine). The first ``pp_size - 2`` rounds yield
    ``await again()`` if not yet done so the SCHEDULER can
    interleave other in-flight batches' polling. The ``else`` arm
    (= the ``pp_size - 1``-th and last round, run when the
    SCHEDULER drives the batch with ``step`` instead of
    ``try_step`` -- see "How the SCHEDULER drives the polling"
    below) does a BLOCKING ``request.wait()`` so HANDOFF_6 always
    completes by the deadline without the SCHEDULER having to
    busy-loop.
* FINALIZE_9: ``request.wait()`` on this batch's own isend handle
  (or skip on second-to-last / source-only paths). Held as a
  coroutine local -- no per-microbatch slot ring, no cross-batch
  shared state.

How the SCHEDULER drives the polling
------------------------------------

A batch enters HANDOFF_6 the first time when the
:func:`scheduler_iter_pp` polling pass first reaches it (one iter
after creation), and gets retired ``pp_size - 1`` iters later.
The SCHEDULER hands the batch ``pp_size - 1`` round-drives
exactly:

* The first ``pp_size - 2`` come from the per-iter polling pass,
  which calls ``try_step(parked, through=HANDOFF_6)`` on every
  batch *currently parked* in the in-flight deque (= every
  in-flight batch EXCEPT the one being retired this iter).
  ``try_step`` allows the concern's ``await again()`` to cascade
  back to the SCHEDULER so the next iter's polling pass picks
  up where this one left off.
* The last round comes from the SCHEDULER's
  ``step(retired, through=FINALIZE_9)`` at the retire iter,
  which drives the deadline batch through HANDOFF_6 with
  ``step`` (not ``try_step``) -- forbidding ``again()``. The
  concern's ``for ... else`` falls into its blocking ``wait()``
  arm, completing HANDOFF_6 in one final round.

The deadline batch is therefore the only batch ``step``-ped past
HANDOFF_6 (everyone else has already exited it via the polling
pass). No SCHEDULER-side busy-loop, no concern-side counter --
the round-bound and the deadline iter line up by construction.

Ring topology summary
---------------------

Reverse ring rk(n-1) -> rk0 -> rk1 -> ... -> rk(n-2):

* **Last rank (rk(n-1))**: source. ``pp_source_isend``. No recv.
* **Intermediate ranks** (rk0 .. rk(n-3)): receive from prev ring
  neighbor (= ``prev_pp_rank`` in PP forward direction, wrapping to
  the last rank for rk0), then ``pp_intermediate_isend`` to next
  ring neighbor.
* **Second-to-last rank (rk(n-2))**: terminus. Receive only.

For ``pp_size == 2`` the ring degenerates to one hop rk1 -> rk0:
rk1 source-isends, rk0 (which is also second-to-last for n=2)
receives only. Note that the polling-pass loop is empty (the
deadline batch IS the only in-flight batch, and it's popped
before the polling pass), so all of rk0's HANDOFF_6 work happens
inside ``step(retired, FINALIZE_9)`` -- the concern's ``for``
loop body is never entered (``range(0)``) and the ``else`` arm
runs immediately to do the blocking ``wait()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from ..batch_storage import BatchPhase, enter_phase
from ..coroutines import again
from ..pp_helpers import (pp_apply_recv_sample_state, pp_intermediate_isend,
                           pp_post_recv_sample_state, pp_source_isend)

if TYPE_CHECKING:
    from ..context import Context


class RingBroadcastSampleConcern:
    """Per-batch HC10 ring-broadcast hop driver."""

    async def handle_batch(self, ctx: "Context") -> None:
        # ---- HANDOFF_6: sync + (source-isend OR irecv + poll loop) ----
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

        if ctx.svc.dist.is_last_pp_rank:
            # Source: isend immediately. Non-blocking so we return
            # right away; FINALIZE_9 below waits on the handle.
            send_handle = pp_source_isend(ctx.svc.dist, sample_state)
        else:
            # Non-source: post irecv, then poll up to ``pp_size - 1``
            # rounds. The SCHEDULER's per-iter polling pass calls
            # ``try_step(.., HANDOFF_6)`` once per iter on this batch;
            # the i-th call advances this loop by one round. Round
            # bound = ``pp_size - 1`` matches the number of iters
            # this batch sits in the in-flight deque before the
            # SCHEDULER pops it for retire, so the last round
            # coincides with the deadline iter -- on which we BLOCK
            # rather than yield ``await again()`` so HANDOFF_6 always
            # completes before retire.
            recv_handle = pp_post_recv_sample_state(ctx.svc.dist)
            recv_payload: Optional[tuple] = None
            for _ in range(ctx.svc.dist.pp_size - 2):
                done, recv_payload = recv_handle.test()
                if done:
                    break
                await again()
            else:
                # Last round = deadline iter. Block until done;
                # ``request.wait()`` returns immediately if MPI
                # already received the message during the
                # previous polls + other batches' polls.
                recv_payload = recv_handle.wait()
            pp_apply_recv_sample_state(sample_state, recv_payload)
            # Forward isend (returns None on second-to-last).
            send_handle = pp_intermediate_isend(ctx.svc.dist, sample_state)

        # ---- FINALIZE_9: wait on our own isend ----
        # Held as a coroutine local -- no per-microbatch slot ring,
        # no cross-batch shared state. The wait must happen before
        # the batch's ``sample_state.host`` storage is freed (the
        # outer scheduler retires this batch immediately after this
        # phase returns).
        await enter_phase(BatchPhase.FINALIZE_9)
        if send_handle is not None:
            send_handle.wait()


__all__ = ["RingBroadcastSampleConcern"]
