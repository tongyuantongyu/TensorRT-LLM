"""Concern coroutine framework for the PyExecutor refactor.

A "concern" is a domain area of the forward loop -- ``schedule``,
``forward``, ``sample``, ``response``, ``disagg``, ``kv_connector``,
``spec_decode``, ``guided_decoder``, ``perf_metric``, ``iter_stats``,
``profile``, ``control``, ``benchmark_disagg_gate``, ``dwdp``,
``kv_cache_events``, ``save_hidden_states``, etc. -- whose work is
intermixed across :class:`BatchPhase` values on every batch.

For the full design rationale (lifecycle, coroutine-vs-method choice,
dependency injection, cross-batch / cross-concern interactions,
planned concerns table, file layout discussion, FAILURE HANDLING,
SHARED SERVICES, the Concerns bag) see the ORIGINAL design notes:
``concerns_design_notes.py.bak`` (renamed once the package was
created so Python doesn't import the doc file by mistake). The
practical summary below covers the parts an implementer reaches for
day-to-day.

What this package contains
==========================

* ``Concerns`` (this file) -- frozen dataclass holding one field per
  concern instance. Built by ``PyExecutorCoro.__init__`` and stashed
  on the executor as ``self._loop_crn``; passed as a SEPARATE
  argument (not via ``ctx``) into ``run_loop`` / ``scheduler_iter``
  / ``batch_body``. Concerns themselves do NOT receive it -- see
  the "no ``ctx.crn``" rule below.
* ``services.py`` -- ``RequestPool``, ``ClientChannel``,
  ``TerminationService``. Loop-thread-only objects exposed as
  ``ctx.svc.pool`` / ``ctx.svc.client`` / ``ctx.svc.termination``.
* ``shared.py`` -- ``fail_requests(ctx, reqs, msg)`` utility for
  per-request fail-fast (Mode B in the original FAILURE HANDLING
  section).
* One file per concern (``schedule.py``, ``forward.py``,
  ``sample.py``, ``response.py``, ``resource.py``, ...).

The plain-loop bring-up implements the PHASE 1 set from the planned-
concerns table:
``ScheduleConcern``, ``ResourceConcern``, ``ForwardConcern``,
``SampleConcern``, ``StateAdvanceConcern``, ``ResponseConcern``.
Optional concerns (``ProfileConcern``, ``IterStatsConcern``,
``SpecDecodeConcern``, ...) are added later as their feature gates
flip on.

NO ``ctx.crn`` -- four homes for cross-concern data
===================================================

Concerns do NOT receive a reference to peer concerns. The
``Concerns`` bag is reachable only by the orchestrators
(``run_loop`` / ``scheduler_iter`` / ``batch_body``) -- it is
deliberately NOT a field on :class:`Context`. Every cross-concern
data flow falls into exactly one of four homes:

1. **Within-batch concern->concern data** flows through
   :class:`BatchStorage`. Producer writes via the SCHEDULE_0 /
   RESOURCE_PREP_1 / ... write view; consumer reads at a strictly
   later phase via the read view. The runtime narrows views per
   phase so the producer / consumer typing is enforced.

2. **Batch-to-batch handoff** (one batch's terminal output is the
   next batch's input -- the canonical example is HC1 in the
   overlap loop) is mediated by the SCHEDULER iter. Only the
   SCHEDULER speaks of "iter"; it reads the prior batch's terminal
   read view and stuffs the bridge value into the next batch's
   SCHEDULE_0 write view. No concern-to-concern channel involved.

3. **Cross-batch shared state** (cumulative state with multiple
   writers / readers across many batches, e.g. spec_decode's
   rolling-acceptance gate written by response and read by schedule
   over many batches) lives on a SHARED SERVICE in ``ctx.svc.*``.
   Both sides reach a single owner-typed surface, not the writer's
   concern internals.

4. **A concern's PRIVATE cross-batch state** (a latch or counter
   that ONLY one concern reads and writes, observed from each
   per-batch invocation) is just an instance attribute on that
   concern. Not on ``ctx`` at all.

Removing ``ctx.crn`` makes (1)-(4) enforceable mechanically:
``rg "ctx\\.crn" tensorrt_llm/_torch/pyexecutor/concerns/`` should
always be empty.

Concern shape (cheat sheet)
===========================

::

    from tensorrt_llm._torch.pyexecutor.batch_storage import (
        BatchPhase, enter_phase,
    )
    from tensorrt_llm._torch.pyexecutor.context import Context

    class XxxConcern:
        def __init__(self, *, owned_a, owned_b, dep_x):
            # No ``ctx`` here -- the bag isn't built yet at this
            # point. Cross-cutting deps come in as kwargs from the
            # caller (PyExecutorCoro.__init__).
            self._a = owned_a
            self._b = owned_b
            self._dep_x = dep_x
            # PRIVATE cross-batch latches / counters (read AND
            # written by ONLY this concern) go here too as
            # instance attributes -- NOT on ``ctx``.

        async def handle_batch(self, ctx: Context):
            # Always a coroutine, even when the concern only
            # touches a single phase -- the runtime then gives it
            # a ``C<ClassName>.handle_batch`` NVTX range
            # automatically. ``ctx`` exposes svc / conf / state /
            # port -- NEVER ``ctx.crn`` (concerns are mechanically
            # denied peer access).
            r0, _ = await enter_phase(BatchPhase.SCHEDULE_0)
            ...
            r2, _ = await enter_phase(BatchPhase.FORWARD_2)
            ...

The BATCH body wraps each per-batch coroutine in a runtime
:class:`Concern` handle and drives them with ``await resume(handle)``
inside ``async with batch_phase(P): ...`` blocks. See
:mod:`py_executor_coro` for the plain-loop ``batch_body``.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .forward import ForwardConcern
    from .resource import ResourceConcern
    from .response import ResponseConcern
    from .ring_broadcast import RingBroadcastSampleConcern
    from .sample import SampleConcern
    from .schedule import PpScheduleConcern, ScheduleConcern
    from .state_advance import StateAdvanceConcern


@dataclasses.dataclass(frozen=True)
class Concerns:
    """Bag of concern instances. Loop-thread-only; orchestrators only.

    Built once at executor startup by ``PyExecutorCoro.__init__``,
    stashed on the executor as ``self._loop_crn``, and passed as a
    SEPARATE argument (NOT a field on :class:`Context`) into the
    orchestrators that legitimately need it (``run_loop`` /
    ``scheduler_iter`` / ``batch_body``). Concerns receive only
    ``ctx`` and so cannot reach for peers -- this dataclass being
    absent from ``ctx`` is what mechanically enforces the
    four-homes routing rule for cross-concern data (within-batch
    -> BatchStorage; batch-to-batch handoff -> SCHEDULER bridge;
    cross-batch shared state -> ``ctx.svc.*``; private latch ->
    concern instance attribute). See "Why no ``crn`` field" on
    :class:`Context`. Frozen so the bag itself is immutable (the
    concern instances inside hold their own mutable state as
    instance attributes; that's fine).

    Required concerns (always present, no default):

    * ``schedule`` -- request fetch + schedule + can_queue.
    * ``resource`` -- KV cache prep / update.
    * ``forward`` -- model forward.
    * ``sample`` -- sampling kernel + apply.
    * ``state_advance`` -- per-request chunk-position advance +
      GENERATION_* state transitions. Independent of
      ``sample_state`` (which is why it's NOT folded into
      ``sample``); plain loop runs at APPLY_7, overlap / PP at
      STATE_UPD_4.
    * ``response`` -- build / enqueue / terminate.

    Optional concerns (default to ``None``; callers must check):
    add as the corresponding feature lands. Examples currently
    planned but not implemented: ``profile``, ``control``,
    ``iter_stats``, ``perf_metric``, ``spec_decode``, ``disagg``,
    ``kv_connector``, ``guided_decoder``, ``dwdp``,
    ``kv_cache_events``, ``save_hidden_states``,
    ``benchmark_disagg_gate``, ``ring_broadcast_sample``.
    """

    schedule: "ScheduleConcern"
    resource: "ResourceConcern"
    forward: "ForwardConcern"
    sample: "SampleConcern"
    state_advance: "StateAdvanceConcern"
    response: "ResponseConcern"

    # Optional fields land below as features are wired up. Each is
    # typed ``Optional[XxxConcern] = None`` and is checked at every
    # orchestrator-side use site with ``if crn.xxx is not None:``.
    # profile: Optional["ProfileConcern"] = None
    # iter_stats: Optional["IterStatsConcern"] = None
    # ...
    ring_broadcast: Optional["RingBroadcastSampleConcern"] = None


import typing as _typing  # noqa: E402

from .forward import ForwardConcern  # noqa: E402
from .resource import ResourceConcern  # noqa: E402
from .response import ResponseConcern  # noqa: E402
from .ring_broadcast import RingBroadcastSampleConcern  # noqa: E402
from .sample import SampleConcern  # noqa: E402
from .schedule import PpScheduleConcern, ScheduleConcern  # noqa: E402
from .services import ClientChannel, RecvOffload, RequestPool, TerminationService  # noqa: E402
from .shared import fail_requests  # noqa: E402
from .state_advance import StateAdvanceConcern  # noqa: E402

# Re-bind the ``Optional[...]`` field default so dataclasses sees a
# concrete type at module-load time (the forward-ref string only
# resolves when ``get_type_hints`` is called, which we don't do).
_ = _typing  # keep ``typing`` imported for consistency with future fields.

__all__ = [
    "ClientChannel",
    "Concerns",
    "ForwardConcern",
    "PpScheduleConcern",
    "RecvOffload",
    "RequestPool",
    "ResourceConcern",
    "ResponseConcern",
    "RingBroadcastSampleConcern",
    "SampleConcern",
    "ScheduleConcern",
    "StateAdvanceConcern",
    "TerminationService",
    "fail_requests",
]
