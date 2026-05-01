"""Concern coroutine framework for the PyExecutor refactor.

A "concern" is a domain area of the forward loop -- ``schedule``,
``forward``, ``sample``, ``response``, ``disagg``, ``kv_connector``,
``spec_decode``, ``guided_decoder``, ``perf_metric``, ``iter_stats``,
``profile``, ``control``, ``benchmark_disagg_gate``, ``dwdp``,
``kv_cache_events``, ``save_hidden_states``, etc. -- whose work is
intermixed across ``BatchPhase`` values on every batch.

This file is the design doc that the per-concern modules (or sections
in this file, TBD) will follow once we start implementing. It also
defines the placeholder ``Concerns`` bag dataclass (currently empty)
that ``ctx.crn`` points at. Pure spec otherwise.

==============================================================================
COROUTINE OR PLAIN METHOD?
==============================================================================

Simple rule, decided by phase count:

- **Multiple phases per batch -> coroutine**, with ``await
  enter_phase(P)`` markers between phase-local blocks. Even if the
  concern doesn't carry any local variables across phases, the
  coroutine body keeps the concern's whole story in one readable
  function -- skim from top to bottom and you see exactly which
  phases it participates in and what it does at each. Several
  separate methods, one per phase, scattered across the class
  surface and called from the BATCH body, hide that flow.

- **Single phase per batch -> method**, called from ``batch_body``
  inside the matching ``async with batch_phase(P): ...`` block. No
  coroutine machinery; just a function call.

Method skeleton (single phase)::

    class ProfileConcern:
        def __init__(self, *, profiler):
            self._profiler = profiler

        def tick(self, ctx: Context) -> None:
            self._profiler.step()

    # In batch_body:
    async with batch_phase(BatchPhase.SCHEDULE_0):
        ctx.crn.profile.tick(ctx)
        # ... other concerns ...

A method is also the right shape for the cross-concern peer hook
(see the ``record_acceptance`` example further down), regardless of
whether the rest of the concern is method-only or coroutine-shaped.

Why a coroutine even without state-threading
--------------------------------------------

Concerns like ``DisaggConcern`` or ``ResourceConcern`` could in
principle expose two or three methods (e.g.,
``Resource.prepare(...)`` at FORWARD_1, ``Resource.update(...)`` at
RESPOND_7), called from the BATCH body at the right phases. We don't
do that. The coroutine form has two real advantages:

1. **One source of truth for the concern's lifecycle.** The body
   reads top-to-bottom in phase order, with ``await enter_phase(P)``
   as the rendezvous between BATCH and concern. A reader audits one
   function instead of cross-referencing N method definitions
   against M BATCH-body call sites.
2. **Locality of intra-concern state, when it appears.** The first
   time a concern needs to thread something across a phase boundary
   (a CUDA event, a matcher result, a flag, ...), you just declare
   a coroutine local. No retrofit; no "what container do I store
   this in" question.

The only place the method form genuinely wins is when the per-batch
work is one short call -- then a coroutine adds boilerplate
(``async def handle_batch(self, ctx): r, _ = await
enter_phase(...); ...``) without saving anything. That's the
single-phase case.

Hang detection
--------------

Hang detection is INTENTIONALLY NOT a concern. It becomes a builtin
feature of the runtime ``Driver`` -- the Driver can timestamp each
``send`` to a coroutine and trip a watchdog when a coroutine doesn't
yield for too long. That keeps the per-concern code free of
``checkpoint()``-style pollution and gives a uniform diagnostic across
every loop variant.

(``HangConcern`` from earlier drafts is dropped.)

==============================================================================
LIFECYCLE
==============================================================================

- **Construction (main thread)**: ``PyExecutorCoro.__init__`` builds
  one instance per concern, then bundles them into a ``Concerns``
  dataclass that goes onto ``ctx.crn``. Construction is plain kwargs
  (no ``ctx`` argument) -- the caller pulls cross-cutting deps from
  its local ``ctx`` and passes them in.
- **Lifetime**: each concern instance lives for the executor's
  lifetime. State that survives across iters is stored as instance
  attributes.
- **Per batch (loop thread)**: the BATCH body asks each concern for a
  fresh per-batch coroutine via ``concern.handle_batch(ctx)``, wraps
  it in a runtime ``Concern`` handle, and drives it phase-by-phase
  with ``await resume(handle)`` inside the matching
  ``async with batch_phase(P): ...`` block.

==============================================================================
THE CONCERN CLASS SHAPE
==============================================================================

Skeleton::

    from tensorrt_llm._torch.pyexecutor.batch_storage import BatchPhase, enter_phase
    from tensorrt_llm._torch.pyexecutor.context import Context

    class DisaggConcern:
        def __init__(
            self,
            *,
            # Cross-cutting deps pulled from ctx by the caller
            # (PyExecutorCoro.__init__) and passed in as plain kwargs.
            # The concern itself never sees ``ctx`` at construction
            # time -- it's loop-thread-only state.
            dist,
            # Owned services -- not in ``ctx``, used by no other
            # concern.
            transceiver,
            async_transfer_manager,
            # Optional config (pulled from ``ctx.conf.<sub>`` by caller):
            kv_transfer_timeout_ms=None,
        ):
            self.dist = dist
            self.transceiver = transceiver
            self.transfer_mgr = async_transfer_manager
            self.kv_transfer_timeout_ms = kv_transfer_timeout_ms

            # Cross-iter state: just instance attributes.
            #  (none for disagg today; see SpecDecodeConcern below for
            #   the canonical example)

        async def handle_batch(self, ctx: Context):
            # Receives ``ctx`` as a runtime arg. Use it for:
            #   - peer concern access: ``ctx.crn.X.method()``
            #   - cross-iter mutable state: ``ctx.state.<...>`` (rare)
            #   - thread-boundary signals: ``ctx.port.<...>``
            # For owned services / cross-cutting deps the concern
            # already pulled in ``__init__``, prefer ``self.<...>``
            # (no need to re-walk ``ctx`` every access).

            # SCHEDULE_0: opportunistic non-blocking probes
            await enter_phase(BatchPhase.SCHEDULE_0)
            if self.transceiver:
                self._check_disagg_ctx_schedulable_status()
                self._check_disagg_gen_transfer_status()

            # FORWARD_1: prepare disagg-gen-transmission-complete
            r1, _ = await enter_phase(BatchPhase.FORWARD_1)
            if self.transceiver:
                self._prepare_disagg_gen_transmission_complete(r1.scheduled_batch)

            # RESPOND_7: send_kv_async + opportunistic probe
            r7, _ = await enter_phase(BatchPhase.RESPOND_7)
            for req in r7.scheduled_batch.all_requests():
                if req.is_context_only_request and ...:
                    self.transfer_mgr.start_transfer(req)
                    self.transceiver.respond_and_send_async(req)
            if self.transceiver:
                self._check_disagg_ctx_cache_transfer_status(0)

The BATCH body driving them (note: a free function, NOT a method of
``PyExecutorCoro`` -- the loop side and the main-thread side are
separate; see ``context.py`` for the threading rationale)::

    from tensorrt_llm._torch.pyexecutor.coroutines import (
        Concern, batch_phase, resume,
    )
    from tensorrt_llm._torch.pyexecutor.context import Context

    async def batch_body(ctx: Context):
        crn = ctx.crn
        # One Concern handle per concern instance, alive for the
        # batch's lifetime. The runtime's ``Concern`` handle wraps the
        # coroutine + carries a ``done`` flag the Driver flips on
        # completion. Optional concerns may be ``None`` -- skip them.
        schedule = Concern(crn.schedule.handle_batch(ctx))
        forward = Concern(crn.forward.handle_batch(ctx))
        sample = Concern(crn.sample.handle_batch(ctx))
        response = Concern(crn.response.handle_batch(ctx))
        disagg = Concern(crn.disagg.handle_batch(ctx)) if crn.disagg else None
        # ...

        async with batch_phase(BatchPhase.SCHEDULE_0):
            await resume(schedule)
            if disagg is not None:
                await resume(disagg)
            # ...

        async with batch_phase(BatchPhase.FORWARD_1):
            if disagg is not None:
                await resume(disagg)
            await resume(forward)
            # ...

        # ...

==============================================================================
RUNTIME GUARANTEES THAT MAKE THIS WORK
==============================================================================

(See ``coroutines.py`` for the implementation.)

- **Phase skipping**. If a concern's next ``enter_phase(P_next)`` is
  at ``P_next > current``, ``await resume(handle)`` short-circuits
  without entering the Driver. So BATCH can call ``resume`` at every
  phase uniformly -- concerns explicitly skip phases they don't care
  about by simply not yielding there.

- **Done short-circuit**. When a concern's coroutine returns, its
  ``Concern.done`` flips True. Subsequent ``resume(handle)`` calls
  return immediately without entering the Driver. So a concern that
  finishes early (e.g., last yield at RESPOND_7) is a no-op for any
  later phases the BATCH still iterates through.

- **Strict progression**. Successive ``enter_phase`` calls from the
  same coroutine MUST yield strictly greater phases. A concern that
  legitimately repeats work across a phase boundary needs ``await
  again()`` (see runtime docs). Don't try to ``enter_phase`` the same
  phase twice in one coroutine.

- **Handle is the canonical reference**. The runtime tracks
  per-coroutine state (suspension record, done flag, debug-mode
  tracked-write log) by the handle, not by the coroutine. Always wrap
  in ``Concern(coro)`` before driving; never pass a raw coroutine to
  ``resume``. ``__del__`` on undriven handles closes the coroutine
  cleanly -- replaces the old ``spawn`` primitive.

- **No ``ctx.batch``**. Per-batch data goes through
  ``(r, w) = await enter_phase(BatchPhase.X)`` views. Read-view at
  phase P sees writes at < P; write-view at phase P accepts writes
  for phase-P fields only. Direct field access on ``BatchStorage`` is
  forbidden by design.

==============================================================================
DEPENDENCY INJECTION
==============================================================================

Construction signature: ``__init__(self, *, owned_a, owned_b, dep_x,
dep_y)`` -- ALL kwargs, no ``ctx`` argument. The caller
(``PyExecutorCoro.__init__``) pulls cross-cutting deps from its local
``ctx`` and passes them in::

    # In PyExecutorCoro.__init__ (main thread):
    disagg = DisaggConcern(
        dist=ctx.svc.dist,                 # cross-cutting from ctx
        transceiver=tx,                    # owned
        async_transfer_manager=atm,        # owned
        kv_transfer_timeout_ms=ctx.conf.disagg.kv_transfer_timeout_ms,
    )

Why this shape:

- **No ``ctx`` at construction time** keeps the construction-vs-
  runtime cut crisp. ``ctx.crn`` doesn't exist yet while concerns are
  being built (chicken-and-egg with ``Concerns`` itself), so a
  concern's ``__init__`` couldn't access peer concerns even if it
  wanted to. The constructor is pure: take kwargs, store them, done.
- **All kwargs** makes construction sites self-documenting and
  type-checkable. Reading the call site shows exactly which deps the
  concern uses, at the cost of a few extra characters per arg.
- **``__init__`` is independent of runtime state**. Trivially mockable
  in tests: build the concern with mock kwargs, drive its
  ``handle_batch`` against a test ``ctx``.

Why not auto-DI? See the design discussion in commit history. Short
version: explicit ``__init__`` is ~5 lines per concern, type-checkable,
grep-friendly, fail-at-construction-time. The ~30-50 line auto-DI
helper we'd otherwise write saves marginal boilerplate at the cost of
runtime errors, single-instance-per-type constraint, ``Optional``
ambiguity, ``from __future__`` annotation resolution caveats, and lost
grep on construction sites.

==============================================================================
CROSS-ITER STATE
==============================================================================

Lives on the concern class as an instance attribute. Example::

    class SpecDecodeConcern:
        def __init__(self, *, dist, drafter, speculation_gate):
            self.dist = dist
            self.drafter = drafter
            self.speculation_gate = speculation_gate
            # Cross-iter latch -- written by `record_acceptance` (called
            # by ResponseConcern at RESPOND_7), read by this concern's
            # coroutine at next iter's SCHEDULE_0.
            self.permanently_disabled = False

        def record_acceptance(self, request):
            # Cross-concern interaction via METHOD CALL on this concern
            # instance. ResponseConcern reaches us via
            # `ctx.crn.spec_decode.record_acceptance(request)`.
            avg = getattr(request, "avg_decoded_tokens_per_iter", None)
            if avg is not None:
                disabled, _ = self.speculation_gate.record_avg_decoded(
                    avg, request_id=request.py_request_id
                )
                if disabled:
                    self.permanently_disabled = True

        async def handle_batch(self, ctx: Context):
            r0, _ = await enter_phase(BatchPhase.SCHEDULE_0)
            if self.permanently_disabled:
                return  # spec_decode is off for the rest of this batch
            # ... per-batch SpecDecode work ...

This is the pattern that lets ``PersistentState`` shrink to ~zero.

==============================================================================
CROSS-CONCERN INTERACTIONS
==============================================================================

At runtime, peer concerns are reached via ``ctx.crn.X.method()``,
NOT through stored peer references. Construction order doesn't
matter, dependency wiring isn't repeated at construction sites, and
optional peers (``ctx.crn.kv_connector is None``) are checked the
same way they're checked everywhere else.

Example::

    class ResponseConcern:
        async def handle_batch(self, ctx: Context):
            r7, _ = await enter_phase(BatchPhase.RESPOND_7)
            ...
            for finished_request in finished:
                # Peer call via ctx.crn -- NOT via a stored attribute.
                if ctx.crn.spec_decode is not None:
                    ctx.crn.spec_decode.record_acceptance(finished_request)
            ...

Construction therefore does NOT need to thread peer references. Every
concern's ``__init__`` lists owned services + cross-cutting deps; peer
concerns are discovered at runtime through ``ctx.crn``.

The trade-off: the dependency graph between concerns isn't visible at
construction sites. If that becomes a problem, we add a lint or test
that walks ``ctx.crn.<X>`` references in concern source and fails on
unwired peers. For now, the runtime-discovery model is simpler.

==============================================================================
THE Concerns BAG
==============================================================================

``Concerns`` is a frozen dataclass holding one field per concern.
Built once at executor startup by ``PyExecutorCoro.__init__`` and
attached as ``ctx.crn``. Optional concerns are typed ``Optional[X]``
and default to ``None`` -- callers must check ``if ctx.crn.X is not
None`` before use.

Currently empty (placeholder); real fields are filled in as concerns
are implemented.

==============================================================================
PLANNED CONCERNS
==============================================================================

Per the loop annotations in ``py_executor.py`` (plain / overlap / PP).
Listed in the recommended IMPLEMENTATION ORDER -- bring up the plain
loop first, then add optional features, then add the PP-specific
ring-broadcast concern. Each row's ``shape`` column is either
``method`` (single per-batch phase, exposed as a method called by
``batch_body``) or ``coroutine`` (multiple per-batch phases or
intra-concern continuity, exposed via ``handle_batch(ctx)`` returning
a coroutine wrapped by the BATCH in a ``Concern`` handle).

PHASE 1 -- minimum viable plain loop
------------------------------------

+--------+---------------------+--------------+--------------------------+
| order  | concern             | shape        | scope                    |
+========+=====================+==============+==========================+
| 1      | ProfileConcern      | method       | torch / CUDA-event       |
|        |                     |              | profiler driver tick     |
+--------+---------------------+--------------+--------------------------+
| 2      | ControlConcern      | method       | control-request          |
|        |                     |              | rendezvous               |
+--------+---------------------+--------------+--------------------------+
| 3      | ScheduleConcern     | coroutine    | request fetch + schedule |
|        |                     |              | + ADP dummy +            |
|        |                     |              | ``can_queue`` +          |
|        |                     |              | inflight_ids (multi-     |
|        |                     |              | phase: SCHEDULE_0 +      |
|        |                     |              | RESPOND_7 inflight       |
|        |                     |              | cleanup)                 |
+--------+---------------------+--------------+--------------------------+
| 4      | ResourceConcern     | coroutine    | ``prepare_resources`` at |
|        |                     |              | FORWARD_1 +              |
|        |                     |              | ``update_resources`` /   |
|        |                     |              | ``revert_gen_alloc`` at  |
|        |                     |              | RESPOND_7                |
+--------+---------------------+--------------+--------------------------+
| 5      | ForwardConcern      | method       | model forward call       |
|        |                     |              | (single-phase per rank;  |
|        |                     |              | distributed in PP via    |
|        |                     |              | NCCL p2p -- HC7)         |
+--------+---------------------+--------------+--------------------------+
| 6      | SampleConcern       | coroutine    | ``sample_async`` at      |
|        |                     |              | SAMPLE_2 +               |
|        |                     |              | ``update_request_states``|
|        |                     |              | / ``update_requests``    |
|        |                     |              | at APPLY_6               |
+--------+---------------------+--------------+--------------------------+
| 7      | ResponseConcern     | coroutine    | first-token (SCHEDULE_0, |
|        |                     |              | disagg gate optional) +  |
|        |                     |              | ``handle_canceled`` +    |
|        |                     |              | ``handle_responses`` at  |
|        |                     |              | RESPOND_7                |
+--------+---------------------+--------------+--------------------------+

After phase 1, the plain loop runs end-to-end with non-disagg,
non-spec, non-guided, non-connector configs. Test it.

PHASE 2 -- iteration telemetry (still plain loop)
-------------------------------------------------

+--------+---------------------+--------------+--------------------------+
| order  | concern             | shape        | scope                    |
+========+=====================+==============+==========================+
| 8      | IterStatsConcern    | coroutine    | iter_stats record at     |
|        |                     |              | SCHEDULE_0 + num_ctx_    |
|        |                     |              | tokens snapshot +        |
|        |                     |              | ``process_iter_stats``   |
|        |                     |              | at FINALIZE_8 (gated by  |
|        |                     |              | ``enable_iter_perf_      |
|        |                     |              | stats``; optional)       |
+--------+---------------------+--------------+--------------------------+
| 9      | PerfMetricConcern   | coroutine    | CUDA timing events       |
|        |                     |              | created at FORWARD_1,    |
|        |                     |              | recorded around forward  |
|        |                     |              | / sample, used at        |
|        |                     |              | RESPOND_7                |
|        |                     |              | (``compute_batch_gpu_    |
|        |                     |              | times``)                 |
+--------+---------------------+--------------+--------------------------+

PHASE 3 -- optional features (any order; pick by need)
------------------------------------------------------

+--------+----------------------+--------------+--------------------------+
| order  | concern              | shape        | scope                    |
+========+======================+==============+==========================+
| 10     | SpecDecodeConcern    | coroutine    | gating (SCHEDULE_0) +    |
|        |                      |              | drafter run (FORWARD_1)  |
|        |                      |              | + ``record_acceptance``  |
|        |                      |              | hook called by Response  |
|        |                      |              | at RESPOND_7. Owns       |
|        |                      |              | ``permanently_disabled`` |
|        |                      |              | latch.                   |
+--------+----------------------+--------------+--------------------------+
| 11     | DisaggConcern        | coroutine    | KV transceiver probes    |
|        |                      |              | (SCHEDULE_0) + transmis- |
|        |                      |              | sion-complete (FORWARD_1)|
|        |                      |              | + ``send_kv_async`` +    |
|        |                      |              | ctx-cache transfer probe |
|        |                      |              | (RESPOND_7)              |
+--------+----------------------+--------------+--------------------------+
| 12     | KvConnectorConcern   | coroutine    | ``handle_metadata`` +    |
|        |                      |              | ``start_batch`` (FORWARD |
|        |                      |              | _1) + ``wait_for_save``  |
|        |                      |              | (FORWARD_1) +            |
|        |                      |              | ``terminate_requests``   |
|        |                      |              | (RESPOND_7)              |
+--------+----------------------+--------------+--------------------------+
| 13     | GuidedDecoderConcern | coroutine    | ``add_batch`` + ``init_  |
|        |                      |              | disagg_gen_requests``    |
|        |                      |              | (FORWARD_1) +            |
|        |                      |              | ``execute(logits)``      |
|        |                      |              | (SAMPLE_2) +             |
|        |                      |              | ``handle_errors``        |
|        |                      |              | (APPLY_6). Failed-       |
|        |                      |              | request list also threads|
|        |                      |              | from SAMPLE_2 to APPLY_6 |
|        |                      |              | as a coroutine local.    |
+--------+----------------------+--------------+--------------------------+
| 14     | DwdpConcern          | method       | first-layer weight       |
|        |                      |              | prefetch at FORWARD_1    |
|        |                      |              | (single phase)           |
+--------+----------------------+--------------+--------------------------+
| 15     | KvCacheEventsConcern | method       | ``flush_iteration_       |
|        |                      |              | events`` at RESPOND_7    |
|        |                      |              | (single phase, gated by  |
|        |                      |              | ``enable_kv_cache_       |
|        |                      |              | events``)                |
+--------+----------------------+--------------+--------------------------+
| 16     | SaveHiddenStates     | method       | ``spec_resource_mgr.     |
|        | Concern              |              | process_and_save`` at    |
|        |                      |              | APPLY_6 (single phase,   |
|        |                      |              | gated by SaveHiddenStates|
|        |                      |              | spec mode)               |
+--------+----------------------+--------------+--------------------------+
| 17     | BenchmarkDisaggGate  | method       | benchmark-disagg fill-   |
|        | Concern              |              | phase gate at SCHEDULE_0 |
|        |                      |              | (returns retry/skip      |
|        |                      |              | decision to BATCH;       |
|        |                      |              | single phase)            |
+--------+----------------------+--------------+--------------------------+

PHASE 4 -- pipeline parallel
----------------------------

+--------+-------------------------------+--------------+----------------+
| order  | concern                       | shape        | scope          |
+========+===============================+==============+================+
| 18     | RingBroadcastSampleConcern    | coroutine    | PP-only long-  |
|        |                               | (long-       | running ring   |
|        |                               | running)     | broadcast      |
|        |                               |              | spanning (n-1) |
|        |                               |              | iters per      |
|        |                               |              | batch (HC10).  |
|        |                               |              | Replaces the   |
|        |                               |              | bcast thread.  |
+--------+-------------------------------+--------------+----------------+

(``ScheduleConcern``, ``ForwardConcern``, ``SampleConcern``, etc.
also gain PP-specific paths -- the PP variants of the existing
concerns, not new concerns. Document those changes inside the per-
concern files when implemented.)

Naming convention: each concern class is ``XxxConcern`` even if the
domain word is short ("disagg", "sample"). The ``Concern`` suffix
distinguishes the class from the ``Concern`` runtime handle without
confusion.

File layout: TBD. Two reasonable options --
  (a) one file per concern under ``concerns/`` (e.g.,
      ``concerns/disagg.py``),
  (b) all concerns in this file, grouped by section.

Pick (a) once concerns get non-trivial. Until then, growing this file
is fine. ``Concerns`` (the bag) stays in ``concerns.py`` (or
``concerns/__init__.py``) so ``Context`` can string-reference it.

==============================================================================
THE LOOP-LAYER ENTRY POINT
==============================================================================

``run_loop(ctx)`` (or whatever we name the loop function) is the loop
thread's entry point. It receives ``ctx`` from the thread launch in
``PyExecutorCoro.__init__`` and runs the SCHEDULER iter forever (until
``ctx.port.shutdown_event`` is set or the request stream drains).

The SCHEDULER iter is also a free function ``async def
scheduler_iter(ctx)`` (or method on a small ``Scheduler`` class
INSIDE the loop layer, if it grows state -- but that's a loop-thread
class, NOT ``PyExecutorCoro``). It owns:

- The deque / slot ring of ``Batch`` handles (one per in-flight
  batch).
- Per-iter SCHEDULER bookkeeping: ``iter_counter``, slot ring (PP),
  ``previous_batch`` / HC2 promotion (overlap), ``has_previous_draft_tokens``
  (HC3 cross-iter side channel, overlap), HC1 cross-iter bridge code
  (overlap).
- Cross-rank scheduler-direct, NOT-tied-to-any-batch collectives:
  HC9 ``retire_vote`` (PP only); ``terminate_pending_requests``
  ballot (PP-only).

Inside the iter, for each fresh batch the SCHEDULER constructs a
``Batch`` handle (with ``BatchStorage``), starts ``batch_body(ctx)``
(which spawns one ``Concern(handle)`` per concern via
``ctx.crn.X.handle_batch(ctx)``), and drives via ``await
step(handle, through=Phase)``.

The thread cut: ``run_loop`` and everything it transitively calls
(SCHEDULER iter, ``batch_body``, concern coroutines) are loop-thread
only. ``PyExecutorCoro`` instance methods are main-thread only. The
two communicate strictly through ``ctx.port``.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Concerns:
    """Bag of concern instances. Lives at ``ctx.crn``.

    Built once at executor startup by ``PyExecutorCoro.__init__`` and
    attached to ``Context``. Frozen so the bag itself is immutable
    (the concern instances inside hold their own mutable state as
    attributes; that's fine).

    Currently empty -- real fields are filled in as the concern
    classes are implemented (see "PLANNED CONCERNS" above).
    """

    # Add fields per the planned-concerns table as concerns are built.
    # Optional concerns are typed ``Optional[X] = None``; required
    # concerns are typed ``X`` and have no default (forcing the
    # caller to wire them).
    pass
