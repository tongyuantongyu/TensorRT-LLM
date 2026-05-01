"""Shared runtime context for the PyExecutor coroutine layer.

The single ``Context`` object is threaded as an argument through EVERY
coroutine in the loop layer:

- the loop function (``run_loop(ctx)``)
- the SCHEDULER iter (``async def scheduler_iter(ctx)``)
- the BATCH body (``async def batch_body(ctx)``)
- every concern's per-batch coroutine (``handle_batch(self, ctx)``)

Everything in ``ctx`` is FREELY ACCESSIBLE to any code in the loop
layer. The flip side: anything you DON'T want freely accessible MUST
NOT live in ``ctx``. Hide it on a concern instance by passing it as a
construction-time kwarg in ``PyExecutorCoro.__init__`` (see
``concerns.py``).

Threading: main thread vs loop thread
=====================================

There is a strict cut between the executor's two threads:

- **Main thread** runs ``PyExecutorCoro`` instance methods (the public
  API: ``submit``, ``shutdown``, result waiters, control actions).
  These methods MUST interact with the loop thread ONLY via
  ``ctx.port`` (and may dereference immutable ``ctx.svc`` /
  ``ctx.conf`` references, never their internals).

- **Loop thread** runs the loop function, SCHEDULER iter, BATCH body,
  and concern coroutines. These all receive ``ctx`` as a parameter.
  The loop thread MUST NOT touch ``PyExecutorCoro`` instance
  attributes — anything it needs is in ``ctx``.

The legacy ``PyExecutor`` mixed both freely (every method ran on
whichever thread happened to call it, accessing the same fields). The
refactor's job is to make every cross-thread access explicit by
forcing it through ``ctx.port``. If you find yourself reaching for a
non-``port`` ``ctx`` field from main thread code OR for a
``PyExecutorCoro`` attribute from loop thread code, the design has
slipped.

A clean ``PyExecutorCoro`` skeleton::

    class PyExecutorCoro:
        def __init__(self, ...):
            # MAIN-THREAD-ONLY work: build everything, then hand to
            # the loop thread.
            svc = Service(dist=...)
            conf = Configuration(...)
            port = MessagePort(...)
            state = PersistentState()
            crn = build_concerns(svc=svc, conf=conf, ...)  # see concerns.py
            ctx = Context(svc=svc, conf=conf, state=state, port=port, crn=crn)

            # Main thread only keeps `port`. Everything else moves to
            # the loop thread on `ctx`.
            self._port = port
            self._loop_thread = threading.Thread(target=run_loop, args=(ctx,))
            self._loop_thread.start()

        def submit(self, request):
            # MAIN-THREAD path: only `_port` is touchable.
            self._port.executor_request_queue.put(request)

What goes in ``ctx``
====================

Five buckets, each with its own membership rule (below). Reach the
right bucket via ``ctx.svc`` / ``ctx.conf`` / ``ctx.state`` /
``ctx.port`` / ``ctx.crn``.

What does NOT go in ``ctx``
===========================

- ``BatchStorage`` -- per-batch data is reached only through the
  runtime's ``enter_phase`` / ``step`` views, never through ``ctx``.
  See "Why no ``batch`` field" on :class:`Context` below.
- A concern's owned services -- those go on the concern instance via
  its ``__init__`` kwargs (see ``concerns.py``). Putting them in
  ``ctx`` would make them freely accessible to other concerns,
  defeating the point of "owned".
- A concern's cross-iter state -- lives as instance attributes on the
  concern class.
- Per-iter SCHEDULER state -- the SCHEDULER iter holds it as locals
  (``iter_counter``, slot ring, ``previous_batch``, HC1/HC2/HC3
  cross-iter bridges).
"""

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed.communicator import Distributed

    from .concerns import Concerns
    from .executor_request_queue import ExecutorRequestQueue


@dataclasses.dataclass(frozen=True)
class Service:
    """Truly cross-cutting service objects. Immutable.

    HIGH BAR for membership. A candidate belongs here ONLY if it is
    used by MULTIPLE concerns AND has no natural single-owner concern.

    Almost every "service" in the legacy ``PyExecutor.__init__``
    actually belongs to a single concern and should live as an
    instance attribute on that concern's class -- e.g.,

    +---------------------------+----------------------------+
    | legacy ``PyExecutor`` field | owning concern             |
    +===========================+============================+
    | ``model_engine``          | ``ForwardConcern``         |
    | ``sampler``               | ``SampleConcern``          |
    | ``scheduler``             | ``ScheduleConcern``        |
    | ``kv_cache_manager``      | ``ResourceConcern``        |
    | ``resource_manager``      | ``ResourceConcern``        |
    | ``drafter``               | ``SpecDecodeConcern``      |
    | ``speculation_gate``      | ``SpecDecodeConcern``      |
    | ``guided_decoder``        | ``GuidedDecoderConcern``   |
    | ``kv_cache_transceiver``  | ``DisaggConcern``          |
    | ``async_transfer_manager``| ``DisaggConcern``          |
    | ``kv_connector_manager``  | ``KvConnectorConcern``     |
    | ``perf_manager``          | ``PerfMetricConcern``      |
    | ``hang_detector``         | runtime ``Driver`` builtin |
    | ``dwdp_manager``          | ``DwdpConcern``            |
    | ``execution_stream``      | ``ForwardConcern`` (owns)  |
    | ``sample_stream``         | ``SampleConcern`` (owns)   |
    +---------------------------+----------------------------+

    Realistically this dataclass stays at 1-3 fields. The canonical
    member is ``dist`` -- the cross-rank communicator, used by
    schedule, forward, sample, response, disagg, kv_connector, etc.
    """

    # --- Always available ---
    dist: "Distributed" = None

    # Add more ONLY if a candidate fails the "single owner concern"
    # test above. If you find yourself adding a 4th field, audit the
    # candidates first.


@dataclasses.dataclass(frozen=True)
class Configuration:
    """Immutable configuration flags + limits.

    Read at executor startup; never mutated after. Concerns pull each
    flag they care about once in ``__init__``.

    Sub-group by concern family as the field set grows -- keep the
    top-level shape small. Suggested sub-groups (each its own frozen
    dataclass) when promoted out of the flat list:

    - ``conf.batching``   -- max_batch_size, max_num_tokens,
                             stream_interval, ...
    - ``conf.pp``         -- num_micro_batches,
                             pp_async_broadcast_sample_state,
                             pp_multi_stream_sample,
                             pp_scheduler_max_retry_count, ...
    - ``conf.disagg``     -- kv_transfer_timeout_ms,
                             enable_partial_reuse_for_disagg, ...
    - ``conf.bench``      -- is_warmup, is_benchmark_disagg,
                             benchmark_req_queues_size, ...
    - ``conf.feature``    -- enable_attention_dp,
                             enable_iter_perf_stats,
                             enable_kv_cache_events,
                             disable_overlap_scheduler,
                             gather_all_responses, ...

    Until that's worth doing, keep cohesive flags inline.
    """

    device_id: int = 0


@dataclasses.dataclass
class PersistentState:
    """Cross-iter mutable state that genuinely belongs to no concern.

    AIM AT ZERO.

    Every legacy ``PyExecutor`` field considered for this bucket has a
    better home:

    - One-way latches owned by a single concern
      (``speculation_permanently_disabled``,
      ``_benchmark_fill_phase_active``, ...) live as an attribute on
      that concern's class. Cross-concern interactions go through
      method calls, not shared state. See ``concerns.py``.
    - SCHEDULER bookkeeping (``iter_counter``, slot ring,
      ``previous_batch``, ``unhandled_batch_counter``,
      ``has_previous_draft_tokens``, ``micro_batches[]``) lives on the
      SCHEDULER iter as locals.
    - Cross-thread signals (``is_shutdown``) live in ``MessagePort``
      because that's their primary identity.
    - Per-iter "current values" (``use_spec_decode`` /
      ``max_total_draft_tokens`` for the iter's batch) live in
      ``BatchStorage``.

    This bucket is for the residue: cross-iter mutable state with
    LEGITIMATELY multiple unrelated mutators AND that cannot be
    modeled as method calls on a single owner.

    Each surviving field MUST document, in a leading ``#`` comment:
    its single rationale, the concerns / scheduler that mutate it, and
    why method-call delegation through one of them did not suffice.
    Anything that grows a second mutator without an updated rationale
    is a refactor flag.
    """

    # Empty by default. The example below shows the kind of thing that
    # MIGHT belong here if a single owner cannot be agreed -- but
    # usually a concern can claim it and expose a method.
    #
    # speculation_permanently_disabled: bool = False
    # ^ -- this actually belongs on `SpecDecodeConcern.permanently_disabled`,
    #      with `ResponseConcern` calling
    #      `ctx.crn.spec_decode.record_acceptance(request)` at runtime.
    #      Kept here only as a shape example.


@dataclasses.dataclass
class MessagePort:
    """Threading boundary between the public API and the event loop.

    Holds data-passing containers and primitives that BOTH threads
    touch. ALL cross-thread data SHALL flow through this class -- if
    you find yourself reaching for a Lock / Queue / Event / Condition
    in concern code OR in ``PyExecutorCoro`` methods, the boundary
    object should be moved here.

    ``PyExecutorCoro`` (main thread) holds a direct reference to a
    ``MessagePort`` and uses it as its sole interface to the loop.
    The loop thread reaches the same ``MessagePort`` via ``ctx.port``.

    Stays small by construction; only true thread-boundary objects.
    Examples that belong here as we wire them up:

    - ``executor_request_queue``      -- new requests from API thread
    - ``response_queue`` + ``response_cv`` -- responses to API thread
    - ``control_request_queue`` + ``control_request_barrier`` +
      ``control_action_done``        -- synchronous control actions
    - ``shutdown_event``              -- shutdown signal
    - ``canceled_req_ids``            -- request cancellation set
    - ``waiting_queue``               -- pre-active queue
    - ``responses`` + ``result_wait_queues`` -- per-request response
                                                fan-out
    - ``is_shutdown``                 -- one-way latch (cross-thread)
    """

    executor_request_queue: "ExecutorRequestQueue" = None


@dataclasses.dataclass(frozen=True)
class Context:
    """The single object passed to every coroutine in the loop layer.

    Wraps the five buckets. Threaded as an argument through the loop
    function, the SCHEDULER iter, the BATCH body, and every concern's
    ``handle_batch(self, ctx)``. Passing it explicitly (rather than
    via a ContextVar) keeps the dependency surface visible at every
    call.

    The "freely accessible" rule
    ----------------------------

    Anything in ``ctx`` is FREELY ACCESSIBLE to any code on the loop
    thread. That is the point: no plumbing for cross-cutting deps,
    no construction-order constraints for peer-concern access. The
    flip side: anything you don't want freely accessible MUST NOT live
    in ``ctx``. Push it onto a concern instance via construction-time
    kwargs in ``PyExecutorCoro.__init__`` (see ``concerns.py``).

    This is the lever that keeps the design honest. A new "service" in
    ``ctx.svc`` is now a public dependency that any concern can grab;
    a kwarg on a single concern's ``__init__`` is private to that
    concern.

    Why no ``batch: BatchStorage`` field
    ====================================

    Three independent reasons:

    1. **Wrong scope.** The overlap and PP loops have multiple
       batches alive simultaneously, each with its own
       ``BatchStorage`` instance. A single ``ctx.batch`` slot cannot
       represent "the active one"; the SCHEDULER would have to
       repoint it between every ``step`` call, defeating ``ctx``'s
       immutable-identity property.

    2. **Bypasses access control.** The runtime gates per-batch
       access through ``(r, w) = await enter_phase(P)`` views and (in
       ``TLLM_COROUTINE_TRACK_STORAGE=1`` debug mode) through
       ``_TrackedReadView`` / ``_TrackedWriteView`` proxies. Direct
       ``ctx.batch.<field>`` access skips all of this and silently
       re-grants every concern "everything access" -- exactly the
       outcome the freely-accessible rule above warns against.

    3. **Wrong source of truth.** The runtime already exposes the
       active batch via the ``_active_storage`` ContextVar, scoped to
       the currently-running ``step`` call. ``ctx.batch`` would
       either duplicate or contradict it.

    Per-batch data goes through ``enter_phase(BatchPhase.X)``. Always.
    """

    svc: Service
    conf: Configuration
    state: PersistentState
    port: MessagePort
    crn: "Concerns"

    # NO ``batch`` field -- see "Why no batch" above.


# --------------------------------------------------------------------------- #
# Notes on the ContextVar pattern (ambient values, sparingly)
# --------------------------------------------------------------------------- #
#
# The runtime uses ContextVars (``_active_storage``, ``_active_phase``)
# to make the active batch / phase available without threading them
# through every coroutine signature. ``ctx`` is the explicit argument-
# passing alternative. Both have a place; the rule is:
#
# - Use ``ctx`` for everything not needed by the runtime itself
#   (services, config, state, port, concerns). Explicit > implicit
#   when the receiver opts in.
#
# - Use ContextVars for runtime-active values that change PER STEP and
#   would otherwise need every coroutine signature to plumb them
#   (current batch storage, current phase). Bounded set, owned by the
#   runtime in ``coroutines.py``.
#
# We MAY add 1-2 more ContextVars for ambient values that every
# coroutine would otherwise read off ``ctx`` (e.g., ``iter_counter``).
# CAP at 2-3 such values total. Anything below that bar belongs in
# ``ctx`` (explicit), or on a concern attribute (explicit ownership).
