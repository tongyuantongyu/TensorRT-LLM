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
            ctx = Context(svc=svc, conf=conf, state=state, port=port)
            crn = build_concerns(svc=svc, conf=conf, ...)  # see concerns.py

            # Main thread only keeps `port`. Everything else moves to
            # the loop thread. ``crn`` is passed AS A SEPARATE ARG --
            # it intentionally does NOT live on ``ctx`` so concerns
            # can't reach for peers. Only the orchestrators
            # ``run_loop`` / ``scheduler_iter`` / ``batch_body``
            # legitimately need ``crn``. See "Why no ``crn`` field"
            # on :class:`Context` for the four-homes routing of
            # cross-concern data (within-batch -> BatchStorage;
            # batch-to-batch handoff -> SCHEDULER bridge; cross-
            # batch shared state -> ``ctx.svc.*``; private latch ->
            # concern instance attribute).
            self._port = port
            self._loop_thread = threading.Thread(
                target=run_loop, args=(ctx, crn))
            self._loop_thread.start()

        def submit(self, request):
            # MAIN-THREAD path: only `_port` is touchable.
            self._port.executor_request_queue.put(request)

What goes in ``ctx``
====================

Four buckets, each with its own membership rule (below). Reach the
right bucket via ``ctx.svc`` / ``ctx.conf`` / ``ctx.state`` /
``ctx.port``. The bag of concern instances (``Concerns``) does NOT
live here -- see "Why no ``crn`` field" on :class:`Context` below.

What does NOT go in ``ctx``
===========================

- ``BatchStorage`` -- per-batch data is reached only through the
  runtime's ``enter_phase`` / ``step`` views, never through ``ctx``.
  See "Why no ``batch`` field" on :class:`Context` below.
- ``Concerns`` -- the bag of concern instances. Lives on
  ``PyExecutorCoro`` as a side-channel and is passed as a
  separate argument to the orchestrators. Mechanically prevents
  concerns from reaching peer concerns; see "Why no ``crn``
  field" on :class:`Context` for the four-homes routing rule.
- A concern's owned services -- those go on the concern instance via
  its ``__init__`` kwargs (see ``concerns.py``). Putting them in
  ``ctx`` would make them freely accessible to other concerns,
  defeating the point of "owned".
- A concern's PRIVATE cross-batch state -- lives as an instance
  attribute on the concern class (no ``ctx`` involvement at
  all).
- Per-iter SCHEDULER state -- the SCHEDULER iter holds it as locals
  (``iter_counter``, slot ring, ``previous_batch``, HC1 / HC2 /
  HC3 batch-to-batch bridges). "Iter" is a SCHEDULER concept;
  concerns and BATCH bodies do not see it.
"""

import dataclasses
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm._torch.distributed.communicator import Distributed

    from .executor_request_queue import ExecutorRequestQueue
    # Forward refs for service types defined alongside concerns
    # (in ``concerns/services.py``).
    from .concerns import (  # noqa: F401
        ClientChannel,
        RecvOffload,
        RequestPool,
        TerminationService,
    )


@dataclasses.dataclass(frozen=True)
class Service:
    """Truly cross-cutting service objects. Immutable.

    Two kinds of members coexist here:

    1. **Cross-rank primitives** with no natural single owner. The
       canonical member is ``dist`` -- the cross-rank communicator
       used by schedule, forward, sample, response, disagg,
       kv_connector, etc.

    2. **Cross-cutting service objects** that own state previously
       smeared across many ``PyExecutor`` methods. These exist
       because the underlying state (``active_requests``, response
       delivery, request termination) is touched by MANY concerns
       AND a few standalone utilities (``fail_requests``); making
       any single concern its owner would force every other concern
       to reach into that one. The services are:

       * ``pool: RequestPool`` -- owns ``active_requests`` and
         ``inflight_req_ids`` (PP). Methods: ``add_active(reqs)``,
         ``remove_active(reqs)``, ``mark_inflight(reqs)`` /
         ``unmark_inflight(reqs)``, ``is_drained()``,
         iteration / filtered views. Replaces direct
         ``self.active_requests.…`` accesses scattered across
         legacy code.
       * ``client: ClientChannel`` -- the loop-side write surface
         of the response side of ``MessagePort``. Owns the body of
         the legacy ``_enqueue_responses`` (TP gather +
         cross-thread put + per-request fan-out to
         ``result_wait_queues``). Methods: ``enqueue(items)``.
         Concerns and utilities call this instead of poking
         ``MessagePort`` directly for response output.
       * ``termination: TerminationService`` -- owns
         ``DisaggPPTerminationHandler`` reference and the
         resource-free + ``result_wait_queues`` cleanup. Methods:
         ``terminate(req)`` (replaces ``_terminate_request`` +
         ``_do_terminate_request``).
       * ``recv_offload: RecvOffload`` (PP only; ``None`` on
         single-rank executors) -- single-worker thread pool that
         runs blocking ``recv_object`` calls so coroutines can
         poll the resulting :class:`concurrent.futures.Future`
         instead of an MPI Request. Required because
         ``mpi4py.util.pkl5`` does not implement non-blocking
         ``irecv`` for pickled objects; the legacy executor's
         dedicated ``broadcast_sample_state_handler`` thread
         played the same role.

    The ``fail_requests(ctx, reqs, msg)`` utility (free function
    in ``concerns/_shared.py``) composes ``pool``, ``client``,
    and ``termination``.

    Almost every other "service" in the legacy
    ``PyExecutor.__init__`` belongs to a single concern and should
    live as an instance attribute on that concern's class -- e.g.,

    +---------------------------+----------------------------+
    | legacy ``PyExecutor`` field | owning concern / service   |
    +===========================+============================+
    | ``model_engine``          | ``ForwardConcern``         |
    | ``sampler``               | ``SampleConcern``          |
    | ``scheduler``             | ``ScheduleConcern``        |
    | ``kv_cache_manager``      | ``ResourceConcern``        |
    | ``resource_manager``      | ``ResourceConcern`` (also  |
    |                           | ``TerminationService``     |
    |                           | uses for ``free_resources``|
    |                           | -- inject the same ref)    |
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
    | ``active_requests``       | ``RequestPool`` (svc.pool) |
    | ``inflight_req_ids``      | ``RequestPool`` (svc.pool) |
    | ``responses``+CV+         | ``ClientChannel``          |
    | ``result_wait_queues``    | (svc.client) +             |
    |                           | ``MessagePort`` (data)     |
    | ``_disagg_pp_termination_ | ``TerminationService``     |
    | handler``                 | (svc.termination)          |
    +---------------------------+----------------------------+

    Member count discipline: HIGH BAR for adding to this list
    beyond the five above (``dist``, ``pool``, ``client``,
    ``termination``, ``recv_offload``). Anything new must be used
    by multiple concerns AND have no natural single-owner concern.
    """

    # --- Always available ---
    dist: "Distributed" = None

    # --- Cross-cutting services (own state previously on PyExecutor) ---
    pool: "RequestPool" = None
    client: "ClientChannel" = None
    termination: "TerminationService" = None
    # PP-only; ``None`` on single-rank executors.
    recv_offload: "RecvOffload" = None

    # Add more ONLY if a candidate fails the "single owner concern"
    # test above. If you find yourself adding a 6th field, audit the
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
    """Mutable state that survives across batches and genuinely belongs to no concern.

    AIM AT ZERO. Every legacy ``PyExecutor`` field considered for
    this bucket has a better home:

    - **A concern's PRIVATE latch / counter** -- e.g. a one-way
      flag that ONLY one concern observes
      (``_benchmark_fill_phase_active``,
      ``speculation_permanently_disabled``) -- lives as an
      instance attribute on that concern's class. No ``ctx``
      involvement.
    - **Cross-concern shared state** -- needed by multiple
      concerns across many batches (e.g. the spec_decode rolling
      gate) -- lives on a SHARED SERVICE in ``ctx.svc.*``. Both
      sides go through that one owner-typed surface.
    - **SCHEDULER bookkeeping** (``iter_counter``, slot ring,
      ``previous_batch``, ``unhandled_batch_counter``,
      ``has_previous_draft_tokens``, ``micro_batches[]``) lives
      on the SCHEDULER iter as locals. "Iter" is a SCHEDULER
      concept -- concerns don't see it.
    - **Cross-thread signals** (``is_shutdown``) live in
      :class:`MessagePort` because that's their primary
      identity (touched by both threads).
    - **Per-batch "current values"** (``use_spec_decode`` /
      ``max_total_draft_tokens`` for the batch's run) live in
      :class:`BatchStorage`.
    - **Batch-to-batch handoffs** (one batch's terminal output is
      next batch's input, e.g. HC1 in overlap) are mediated by
      the SCHEDULER iter -- it reads the prior batch's terminal
      view and stuffs the bridge value into the next batch's
      SCHEDULE_0 write view. Not a ``PersistentState`` field.

    Each surviving field MUST document, in a leading ``#`` comment,
    why none of the above buckets fits.
    """

    # Empty by default. Most candidates fit one of the buckets
    # above; ``PersistentState`` is the residue. The example below
    # shows what a residue field would look like.
    #
    # Example: a hypothetical legacy flag with multiple unrelated
    # concern-side mutators that defies single-owner placement
    # would land here with a comment justifying the routing.


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

    Plain-loop bring-up scope
    -------------------------

    The fields below are what the plain executor loop needs. PP /
    disagg / control-action additions land as those features are
    wired:

    * ``executor_request_queue`` (``ExecutorRequestQueue``) --
      cross-thread request inbox. Main thread enqueues; loop thread
      drains via ``ScheduleConcern.handle_batch``. Cancellations
      and shutdown also flow through this queue (as marker items)
      so a single thread-safe channel carries every API->loop
      message; the loop's ``ScheduleConcern`` dispatches each
      marker to the appropriate per-iter destination
      (``ctx.port.is_shutdown`` for shutdown; cancel IDs go into
      the SCHEDULE_0 write view's ``canceled_req_ids`` field, which
      ``ResponseConcern`` reads at RESPOND_8).
    * ``shutdown_event`` (``threading.Event``) -- set by the loop
      thread after the loop exits cleanly. Main thread waits on
      this in :meth:`PyExecutorCoro.shutdown`.
    * ``is_shutdown`` (bool) -- set by ``ScheduleConcern`` when it
      sees the shutdown marker in the request queue. Read by the
      SCHEDULER iter's drain check. Also drives early-return in
      ``ClientChannel.await_*`` so main-thread waiters don't block
      forever past shutdown. NOT a one-shot event because both
      threads only READ it after the loop sets it -- a plain bool
      with the publishing happens-before the read covers it.
    * ``is_warmup`` (bool) -- set by main thread via
      :attr:`PyExecutorCoro.is_warmup` (used e.g. by
      ``_util.py``'s KV-cache memory estimation pass to flag a
      dummy run). The loop thread consumes it in two places:
      :meth:`ForwardConcern.handle_batch` reflects it onto
      ``model_engine.is_warmup`` once per batch (just before
      invoking ``forward()``) so model-internal gates see the same
      value; the SCHEDULER-layer ``profiler`` drains a warmup pass
      at the top of its try-body
      (``while ctx.port.is_warmup: yield``) before the post-warmup
      ``itertools.count`` block takes over. Plain bool -- the GIL
      serializes the cross-thread attribute set / get, and the
      only consumers read it after the main-thread write happens-
      before (loop start AND per-batch FORWARD_2 boundary).
      One-shot (caller-side; not runtime-enforced): callers
      transition the value ``False -> True -> False`` at most
      once over the executor's lifetime; ``profiler``'s warmup-
      drain block relies on this to never re-enter warmup once
      cleared. Matches the legacy ``PyExecutor.is_warmup``
      (plain attribute, no runtime check).

    What does NOT belong here
    -------------------------

    Anything LOOP-thread-only -- even if it's a buffer that a few
    concerns share -- belongs in :class:`BatchStorage` (within-
    batch concern->concern data), in ``ctx.svc.*`` (cross-batch
    shared state), in a SCHEDULER iter local (batch-to-batch
    handoff), or on a concern instance (private latch), NOT on
    :class:`MessagePort`. Putting it here advertises "cross-thread,
    may need locking" and misleads readers and future maintainers.
    Examples that belonged here in earlier drafts but moved out
    once the threading audit clarified:

    * ``canceled_req_ids`` -- main thread NEVER touches it
      directly (cancellations cross the boundary as queue
      markers). Now flows through ``BatchStorage`` --
      ``ScheduleConcern`` writes ``w0.canceled_req_ids`` at
      SCHEDULE_0 from the dispatched marker IDs;
      ``ResponseConcern`` reads ``r8.canceled_req_ids`` at
      RESPOND_8.

    The response output side of the boundary -- ``responses`` dict +
    cv + per-request streaming sinks -- is owned by
    :class:`ClientChannel` (in :mod:`concerns.services`) rather than
    here, because ClientChannel adds the loop-side write-API
    (:meth:`enqueue`) that concerns call. ``MessagePort`` would
    otherwise grow a write-only / read-only split surface that
    duplicates ClientChannel's role.

    Future fields (PP / disagg / control):

    * ``control_request_queue`` + ``control_request_barrier`` +
      ``control_action_done`` -- ``control_action`` CM rendezvous.
    """

    executor_request_queue: "ExecutorRequestQueue" = None
    shutdown_event: threading.Event = dataclasses.field(
        default_factory=threading.Event)
    is_shutdown: bool = False
    is_warmup: bool = False


@dataclasses.dataclass(frozen=True)
class Context:
    """The single object passed to every coroutine in the loop layer.

    Wraps the four buckets. Threaded as an argument through the loop
    function, the SCHEDULER iter, the BATCH body, and every
    concern's ``handle_batch(self, ctx)``. Passing it explicitly
    (rather than via a ContextVar) keeps the dependency surface
    visible at every call.

    The "freely accessible" rule
    ----------------------------

    Anything in ``ctx`` is FREELY ACCESSIBLE to any code on the loop
    thread. That is the point: no plumbing for cross-cutting deps,
    no construction-order constraints for service access. The flip
    side: anything you don't want freely accessible MUST NOT live in
    ``ctx``. Push it onto a concern instance via construction-time
    kwargs in ``PyExecutorCoro.__init__`` (see ``concerns.py``).

    This is the lever that keeps the design honest. A new "service"
    in ``ctx.svc`` is now a public dependency that any concern can
    grab; a kwarg on a single concern's ``__init__`` is private to
    that concern.

    Why no ``crn: Concerns`` field
    ==============================

    Concerns DO NOT receive ``ctx.crn`` -- they MUST NOT reach for
    peer concerns. The constraint is mechanical, not just
    convention: the bag of concern instances lives on
    :class:`PyExecutorCoro` as a side-channel and is passed as a
    SEPARATE argument to ``run_loop`` / ``scheduler_iter`` /
    ``batch_body`` (the orchestrators that legitimately need it).
    Concerns receive only ``ctx``.

    Rationale -- the four homes for cross-concern data, and which
    one a given case goes to:

    1. **Within-batch concern->concern data** (e.g.,
       ``ScheduleConcern`` writes ``scheduled_batch`` at
       SCHEDULE_0; everyone else reads it at later phases) goes
       through ``BatchStorage`` via the typed read / write views
       returned by ``enter_phase`` / ``batch_phase``. The
       happens-before rule and per-phase view narrowing are the
       design's contract; "secret" peer method calls
       (``ctx.crn.X.method(...)``) would bypass it.

    2. **Batch-to-batch handoff** (one batch's terminal output is
       the next batch's input -- the canonical example is HC1 in
       the overlap loop: batch N-1's ``sample_state`` becomes
       batch N's ``previous_tensors_device`` at SCHEDULE_0) is
       the SCHEDULER iter's job. ONLY the SCHEDULER speaks of
       "iter": it reads the prior batch's terminal read view,
       computes the bridge value, and stuffs it into the next
       batch's SCHEDULE_0 write view. No concern-to-concern
       channel involved.

    3. **Cross-batch shared state** -- cumulative state with
       multiple writers / multiple readers across many batches,
       e.g., spec_decode's rolling-acceptance gate written by
       ``response`` after each finished request and read by
       ``schedule`` at every batch's SCHEDULE_0 -- lives on a
       SHARED SERVICE in ``ctx.svc.*``. Both sides go through a
       single owner-typed surface; neither reaches the other's
       concern internals.

    4. **A concern's PRIVATE cross-batch state** -- a latch or
       counter that ONLY one concern reads and writes, observed
       from each per-batch invocation -- is just an instance
       attribute on that concern. No mention on ``ctx`` at all.

    Removing ``ctx.crn`` makes (1)-(4) enforceable at
    compile-by-grep: ``rg "ctx\\.crn" tensorrt_llm/_torch/
    pyexecutor/concerns/`` should always be empty.

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
       access through ``(r, w) = await enter_phase(P)`` views and
       (in ``TLLM_COROUTINE_TRACK_STORAGE=1`` debug mode) through
       ``_TrackedReadView`` / ``_TrackedWriteView`` proxies. Direct
       ``ctx.batch.<field>`` access skips all of this and silently
       re-grants every concern "everything access" -- exactly the
       outcome the freely-accessible rule above warns against.

    3. **Wrong source of truth.** The runtime already exposes the
       active batch via the ``_active_storage`` ContextVar, scoped
       to the currently-running ``step`` call. ``ctx.batch`` would
       either duplicate or contradict it.

    Per-batch data goes through ``enter_phase(BatchPhase.X)``. Always.
    """

    svc: Service
    conf: Configuration
    state: PersistentState
    port: MessagePort

    # NO ``batch`` field -- see "Why no batch" above.
    # NO ``crn`` field -- see "Why no crn" above.


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
#   (services, config, state, port). Explicit > implicit when the
#   receiver opts in. The ``Concerns`` bag is NOT on ``ctx``; it's
#   passed as a separate orchestrator-only argument.
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
