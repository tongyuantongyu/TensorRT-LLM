# Concern coroutine framework for the PyExecutor refactor

A “concern” is a domain area of the forward loop — `schedule`,
`forward`, `sample`, `response`, `disagg`, `kv_connector`,
`spec_decode`, `guided_decoder`, `perf_metric`, `iter_stats`,
`profile`, `control`, `benchmark_disagg_gate`, `dwdp`,
`kv_cache_events`, `save_hidden_states`, etc. — whose work is
intermixed across `BatchPhase` values on every batch.

This file is the design doc that the per-concern modules (or sections
in this file, TBD) will follow once we start implementing. It also
defines the placeholder `Concerns` bag dataclass (currently empty)
that `ctx.crn` points at. Pure spec otherwise.

## Coroutine or plain method?

Simple rule, decided by phase count:

- **Multiple phases per batch → coroutine**, with `await enter_phase(P)` markers between phase-local blocks. Even if the concern doesn’t carry any local variables across phases, the coroutine body keeps the concern’s whole story in one readable function — skim from top to bottom and you see exactly which phases it participates in and what it does at each. Several separate methods, one per phase, scattered across the class surface and called from the BATCH body, hide that flow.

- **Single phase per batch → method**, called from `batch_body` inside the matching `async with batch_phase(P): ...` block. No coroutine machinery; just a function call.

Method skeleton (single phase):

```python
class ProfileConcern:
    def __init__(self, *, profiler):
        self._profiler = profiler

    def tick(self, ctx: Context) -> None:
        self._profiler.step()

# In batch_body:
async with batch_phase(BatchPhase.SCHEDULE_0):
    ctx.crn.profile.tick(ctx)
    # ... other concerns ...
```

A method is also the right shape for the cross-concern peer hook
(see the `record_acceptance` example further down), regardless of
whether the rest of the concern is method-only or coroutine-shaped.

### Why a coroutine even without state-threading

Concerns like `DisaggConcern` or `ResourceConcern` could in principle expose two or three methods (e.g., `Resource.prepare(...)` at RESOURCE_PREP_1, `Resource.update(...)` at RESPOND_8), called from the BATCH body at the right phases. We don’t do that. The coroutine form has two real advantages:

1. **One source of truth for the concern’s lifecycle.** The body reads top-to-bottom in phase order, with `await enter_phase(P)` as the rendezvous between BATCH and concern. A reader audits one function instead of cross-referencing N method definitions against M BATCH-body call sites.

2. **Locality of intra-concern state, when it appears.** The first time a concern needs to thread something across a phase boundary (a CUDA event, a matcher result, a flag, …), you just declare a coroutine local. No retrofit; no “what container do I store this in” question.

The only place the method form genuinely wins is when the per-batch work is one short call — then a coroutine adds boilerplate (`async def handle_batch(self, ctx): r, _ = await enter_phase(...); ...`) without saving anything. That’s the single-phase case.

### Hang detection

Hang detection is **intentionally not** a concern. It becomes a builtin feature of the runtime `Driver` — the Driver can timestamp each `send` to a coroutine and trip a watchdog when a coroutine doesn’t yield for too long. That keeps the per-concern code free of `checkpoint()`-style pollution and gives a uniform diagnostic across every loop variant.

(`HangConcern` from earlier drafts is dropped.)

## Lifecycle

- **Construction (main thread):** `PyExecutorCoro.__init__` builds one instance per concern, then bundles them into a `Concerns` dataclass that goes onto `ctx.crn`. Construction is plain kwargs (no `ctx` argument) — the caller pulls cross-cutting deps from its local `ctx` and passes them in.

- **Lifetime:** each concern instance lives for the executor’s lifetime. State that survives across iters is stored as instance attributes.

- **Per batch (loop thread):** the BATCH body asks each concern for a fresh per-batch coroutine via `concern.handle_batch(ctx)`, wraps it in a runtime `Concern` handle, and drives it phase-by-phase with `await resume(handle)` inside the matching `async with batch_phase(P): ...` block.

## The concern class shape

Skeleton:

```python
from tensorrt_llm._torch.pyexecutor.batch_storage import BatchPhase, enter_phase
from tensorrt_llm._torch.pyexecutor.context import Context

class DisaggConcern:
    def __init__(
        self,
        *,
        # Cross-cutting deps pulled from ctx by the caller
        # (PyExecutorCoro.__init__) and passed in as plain kwargs.
        # The concern itself never sees `ctx` at construction
        # time -- it's loop-thread-only state.
        dist,
        # Owned services -- not in `ctx`, used by no other concern.
        transceiver,
        async_transfer_manager,
        # Optional config (pulled from `ctx.conf.<sub>` by caller):
        kv_transfer_timeout_ms=None,
    ):
        self.dist = dist
        self.transceiver = transceiver
        self.transfer_mgr = async_transfer_manager
        self.kv_transfer_timeout_ms = kv_transfer_timeout_ms

        # Cross-iter state: just instance attributes.
        # (none for disagg today; see SpecDecodeConcern below for
        #  the canonical example)

    async def handle_batch(self, ctx: Context):
        # Receives `ctx` as a runtime arg. Use it for:
        #   - peer concern access: `ctx.crn.X.method()`
        #   - cross-iter mutable state: `ctx.state.<...>` (rare)
        #   - thread-boundary signals: `ctx.port.<...>`
        # For owned services / cross-cutting deps the concern
        # already pulled in `__init__`, prefer `self.<...>`
        # (no need to re-walk `ctx` every access).

        # SCHEDULE_0: opportunistic non-blocking probes
        await enter_phase(BatchPhase.SCHEDULE_0)
        if self.transceiver:
            self._check_disagg_ctx_schedulable_status()
            self._check_disagg_gen_transfer_status()

        # FORWARD_2: prepare disagg-gen-transmission-complete
        r1, _ = await enter_phase(BatchPhase.FORWARD_2)
        if self.transceiver:
            self._prepare_disagg_gen_transmission_complete(r1.scheduled_batch)

        # RESPOND_8: send_kv_async + opportunistic probe
        r7, _ = await enter_phase(BatchPhase.RESPOND_8)
        for req in r7.scheduled_batch.all_requests():
            if req.is_context_only_request and ...:
                self.transfer_mgr.start_transfer(req)
                self.transceiver.respond_and_send_async(req)
        if self.transceiver:
            self._check_disagg_ctx_cache_transfer_status(0)
```

The BATCH body driving them (note: a free function, **not** a method of `PyExecutorCoro` — the loop side and the main-thread side are separate; see `context.py` for the threading rationale):

```python
from tensorrt_llm._torch.pyexecutor.coroutines import Concern, batch_phase, resume
from tensorrt_llm._torch.pyexecutor.context import Context

async def batch_body(ctx: Context):
    crn = ctx.crn
    # One Concern handle per concern instance, alive for the batch's
    # lifetime. The runtime's `Concern` handle wraps the coroutine and
    # carries a `done` flag the Driver flips on completion. Optional
    # concerns may be `None` — skip them.
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

    async with batch_phase(BatchPhase.FORWARD_2):
        if disagg is not None:
            await resume(disagg)
        await resume(forward)
        # ...

    # ...
```

## Runtime guarantees that make this work

(See `coroutines.py` for the implementation.)

- **Phase skipping.** If a concern’s next `enter_phase(P_next)` is at `P_next > current`, `await resume(handle)` short-circuits without entering the Driver. So BATCH can call `resume` at every phase uniformly — concerns explicitly skip phases they don’t care about by simply not yielding there.

- **Done short-circuit.** When a concern’s coroutine returns, its `Concern.done` flips True. Subsequent `resume(handle)` calls return immediately without entering the Driver. So a concern that finishes early (e.g., last yield at RESPOND_8) is a no-op for any later phases the BATCH still iterates through.

- **Strict progression.** Successive `enter_phase` calls from the same coroutine **must** yield strictly greater phases. A concern that legitimately repeats work across a phase boundary needs `await again()` (see runtime docs). Don’t try to `enter_phase` the same phase twice in one coroutine.

- **Handle is the canonical reference.** The runtime tracks per-coroutine state (suspension record, done flag, debug-mode tracked-write log) by the handle, not by the coroutine. Always wrap in `Concern(coro)` before driving; never pass a raw coroutine to `resume`. `__del__` on undriven handles closes the coroutine cleanly — replaces the old `spawn` primitive.

- **No `ctx.batch`.** Per-batch data goes through `(r, w) = await enter_phase(BatchPhase.X)` views. Read-view at phase P sees writes from strictly earlier phases; write-view at phase P accepts writes for phase-P fields only. Direct field access on `BatchStorage` is forbidden by design.

## Dependency injection

Construction signature: `__init__(self, *, owned_a, owned_b, dep_x, dep_y)` — **all** kwargs, no `ctx` argument. The caller (`PyExecutorCoro.__init__`) pulls cross-cutting deps from its local `ctx` and passes them in:

```python
# In PyExecutorCoro.__init__ (main thread):
disagg = DisaggConcern(
    dist=ctx.svc.dist,  # cross-cutting from ctx
    transceiver=tx,  # owned
    async_transfer_manager=atm,  # owned
    kv_transfer_timeout_ms=ctx.conf.disagg.kv_transfer_timeout_ms,
)
```

Why this shape:

- **No `ctx` at construction time** keeps the construction-vs-runtime cut crisp. `ctx.crn` doesn’t exist yet while concerns are being built (chicken-and-egg with `Concerns` itself), so a concern’s `__init__` couldn’t access peer concerns even if it wanted to. The constructor is pure: take kwargs, store them, done.

- **All kwargs** makes construction sites self-documenting and type-checkable. Reading the call site shows exactly which deps the concern uses, at the cost of a few extra characters per arg.

- **`__init__` is independent of runtime state.** Trivially mockable in tests: build the concern with mock kwargs, drive its `handle_batch` against a test `ctx`.

Why not auto-DI? See the design discussion in commit history. Short version: explicit `__init__` is ~5 lines per concern, type-checkable, grep-friendly, fail-at-construction-time. The ~30–50 line auto-DI helper we’d otherwise write saves marginal boilerplate at the cost of runtime errors, single-instance-per-type constraint, `Optional` ambiguity, `from __future__` annotation resolution caveats, and lost grep on construction sites.

## Cross-iter state

Lives on the concern class as an instance attribute. Example:

```python
class SpecDecodeConcern:
    def __init__(self, *, dist, drafter, speculation_gate):
        self.dist = dist
        self.drafter = drafter
        self.speculation_gate = speculation_gate
        # Cross-iter latch -- written by `record_acceptance` (called
        # by ResponseConcern at RESPOND_8), read by this concern's
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
```

This is the pattern that lets `PersistentState` shrink to ~zero.

## Cross-concern interactions

At runtime, peer concerns are reached via `ctx.crn.X.method()`, **not** through stored peer references. Construction order doesn’t matter, dependency wiring isn’t repeated at construction sites, and optional peers (`ctx.crn.kv_connector is None`) are checked the same way they’re checked everywhere else.

Example:

```python
class ResponseConcern:
    async def handle_batch(self, ctx: Context):
        r7, _ = await enter_phase(BatchPhase.RESPOND_8)
        ...
        for finished_request in finished:
            # Peer call via ctx.crn -- NOT via a stored attribute.
            if ctx.crn.spec_decode is not None:
                ctx.crn.spec_decode.record_acceptance(finished_request)
        ...
```

Construction therefore does **not** need to thread peer references. Every concern’s `__init__` lists owned services + cross-cutting deps; peer concerns are discovered at runtime through `ctx.crn`.

The trade-off: the dependency graph between concerns isn’t visible at construction sites. If that becomes a problem, we add a lint or test that walks `ctx.crn.<X>` references in concern source and fails on unwired peers. For now, the runtime-discovery model is simpler.

## Shared services (`ctx.svc.*`)

Three loop-thread service objects own state previously smeared across many `PyExecutor` methods. They live on `ctx.svc` (see `context.py`’s `Service` dataclass) and are reachable from any concern and any standalone utility.

### `ctx.svc.pool`: `RequestPool`

Owns the `active_requests` list and the PP-only `inflight_req_ids` set. Single owner of “what’s currently in flight in the executor (across iterations)”. Methods:

- `add_active(reqs)` — admit new fetched requests + ADP dummy.
- `remove_active(reqs)` — evict finished or failed requests.
- `mark_inflight(reqs)` / `unmark_inflight(reqs)` — PP slot-ring exclusion set add / remove.
- Iteration helpers (`__iter__`, `count_schedulable()`, filtered views for disagg-state subsets, etc.).
- `is_drained() -> bool` — empty pool predicate (used by the SCHEDULER’s `should_stop_processing` check).

**Used by:** schedule concern (admit, schedule iteration), response concern (evict finished), disagg concern (filtered views, mark/unmark inflight on PP), `fail_requests` utility, SCHEDULER iter (`is_drained`).

### `ctx.svc.client`: `ClientChannel`

The loop-side write surface of the response side of `MessagePort`. Encapsulates the body of the legacy `_enqueue_responses` (TP gather, cross-thread put on `MessagePort.responses` + `response_cv.notify_all`, per-request fan-out to `MessagePort.result_wait_queues`). Methods:

- `enqueue(items: List[Tuple[req_id, LlmResponse]])`.

**Used by:** response concern (`handle_responses`, `handle_first_token`, `handle_canceled`), `fail_requests` utility (error responses). Concerns and utilities call this instead of poking `MessagePort` directly for response output.

### `ctx.svc.termination`: `TerminationService`

Owns the `DisaggPPTerminationHandler` reference and the resource-free + `result_wait_queues` cleanup logic. Body of the legacy `_terminate_request` + `_do_terminate_request`. Methods:

- `terminate(req)` — terminate one request (PP-aware dispatch).

**Used by:** response concern (response retire, cancellation), schedule concern (paused-request termination), disagg concern (ctx-cache transfer probe, KV transfer timeout), `fail_requests` utility, SCHEDULER’s catastrophic handler.

### Why services and not concerns?

- These objects own **executor-wide** mutable state (the request pool, the cross-thread response queues, the termination handler). Concerns own per-batch lifecycles; this state outlives any one batch.

- They have **no** per-batch coroutine. Their methods are called synchronously from concerns (and from `fail_requests`); no `handle_batch` involved.

- Multiple concerns mutate them, so a single concern can’t be the owner without forcing the others to reach into it.

## Failure handling

Two distinct failure modes; the legacy `_handle_errors` conflated them.

### Mode A — catastrophic

A concern can’t recover; the rest of the loop can’t proceed. In the new design the concern’s coroutine just `raise`s — no special helper, no `try/except` wrapper around the body. The exception propagates:

```text
concern body -- raise --> BATCH body -- (re-raise) -->
SCHEDULER iter (catches at the top of its loop) -->
fail_requests(ctx, list(ctx.svc.pool), msg) +
ctx.port.shutdown_event.set()
```

The intermediate `GeneratorExit` thrown into still-suspended concerns by the Driver runs their `finally:` for concern-**local** cleanup (CUDA event handles, partial buffers) — but **not** for the failing of requests, which is the SCHEDULER’s job.

### Mode B — per-request fail-fast

A specific subset of requests is bad; the rest of the batch (and the loop) continues. The concern calls `fail_requests(ctx, reqs, msg)` and continues its own work. The utility (free function in `concerns/_shared.py`):

```python
def fail_requests(ctx, reqs, msg):
    error_responses = []
    for req in reqs:
        req.state = LlmRequestState.GENERATION_COMPLETE
        error_responses.append((req.py_request_id, LlmResponse(
            request_id=req.py_request_id,
            error_msg=msg,
            client_id=req.py_client_id,
        )))
    ctx.svc.pool.remove_active(reqs)
    ctx.svc.client.enqueue(error_responses)
    for req in reqs:
        ctx.svc.termination.terminate(req)
```

### `scheduled_batch` is immutable

After fail-fast, the failed request stays in any `BatchStorage.scheduled_batch` snapshot that included it. Concerns that subsequently iterate `scheduled_batch` either:

- iterate `scheduled_batch.context_requests` / `.generation_requests` and check `req.state` (the fail-fast marker); or
- iterate `ctx.svc.pool` instead, which the fail-fast removed from already; or
- call resource / sample functions that are idempotent on already-terminated requests.

Today’s legacy code already follows this discipline. Only **one** explicit `state != GENERATION_COMPLETE` check survives in the hot path (in `_update_request_states_tp`); other sites either iterate `active_requests` (auto-excluded after fail-fast) or rely on idempotency. The new design preserves this — with a small `iter_live(reqs)` helper available where the discipline is awkward.

## The `Concerns` bag

`Concerns` is a frozen dataclass holding one field per concern. Built once at executor startup by `PyExecutorCoro.__init__` and attached as `ctx.crn`. Optional concerns are typed `Optional[X]` and default to `None` — callers must check `if ctx.crn.X is not None` before use.

Currently empty (placeholder); real fields are filled in as concerns are implemented.

The services live in `ctx.svc` (**not** `ctx.crn`). Concerns and utilities reach them via `ctx.svc.pool` / `ctx.svc.client` / `ctx.svc.termination`.

## Planned concerns

Per the loop annotations in `py_executor.py` (plain / overlap / PP). Listed in the recommended **implementation order** — bring up the plain loop first, then add optional features, then add the PP-specific ring-broadcast concern. Each row’s `shape` column is either `method` (single per-batch phase, exposed as a method called by `batch_body`) or `coroutine` (multiple per-batch phases or intra-concern continuity, exposed via `handle_batch(ctx)` returning a coroutine wrapped by the BATCH in a `Concern` handle).

### Phase 1 — minimum viable plain loop

| order | concern | shape | scope |
|------:|---------|-------|-------|
| 1 | ProfileConcern | method | torch / CUDA-event profiler driver tick |
| 2 | ControlConcern | method | control-request rendezvous |
| 3 | ScheduleConcern | coroutine | request fetch + schedule + ADP dummy + `can_queue` + inflight_ids (multi-phase: SCHEDULE_0 + RESPOND_8 inflight cleanup). Uses `ctx.svc.pool` for all `active_requests` + inflight_ids access. |
| 4 | ResourceConcern | coroutine | `prepare_resources` (main scheduled batch + disagg-gen-init holder) at RESOURCE_PREP_1 + `update_resources` / `revert_gen_alloc` at RESPOND_8 |
| 5 | ForwardConcern | method | model forward call (single-phase per rank; distributed in PP via NCCL p2p — HC7) |
| 6 | SampleConcern | coroutine | `sample_async` at SAMPLE_3 + `update_request_states` / `update_requests` at APPLY_7 |
| 7 | ResponseConcern | coroutine | first-token (SCHEDULE_0, disagg gate optional) + `handle_canceled` + `handle_responses` at RESPOND_8. Uses `ctx.svc.client.enqueue` for all response output, `ctx.svc.pool.remove_active` to evict finished, `ctx.svc.termination.terminate` for cancellation / retire termination. |

After phase 1, the plain loop runs end-to-end with non-disagg, non-spec, non-guided, non-connector configs. Test it.

### Phase 2 — iteration telemetry (still plain loop)

| order | concern | shape | scope |
|------:|---------|-------|-------|
| 8 | IterStatsConcern | coroutine | `iter_stats` record at SCHEDULE_0 + num_ctx_tokens snapshot + `process_iter_stats` at FINALIZE_9 (gated by `enable_iter_perf_stats`; optional) |
| 9 | PerfMetricConcern | coroutine | CUDA timing events created at FORWARD_2, recorded around forward / sample, used at RESPOND_8 (`compute_batch_gpu_times`) |

### Phase 3 — optional features (any order; pick by need)

| order | concern | shape | scope |
|------:|---------|-------|-------|
| 10 | SpecDecodeConcern | coroutine | gating (SCHEDULE_0) + drafter run (FORWARD_2) + `record_acceptance` hook called by Response at RESPOND_8. Owns `permanently_disabled` latch. |
| 11 | DisaggConcern | coroutine | KV transceiver probes + package `disagg_gen_init_to_prepare` (SCHEDULE_0) + async KV recv submission (RESOURCE_PREP_1) + transmission-complete (FORWARD_2) + `send_kv_async` + ctx-cache transfer probe (RESPOND_8) |
| 12 | KvConnectorConcern | coroutine | `handle_metadata` + `start_batch` (FORWARD_1) + `wait_for_save` (FORWARD_2) + `terminate_requests` (RESPOND_8) |
| 13 | GuidedDecoderConcern | coroutine | `add_batch` + `init_disagg_gen_requests` (FORWARD_2) + `execute(logits)` (SAMPLE_3) + `handle_errors` (APPLY_7). Failed-request list also threads from SAMPLE_3 to APPLY_7 as a coroutine local. |
| 14 | DwdpConcern | method | first-layer weight prefetch at FORWARD_2 (single phase) |
| 15 | KvCacheEventsConcern | method | `flush_iteration_events` at RESPOND_8 (single phase, gated by `enable_kv_cache_events`) |
| 16 | SaveHiddenStatesConcern | method | `spec_resource_mgr.process_and_save` at APPLY_7 (single phase, gated by SaveHiddenStates spec mode) |
| 17 | BenchmarkDisaggGateConcern | method | benchmark-disagg fill-phase gate at SCHEDULE_0 (returns retry/skip decision to BATCH; single phase) |

### Phase 4 — pipeline parallel

| order | concern | shape | scope |
|------:|---------|-------|-------|
| 18 | RingBroadcastSampleConcern | coroutine (long-running) | PP-only long-running ring broadcast spanning (n−1) iters per batch (HC10). Replaces the bcast thread. |

(`ScheduleConcern`, `ForwardConcern`, `SampleConcern`, etc. also gain PP-specific paths — the PP variants of the existing concerns, not new concerns. Document those changes inside the per-concern files when implemented.)

**Naming convention:** each concern class is `XxxConcern` even if the domain word is short (“disagg”, “sample”). The `Concern` suffix distinguishes the class from the `Concern` runtime handle without confusion.

**File layout:** TBD. Two reasonable options:

1. One file per concern under `concerns/` (e.g., `concerns/disagg.py`).
2. All concerns in this file, grouped by section.

Pick (1) once concerns get non-trivial. Until then, growing this file is fine. `Concerns` (the bag) stays in `concerns.py` (or `concerns/__init__.py`) so `Context` can string-reference it.

## The loop-layer entry point

`run_loop(ctx)` (or whatever we name the loop function) is the loop thread’s entry point. It receives `ctx` from the thread launch in `PyExecutorCoro.__init__` and runs the SCHEDULER iter forever (until `ctx.port.shutdown_event` is set or the request stream drains, as detected via `ctx.svc.pool.is_drained()`).

The SCHEDULER iter is also a free function `async def scheduler_iter(ctx)` (or method on a small `Scheduler` class **inside** the loop layer, if it grows state — but that’s a loop-thread class, **not** `PyExecutorCoro`). It owns:

- The deque / slot ring of `Batch` handles (one per in-flight batch).

- Per-iter SCHEDULER bookkeeping: `iter_counter`, slot ring (PP), `previous_batch` / HC2 promotion (overlap), `has_previous_draft_tokens` (HC3 cross-iter side channel, overlap), HC1 cross-iter bridge code (overlap).

- Cross-rank scheduler-direct, **not** tied-to-any-batch collectives: HC9 `retire_vote` (PP only); `terminate_pending_requests` ballot (PP-only).

- The catastrophic exception handler at the top of its loop — when a batch’s coroutine propagates an exception out, SCHEDULER calls `fail_requests(ctx, list(ctx.svc.pool), msg)` and sets `ctx.port.shutdown_event`. See [Failure handling](#failure-handling) above.

Inside the iter, for each fresh batch the SCHEDULER constructs a `Batch` handle (with `BatchStorage`), starts `batch_body(ctx)` (which spawns one `Concern(handle)` per concern via `ctx.crn.X.handle_batch(ctx)`), and drives via `await step(handle, through=Phase)`.

**The thread cut:** `run_loop` and everything it transitively calls (SCHEDULER iter, `batch_body`, concern coroutines) are loop-thread only. `PyExecutorCoro` instance methods are main-thread only. The two communicate strictly through `ctx.port` (with `ctx.svc.client` as the loop-side write surface for the response side of port).

---

### `Concerns` placeholder (Python)

```python
import dataclasses


@dataclasses.dataclass(frozen=True)
class Concerns:
    """Bag of concern instances. Lives at `ctx.crn`.

    Built once at executor startup by `PyExecutorCoro.__init__` and
    attached to `Context`. Frozen so the bag itself is immutable
    (the concern instances inside hold their own mutable state as
    attributes; that's fine).

    Currently empty -- real fields are filled in as the concern
    classes are implemented (see "Planned concerns" above).
    """

    # Add fields per the planned-concerns table as concerns are built.
    # Optional concerns are typed `Optional[X] = None`; required
    # concerns are typed `X` and have no default (forcing the
    # caller to wire them).
    pass
```
