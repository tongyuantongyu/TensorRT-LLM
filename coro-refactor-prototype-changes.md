# Prototype-vs-design changes

The prototype at commit `a3d063bb7b` validates that the design in
[`coro-refactor-design-doc.md`](coro-refactor-design-doc.md) is
implementable, and it powers the end-to-end smoke + perf tests
referenced in the design doc. It predates several of the design
decisions documented there, though, so its current shape diverges
from the design on a small number of bounded points.

This file is the **porting checklist** that closes the gap, plus a
record of **open design questions** still under debate (see
[Open questions](#open-questions) at the end). Each porting item:

- Names the area of divergence.
- States what the prototype has today.
- States what the design calls for.
- Points at the design-doc section that motivates the change.

Reading the design doc does NOT require reading this file. This
is implementation guidance, not architecture.

---

## 1. Cross-thread boundary services bag — `ctx.io.*`

**Today.** A passive `MessagePort` dataclass on `ctx.port` holds
the cross-thread request queue, the shutdown `threading.Event`,
the `is_shutdown` bool, and the `is_warmup` bool. A separate
`ClientChannel` (the loop-side write surface for responses + the
per-request streaming-sink registry) lives in `ctx.svc.client`
next to the loop-only services.

**Design.** Four typed boundary services on `ctx.io.*`, each
owning one piece of cross-thread state behind a thread-safe
method API:

- `ctx.io.inbox: RequestInbox` — wraps the request queue.
- `ctx.io.client: ClientChannel` — already exists in this shape;
  moves from `ctx.svc.*` to `ctx.io.*`.
- `ctx.io.shutdown: ShutdownSignal` — wraps the event + inbound
  flag; methods `mark_inbound()`, `is_marked()`, `signal_done()`,
  `wait_done()`.
- `ctx.io.warmup: WarmupFlag` — wraps the bool with the one-shot
  `False → True → False` invariant baked into the API
  (`start()` / `stop()` / `is_warmup()`).

`MessagePort` and `ctx.port.*` go away.

**Design-doc refs.** §2.4 (Threading model: a clear ownership
cut), §4.4 (Services — two buckets).

**Why.** §2.4's rule "every piece of cross-thread state has a
typed object with a thread-safe method API" applies uniformly to
all four pieces. The current passive dataclass plus a sibling
service in `ctx.svc` predates that rule.

---

## 2. Unified `Requests` service for the request lifecycle

**Today.** Four separate objects together handle the request
lifecycle:

- `RequestPool` (`ctx.svc.pool`) — active list + PP inflight set,
  with `add_active` / `remove_active` / `mark_inflight` /
  `unmark_inflight` methods.
- `ClientChannel` (`ctx.svc.client`) — response delivery + per-
  request streaming sinks.
- `TerminationService` (`ctx.svc.termination`) — resource free +
  sink unregister via `terminate(req)`.
- `fail_requests(ctx, reqs, msg)` — a free function in
  `concerns/shared.py` that composes the first three for the
  fail-fast path.

Every logical request operation (admit, complete, fail) requires
the calling concern to compose 2–4 of these in a specific order;
the correctness of the composition (call ordering, no-op
idempotency, which preconditions are OK) is implicit at the call
site.

**Design.** One `Requests` service on `ctx.svc.requests` owning
the whole lifecycle and exposing high-level operations:

```
admit(items) -> AdmissionOutcome   # validate + waiting-queue + pop to active
cancel(ids)                        # scan waiting + active, mark cancelled
complete(req, response)            # remove + enqueue + free + unregister
fail(reqs, msg)                    # like complete but with error response
pause(reqs)                        # V1: free KV, keep in pool, mark paused
active() -> Iterable[LlmRequest]
is_drained() -> bool
waiting_empty() -> bool
track_inflight(scheduled)          # CM, see item 3
```

`pool` and `termination` disappear as separate public services.
`fail_requests` and `concerns/shared.py` disappear. `client`
stays as a boundary service (item 1) for response delivery and
sink registration; `Requests` composes `client.enqueue` and
`client.unregister_wait_queue` internally.

**Design-doc refs.** §4.4 (Request lifecycle — `ctx.svc.requests`).

**Why.** Distributing a coherent lifecycle across four objects
with implicit composition rules is the kind of hand-written
state machine §5.2 warns against. A single owner moves the
lifecycle invariants inside one class where they can be enforced
internally, and concerns shrink dramatically at the call site.

---

## 3. PP inflight tracking as a context manager

**Today.** `RequestPool.mark_inflight(scheduled)` and
`RequestPool.unmark_inflight(scheduled)` are separate public
methods, called by `PpScheduleConcern` at `SCHEDULE_0` and
`ResponseConcern` at `RESPOND_8`. The pair must match per batch;
nothing in the type system enforces the pairing.

**Design.** A context manager `Requests.track_inflight(scheduled)`
held in `PpScheduleConcern` across its `SCHEDULE_0` → `RESPOND_8`
span:

```python
async def handle_batch(self, ctx):
    r0, w0 = await enter_phase(BatchPhase.SCHEDULE_0)
    # ... fetch + classify + admit + schedule + write to w0 ...
    if scheduled.batch_size > 0:
        with ctx.svc.requests.track_inflight(scheduled):
            await enter_phase(BatchPhase.RESPOND_8)
            # set holds these ids for the full SCHEDULE_0–RESPOND_8 span
```

`Requests.inflight_ids` (a read-only attribute exposing the set
for the scheduler binding) stays public; the mark/unmark
methods do not.

**Design-doc refs.** §5.2 (Sequence-coupled methods are coroutines
in disguise), §4.4 (Request lifecycle).

**Why.** Mark/unmark is the canonical 2-state automaton across
batch phases — exactly the sequence-coupled-methods smell §5.2
calls out. The CM hides the pair behind one syntactic call;
callers can't forget the unmark, and no public `mark` exists to
misuse on its own.

---

## 4. Remove `BatchStorage.canceled_req_ids`

**Today.** `ScheduleConcern` extracts cancel-marker IDs from the
cross-thread queue at `SCHEDULE_0` and writes them to
`BatchStorage.canceled_req_ids`. `ResponseConcern` reads them at
`RESPOND_8`, scans the pool for matching requests, calls
`req.finish_by_reason(CANCELLED)`, and the response-build loop
in the same `handle_batch` picks them up as finished.

**Design.** `ScheduleConcern` calls
`ctx.svc.requests.cancel(ids)` directly at `SCHEDULE_0`; the
`Requests` service marks matching requests in the waiting queue
or the active pool. `ResponseConcern` sees cancelled requests
via `req.is_finished` like any other completion. The
`canceled_req_ids` field disappears from `BatchStorage`.

**Design-doc refs.** §4.4 (Request lifecycle — knock-on
cleanups).

**Why.** Cancel IDs don't conceptually belong to a batch (a
cancel may target a request not in the current iter's pool, or
still in the waiting queue). Routing them through the request
lifecycle service puts them in the right home and consolidates
all "search across waiting + active" logic into one place.

---

## 5. Drop `sampler=` from `ScheduleConcern.__init__`

**Today.** `ScheduleConcern.__init__` takes `sampler=` purely so
it can call `sampler.validate_request(req)` during the admission
loop. The sampler is also held by `SampleConcern` for its actual
sampling work; two concerns reference the same object for
unrelated reasons.

**Design.** Validation moves into `Requests` (which holds an
injected `RequestValidator`). `ScheduleConcern.__init__` no
longer takes a sampler reference; `SampleConcern` is the sole
owner of the sampler. Because validation now lives inside `admit`,
a rejected request is routed straight to an error response on the
same path a mid-flight failure uses — the caller never distinguishes
"rejected at admission" from "failed in flight."

**Design-doc refs.** §4.4 (Request lifecycle — knock-on
cleanups).

**Why.** Single-owner-per-dependency cleanliness; validation is
a request-lifecycle concern, not a scheduling concern.

---

## Open questions

Unlike the porting items above (decided changes awaiting
implementation), these are design points still under debate. They
are recorded here so the reasoning isn't lost, not because the
design doc commits to an answer yet.

### A. Should downstream concerns iterate `scheduled_batch` instead of the active pool?

**The tension.** A lot of concerns iterate the active-request pool
(`ctx.svc.requests.active()`, `ctx.svc.pool` in the prototype).
That pool is the scheduler's cross-iteration working set — the
scheduler writes it, everyone else reads it. Wiring it that way
makes the pool do double duty: a scheduler-global registry AND the
per-batch "requests to handle" list. There are then *two* lists of
"requests to handle" — the pool and `BatchStorage.scheduled_batch` —
and it is not obvious why a downstream concern reads one versus the
other.

**They are not the same thing.**

- The pool (`active_requests`) is the scheduler's cross-iteration
  working set + lifecycle registry: everything admitted and still in
  flight, *across* iterations.
- `scheduled_batch` is the per-batch selection the scheduler
  produced at `SCHEDULE_0`, carried in `BatchStorage` and handed to
  every later phase through the typed read view.
- `scheduled_batch ⊆ pool`; `scheduled_batch` is exactly "the
  requests THIS batch ran."

**Evidence — the current read pattern is a filtered superset.**
`ResponseConcern` iterates the pool for its build + finish-detection
loop:

```python
# ResponseConcern today, RESPOND_8 (concerns/response.py):
for req in ctx.svc.pool:
    do_emit = (req.py_decoding_iter == 1
               or req.is_finished
               or req.py_decoding_iter % self._stream_interval == 0)
    if do_emit:
        ...
    if req.is_finished:
        finished.append(req)
```

Every request that can emit a response or finish this iter is one
that advanced this iter — i.e. it is in `scheduled_batch`. A pool
entry that wasn't scheduled produced no token, so the per-request
`do_emit` / `is_finished` flags skip it anyway. The pool iteration
"works" only because per-request flags filter it back down to
effectively `scheduled_batch` — it reads the wrong set and survives
on the filters.

**Who could switch to `scheduled_batch`:** forward, sample,
response-build, finish-detection — everything that "operates on the
requests this batch ran." The pattern is already established:
`DisaggConcern` reads `r7.scheduled_batch.all_requests()`, and the
PP inflight release reads `r8.scheduled_batch`.

**Who legitimately stays on the pool** (genuinely cross-iteration,
not batch-local):

- `ScheduleConcern` reads the pool to feed the scheduler — it is the
  *producer* of `scheduled_batch`; the pool is its input.
- Cancellation marking — a cancel ID can target a queued / paused
  request that is in no `scheduled_batch` this iter, so it must scan
  the pool (could live on the schedule/admission side, which ingests
  the cancel markers anyway).
- Disagg waiting-state probes (KV-transfer timeout,
  transmission-in-progress) — those requests sit in the pool
  *between* iterations, never in the current batch.
- Pool *mutations*: `remove_active` (evict finished), `fail`, the
  catastrophic fail-all, `is_drained`.

So `finished` is computed by iterating `scheduled_batch`; the pool
is only *written* (for eviction), never read for "the batch's
requests."

**The decisive argument — overlap / PP correctness.** In the plain
loop this is only cleanliness. In overlap / PP it is *correctness*:
by the time batch N's `RESPOND_8` runs, the scheduler has already
run batch N+1's `SCHEDULE_0`, which admitted N+1's requests into the
pool. A response handler that iterates the *pool* is iterating a set
that already contains batch N+1's members; it survives today only
because a brand-new request's `py_decoding_iter` doesn't trip
`do_emit`. `scheduled_batch` is the batch's own immutable snapshot —
the only set that is correct regardless of how far the pipeline has
advanced.

**The tradeoff against switching.** Pool iteration auto-excludes
fail-fast'd requests (`fail` removes them from the pool before later
concerns run). `scheduled_batch` retains them in the snapshot, so a
concern iterating it needs a live-filter (`iter_live(reqs)` / skip
`GENERATION_COMPLETE`). This is the discipline the design was
already heading toward — per-request fail-fast (§2.5) marks a failed
request as failed, so a concern iterating `scheduled_batch` skips it
with that same live-filter. The cost is one helper.

**Connection to the strict service rule (§5.1).** "A service is not
a data channel" already points at the answer at the principle level:
using `Requests.active()` to feed downstream concerns "the batch's
requests" is using a service as a data channel; `scheduled_batch` (a
phase field) is the right medium. The scheduler reading `active()`
to *produce* `scheduled_batch` is fine — that's the producer reading
its own working set, not concern→concern data flow.

**Current lean.** Make `scheduled_batch` the single source of truth
for "the batch's requests" (downstream concerns read
`r.scheduled_batch` through `iter_live`), and demote the pool to
scheduler-input + cross-iteration lifecycle only. Possibly move
cancellation into `ScheduleConcern` so the schedule/admission side
becomes the *only* pool iterator and disagg the only reader of its
filtered waiting-state views.

**Why still open.** It interacts with item 2 (the unified `Requests`
service) and changes the failure-handling discipline from "pool
auto-exclusion" to "explicit `iter_live`." Worth settling the
`iter_live` ergonomics and the cancellation home before committing
the concern rewrites.

## Not blocking (scope notes)

These are already documented in the design doc as known
limitations of the prototype's scope, not as porting tasks:

- **Phases 2–4 of the planned-concerns table** — `PerfMetricConcern`,
  `IterStatsConcern`, `SpecDecodeConcern`, `DisaggConcern`,
  `GuidedDecoderConcern`, `KvConnectorConcern`, `DwdpConcern`. The
  prototype's `PyExecutorCoro.__init__` raises `NotImplementedError`
  for any configuration that requires them. They ship as each
  concern lands. See §8.2.
- **Legacy `py_executor.py`** is untouched; selecting between the
  legacy and the new executor is per-construction. Full CI parity
  work happens before the legacy is removed. See §4.1 / §8.1.
- **`_broadcast_sample_state_loop` thread + flush-when-idle hack**
  already removed in the prototype; replaced by
  `RingBroadcastSampleConcern` + `RecvOffload`. See §4.5 (PP) / §8.4.
- **`has_previous_draft_tokens` stale-flag bug** persists in the
  prototype (only reset on the overlap path). Moves into
  `SpecDecodeConcern` coroutine locals when that concern is
  implemented. See §8.4.
