# PyExecutor Coroutine Refactor — Design Doc

Scope: replace the legacy `PyExecutor` and its three `_executor_loop*`
methods with a coroutine-based loop layer (`PyExecutorCoro`).

This design has two parts that are at very different maturity
levels, and the rest of the doc should be read with that split in
mind:

- **The coroutine runtime** a custom runtime fully tailored to express the executor's logic, but nothing more — keeps the overhead minimal. This part should already in a fairly good shape: its design decisions are backed by observations from actually implementing the prototype `PyExecutorCoro` on top of it.

- **The executor layer** was built on top of the runtime to **verify that this design actually works end-to-end** — and it does, for all three executor loops. But the prototype is a **reference implementation, not production code**: it predates
  several of the decisions documented here, it cuts corners the
  design does not, and the plan is to **reimplement most of the executor layer** once this doc is revised, accepted, and more
  contributors are involved.

---

## 1. Introduction

### 1.1 The problem

The legacy `PyExecutor` runs the forward loop in three variants —
`_executor_loop`, `_executor_loop_overlap`, `_executor_loop_pp` —
plus a helper thread, `_broadcast_sample_state_loop`. The three do
the **same work**: schedule requests, prepare the KV cache, run the
model forward, sample tokens, build responses, transfer KV cache,
and so on. In this doc, we call rach of these domain area as a **concern**. What differs is
the **order** — plain, overlap, and pipeline-parallel execution
interleave the concerns differently to keep the GPU busy. Each loop
is a few hundred lines, and every helper they call
(`_handle_responses`, `_sample_async`, `_schedule`, `_forward_step`,
…) is a method on `PyExecutor` itself: ~4300 lines and ~120 methods
on one class.

All kinds of state, as long as being cross-iteration, all lives on `self` as scattered fields
(`previous_batch`, `has_previous_draft_tokens`, `can_forward`, several
`send_*_handles[microbatch_id]` slot lists, …). Two consequences:

- **Reading any one loop means re-assembling every concern in your head.**
  A concern's logic is split across the loop body and the
  helpers it calls — resource management preps the KV cache before
  the forward, then updates and frees it after responses. Overlap and
  PP reorder the work, so a concern's *later* step can sit *earlier*
  in the code than its earlier one: the sampler's apply-tokens runs
  near the top of the overlap loop, ahead of the launch it follows.
  And there are three orderings to hold at once, not one.
- **Adding a concern means editing all three loops and `PyExecutor`.**
  Disagg, KV connector, guided decoder, spec decode, DWDP — each
  added a `self.<flag>`, a new `_handle_<x>` method, and a call
  spliced into every loop at the right point. The right point differs
  per loop, so a contributor must understand all three orderings
  to add some feature. Everyone have to pay this burden.

The refactor's headline goal is **one linear story per concern**: a
concern's logic, today scattered and reordered across three loops,
becomes a single function that reads top-to-bottom in the order
things happen for it — for resource management, prepare then update
then free — with the state it carries between those steps held in
ordinary local variables. Such a function has to run partway, hand
control back so other concerns can run, then resume where it left
off. That is a **coroutine**, and adopting it is the core move of
this refactor.

Two key decisions follow.

### 1.2 Why coroutines

A concern's work is sequential — resource management prepares, then
updates, then frees; the sampler launches, then applies the result.
But the concerns are **multiplexed**: at each point in the loop
several of them have a step to run, in a defined order, before any
one of them is finished. A coroutine runs a sequential body yet can
suspend at defined points and resume later — exactly the shape this
multiplexing needs.

Two benefits over the legacy "helpers called from a loop" shape:

1. **Per-concern logic and state, kept together and local.** A
   concern is one function holding its own logic; the state it
   threads from one step to the next — a CUDA event, a partial
   result, a flag — is a local variable in that function, alive only
   while the concern is mid-flight. Nothing lands on a shared `self`
   for every loop to set, reset, and avoid colliding on.
2. **Top-to-bottom lifecycle.** That function reads in the order
   things happen for the concern, with `await` marking each point it
   yields to others. Following a concern is reading one body, not
   re-assembling it from three loops and their helpers.

### 1.3 Why not asyncio

`asyncio` is Python's standard coroutine runtime, but it solves a
different problem. It exists to multiplex coroutines that wait on
**real I/O**: each `await` is paired with an event loop that watches
file descriptors / sockets / timers and schedules whoever's ready
next. Its scheduling policy is opaque — the order coroutines resume
depends on event-loop internals, the OS, and which futures happen to
complete first.

The forward loop has the opposite shape:

- No real I/O is waited on at the coroutine boundary. Every kernel
  launch is non-blocking; every MPI call is either non-blocking
  (`isend`) or offloaded to a worker thread that returns a `Future`.
  The driver doesn't need to *wait* on anything to decide what to
  resume next — it just decides.
- The schedule must be **deterministic and explicit**. Whether two
  batches' phases interleave a particular way determines correctness
  (see §4.5's overlap-loop ordering invariants) and performance.
  Implicit scheduling is exactly what we don't want.
- asyncio's native surface gives you `await coroutine` (run to
  completion) or `asyncio.create_task(coroutine)` (run independently
  as a background task) — not "advance batch A to the next phase,
  hand control to batch B for a bit, then come back and advance A
  further". That middle ground has to be layered on top of asyncio's
  primitives if you want it.

It *is* possible to build the middle ground on asyncio: an
`asyncio.Event` per phase, an `asyncio.create_task` per batch /
concern handle, contextvars for the active state. A side-by-side
microbenchmark with that adapter confirms that the same
`scheduler_iter_overlap`, `batch_body`, and concern coroutines run
**unmodified** against either the asyncio adapter or the custom
driver. So coroutine bodies are portable across runtimes by
construction. But I still consider asyncio not the right choice, for 2 reasons:

1. **Cost.** Benchmarked on the exact scheduler logic with empty concern body, the asyncio path is more than ~2× slower in steady state:

| runtime | phase-gated | empty (concerns no-op) |
|---|---|---|
| custom driver | ~41 µs/iter | ~33 µs/iter |
| asyncio | ~106 µs/iter (≈2.6× slower) | ~63 µs/iter (≈1.9× slower) |

This is ~65 µs of per-iter scheduling overhead to pay if we want asyncio, and we pay it for no real benefit:

2. **Sharing the event loop is the wrong default for a tight loop.**
  asyncio is designed to multiplex unrelated tasks — file
  descriptors, sockets, timers, application-level coroutines —
  through one event loop. Compatibility with that ecosystem sounds
  attractive, but the forward loop is a tight execution path that
  doesn't *want* unrelated coroutines landing in its scheduler
  queue and injecting latency between its yields. The custom
  driver's only customer is the scheduler iter; nothing else can
  delay it.

So the design builds a **bare coroutine driver** that exposes
the "advance to next suspension point" primitive and nothing else.
The schedule is fully spelled out by the top-level coroutine; the
runtime just pumps. No event loop, no callback registration, no
future scheduling. Coroutine bodies stay portable to asyncio if a
future use case ever needs it (one adapter file away), but
production runs against the bare driver.

---

## 2. Concept design

This section lays out the conceptual shape the refactor is built
around — the abstractions, invariants, and rules — without committing
to specific types, primitives, or APIs. The next three sections (§3
the runtime, §4 the executor, §5 the rules for writing concerns)
cover how each concept is realized in code.

### 2.1 Three layers: scheduler, batch, concern

The **concern** layer is the obvious one. A concern is one of the
cross-cutting responsibilities the forward loop interleaves:
scheduling, request fetch, KV-cache prep, model forward, sampling,
response build, KV transfer, and so on. Each concern's own work is
sequential per batch (resource management prepares the KV cache,
then later updates and frees it), even though concerns are
multiplexed across the loop. The legacy executor already gropes
toward this idea — its `_handle_*` / `_*_step` helpers are concern
fragments lifted out of the loop bodies — but without coroutines a
concern stays splintered across several helpers, and fragments of
*different* concerns get fused into one helper merely because they
run next to each other in the loop. Making each concern's body a
single coroutine that reads top-to-bottom across the phases it cares
about is the headline goal.

The **batch** layer above it is less obvious — the legacy executor
never separated it out. A *batch* is one group of requests driven
through the work; an *iteration* is one turn of the outer loop. In
the plain loop the two coincide: each iteration creates a batch, runs
every concern on it, and ends. That is exactly how the legacy
`_executor_loop` treats them — as one thing — and for the plain loop
a two-layer scheduler ↔ concerns design would be enough.

The overlap and PP loops break the coincidence. Both keep **multiple
batches in flight at once**, each at a different stage: in the overlap
loop, while batch N is in forward + sample, batch N−1 is in apply +
respond; in PP, `pp_size − 1` batches are parked at various stages of
the cross-rank ring. One iteration now drives *several* batches
through *different* phases. The legacy loops never untangled
iteration from batch when overlap and PP were added — that should
have happened then, and the tangle is much of why those two loops are
the hardest to follow.

Folding batch-lifecycle work into the iteration body — "this iter is
batch N's forward AND batch N−1's respond AND batch N−2's KV
transfer" — costs the top-to-bottom reading: the single thread "this
batch goes schedule → forward → sample → respond" is lost in the
interleave between concerns acting on *different* batches. The key
realization of this design is that **separating batch from
iteration** is what restores linear, top-to-bottom logic. With batch
as its own layer, a batch's full lifecycle is one body that reads
top-to-bottom, just like a concern's; the scheduler iter above it
carries only the inter-batch interleave and says nothing about a
single batch's lifecycle.

So:

- **Concern** = one cross-cutting responsibility. Top-to-bottom body
  across the phases it cares about. One instance per concern type;
  invoked once per batch.
- **Batch** = one group of requests. Top-to-bottom body across every
  phase. Concerns participate at each phase.
- **Scheduler** = how multiple batches' lifecycles interleave across
  outer-loop iterations. Different loop variants (plain, overlap, PP)
  are different schedulers operating on the same batch lifecycle.
  Only the scheduler thinks in iterations; concerns and batch bodies
  never see an iter counter.

Each layer is its own top-to-bottom body, and each layer talks
only to its immediate neighbor: the scheduler addresses batches, a
batch addresses its concerns, a concern works on its own tasks.

### 2.2 Phase as a named suspension point

Explicit scheduling (§1.3) needs a vocabulary for *where* a coroutine
is suspended. asyncio gets away without naming because the schedule is
implicit in the data dependency graph: a coroutine awaits a future,
the event loop resumes it whenever the future resolves; "where it's
suspended" is just "wherever it last awaited", and the runtime
doesn't need to refer to that point by name.

An explicit scheduler is making decisions like "advance batch N to
forward, but no further" or "drive batch N−1 through respond now".
Without a name for "to forward" or "through respond", these
decisions can't be expressed.

A **phase** is exactly that name: an enumerated point in a
coroutine's lifecycle where it can be suspended, named so the
scheduler can refer to it. Both batch coroutines and concern
coroutines yield at phase boundaries; the scheduler asks for
advancement to a specific phase.

The phases form a **single ordered timeline** shared across all
three layers. Every layer ultimately suspends at the same boundaries
— a batch suspends between schedule and forward; a concern in that
batch yields at the same boundary; the scheduler asking for
"advance to forward" picks the same name. Collapsing onto one
ordering removes the duplication of having per-layer enums; "code at
phase A happens before code at phase B" becomes a single integer
comparison across the entire design.

### 2.3 Data flow

A phase (§2.2) says *when* code runs, and by the same token it fixes
*what data is already available* and *what data this step must
produce*: data flow becomes phase-defined. Cataloguing that flow
turns up four categories, each with its own home. The first three
stay within the loop thread; the fourth crosses out of it.

#### Intra-concern (trivial)

Data a single concern threads across its own phases — a CUDA event
recorded at one phase and consumed at a later phase, a matcher
result, a per-batch flag — is just a coroutine local. Coroutines
preserve locals across `await`, so this category needs no machinery
beyond declaring a Python variable in the concern body.

#### Per-batch cross-concern: data flow follows the schedule

This is where explicit and implicit scheduling diverge most sharply.

In implicit scheduling (asyncio), the data flow *defines* the
schedule: a coroutine awaits a future, the runtime resumes it when
the future resolves. The order coroutines run is whatever order
their data dependencies happen to satisfy.

In explicit scheduling, the relationship is reversed: the schedule
order **defines** what data is visible. The scheduler decides
"concern A runs at phase P", and from that the runtime can derive
"data produced at phases earlier than P is visible to A; data A
produces is visible to whoever runs at phases later than P". A
field's lifecycle is determined by the phase that produces it; the
scheduler's order of resume calls is what produces a coherent
dataflow.

Two consequences fall out of this:

- **The current phase determines what data is available now and
  what data the current code is expected to produce.** A concern at
  phase P sees all fields owned by earlier phases (they were
  produced before P) and writes only fields owned by P (later
  concerns will read them at later phases).
- **Time never goes backward.** A coroutine cannot
  yield a phase **earlier** than its previous suspension — that
  would mean reading data that hasn't been produced yet (the
  would-be producers run at later phases). And, assuming forward
  progress, it shouldn't yield the **same** phase twice either —
  each phase is reached exactly once per batch. The backward-phase
  rule is **hard**: data availability forbids it always. The
  same-phase rule is **soft**: it's a forward-progress assumption
  the caller can choose to relax. §3.1 covers when and how.

#### Cross-batch: scheduler-mediated

When one batch's terminal output is the input to the next batch
(the canonical example: in the overlap loop, batch N−1's sampled
tokens become batch N's forward input), the handoff is the
**scheduler's** job. The scheduler reads batch N−1's terminal phase
data, computes the bridge value, and stuffs it into batch N's
first-phase data slot.

Batches do not reach across to each other directly. If they did,
"batch" would stop being a useful unit of encapsulation — every
batch would carry knowledge of its peers, and the inter-batch
ordering invariants of overlap and PP would be split between the
batch body and the scheduler. Routing all cross-batch data through
the scheduler keeps each batch's body self-contained: it sees only
its own storage and the concerns it drives.

#### Cross-thread: through boundary objects

The previous three categories all live within the loop thread. There
is also a fourth category that crosses out of it: the public API
runs on the main thread (§2.4 covers this in detail), so any data
that flows between user-facing API calls and the loop has to cross
the thread boundary. Concretely:

- Requests submitted via the public API (main thread) need to reach
  the scheduler at `SCHEDULE_0` (loop thread).
- Cancel markers submitted via the API need to reach the scheduler
  in the same channel as requests.
- Responses produced by the response concern (loop thread) need to
  reach `await_responses` blockers (main thread).
- Lifecycle signals — warmup mode set by the API, shutdown
  initiated either by the API or detected by the loop — flow in
  whichever direction the signal needs.

Each piece of cross-thread state lives on a typed **boundary
object** — a class that owns the state and exposes thread-safe
methods. The synchronization (a `threading.Event` for one-shot
signals, a condition variable for buffered queues, a plain bool for
one-shot flags whose write happens-before any read) is encapsulated
inside the object, not at the call site. Concerns and the public
API never touch raw cross-thread data containers; they go through
the boundary object's methods.

The threading model in §2.4 sharpens this further: **the boundary
objects are the entire cross-thread surface**. Any new piece of
cross-thread state lands on one of them, or it's a structural error.

### 2.4 Threading model: a clear ownership cut

The fourth data-flow category (§2.3) — the one piece that crosses out
of the loop thread — is what makes the threading boundary something
to state explicitly rather than leave implicit.

Both the legacy executor and the new design run on two threads — a
main thread for the public API and a dedicated loop thread for the
forward loop. The threading shape itself is inherited.

What's new is **a clear cut about which side owns what**.

In the legacy `PyExecutor`, both threads accessed the same instance
fields freely; whichever method happened to be called from a given
thread did its work against shared state. Cancellation flags,
request queues, response containers, internal bookkeeping — all sat
together on `self`, with implicit assumptions about which thread
might touch which field at which time. Locks and condition variables
existed where data races would have been obvious, but the
discipline was "don't cause races", not a structural rule.

The new design draws the cut **structurally**: the loop thread is
the only thread that runs coroutines and mutates loop-side state;
the main thread is the only thread that runs API methods and holds
API-side state. Cross-thread communication goes through a small set
of named **boundary objects** that hold *only* the data both threads
touch and expose thread-safe APIs. Anything else lives on one side
or the other, never both.

The benefit is **understandability**: a reader of any line knows
which thread it runs on by where it lives, and a reader of the
boundary objects knows exactly what crosses. Adding new cross-thread
state is impossible without either landing on a boundary object
(and being noticed) or violating the rule (and standing out).

### 2.5 Failure handling: catastrophic vs per-request

Two distinct failure modes appear in the loop, and they need
different handling.

**Catastrophic** failure means the loop can't continue: a kernel
raised an exception, a service is unreachable, a runtime invariant
tripped. The corrective action is to fail every in-flight request
(so each client side unblocks with an error response), set the
shutdown latch, and tear down the loop.

**Per-request fail-fast** is different in scope: a particular
subset of requests is bad — validation failed on those specific
requests, an inference invariant tripped on a few of them — but
the rest of the batch and the rest of the loop should keep going.
The corrective action is to fail those requests individually
(mark as failed, send error responses, free their per-request
resources) and continue the loop body unchanged.

The two modes need different propagation paths:

- **Catastrophic** propagates as an exception — so Python's own
  unwinding rules do almost all the work for free. The concern that
  detected it just `raise`s; the exception travels up through the
  batch coroutine and the scheduler to a single top-level
  `try / except` in the scheduler, which runs the global shutdown
  sequence. That one handler covers a failure raised *anywhere* —
  any concern, any phase — with no per-concern error plumbing to
  write. Cleanup that in-flight concern coroutines need is delivered
  the way Python coroutines always clean up: their `try / finally`
  and `with` blocks unwind when the runtime closes them on the way
  down.
- **Per-request fail-fast** is a normal control-flow path inside
  the concern that detected it. The concern calls a shared utility
  that does the per-request bookkeeping (mark request, build error
  response, free resources, terminate streaming sink) and continues
  its own work for the remaining requests.

---

## 3. The runtime

The runtime is the generic, feature-complete foundation. It knows
only `IntEnum` phases and an unconstrained storage object — nothing
about concerns, `BatchStorage`, or the executor's loop variants.
This section documents the primitives a coroutine calls, the Driver
that pumps them, and the production aids baked into it. The concrete
data model and the concerns that use these primitives are §4.

### 3.1 The coroutine surface

The runtime provides six primitives. They split into two roles: a
coroutine **announces where it wants to suspend** (`enter_phase`,
`batch_phase`, `again`), and the layer above **drives a child
forward** to a phase (`resume`, `step`). `disable_hang_detect` is a
side utility for the watchdog.

| Layer | Primitive | Purpose |
|---|---|---|
| Concern | `r, w = await enter_phase(P)` | Suspend until the parent batch is at phase `P`; resume with `(read_view_at_P, write_view_at_P)`. |
| Batch | `async with batch_phase(P): ...` | Enter phase `P` — drives concerns called inside this block to `P` via `resume`. |
| Batch | `await resume(child)` / `await try_resume(child)` | Drive a `Concern` handle through the current `batch_phase`. |
| Scheduler | `read, write = await step(handle, through=P)` / `await try_step(handle, through=P)` | Drive a `Batch` handle through phase `P`; return its read/write views. |
| Any | `await again()` | "I'm at this phase, ask me again later" (POSIX `EAGAIN`). |
| Any | `async with disable_hang_detect(): ...` | Pause the runtime watchdog over an intentionally-long body. |

#### `enter_phase` — a concern parks until its phase

A concern calls `await enter_phase(P)` to say "wake me when the batch
reaches phase `P`." It suspends there; when resumed it gets the typed
`(read, write)` views for `P` (§4.3), does its work for that phase,
then loops to the next `enter_phase`, which parks it again.

```python
r, w = await enter_phase(BatchPhase.SAMPLE_3)   # park until SAMPLE
w.sample_state = self._sampler.launch(r.batch_outputs)

r, _ = await enter_phase(BatchPhase.APPLY_7)    # park until APPLY
self._sampler.apply(r.sample_state)
```

The statements between two `enter_phase` calls are the concern's work
at the first of the two phases. A concern that participates at one
phase calls it once; one that participates at several calls it once
per phase.

Each `enter_phase` must name a phase strictly greater than the
previous one — a concern only ever moves forward through the
timeline. Re-entering the same phase or naming an earlier one is a
hard error raised at that call.

#### `resume` and `step` — the layer above says "it's time"

`enter_phase` is the request to be woken; `resume` and `step` are the
wake-up call from the layer above. A batch wakes its concerns with
`resume`; the scheduler wakes a batch with `step`. Each drives the
child up through the named phase and returns once the child has
finished its work there.

```python
# a batch drives two of its concerns through SAMPLE_3
async with batch_phase(BatchPhase.SAMPLE_3):
    await resume(sample)
    await resume(guided_decoder)

# the scheduler drives a whole batch through SAMPLE_3
read, write = await step(batch_handle, through=BatchPhase.SAMPLE_3)
```

"Through `P`" is strict: the call returns once the child's *next*
suspension is past `P` — the child has run its `P` work and parked
itself at some later phase. Read `enter_phase(P)` as "I'm now *at*
`P`, doing `P` work"; being driven *through* `P` means the child has
left its `P` block. A child that doesn't participate at `P` at all —
its next suspension is already past `P` — is simply a no-op: the call
returns without running anything.

Two more inputs short-circuit to a no-op, for the caller's
convenience:

- **`None`.** `resume(None)` and `step(None, …)` return immediately,
  so a batch can keep an optional concern in its `resume` list even
  on a run where that concern isn't wired — the line stays, it just
  does nothing.
- **An already-returned child.** A coroutine that has returned is
  treated as parked at a virtual phase later than every real one, so
  driving it to any `P` does nothing. A concern that finished at its
  last `enter_phase` can stay in every later phase's `resume` list
  without a guard.

> Why `resume` and `step` are separate primitives?

Both drive a child until it has finished phase `P`; what differs is
the return, and that is why they aren't one call:

- `resume(child)` drives a concern through the current phase's work
  and returns `None`. Whatever the concern published is read back by
  the batch through its *next* `batch_phase(...)` views, not from
  `resume`.
- `step(handle, through=P)` drives a batch and returns
  `(read_view, write_view)` at `P`, so the scheduler can read what
  the batch produced and inject anything the next phase needs (the
  batch-to-batch handoff of §4.5 uses this).

Keeping them separate also lets the type checker narrow `step`'s
return to the right `(_ReadAtP, _WriteAtP)` per call site via the
overload chain (§4.3); a single merged primitive would lose that.

#### `batch_phase` — declare the phase, then enter it

`batch_phase(P)` is the batch layer's `enter_phase`: it parks the
batch until `P` (so the batch body reads top-to-bottom across phases
just like a concern) and, in addition, declares `P` as the **active
phase** for every `resume` inside its block. That is why
`resume(child)` takes no phase argument — it reads the phase from the
enclosing `batch_phase`. The CM is batch-only; concerns use the bare
`enter_phase`. And as with `enter_phase`, the blocks only move
forward: each `batch_phase` must name a phase greater than the
previous block's; repeating or going back is a hard error.

```python
async with batch_phase(BatchPhase.FORWARD_2):
    await resume(forward)
    await resume(spec_decode)
    await resume(dwdp)

async with batch_phase(BatchPhase.SAMPLE_3):
    await resume(sample)
    await resume(guided_decoder)
```

Naming the phase on the block, rather than on each `resume`, buys two
things:

- **The phase is spelled once.** The alternative —
  `await resume(child, at=BatchPhase.X)` per call — repeats the phase
  name once per participant, and a single typo (`SAMPLE_4` for
  `SAMPLE_3` on one of N lines) becomes a silent ordering bug.
- **Indentation groups participants by phase.** Every `resume` at a
  phase shares one `async with` parent and one indent level; "what
  runs at `FORWARD_2`" is exactly the body of the `FORWARD_2` block,
  and reordering within a phase is a vertical edit, never a
  search-and-replace.

#### `again()` and the strict / tolerant pair

The backward-phase rule from §2.3 is hard — data availability forbids
yielding an earlier phase. The same-phase rule is the soft one: there
are legitimate cases where a coroutine is *at* the right phase but
genuinely cannot finish its work yet (the canonical case: a PP rank
polling for an MPI message that hasn't arrived). It would rather be
asked again later than block.

Whether a same-phase retry is an error or an expected occurrence is
the **caller's** choice, not the callee's. The coroutine asks with
`await again()`; the caller decides what that means by which driver
variant it used:

```python
await again()        # "I'm at this phase, ask me again later"

await step(...)      # strict: raises RuntimeError on retry
await try_step(...)  # tolerant: returns None on retry

await resume(...)    # strict: raises RuntimeError on retry
await try_resume(...) # tolerant: returns False on retry
```

- **Strict** (`step` / `resume`): "drive the child to its next phase,
  and raise if it asks to stay where it is." The default — an
  unexpected retry surfaces immediately rather than silently looping.
- **Tolerant** (`try_step` / `try_resume`): "drive the child to its
  next phase, OR report back that it asked to stay." The caller
  decides what to do — try again later, cascade the retry up to its
  own caller, or skip this round.

From the coroutine's point of view, `await again()` is just a
suspension point that resumes from the same world it left: same
active phase, same storage views, same read/write barrier (at `P`,
reads see writes at `< P`). The next time it runs, code continues
right after the `await again()` as if the yield had never moved the
phase forward.

The contract is "**`again()` must be paired with a `try_*` driver
upstream**". The strict variants exist so accidental retries surface
as a clear exception at the immediate boundary instead of looping
silently. Cascading is allowed: a batch may itself `await again()`
after a child concern retries, as long as the scheduler used
`try_step` — the `RingBroadcastSampleConcern` HANDOFF_6 polling loop
relies on this cascade.

So the runtime distinguishes three outcomes per resume: advance to a
strictly later phase (success), retry at the same phase (legal only
under a tolerant caller), regress (always an error).

#### `disable_hang_detect`

```python
async with disable_hang_detect():
    items = self._inbox.dequeue_for_iter(timeout=...)
```

Pauses the Driver-global hang watchdog over the body. Used for
intentional long-blocking sections (idle queue waits, one-time startup
work). Does NOT nest for simplicity.

This is strictly for the calling coroutine to perform its own long,
blocking work. Inside the CM you must not advance a phase
(`enter_phase` / `batch_phase`) or drive another coroutine
(`resume` / `step`): the pause is meant to cover your own blocking
call, never a span of the schedule that hands control to other
coroutines.

### 3.2 The Driver

A concern author rarely touches the internal of the Driver directly; this is
the minimum worth knowing they exist.

**Handles.** Every coroutine the runtime drives is wrapped in a
handle — a `Batch` for a batch coroutine, a `Concern` for a concern
coroutine — and handles are what `step` / `resume` take. A handle
tracks whether its coroutine has finished, so driving one that has
already returned is a no-op; that is why the batch body can `resume`
a concern at every phase even after the concern's last `enter_phase`.

**The Driver.** The `Driver` pumps the top-level coroutine, interpreting
each primitive acoroutine awaits and advancing the appropriate coroutine
in response. It does the dirty work to implement the primitives. 

**Shutdown.** `Driver.close()` closes every live coroutine, which
raises `GeneratorExit` into each — so a concern's `try / finally` and
`with` blocks run on teardown exactly as Python guarantees. A concern
needs no special shutdown hook; ordinary cleanup blocks suffice.

### 3.3 Production aids

#### Hang watchdog

Built into the Driver, not a concern. Pass `hang_timeout` (seconds)
to the Driver constructor; the Driver spawns a daemon thread that
gets a `notify(coro_name, phase)` heartbeat before each `coro.send` /
`coro.throw`. If no message arrives within `hang_timeout`, fires
`on_hang(context)` (default: log error + `print_all_stacks`).

`coro_name` is a STRING (`hang_label`) — the watchdog never holds a
coroutine reference, so a finished coroutine GCs promptly even if
the watchdog thread is still alive.

`disable_hang_detect` (§3.1) pauses globally; the pause flag is
reset on `Driver._close_all` so a reused Driver starts clean.

Putting it in the runtime rather than in a `HangConcern` keeps the
per-concern code free of `checkpoint()`-style pollution and gives a
uniform diagnostic across every loop variant.

#### NVTX

`TLLM_COROUTINE_NVTX=1` opens an NVTX range per handle on the Driver
stack: push when the handle enters, pop when it leaves. Mirrors the
driver stack on the host timeline:

- `Batch` ranges: `B<idx>` (one color per `idx % palette_size`) so
  concurrent batches in overlap / PP get distinct hues.
- `Concern` ranges: `C:<qualname>` (one color per concern type, via
  a registry keyed by qualname).
- `_RootHandle` is suppressed (a scheduler-rooted range covers the
  entire run and adds no per-iter signal).

Off by default — message + color resolution is lazy via
`cached_property`, so when `TLLM_COROUTINE_NVTX` is off there is no
work to do. When on, the first push pays one f-string + one palette
lookup per handle; subsequent pushes are cached.

#### Storage tracking

`TLLM_COROUTINE_TRACK_STORAGE=1` enables the `_TrackedReadView` /
`_TrackedWriteView` proxies (§4.3). Default off so production runs
get zero overhead; the `@overload` chain plus mypy guard production.

#### Traceback hiding

Driver and primitive frames are stripped from exception tracebacks
by default, so a concern's exception reads against its own code
rather than the runtime's plumbing. `TLLM_COROUTINE_SHOW_FRAMES=1`
keeps the framework frames when debugging the runtime itself.

---

## 4. The executor

The executor layer is where the design work lives (it is the
reference prototype slated for reimplementation; see the
introduction). This section covers its fundamental shape: the
three-layer hierarchy realized in code, the concrete phase timeline
and `BatchStorage` data model, what a concern and a service are, and
how the three scheduler variants weave batches. The rules an
implementer follows when writing a new concern are §5.

### 4.1 Architecture at a glance

The loop layer is a strict three-layer hierarchy:

```
┌─────────────────────────────────────────────────────────────────┐
│ PyExecutorCoro (main thread)                                    │
│   public API: enqueue / await / cancel / shutdown               │
│   holds direct refs to the boundary services                    │
└──────────────────────────────┬──────────────────────────────────┘
                               │ ctx.io.* (cross-thread surface)
┌──────────────────────────────▼──────────────────────────────────┐
│ Scheduler         scheduler_iter_{plain,overlap,pp}             │
│ ─────────                                                       │
│   pumps Batch handles in some interleave; owns iter counter,    │
│   in-flight ring, batch-to-batch handoffs                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │ await step(handle, through=Py)
┌──────────────────────────────▼──────────────────────────────────┐
│ Batch             batch_body                                    │
│ ─────────                                                       │
│   one coroutine per batch of requests; drives concerns through  │
│   the batch's lifecycle phases                                  │
└──────────────────────────────┬──────────────────────────────────┘
                               │ await resume(handle)
┌──────────────────────────────▼──────────────────────────────────┐
│ Concern           XxxConcern.handle_batch                       │
│ ─────────                                                       │
│   one coroutine per concern per batch; reads / writes typed     │
│   views over BatchStorage; holds the services it needs,         │
│   injected at construction (receives no ctx)                    │
└─────────────────────────────────────────────────────────────────┘
```

Each layer talks only to its immediate neighbor. Why three layers
rather than fewer is covered in §6.

**The two threads.** `PyExecutorCoro` instance methods run on the
main thread; everything below the scheduler runs on the loop thread.
The cut is enforced by what each side can reach: the main thread
holds direct references to the **boundary services** (see §4.4 for
the inventory) and uses only their thread-safe methods; the loop
thread reaches the same boundary services — orchestrators via
`ctx.io.*`, concerns via the refs injected into them — plus the
loop-only services in `ctx.svc.*`. The main thread never touches
`ctx.svc.*` (it's loop-only), and the loop thread never touches
`PyExecutorCoro` instance attributes. The boundary services are the
*entire* cross-thread surface — anything else lives strictly on one
side.

**`ctx` is the orchestrators' environment, not the concerns'.**
Only the loop's orchestrators — `run_loop`, the scheduler iters,
`batch_body` — receive `ctx`. A **concern receives no `ctx` at
all**; every dependency it needs (services, config values) is
handed to it through `__init__` (§4.4). This is what lets `ctx`
carry the `Concerns` bag (`ctx.crn`) without re-opening peer access:
a concern simply has no `ctx` to reach `ctx.crn` through (§5.1).

**Where state lives.**

| State | Home |
|---|---|
| Cross-thread surface (request inbox, response channel, shutdown signal, warmup flag) | Boundary services (`ctx.io.*` for orchestrators; injected into the concerns that use them) |
| Loop-thread-only services (request lifecycle, recv offload, dist communicator) | `ctx.svc.*` (orchestrators); injected into concerns |
| Immutable configuration | `Configuration` (`ctx.conf`); concerns get the specific values via `__init__` |
| Per-batch fields shared between concerns | `BatchStorage` (per batch; reached only via typed views, never via `ctx`) |
| The bag of concern instances | `Concerns` (`ctx.crn`) — used by the orchestrators; concerns can't reach it (no `ctx`), see §5.1 |
| Cross-iter scheduler bookkeeping (`previous_batch`, in-flight deque, vote isend slot) | Locals on the scheduler iter |
| A concern's private cross-batch latch | Instance attribute on the concern class |
| A concern's per-batch intermediate (CUDA event, matcher result) | Coroutine local |

There is no "mutable state nobody owns" bucket: every piece of
cross-batch mutable state is owned by a service, so a concern reads
and writes it only through that service's methods. (The prototype
still has an empty `PersistentState`; the design drops it.)

Discipline: **`ctx` is reachable only by the orchestrators.** A
concern holds exactly the dependencies it was constructed with —
nothing ambient. This is the lever that keeps cross-concern data
flow honest and understandable (§5.1).

### 4.2 The phase timeline

Every layer ultimately yields at the same phase boundaries: a batch
suspends between schedule and forward; a concern that participates in
schedule and forward yields at the same two boundaries; the scheduler
that drives a batch through `SCHEDULE_0` is asking it to reach the
same boundary the concern just yielded at. So one ordered enum is
enough for the entire timeline, and that's what `BatchPhase` is:

```python
class BatchPhase(IntEnum):
    SCHEDULE_0       = 0
    RESOURCE_PREP_1  = 1
    FORWARD_2        = 2
    SAMPLE_3         = 3
    STATE_UPD_4      = 4
    SYNC_EVT_5       = 5
    HANDOFF_6        = 6
    APPLY_7          = 7
    RESPOND_8        = 8
    FINALIZE_9       = 9
```

The numeric suffix on each name *is* the enum value. A reader at any
call site can see immediately that
`BatchPhase.STATE_UPD_4 < BatchPhase.HANDOFF_6` without consulting the
definition. With one enum, comparing two pieces of code across layers
is a single integer comparison; cross-layer rules ("concern A's
`RESPOND_8` happens before concern B's `FINALIZE_9` because…") are
visible at the call site.

**The phase set is open, not final.** The 10 phases above are
what the current shared `batch_body` (§4.5) and concern set need.
The current single batch body yields at every phase, so every one
is used in code today. New phases get added when a cross-concern
data-flow pair forces a split — the producer's phase must be
strictly less than the consumer's phase (§2.3, §4.3), so two
concerns touching the same logical step but acting as
producer / consumer of the same field need two different phases.
Two of the current phases exist for this reason alone:
`RESOURCE_PREP_1` (so the schedule concern's `disagg_gen_init`
output reaches the resource concern at a strictly-later phase) and
`FINALIZE_9` (so the response concern's `finished_requests`
reaches the iter-stats concern). Adding a phase is cheap (one
enum entry, one row of generated views) — the door stays open
rather than forcing every cross-concern producer-consumer pair
into the nearest existing phase.

**Multiple batch bodies are not ruled out** if a future variant
needs one. The runtime doesn't require a coroutine to yield at
every phase — yielding only at the phases its concerns participate
in is enough. The current design uses a single shared body
because feature configurations are driven by which concerns are
wired up rather than by alternative batch shapes; that's the
simplest path until a real reason to fork appears.

### 4.3 Data flow: typed views over `BatchStorage`

Per-batch fields live on a single `BatchStorage`
dataclass — one instance per `Batch` handle. Each field carries the
phase that produces it:

```python
@dataclasses.dataclass
class BatchStorage:
    scheduled_batch:   ScheduledRequests       = phased_field(BatchPhase.SCHEDULE_0)
    can_queue:         bool                    = phased_field(BatchPhase.SCHEDULE_0)
    batch_outputs:     Dict[str, torch.Tensor] = phased_field(BatchPhase.FORWARD_2)
    sample_state:      SampleState             = phased_field(BatchPhase.SAMPLE_3)
    finished_requests: List[LlmRequest]        = phased_field(BatchPhase.RESPOND_8)
    # ...
```

`phased_field(phase)` tags a field with the phase that produces it —
the single declaration the views and the generator below read off.
Its optional `default=` governs read-before-write:

- **No `default`** (the common case): the field starts as a sentinel
  that makes any read before its producer has written it a hard
  error. A field should be read only after the phase that produces
  it has run, and the sentinel catches violations of that.
- **`default=<value>`**: the field is initialized to `<value>` and
  may be read with no producer having written it, returning the
  default. Loop-specific fields use `default=None` for exactly this
  — on a loop that never produces `previous_tensors_device`, its
  consumer reads `None` instead of tripping the read-before-write
  error.

#### Read view at P, write view at P

The runtime gives a coroutine at phase `P` two distinct views over
the storage:

- **Read view at P** — exposes only fields whose `phased_field`
  declared a phase **strictly less than** `P`. Cumulative.
- **Write view at P** — exposes only fields whose `phased_field`
  declared exactly `P`. Phase-local.

This is happens-before, surfaced through the *view types*. Python
types don't enforce at runtime, but they drive the IDE so it shows
the author exactly the fields legal at the current phase: at
`RESOURCE_PREP_1` the read view offers `scheduled_batch` (written at
`SCHEDULE_0`) and not `batch_outputs` (not produced until
`FORWARD_2`); at `FORWARD_2` the write view offers `batch_outputs` to
assign and the read view still won't surface it. Guiding the author
toward the right fields is what keeps the producer / consumer
ordering hard to get wrong by accident; actual runtime checking is
the opt-in tracked views below.

#### Generated views, single source of truth

The phase metadata on `BatchStorage` is the only declaration. From it,
`scripts/generate_coroutine_views.py`
emits — into a `BEGIN GENERATED` / `END GENERATED` block in
`batch_storage.py` — the per-phase Protocols, dataclasses, and
`@overload` chains that narrow `step` / `try_step` / `enter_phase` /
`batch_phase` per `Literal[BatchPhase.X]`. The generator's `--check`
mode runs in CI; its paired test fails on drift.

Hand-maintained view types and `phased_field` declarations would
duplicate the same information; one would inevitably drift from the
other. Generation is the cheapest way to keep them in lockstep.

#### Optional runtime enforcement

Production runs return the storage object aliased as both views (zero
overhead); the `@overload` chains plus a static type checker do the
policing. With `TLLM_COROUTINE_TRACK_STORAGE=1`, `_views_at` instead
returns
`_TrackedReadView` / `_TrackedWriteView`
proxies that enforce phase-scoped reads/writes at attribute access
and distinguish "never written" from "explicitly written `None`"
(via a per-storage write set with `weakref.finalize` cleanup). Useful
on correctness-focused CI and when chasing dataflow bugs.

### 4.4 Concerns and services

A concern is a coroutine class; the rules an implementer follows
when adding one — how concerns coordinate without reaching for each
other, when a coroutine should instead be a context manager — are in
§5. This section covers the shapes: what a concern is and the two
service buckets. A concern
receives **no `ctx`**: every dependency arrives through `__init__`
(below), and `handle_batch` takes no arguments. The bag of concern
instances lives on `ctx.crn` for the orchestrators; concerns can't
reach it precisely because they have no `ctx`, which is what keeps
them from calling peers (§5.1).

#### Concern shape

Every concern is a coroutine, even single-phase ones whose body
amounts to one `enter_phase` line and a handful of statements:

```python
class ForwardConcern:
    def __init__(self, *, model_engine, resource_manager, sampler,
                 execution_stream):
        self._model_engine = model_engine
        self._resource_manager = resource_manager
        self._sampler = sampler
        self._execution_stream = execution_stream

    async def handle_batch(self) -> None:
        r, w = await enter_phase(BatchPhase.FORWARD_2)
        if not r.can_queue:
            return
        # ... run forward ...
        w.batch_outputs = outputs
        w.attn_metadata = self._model_engine.attn_metadata
```

The reason for the no-exceptions uniform shape is the
Driver-provided machinery in §3.3 — the hang watchdog heartbeats
every coroutine it pumps, and the NVTX integration opens a range
per handle on the Driver stack. A plain method that the BATCH
body merely called wouldn't sit on the Driver stack and would be
invisible to both. Wrapping that work as a coroutine costs one
`enter_phase` line and gets both for free.

#### Class shape: persistent data on the instance, per-batch data in the coroutine

A concern is a class with one async method:

```python
class XxxConcern:
    def __init__(self, *, ...): ...
    async def handle_batch(self) -> None: ...
```

This shape splits state cleanly along its actual lifetime:

- **Persistent data** (config flags, the services it uses —
  `requests`, `client`, `dist`, … — references to long-lived
  collaborators like `model_engine`, and cross-batch latches that
  only this concern reads and writes) goes in `__init__` as
  instance attributes. It survives the whole executor lifetime.
- **Per-batch ephemeral data** (CUDA events created at one phase
  and consumed at another, intermediate accumulators, matcher
  results that aren't worth publishing to `BatchStorage`) lives
  as locals in `handle_batch`. A fresh coroutine is constructed
  per batch, so locals are discarded at batch end without any
  cleanup plumbing.

There's no "where do I park this on `self`?" question for
transient state, and no per-batch reset code for instance
attributes that secretly only matter for one batch — the class
shape does the separation, the coroutine lifetime does the
cleanup.

Construction takes plain kwargs and is the concern's *only* channel
for dependencies — there is no `ctx` to fall back on at runtime.
`PyExecutorCoro.__init__` builds each service once and passes the
ones a concern needs into its constructor by name; the same service
instance handed to two concerns is how they share it, and that
sharing is visible at the single wiring site. Each concern's full
dependency surface is therefore its `__init__` signature — `rg` it
and you see everything it can touch — and a unit test builds the
concern with mock services and drives `handle_batch()` with no `ctx`
to assemble.

#### Services — two buckets

Cross-cutting state that doesn't belong to any single concern lives
in service objects, split into two buckets by threading scope. A
service earns a bucket when its state outlives any single batch AND
no one concern can naturally own it (multiple concerns touch it).
Which bucket depends only on threading scope.

**Loop-thread services (`ctx.svc.*`)** own state that never leaves
the loop thread; their methods are not thread-safe and aren't
expected to be, since only loop-thread coroutines call them. The
canonical example is `dist: Distributed`, the cross-rank
communicator every concern uses for TP / PP collectives — it is
cross-*rank* but never cross-*thread*, so it is a loop service, not
a boundary one.

**Boundary services (`ctx.io.*`)** are the cross-thread surface
(§2.3, §2.4): each owns one piece of state that both the main thread
and the loop thread touch, behind a thread-safe method API. The main
thread holds direct references (that is how `PyExecutorCoro`'s public
methods reach the loop side); the loop side reaches them via
`ctx.io.*` for orchestrators and via injected refs for concerns. The
bucket is deliberately closed — it *is* the enumeration of what
crosses the thread boundary, so a new entry is a real extension of
the contract, not a convenience. The canonical example is
`shutdown: ShutdownSignal`, a two-step latch: the loop sets an
inbound flag when it dequeues a shutdown marker (`mark_inbound`) and
a done event when teardown finishes (`signal_done`); the main thread
blocks on `wait_done`, and anyone reads `is_marked`.

Member-count discipline is **HIGH** in both buckets: anything new
must be touched by multiple concerns AND have no natural
single-owner concern. The legacy `model_engine`, `sampler`,
`scheduler`, `drafter`, `kv_cache_transceiver`, etc. each fit a
single concern and live as instance attributes there.

### 4.5 Scheduler variants

Three top-level scheduler-iter coroutines live in
[`py_executor_coro.py`](tensorrt_llm/_torch/pyexecutor/py_executor_coro.py).
The same `batch_body` is shared across all three — only the
interleave changes.

#### `batch_body` — the shared per-batch script

```python
async def batch_body(ctx: Context) -> None:
    crn = ctx.crn
    schedule       = Concern(crn.schedule.handle_batch())
    resource       = Concern(crn.resource.handle_batch())
    forward        = Concern(crn.forward.handle_batch())
    sample         = Concern(crn.sample.handle_batch())
    state_advance  = Concern(crn.state_advance.handle_batch())
    response       = Concern(crn.response.handle_batch())
    ring_broadcast = (Concern(crn.ring_broadcast.handle_batch())
                      if crn.ring_broadcast is not None else None)

    async with batch_phase(BatchPhase.SCHEDULE_0):     await resume(schedule)
    async with batch_phase(BatchPhase.RESOURCE_PREP_1): await resume(resource)
    async with batch_phase(BatchPhase.FORWARD_2):      await resume(forward)
    async with batch_phase(BatchPhase.SAMPLE_3):       await resume(sample)
    async with batch_phase(BatchPhase.STATE_UPD_4):    await resume(state_advance)
    async with batch_phase(BatchPhase.SYNC_EVT_5):     await resume(ring_broadcast)
    async with batch_phase(BatchPhase.HANDOFF_6):
        for _ in range(ctx.svc.dist.pp_size - 2):
            if await try_resume(ring_broadcast): break
            else: await again()
        else:
            await resume(ring_broadcast)
    async with batch_phase(BatchPhase.APPLY_7):        await resume(sample)
    async with batch_phase(BatchPhase.RESPOND_8):
        await resume(response)
        await resume(resource)
    async with batch_phase(BatchPhase.FINALIZE_9):
        await resume(schedule)
        await resume(ring_broadcast)
```

Optional concerns are `None` in the bag and the `resume(None)` /
`try_resume(None)` no-op fast path carries the body through unchanged
when they aren't wired. This is what makes the same body work across
loops: every loop variant participates in the same script;
configurations differ only in which concerns are present and which
phases are folded by yielded subset (§4.2).

#### Plain — one batch per iter

```python
for it in profiler(ctx):
    if (ctx.io.shutdown.is_marked()
            and ctx.svc.requests.is_drained()
            and ctx.svc.requests.waiting_empty()):
        return
    storage = BatchStorage()
    handle  = Batch(batch_body(ctx), storage, idx=it)
    try:
        await step(handle, through=BatchPhase.FINALIZE_9)
    except Exception:
        ...   # catastrophic (§2.5): fail in-flight requests, signal shutdown
        raise
```

One batch in flight, every phase. No batch-to-batch handoff.

#### Overlap — two batches in flight, three barriers

Each iter splits `current`'s phases at three barrier points so
`previous`'s remaining work (and one scheduler-side data injection)
can interleave **and** the cross-batch ordering invariants are
honored:

1. **`previous_tensors_device` for curr's `FORWARD_2`** — the
   SCHEDULER reads prev's `sample_state.device` (produced at prev's
   `SAMPLE_3` last iter) and writes it into curr's `SCHEDULE_0`
   write view as `previous_tensors_device`; `ForwardConcern` reads
   it back through the `FORWARD_2` read view to feed prev's
   just-sampled tokens into curr's forward.
2. **Fresh `seq_lens` at curr's `SAMPLE_3`** — requires `prev.APPLY_7`
   first so the sampler kernel sees fresh `num_tokens` and tags
   the iter's last-eligible token for `GENERATION_COMPLETE`
   transition.
3. **State advance after prev's `RESPOND`** — `StateAdvanceConcern`
   flips `set_exclude_last_generation_logits(False)` on
   `GENERATION_TO_COMPLETE` transitions; flipping before prev's
   RESPOND corrupts streaming-response indices for shared-by-id
   requests.

Per-iter steps:

```python
_, w_sched = await step(current, through=BatchPhase.SCHEDULE_0)
if w_sched and previous and previous_view.sample_state:
    # batch-to-batch bridge for invariant 1
    w_sched.previous_tensors_device = previous_view.sample_state.device
await step(current,  through=BatchPhase.FORWARD_2)
await step(previous, through=BatchPhase.APPLY_7)
r_sample, _ = await step(current, through=BatchPhase.SAMPLE_3)
await step(previous, through=BatchPhase.FINALIZE_9)
await step(current,  through=BatchPhase.STATE_UPD_4)
previous, previous_view = current, r_sample
```

The batch-to-batch bridge — write `previous_tensors_device` on
curr's `SCHEDULE_0` slot using prev's `sample_state.device` — is
the only place the SCHEDULER writes directly into a batch's
storage rather than going through the batch's own coroutine. This
is where the "scheduler-only batch-to-batch handoff" home for
cross-concern data (§5.1) materializes.

Shutdown drain: when no more work to admit AND `previous` is still
parked, `current=None`; the per-step `step(None, …)` no-op carries
the body through unchanged so the only useful work that iter is
draining `previous` to `FINALIZE_9`.

#### PP — the deque, the vote, the ring

Works for any `pp_size >= 2`. Each rank runs an identical scheduler
iter; cross-rank coordination happens inside per-batch concerns and
inside two scheduler-direct calls.

The SCHEDULER holds a `collections.deque` of in-flight `Batch`
handles, age order **left=oldest, right=newest**, max size
`pp_size − 1`. The improvement over the legacy slot ring isn't
that it's explicit (the legacy ring was an explicit list too) —
it's **bundling**. A `Batch` handle holds everything that belongs
to that batch: its storage, its concern coroutines, the per-batch
isend handles those coroutines own as locals. So the SCHEDULER
needs **one** deque indexed by batch identity instead of the
legacy's set of parallel slot lists keyed by `microbatch_id`
(`send_handles[mid]`, `send_schedule_handles[mid]`,
`send_expected_batch_num_handles[mid]`, …) plus the active-request
mutations they implicitly synchronized.

Per-iter steps. The first three are deliberately interleaved: the
ring hop is **posted** (step *post-hop*) between the current
batch's **forward** and **sample** steps, so the recv has a full
iter of wall-clock to land before it's checked.

1. **forward-current** — drive `current` through `FORWARD_2`. Queue
   this batch's forward; no GPU sync yet.
2. **post-hop** — drive the newest parked batch through `SYNC_EVT_5`,
   uniform across ranks: the source rank syncs its sampler event and
   posts the non-blocking send (`pp_source_isend`); every non-source
   rank submits its blocking `recv_object` to the offload pool and
   gets a future back. One-shot per batch, no polling. (Older parked
   batches were posted at the iter after they were admitted.)
3. **sample-current** — drive `current` through `STATE_UPD_4` (queue
   sampling + state advance). The CPU wall-clock spent here is the
   window in which the recv worker thread runs, so the
   **opportunistic-poll** step below sees `future.done() == True` for
   recvs that have landed.
4. **forced-retire** — when the ring is full or we're draining, pop
   the oldest batch (`deque.popleft()`) and drive it through
   `FINALIZE_9`.
5. **opportunistic-poll** (rk0 only) —
   `try_step(parked, through=HANDOFF_6)` for each parked batch; count
   the contiguous run of head batches whose recv has landed as `opp`.
   Only rk0 polls, and the reason is fundamental: all ranks must
   retire the *same* batches in the *same* iter (the lockstep
   invariant below). Recv timing differs per rank, so if each rank
   decided locally how many batches had landed they would diverge —
   so instead **one rank decides and the rest follow** its count
   (broadcast in **retire-count-vote**). rk0 is the natural decider:
   it sits at the head of the sample-state ring and sees recvs land
   first.
6. **retire-count-vote** — rk0 broadcasts `opp` (the count of
   opportunistically-retired batches) along the PP forward chain
   (`isend_object` + `recv_object`). Each rank stores the received
   value back into `opp`. Only `opp` rides the wire; `forced`
   (whether **forced-retire** fired) is recomputed locally on every
   rank (see "Lockstep invariant" below).
7. **opportunistic-retire** — pop `opp` more batches from the deque
   head and drive each through `FINALIZE_9`. On rk0 these are already
   past `HANDOFF_6` from **opportunistic-poll**; on intermediate
   ranks each `step` enters the ring concern's blocking `result()`
   wait.
8. **admit-current** — push `current` onto the newest end of the ring.

```python
in_flight = collections.deque()          # left = oldest, right = newest
in_flight_max = pp_size - 1

async def opportunistic_poll() -> int:    # rk0 only
    # poll and count finished head batches in in_flight
    ...

for it in profiler(ctx):
    more_to_admit = ...
    if not more_to_admit and not in_flight:
        return
    current = ...

    await step(current, through=BatchPhase.FORWARD_2)             # forward-current
    if in_flight:
        await step(in_flight[-1], through=BatchPhase.SYNC_EVT_5)  # post-hop
    await step(current, through=BatchPhase.STATE_UPD_4)           # sample-current

    if len(in_flight) >= in_flight_max or not more_to_admit:      # forced-retire
        await step(in_flight.popleft(), through=BatchPhase.FINALIZE_9)

    opp = await opportunistic_poll() if ctx.svc.dist.is_first_pp_rank else 0
    opp = ring_broadcast_executed_batch_num(ctx.svc.dist, opp)    # retire-count-vote
    for _ in range(opp):                                          # opportunistic-retire
        await step(in_flight.popleft(), through=BatchPhase.FINALIZE_9)

    if current is not None:
        in_flight.append(current)                                 # admit-current
```

**Lockstep invariant.** Every rank computes `len(in_flight) >=
in_flight_max or not more_to_admit` locally and reaches the same
answer; only `opp` is on the wire. This holds by induction (shutdown
markers + request items reach every rank via
`pp_broadcast_request_items` at `SCHEDULE_0`; pool mutations take
lockstep inputs).

**The broadcast thread.** The legacy PP path ran the sample-state
broadcast on its own always-on thread
(`_broadcast_sample_state_loop`), which co-owned MPI with the worker
thread and needed a "flush-when-idle" hack to avoid deadlock. The
coroutine design eliminates the thread conceptually: all of its logic moves
onto the loop thread as a concern `RingBroadcastSampleConcern`,
and the driver owns MPI scheduling, so the queues and the flush hack
are gone. A thread does survive — mpi4py's `pkl5` communicator has no non-blocking `irecv` for pickled objects, so the
blocking `recv_object` runs on a single-worker pool (`RecvOffload`) —
but it carries no logic: it merely emulates the MPI recv as a
pollable `concurrent.futures.Future`, so the
loop-thread coroutine can treat it as an async MPI operation.

---

## 5. Rules & guides for concern implementation

The shapes in §4 say what the pieces are. This section is the
rulebook an implementer follows when adding or changing a concern —
the conventions that keep the pieces honest. Most are mechanically
checkable (§5.4).

### 5.1 No peer calls: the four homes for cross-concern data

§2.1 stated the rule conceptually — concerns never call each other
directly. This is the mechanism that enforces it, plus the routing
table for where any given piece of cross-concern data must go.

The mechanism is simply that **a concern receives no `ctx`** (§4.4).
The [`Concerns`](tensorrt_llm/_torch/pyexecutor/concerns/__init__.py#L138-L192)
bag — a frozen dataclass with one field per concern instance, built
once in `PyExecutorCoro.__init__` — lives on `ctx.crn`, and the
orchestrators (`run_loop` / `scheduler_iter*` / `batch_body`) reach
peers through it freely. A concern cannot: it has no `ctx`, so
`ctx.crn.X` is unreachable from inside `handle_batch`. There is no
peer-call syntax to write. Every cross-concern interaction must
therefore fall into one of **four homes**:

| Case | Home |
|---|---|
| Within-batch concern → concern data (e.g., `schedule` writes `scheduled_batch`; `forward` reads it) | `BatchStorage` via the typed `enter_phase` views |
| Batch-to-batch handoff (one batch's terminal output is the next batch's input — e.g., the overlap loop feeding prev's just-sampled tokens into curr's forward) | SCHEDULER iter local; reads prev's terminal read view, stuffs into curr's `SCHEDULE_0` write view |
| Cross-batch shared state, loop-thread only (rolling acceptance gate, …) | A loop-only service on `ctx.svc.*` |
| Cross-batch shared state that also crosses threads (request inbox, response channel, lifecycle signals) | A boundary service on `ctx.io.*` |
| A concern's PRIVATE cross-batch state (one-way latch only this concern observes) | Instance attribute on the concern class |

Two of those homes are services (`ctx.svc.*` / `ctx.io.*`), and they
carry a sharp negative rule: **a service is not a data channel.** A
service is the home for cross-batch *shared state* or an *effect* — a
communicator, a shutdown latch, a lifecycle registry — touched by
multiple concerns across iterations. It is never the medium by which
one concern hands per-batch data to another. The moment a value is
written into a service in one concern so that a later concern can
read it back, a concern→concern data channel has been built out of a
service, and "who produced this, and when" stops being answerable
from one batch's storage. Per-batch concern→concern data belongs in
`BatchStorage` (row 1); cross-batch handoff is scheduler-mediated
(row 2). A service holds state and effects, not in-flight data.

The shape is deliberately inconvenient. The tempting shortcut is
`ctx.crn.X.method()` — "the state I need is already on that other
concern" — easy to type, easy to merge, easy to leave there.
Withholding `ctx` from concerns makes that shortcut not merely
discouraged but *unwritable*: there is no `ctx` in scope, so the
peer call won't even type-check. The author has to find the right
home for the data (one of the rows in the table above), or add a
dependency to `__init__` in a way that's structurally obvious in
review. Friction is the point. AI coding assistants in particular
are prone to take the local-minimum path when a quick
import / one-line peer call works, even when the right answer is
to add a `BatchStorage` field and route the data through the typed
views — trading long-term maintainability for short-term
implementation effort. Making the local-minimum path *absent*
rather than just discouraged keeps the long-term-maintainability
shape self-enforcing.

### 5.2 Sequence-coupled methods are coroutines in disguise

Coroutinization is the theme of this refactor: the legacy
hand-written automatons (three forward loops, the broadcast thread,
the implicit per-batch state machine encoded in scattered `self.*`
fields) become coroutines whose bodies read top-to-bottom in
lifecycle order. The same lens applies in the other direction
when designing new state — we should avoid *reintroducing* the
shape we just removed.

The signature smell of a hand-written class automaton:

> Method `B` on a class must be called only after method `A`, and
> calling them out of order silently corrupts state rather than
> raising. The class is encoding a state machine; the methods are
> events that drive it.

The rule:

> **For every pair of public methods on a new service, ask: does
> one need to be called before the other for correctness? If yes,
> express the sequence as a coroutine, an async / sync context
> manager, or an owner-controlled per-batch concern — not as two
> exposed methods on the class.**

Three patterns are exempt because their sequencing is bounded by
language constructs, not by API contract:

- **Pure container operations.** `add` / `remove`, `put` / `get` —
  individually meaningful, commutative-ish, no protocol.
- **Constructor / destructor / CM enter / CM exit.** The order is
  enforced by Python's object lifecycle. A `start()` / `stop()`
  pair exposed as separate public methods *would* fail the test
  and should be reshaped as a CM.
- **Single-method APIs.** Trivially can't have ordering.

The request-lifecycle design in §4.4 is the first concrete
application: the alternative shape — separate `mark_inflight` /
`unmark_inflight` methods — is a genuine 2-state automaton across
batch phases. The design uses a `track_inflight` context manager
instead, whose `__enter__` / `__exit__` halves are paired by
syntax and impossible to mis-sequence.

### 5.3 CM for state lifetime, concern for phase work

§5.2's rule offers three coroutinization tools (a coroutine, an
async / sync context manager, or an owner-controlled per-batch
concern). The choice between *concern* and *context manager* isn't
arbitrary — each fits a different shape, and conflating them
either bloats the BATCH body with phantom concerns or buries
real cross-phase work inside CMs that can't express it.

The distinction:

- **Concern** = a participant the BATCH body drives. Its
  coroutine yields at each batch phase it cares about; the
  BATCH body's `await resume(handle)` calls advance it. Right
  shape when there is non-trivial *work* to do at multiple
  phases, the work at each phase reads top-to-bottom, and the
  thing can stand alone as a domain (forward, sample, schedule,
  response).

- **Context manager** = a tool a single owning coroutine uses to
  bracket the lifetime of one piece of *state* across phases.
  Entered by one coroutine (a concern, or the BATCH body
  itself); the contained "..." is whatever the owning coroutine
  does between entry and exit. Right shape when the thing
  manages one piece of state with short setup and cleanup, the
  state's lifetime is bounded by phases of one coroutine, and
  no external work depends on this state being open at
  specific intermediate phases.

The two are not substitutes:

- **A CM can't be a concern.** The BATCH body drives Concern
  handles via `resume`; a CM can't yield at multiple phases
  independently of its containing body. Forcing a CM into the
  concern role means nesting it over the BATCH body's phase
  blocks — inverting the orchestration so one would-be-concern
  wraps every other concern's phase work.
- **A concern can't replace some CMs.** State managed at phase
  boundaries inside a single coroutine (a CUDA stream context,
  a recorded NVTX range, the inflight set across SCHEDULE_0 –
  RESPOND_8) doesn't need a Concern handle on the BATCH body's
  roster. It needs `with` / `async with` inside the coroutine
  that owns it.

The boundary test: **does the BATCH body need to drive this thing
through phases independently of its peers?** If yes (other
concerns interleave with it at intermediate phases), it's a
concern. If no (it's local to one body), it's a CM.

Examples from this design:

- `track_inflight(scheduled)` is a CM. The inflight set's
  lifetime is owned by PpScheduleConcern alone; no other
  concern's work depends on the set being open at intermediate
  phases. (An alternative `InflightTrackingConcern` was
  considered — it works, but it places a one-line phantom
  concern on the BATCH body's roster solely to manage a set,
  and loses the syntactic pairing.)
- `RingBroadcastSampleConcern` is a concern. The BATCH body
  drives it at SYNC_EVT_5 separately from HANDOFF_6, with the
  scheduler running other batches' phases in between. The
  work-at-each-phase is structurally interleavable; the
  per-phase bodies are non-trivial.
- `disable_hang_detect()` and `torch.cuda.stream(...)` are CMs.
  Each manages runtime context for one block of code inside
  one coroutine body.

The same test applied across the existing concern set
(ScheduleConcern, ResourceConcern, ForwardConcern,
SampleConcern, StateAdvanceConcern, ResponseConcern,
PpScheduleConcern, RingBroadcastSampleConcern) returns "concern"
for every one. None of the eight is degenerate enough to be a
CM. Future concerns from the planned set whose shape might
*look* CM-ish at first glance (IterStatsConcern with its
SCHEDULE_0 / FINALIZE_9 yields, PerfMetricConcern with its
event-recording bookends) still fail the test: in each case
the BATCH body interleaves other concerns with them at
intermediate phases, so they must be drivable independently.

### 5.4 The extension checklist

The mechanical checks the design relies on stay green only if new
work follows the rules. Most are catchable at PR review by grep:

- `rg "\bctx\b" tensorrt_llm/_torch/pyexecutor/concerns/` should be
  empty: a concern receives no `ctx`, so it can't reach peers or
  ambient services — every dependency comes through `__init__`
  (§4.4, §5.1).
- `python scripts/generate_coroutine_views.py --check` must pass
  (`BatchStorage` field metadata is the single source of truth for
  views).
- New `BatchStorage` fields declare a `phased_field(P)`; producer's
  P must be strictly less than every consumer's read phase; each
  field's leading `#` block names producers and consumers
  explicitly.
- New cross-iter state lands in one of the homes documented in
  §2.3 / §5.1 (intra-concern local, BatchStorage, scheduler local,
  a service in `ctx.svc.*` / `ctx.io.*`, concern instance
  attribute). There is no "nobody owns it" bucket — mutable
  cross-batch state is always service-owned.
- Concerns are coroutines, even single-phase ones, so NVTX tracing
  and stack-uniformity hold (§4.4).
- **Sequence-coupled methods on a class** (method B requires method
  A to have been called first; calling out of order silently
  corrupts) are coroutines / context managers in disguise. Express
  the pair as a CM or extract into a per-batch concern; do not
  expose two methods that must be paired by convention (§5.2, §5.3).

---

## 6. Designs we considered or tried first

Several earlier shapes turned out worse on contact with the actual
loops; each is recorded here so the rationale for the chosen design
isn't lost. Cross-references point back to the section that landed on
the design we kept. Read in any order.

#### Outer loop wraps the Driver (vs. §3.2)

An intermediate framing was that the executor's outer loop is itself
the event loop, with the Driver as that loop. The design goes one
step further: the loop variants (plain, overlap, PP) are themselves
extracted as **root-level coroutines** the Driver pumps. The Driver
does no scheduling on its own; it just follows what the
currently-running coroutine yields. There is no outer loop wrapping
the Driver, and the schedule is fully spelled out by the chain of
`step` / `resume` calls inside the coroutines.

#### Per-layer / per-concern phase enums (vs. §2.2)

A version with separate enums per layer (`LoopPhase` for batches,
`SchedPhase` for the scheduler, plus per-concern step tiebreakers)
duplicates labels: every layer's "between forward and sample"
boundary is the same boundary. Collapsing onto one `BatchPhase` makes
"this code happens before that code" a single integer comparison
across layers, and removes the cross-enum mismatches a separate-enum
design needs runtime asserts to catch.

#### Raw coroutine handles for `resume` (vs. §3.2)

Originally `resume(coro)` took a raw coroutine; only `step` had a
handle (`Batch`). This was asymmetric and had a sharp edge: calling
`resume` on a concern that had already returned tripped
`RuntimeError: cannot reuse already awaited coroutine`. Wrapping both
sides in handles (`Batch` and `Concern`) with a `done` flag the
Driver flips on completion lets the user-side fast paths
short-circuit on `done` *before* re-entering the Driver. As a bonus,
`__del__` on undriven handles closes the coroutine cleanly,
silencing `RuntimeWarning: coroutine '...' was never awaited` and
replacing the old `spawn` primitive that primed coroutines eagerly to
the same end.

#### Method-shaped single-phase concerns (vs. §4.4)

A mix of "method for single-phase concerns, coroutine for multi-phase"
was considered. The savings (one less `enter_phase` line per
single-phase concern) didn't survive contact with three real benefits
of uniform coroutine shape: locality of intra-concern state when it
appears (a CUDA event handle, a matcher result — declare a coroutine
local), uniform NVTX annotation for free (every `Concern`-wrapped
handle is on the Driver stack for its lifetime), and top-to-bottom
lifecycle reading.

#### Passing `ctx` to concerns (vs. §4.4 / §5.1)

The prototype passed `ctx` into every `handle_batch`, so a concern
reached its services as `ctx.svc.*` / `ctx.io.*`. It works, but it
makes a concern's dependency surface ambient and invisible: you
read the body to discover what it touches, it can reach *any*
service whether it needs it or not, and — had the `Concerns` bag
been on `ctx` — it could call peers via `ctx.crn.X`. The design
withholds `ctx` from concerns entirely and injects each
dependency through `__init__` (§4.4): the surface becomes the
constructor signature, least privilege holds, and peer access is
structurally impossible. A useful side effect: with concerns unable
to reach `ctx` at all, the `Concerns` bag can safely move back onto
`ctx.crn` for the orchestrators (§5.1) — the protection no longer
depends on keeping `crn` off `ctx`.

---

## 7. File layout

LoC counts are snapshots of the prototype tree. The files split
into two tiers with very different maturity — see the introduction.

**Runtime + tooling — feature-complete and exhaustively tested.**
This is the foundation; its shape is backed by the prototype
implementation observations recorded throughout the doc.

| Path | LoC | Role |
|---|---|---|
| `tensorrt_llm/_torch/pyexecutor/coroutines.py` | 1884 | Generic runtime: `Driver`, `Batch`, `Concern`, primitives, watchdog, tracked-view proxies |
| `scripts/generate_coroutine_views.py` | 646 | Generate `BatchStorage` views + overloads |
| `tests/unittest/_torch/executor/test_coroutines.py` | 1966 | Runtime + production-binding tests |

**Executor layer — reference prototype, to be reimplemented.**
These verify the design end-to-end but are not production targets;
expect most of them to be rewritten as the design is accepted and
the remaining concerns land.

| Path | LoC | Role |
|---|---|---|
| `tensorrt_llm/_torch/pyexecutor/batch_storage.py` | 1029 | Data model: `BatchPhase`, `BatchStorage`, generated views + overloads |
| `tensorrt_llm/_torch/pyexecutor/context.py` | 553 | `Context` (incl. the `Concerns` bag), `Service`, `Configuration`, and the cross-thread boundary services. (Prototype still has a `PersistentState`; the design drops it and the prototype keeps `Concerns` as a separate arg — see §6.) |
| `tensorrt_llm/_torch/pyexecutor/py_executor_coro.py` | 1570 | `PyExecutorCoro`, `scheduler_iter_{plain,overlap,pp}`, `batch_body`, `profiler` |
| `tensorrt_llm/_torch/pyexecutor/pp_helpers.py` | 575 | PP comm helpers shared between legacy and coro paths |
| `tensorrt_llm/_torch/pyexecutor/concerns/__init__.py` | 227 | `Concerns` bag + module docstring with the four-homes rule |
| `tensorrt_llm/_torch/pyexecutor/concerns/concerns.md` | 445 | Per-concern design notes (planned-concerns table, etc.) |
| `tensorrt_llm/_torch/pyexecutor/concerns/services.py` | 515 | Loop-only services (request lifecycle, recv offload, dist) |
| `tensorrt_llm/_torch/pyexecutor/concerns/forward.py` | 135 | `ForwardConcern` |
| `tensorrt_llm/_torch/pyexecutor/concerns/resource.py` | 89 | `ResourceConcern` |
| `tensorrt_llm/_torch/pyexecutor/concerns/response.py` | 156 | `ResponseConcern` |
| `tensorrt_llm/_torch/pyexecutor/concerns/sample.py` | 157 | `SampleConcern` |
| `tensorrt_llm/_torch/pyexecutor/concerns/schedule.py` | 402 | `ScheduleConcern`, `PpScheduleConcern` |
| `tensorrt_llm/_torch/pyexecutor/concerns/state_advance.py` | 168 | `StateAdvanceConcern` |
| `tensorrt_llm/_torch/pyexecutor/concerns/ring_broadcast.py` | 302 | `RingBroadcastSampleConcern` (PP sample-state ring broadcast) |
| `tests/unittest/_torch/executor/test_py_executor_coro_sanity.py` | 183 | End-to-end sanity for `PyExecutorCoro` |

Generic runtime (`coroutines.py`) and the concrete data model
(`batch_storage.py`) are deliberately split. The runtime knows only
`IntEnum` phases and an unconstrained storage object; the data
model pins the concrete `BatchPhase` and `BatchStorage`. Two
payoffs:

- Runtime tests use a local `_TestPhase` and `_TestStorage`; they
  stay stable across any data-model change — so the reimplementation
  of the executor layer does not destabilize the runtime tests.
- Nothing prevents a future module from defining a different phase
  enum + storage class against the same generic runtime.

---

## 8. Status and future work

### 8.1 Validation status

**The runtime is exhaustively tested.**
[`tests/unittest/_torch/executor/test_coroutines.py`](tests/unittest/_torch/executor/test_coroutines.py)
exercises the generic mechanics against a test-local `_TestPhase`
and `_TestStorage`: step comparison logic, strict progression at the
suspension point, two-branch shared step, voluntary opt-out via
early return, `again()` semantics (active state preserved on retry),
strict / tolerant retry surfacing, cascading `again()`,
`disable_hang_detect` non-nesting, `Driver.close()` LIFO unwind via
`GeneratorExit`, handle `__del__` closing undriven coroutines, and
tracked-view enforcement. Because these tests bind to a test-local
phase enum and storage class, they stay green regardless of how the
executor layer is reshaped — the runtime's correctness does not
depend on the prototype's executor code, and the planned
reimplementation cannot regress it. A **production-binding** group
in the same file imports the concrete `BatchPhase` / `BatchStorage`
and asserts the generated view block matches the field metadata
(the generator's `--check` invariant).

**The executor prototype is validated end-to-end at single rank,
as a design proof.**
[`tests/unittest/_torch/executor/test_py_executor_coro_sanity.py`](tests/unittest/_torch/executor/test_py_executor_coro_sanity.py)
drives the plain and overlap loops end-to-end. That is enough to
demonstrate the design holds together — concerns, services,
`BatchStorage` dataflow, and the scheduler-iter shape all compose
and produce correct output. It is **not** a production test bar:
the prototype's executor code is reference-grade, and the
reimplementation is where production-quality coverage (parity
suites below, multi-rank, multi-feature) gets built out.

Behavioral-parity suites still to be wired against the new
executor (these guard the reimplementation, not the prototype):

- `test_overlap_scheduler.py` — overlap / non-overlap behavioral
  parity.
- `test_disaggregated_serving.py`,
  `test_dwdp_disaggregated_serving.py` — gated on the disagg /
  `DwdpConcern` work landing.
- `tests/unittest/disaggregated/` — KV-transfer / connector coverage,
  gated similarly.
