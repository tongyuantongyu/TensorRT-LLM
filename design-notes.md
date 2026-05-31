## Coroutine runtime: architectural decisions, in the order we made them

This is the design log for `tensorrt_llm/_torch/pyexecutor/coroutines.py` and
its surrounding files (`batch_storage.py`, the generator script, the test
suite). Each entry records *why* we ended up at the current design, and what
alternative we rejected. Read top-to-bottom — every entry assumes the ones
above. Code-style decisions (naming conventions, formatter quirks,
docstring shape) are out of scope.

---

### 1. Use Python coroutines, not asyncio

The forward loop interleaves several concerns whose code is fundamentally
sequential per concern but multiplexed across concerns. That is what
coroutines are *for*. We deliberately do not use `asyncio`: the executor's
outer loop itself plays the role of the event loop. Coroutines suspend with
`await` only to communicate suspension / resumption with their driver, not
to wait on real I/O.

This keeps the schedule explicit and deterministic. There is no hidden
ordering coming from `asyncio` policies; what runs when is whatever the
top-level scheduler coroutine spells out.

### 2. Three-layer hierarchy: Scheduler → batch → concerns

Two layers (scheduler ↔ concerns) wasn't enough — when we tried to fold
batch-lifecycle work into the scheduler, the scheduler became a copy of
itself for every loop variant (plain / overlap / PP). Three layers
matched the actual structure of the code:

- **Scheduler** drives a small number of batches forward, deciding when
  each batch runs which work. Its body *is* the executor's outer loop.
- **Batch coroutine** owns the lifecycle of one batch of requests across
  its phases. There is one batch coroutine instance per batch; for
  pipeline-parallel/overlap loops the scheduler runs several batch
  coroutine instances concurrently.
- **Concerns** handle a single cross-cutting task (sampling, response
  building, KV transfer, etc.). They are spawned by a batch and yield
  inside its phases.

The scheduler weaves batches; a batch weaves concerns. Each layer only
talks to its immediate neighbors.

### 3. `LoopPhase` is the single, unified happens-before timeline

After exploring two separate enums (`LoopPhase` for batches,
`SchedPhase` for the scheduler, plus `StepEnum` tiebreakers per
concern), we collapsed everything onto one ordered enum. Reasons:

- Every layer ultimately yields at the same phase boundaries —
  pretending otherwise just duplicates labels.
- A single timeline makes it trivial to see, locally, whether code
  in concern A happens before or after code in concern B: compare
  the phase values.
- Within one layer the timeline is strictly monotonic per coroutine
  (see strict progression, item 12).

Phase values are placeholder integers in the generic runtime; the
production module pins concrete names (`SCHEDULE_0`, `FORWARD_1`, …)
later (item 17). The numeric suffix exists so a reader sees relative
order without consulting an enum definition.

### 4. Phase as the data-passing contract: typed read / write views

Rather than passing a mutable context bag around (which we tried first
and which produced read-before-write bugs), the runtime gives each
phase its own pair of *views* over the storage:

- **Read view at phase P** exposes only fields produced at phases `< P`
  (cumulative).
- **Write view at phase P** exposes only fields produced *at* P
  (phase-local).

This is happens-before, enforced statically. The view types are
`Protocol` classes for reads and `dataclass` classes for writes, both
generated from a single source of truth — the per-field `phase`
metadata on the storage dataclass (item 16).

The data-passing contract is the same at every layer. Scheduler hands
a batch some seed via the batch's write-at-P view; the batch reads it
back via its read-at-(P+1) view inside the next phase block; a
concern publishes its result via the concern-level write view; later
concerns / the batch / the scheduler read it back via their next
phase's read view.

### 5. Storage is implicit (ContextVar), not threaded

The first attempt threaded an `IterStorage` argument through every
coroutine call. This (a) clutters every signature and (b) gives
concerns raw access to the storage, sidestepping the typed views.

The runtime instead keeps two `ContextVar`s — `_active_storage` and
`_active_phase` — that the scheduler-side `step()` and the
batch-side `batch_phase()` context manager set / clear at their
boundaries. Concerns just call `enter_phase(P)`; the active storage
is read from the ContextVar.

Running each Driver inside its own `contextvars.copy_context().run(...)`
isolates these ContextVars between concurrent Drivers — no leakage
back to the caller.

### 6. Pop predicate is strict `>`; "Py is now" semantics

When a parent yields `await advance(child, at=Py)`, the Driver pumps
the child *past* Py — pops only when the child's next wait is
strictly greater than Py. Reasoning:

- `enter_phase(Py)` should mean "I'm now at Py, doing Py work". The
  caller's view of "the child finished Py work" should therefore mean
  the child has *exited* its Py block and is waiting somewhere later.
- The earlier `>=` predicate left a confusing off-by-one between
  caller and callee at the same phase value.

Combined with item 4: at the caller's phase P, the caller can read
data produced *at and below* P from the child; producing data *at*
P themselves stays in the caller's own write view.

### 7. Two halves of the advance primitive: `resume` and `step`

`resume` (batch → concern) and `step` (scheduler → batch) are the
same yield-an-`_AdvanceRequest` mechanism, but the *meaning* differs:

- `await resume(child)` is "drive the child through the current
  phase's work". The data the child produced becomes visible to the
  batch through the batch's *next* phase block, not through this
  call's return value. So `resume` returns `None`.
- `await step(handle, through=Py)` is "drive the batch until it's
  finished Py and produce its read/write views at Py for me to use".
  So `step` returns `(read_view, write_view)`.

This split lets the type checker narrow `step`'s return type per
`through=Literal[BatchPhase.X]`, while `resume` stays a plain
`-> None`.

The batch-level CM (`batch_phase(P)`) is sugar around `enter_phase` that
*also* sets `_active_phase = P`, so `resume(child)` inside the block
knows what phase to advance the child to without an explicit argument.

### 8. Driver is policy-free, traceback-friendly, and isolated

The Driver knows three sentinel request types and a stack. Nothing
else. In particular:

- It does not interpret a normal return from the root coroutine as
  an error — higher layers decide whether that's legitimate.
- Its own stack frames are stripped from exception tracebacks by
  default (`TLLM_COROUTINE_SHOW_FRAMES=1` opts in to keeping them).
  This was a prerequisite for the `await advance(child)` design over
  a synchronous `advance(child)` — once the framework frames are
  hidden, the `await` is no longer a readability cost.
- It runs inside a fresh `contextvars.Context` so multiple Drivers
  (or the same Driver across `run` calls) are independent.

The deliberate non-decision: the Driver has no scheduling policy.
The scheduler coroutine's own sequence of `step` calls *is* the
schedule.

### 9. Shutdown uses Python-native `close()` / `GeneratorExit`

We considered a custom `Shutdown` exception class. Python already has
this primitive: `Coroutine.close()` throws `GeneratorExit`, which
unwinds `with` / `try` / `finally` blocks naturally. Reinventing it
just creates a second concept readers have to learn.

The Driver closes children LIFO so deep unwinds run their cleanup
before parents. `BaseException` (Ctrl-C, `SystemExit`) is allowed to
propagate through shutdown so a panic doesn't deadlock the close.

### 10. Field metadata is the single source of truth; views are generated

`phased_field(BatchPhase.X)` tags a storage field with the phase that
produces it. From this metadata:

- The generator script renders `_ReadAtP*` / `_WriteAtP*` /
  `_ReadAtAll` types and the `@overload` chains for `step` /
  `try_step` / `batch_phase` in `batch_storage.py`.
- A debug-mode runtime proxy (`_TrackedReadView` /
  `_TrackedWriteView`, enabled by `TLLM_COROUTINE_TRACK_STORAGE=1`)
  enforces happens-before at attribute access — a field read at a
  phase that didn't yet write it raises immediately, not silently
  observes `None`.

The generator has a `--check` mode and a paired test that asserts the
generated block is up to date in CI / pre-commit.

### 11. Strict progression check on `enter_phase`

A coroutine yielding `enter_phase(P)` after a previous wait at `Pprev`
raises if `P <= Pprev`. This catches both regressions (P2 → P1) and
stay-at-same-phase loops at the offending call site, instead of
letting them surface later as confusing read-of-unwritten-field
errors via the tracked-view proxy.

The check is per-coroutine, since each layer has its own monotonic
timeline.

### 12. Runtime module is generic; production data model lives separately

The generic runtime (`coroutines.py`) speaks only `IntEnum` phases
and an unconstrained storage object. The production data model —
the concrete `BatchPhase` enum and the `BatchStorage` dataclass with
its field-level lifecycle docstrings — lives in
`batch_storage.py`. The generator script and the `@overload` chains
also live there.

Two consequences:

1. Runtime tests in `test_coroutines.py` use a local `_TestPhase` and
   `_TestStorage`. They stay stable when production phases change.
   A separate "production-binding" group of tests (clearly fenced at
   the bottom of the file) is the only thing that imports the
   concrete types — exactly to verify the generator and the metadata
   agree.
2. There's nothing stopping a future module from defining a
   *different* phase enum and storage class against the same generic
   runtime.

### 13. Iteration → Batch rename

The lifetime "process one batch of requests through all its phases"
collided with the existing PyExecutor's notion of an "iter" =
"one outer-loop body". We renamed our coroutine-lifecycle term from
"iteration" to "batch" (so `Iteration` became `Batch`, `iter_phase`
became `batch_phase`, `iter_storage` became `batch_storage`, etc.).
"Iter" is left available for the outer-loop notion, and several
follow-up decisions hinge on this distinction (see items 14 and 15).

### 14. Retry: `await again()` + paired strict/tolerant drivers

PP loops created the question: what if a batch has reached phase P
but cannot finish P's work yet because some external state isn't
ready? The general "loops inside coroutines" framing led to several
options; the narrow one we picked is the cleanest:

- The coroutine says `await again()` — semantically POSIX `EAGAIN`,
  *"I'm at this phase and want to be at the same phase when resumed"*.
- The Driver pops the coroutine without touching its suspension
  record, `_active_phase`, `_active_storage`, or the tracked-write
  log. The coroutine, when next driven, resumes from right after the
  `again()` call with the exact same world view.
- Drivers come in two flavours: strict (`step`, `resume`) raise on
  unexpected retry; tolerant (`try_step`, `try_resume`) report it
  back as `None` / `False`. The contract is *retry must be paired
  with a `try_*` driver upstream* — accidental retry surfaces as an
  exception at the immediate boundary.

This change required no relaxation of strict progression (item 11):
`again()` is not a phase yield, so it doesn't poke the suspension
table. The strict progression rule still rejects same-phase
re-entries — which is the right behavior, since `again()` is the
explicit channel for "stay where you are, but yield".

### 15. Concern and Batch are *handles* that track their own completion

Originally `resume(coro)` took a raw coroutine; only `step` had a
handle (`Batch(coro, storage, saved_phase)`). This was asymmetric and
had a sharp edge: calling `resume` on a concern that had already
returned would `RuntimeError: cannot reuse already awaited coroutine`.

We made both layers handle-based and unified the lifecycle:

- `Concern(coro)` wraps a concern coroutine.
- `Batch(coro, storage, ...)` wraps a batch coroutine.
- Both expose `done: bool`. The Driver flips it `True` whenever the
  wrapped coroutine terminates (return, exception, or forcible
  close). The user-side fast paths in `resume` / `step` /
  `try_resume` / `try_step` consult `handle.done` and short-circuit
  *before* yielding into the Driver. Result: re-driving a finished
  handle is a cheap no-op, semantically *"the coroutine completed,
  which means it made it past every phase"*.
- Both handles' `__del__` calls `coro.close()` if `not done` —
  silencing `RuntimeWarning: coroutine '...' was never awaited` for
  handles that get discarded before being driven. This replaces the
  old `spawn` primitive (which primed coroutines eagerly to silence
  the warning); handle lifetime now owns this cleanup.
- The root scheduler coroutine is wrapped in a private `_RootHandle`
  inside the Driver, so its stack contains exactly one shape —
  handles.

A small structural `_Handle` Protocol (internal only) gives the
Driver's internals a precise type for its stack entries and for
`_AdvanceRequest.child`, while public APIs keep their concrete
`Batch` / `Concern` types so call sites and `@overload` chains stay
specific.

---

### Design invariants that survived every refactor

These are properties to preserve when extending the runtime:

- **Single happens-before timeline.** One `LoopPhase`-shaped enum at
  a time. If a new participant needs phases, it uses the same enum
  values.
- **Phase determines view types.** Read at P sees `< P` only; write
  at P produces fields *at* P only. No "carry-over" fields with
  ambiguous lifetimes.
- **Field metadata is the source of truth.** Both the generated
  view types and the runtime tracked proxies derive from
  `phased_field` metadata. Never edit the generated block by hand.
- **Handles own lifecycle.** Every coroutine other than the root is
  wrapped in a handle that tracks `done` and self-closes on GC.
  No raw-coroutine APIs in the public surface.
- **Drivers come in strict / tolerant pairs.** Strict raises on
  unexpected retry; tolerant reports. Pair `again()` with the
  tolerant counterpart at the immediate caller.
- **The Driver is policy-free.** It interprets request sentinels,
  manages a stack, runs in an isolated context. Scheduling policy
  belongs to the top-level coroutine.
- **No `asyncio`.** The executor's outer loop is the event loop.
