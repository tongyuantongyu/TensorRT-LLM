# Coroutine scheduling overhead measurement
_Exported on 5/28/2026 at 18:07:39 GMT+8 from Cursor (3.5.38)_

---

**User**

@tensorrt_llm/_torch/pyexecutor/coroutines.py @tensorrt_llm/_torch/pyexecutor/py_executor_coro.py 

Microbench the overhead of using coroutines to schedule:

In a temporary test file, mirror the definition of `scheduler_iter_overlap` and its subcoroutines, but make the their body empty. Time the speed it can spinning the iteration loop at.

---

**Cursor**

I’ll set up a small temporary microbenchmark that mirrors the coroutine scheduling shape with no-op bodies, then run it to measure iteration throughput. First I’m going to inspect the existing coroutine definitions so the benchmark matches the real structure closely.

I found the overlap scheduler entry point and the coroutine runtime primitives. Next I’m reading the actual scheduler body and driver behavior so the no-op benchmark measures the coroutine machinery, not unrelated executor work.

The steady-state overlap path creates one batch per iteration, drives current through `SCHEDULE_0`, `FORWARD_2`, `SAMPLE_3`, `STATE_UPD_4`, and drains the previous batch through `APPLY_7` and `FINALIZE_9`. I’m going to benchmark that same shape with fixed iteration counts and no payload work in the concerns.

I added `tmp_coro_sched_bench.py` with the overlap scheduler shape and two no-payload modes: `phase` keeps the concern phase waits, while `empty` makes concern coroutines return immediately. I’ll smoke-run it first, then run a larger sample for the actual number.

The smoke run hit the system Python missing `libpython3.12.so` during TensorRT-LLM package preload, so I’m switching to the local `trtllm` Conda Python from the project notes for the benchmark run.

The Conda command didn’t actually expand `CONDA_PREFIX` for the executable path because it was set inline in the same shell command. I’ll rerun with `CONDA_PREFIX` exported first so it uses the intended environment.

---

**User**

You cannot run inside sandbox

---

**Cursor**

Got it. The smoke run was blocked by sandbox restrictions around MPI/socket initialization, so I’ll rerun the benchmark outside the sandbox and then clean up the temporary file.

---

**User**

Perf number confirmed.

Now test the overhead if using `asyncio` to schedule. You should provide parallel implementations for @tmp_coro_sched_bench.py:14-25  that works with asyncio, but they should make `scheduler_iter_overlap` and its callees in this file work as is. You can decide how to schedule the `scheduler_iter_overlap` coroutine whatever you want.

Hint: automatically `asyncio.create_task` the batch and concern coroutines. Use `asyncio.Event` to block the coroutines at `enter_phase`. Use contextvars to pass data without changing the API shape.

---

**Cursor**

I’ll extend the temporary benchmark with an asyncio-backed runtime that exposes the same `Batch`, `Concern`, `Driver`, `enter_phase`, `batch_phase`, `resume`, `step`, and `try_resume` names, so the existing overlap scheduler and batch body can run unchanged. Then I’ll run comparable `phase` and `empty` measurements outside the sandbox.

I’m going to keep the benchmark’s scheduler, `batch_body`, and concern call sites intact, and swap only the runtime API layer behind those names. The new `--runtime asyncio` path will use tasks plus phase gates, while `--runtime driver` can still use the existing TRT-LLM coroutine driver for comparison.

The file now defers the TRT-LLM driver import until `--runtime driver`, so the asyncio benchmark can run without importing the whole package. I’ll smoke-test the asyncio path first, then run the larger timing sample outside the sandbox.

The asyncio path is working and is much slower than the custom driver. I’m doing one small `--runtime driver` smoke run too, to make sure the deferred import still preserves the original benchmark path after the edits.

The 20k-iteration samples were stable: asyncio phase-gated scheduling is around `110 us/iter`, and empty concern bodies are around `63 us/iter`. I’ll run one longer phase sample to reduce short-run noise before I summarize.

Implemented `--runtime asyncio` in `tmp_coro_sched_bench.py` while keeping `scheduler_iter_overlap` and its callees using the same API names.

The asyncio runtime:
- Auto-creates tasks for `Batch` and `Concern`.
- Uses `asyncio.Event` for phase gates.
- Uses `contextvars` for active storage/phase/handle.
- Keeps original driver path as `--runtime driver`.

Results outside sandbox:

`--runtime asyncio --mode phase --iters 100000 --repeat 3 --warmup 1`
- Best: `9,406 iter/s`
- Best overhead: `106,314 ns/iter`
- Median overhead: `106,343 ns/iter`

`--runtime asyncio --mode empty --iters 20000 --repeat 5 --warmup 1`
- Best: `16,002 iter/s`
- Best overhead: `62,492 ns/iter`
- Median overhead: `62,957 ns/iter`

Also smoke-tested `--runtime driver`; it still runs. Lints are clean for `tmp_coro_sched_bench.py`.

---

**User**

Test and report driver runtime result as well.

---

**Cursor**

I’ll run the same benchmark with `--runtime driver` outside the sandbox, using the local `trtllm` Conda Python so it can import the TensorRT-LLM coroutine runtime. I’ll collect both `phase` and `empty` results for comparison with the asyncio numbers.

Driver runtime results:

`--runtime driver --mode phase --iters 100000 --repeat 3 --warmup 1`
- Best: `24,141 iter/s`
- Best overhead: `41,423 ns/iter`
- Median overhead: `41,429 ns/iter`

`--runtime driver --mode empty --iters 100000 --repeat 3 --warmup 1`
- Best: `30,733 iter/s`
- Best overhead: `32,539 ns/iter`
- Median overhead: `32,635 ns/iter`

For comparison, the prior asyncio phase result was about `106,314 ns/iter`, so asyncio is roughly `2.6x` slower on the phase-gated benchmark.

---

**User**

Make a table for comparison.

---

**Cursor**

| Runtime | Mode | Iter/s Best | Best ns/iter | Median ns/iter |
|---|---:|---:|---:|---:|
| driver | phase | 24,141 | 41,423 | 41,429 |
| driver | empty | 30,733 | 32,539 | 32,635 |
| asyncio | phase | 9,406 | 106,314 | 106,343 |
| asyncio | empty | 16,002 | 62,492 | 62,957 |

Asyncio vs driver:
- `phase`: `~2.57x` slower
- `empty`: `~1.92x` slower

