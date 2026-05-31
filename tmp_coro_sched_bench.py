# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import contextvars
import dataclasses
import gc
import statistics
import time
from contextlib import asynccontextmanager
from enum import IntEnum
from typing import Any, Coroutine, Optional


def phased_field(phase: IntEnum, default: Any = None) -> Any:
    return dataclasses.field(default=default, metadata={"phase": phase})


class BatchPhase(IntEnum):
    SCHEDULE_0 = 0
    RESOURCE_PREP_1 = 1
    FORWARD_2 = 2
    SAMPLE_3 = 3
    STATE_UPD_4 = 4
    SYNC_EVT_5 = 5
    HANDOFF_6 = 6
    APPLY_7 = 7
    RESPOND_8 = 8
    FINALIZE_9 = 9


@dataclasses.dataclass
class BatchStorage:
    previous_tensors_device: object = phased_field(BatchPhase.SCHEDULE_0)
    sample_state: object = phased_field(BatchPhase.SAMPLE_3)


@dataclasses.dataclass
class _Pool:

    def is_drained(self) -> bool:
        return True


@dataclasses.dataclass
class _Dist:
    pp_size: int = 1


@dataclasses.dataclass
class _Services:
    pool: _Pool = dataclasses.field(default_factory=_Pool)
    dist: _Dist = dataclasses.field(default_factory=_Dist)


@dataclasses.dataclass
class _Port:
    is_shutdown: bool = False


@dataclasses.dataclass
class Context:
    iters: int
    skipping: bool
    port: _Port = dataclasses.field(default_factory=_Port)
    svc: _Services = dataclasses.field(default_factory=_Services)


class _ScheduleConcern:

    def waiting_queue_empty(self) -> bool:
        return True

    async def handle_batch(self, ctx: Context) -> None:
        if ctx.skipping:
            return
        await enter_phase(BatchPhase.SCHEDULE_0)
        await enter_phase(BatchPhase.FINALIZE_9)


class _ResourceConcern:

    async def handle_batch(self, ctx: Context) -> None:
        if ctx.skipping:
            return
        await enter_phase(BatchPhase.RESOURCE_PREP_1)
        await enter_phase(BatchPhase.RESPOND_8)


class _ForwardConcern:

    async def handle_batch(self, ctx: Context) -> None:
        if ctx.skipping:
            return
        await enter_phase(BatchPhase.FORWARD_2)


class _SampleConcern:

    async def handle_batch(self, ctx: Context) -> None:
        if ctx.skipping:
            return
        await enter_phase(BatchPhase.SAMPLE_3)
        await enter_phase(BatchPhase.APPLY_7)


class _StateAdvanceConcern:

    async def handle_batch(self, ctx: Context) -> None:
        if ctx.skipping:
            return
        await enter_phase(BatchPhase.STATE_UPD_4)


class _ResponseConcern:

    async def handle_batch(self, ctx: Context) -> None:
        if ctx.skipping:
            return
        await enter_phase(BatchPhase.RESPOND_8)


@dataclasses.dataclass
class Concerns:
    schedule: _ScheduleConcern = dataclasses.field(default_factory=_ScheduleConcern)
    resource: _ResourceConcern = dataclasses.field(default_factory=_ResourceConcern)
    forward: _ForwardConcern = dataclasses.field(default_factory=_ForwardConcern)
    sample: _SampleConcern = dataclasses.field(default_factory=_SampleConcern)
    state_advance: _StateAdvanceConcern = dataclasses.field(default_factory=_StateAdvanceConcern)
    response: _ResponseConcern = dataclasses.field(default_factory=_ResponseConcern)
    ring_broadcast: None = None


async def batch_body(ctx: Context, crn: Concerns) -> None:
    schedule = Concern(crn.schedule.handle_batch(ctx))
    resource = Concern(crn.resource.handle_batch(ctx))
    forward = Concern(crn.forward.handle_batch(ctx))
    sample = Concern(crn.sample.handle_batch(ctx))
    state_advance = Concern(crn.state_advance.handle_batch(ctx))
    response = Concern(crn.response.handle_batch(ctx))
    ring_broadcast = None

    async with batch_phase(BatchPhase.SCHEDULE_0):
        await resume(schedule)

    async with batch_phase(BatchPhase.RESOURCE_PREP_1):
        await resume(resource)

    async with batch_phase(BatchPhase.FORWARD_2):
        await resume(forward)

    async with batch_phase(BatchPhase.SAMPLE_3):
        await resume(sample)

    async with batch_phase(BatchPhase.STATE_UPD_4):
        await resume(state_advance)

    async with batch_phase(BatchPhase.SYNC_EVT_5):
        await resume(ring_broadcast)

    async with batch_phase(BatchPhase.HANDOFF_6):
        for _ in range(ctx.svc.dist.pp_size - 2):
            if await try_resume(ring_broadcast):
                break
            await again()
        else:
            await resume(ring_broadcast)

    async with batch_phase(BatchPhase.APPLY_7):
        await resume(sample)

    async with batch_phase(BatchPhase.RESPOND_8):
        await resume(response)
        await resume(resource)

    async with batch_phase(BatchPhase.FINALIZE_9):
        await resume(schedule)
        await resume(ring_broadcast)


async def scheduler_iter_overlap(ctx: Context, crn: Concerns) -> None:
    previous: Optional[Batch] = None
    previous_view = None

    for it in range(ctx.iters + 1):
        more_to_admit = it < ctx.iters
        if not more_to_admit and previous is None:
            return

        current: Optional[Batch] = None
        if more_to_admit:
            storage = BatchStorage()
            current = Batch(batch_body(ctx, crn), storage, idx=it)

        _, w_sched = await step(current, through=BatchPhase.SCHEDULE_0)
        if w_sched is not None and previous is not None and previous_view.sample_state is not None:
            w_sched.previous_tensors_device = previous_view.sample_state.device
        await step(current, through=BatchPhase.FORWARD_2)
        await step(previous, through=BatchPhase.APPLY_7)
        r_sample, _ = await step(current, through=BatchPhase.SAMPLE_3)
        await step(previous, through=BatchPhase.FINALIZE_9)
        await step(current, through=BatchPhase.STATE_UPD_4)
        previous, previous_view = current, r_sample


_asyncio_active_storage: contextvars.ContextVar[Optional[object]] = contextvars.ContextVar(
    "_asyncio_active_storage", default=None
)
_asyncio_active_phase: contextvars.ContextVar[Optional[IntEnum]] = contextvars.ContextVar(
    "_asyncio_active_phase", default=None
)
_asyncio_active_handle: contextvars.ContextVar[Optional["_AsyncioBaseHandle"]] = contextvars.ContextVar(
    "_asyncio_active_handle", default=None
)


class _AsyncioBaseHandle:

    def __init__(self, coro: Coroutine[Any, Any, Any]) -> None:
        self.coro = coro
        self.done = False
        self.exception: Optional[BaseException] = None
        self.target: Optional[IntEnum] = None
        self.wait_phase: Optional[IntEnum] = None
        self.last_phase: Optional[IntEnum] = None
        self.changed = asyncio.Event()
        self.progress = asyncio.Event()
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        token = _asyncio_active_handle.set(self)
        try:
            await self.coro
        except BaseException as exc:
            self.exception = exc
        finally:
            _asyncio_active_handle.reset(token)
            self.done = True
            self.progress.set()


class _AsyncioBatch(_AsyncioBaseHandle):

    def __init__(self, coro: Coroutine[Any, Any, Any], storage: object, idx: int = 0) -> None:
        self.storage = storage
        self.idx = idx
        super().__init__(coro)

    async def _run(self) -> None:
        storage_token = _asyncio_active_storage.set(self.storage)
        phase_token = _asyncio_active_phase.set(None)
        handle_token = _asyncio_active_handle.set(self)
        try:
            await self.coro
        except BaseException as exc:
            self.exception = exc
        finally:
            _asyncio_active_handle.reset(handle_token)
            _asyncio_active_phase.reset(phase_token)
            _asyncio_active_storage.reset(storage_token)
            self.done = True
            self.progress.set()


class _AsyncioConcern(_AsyncioBaseHandle):
    pass


class _AsyncioDriver:

    def __init__(self, main: Coroutine[Any, Any, Any]) -> None:
        self.main = main

    def run(self) -> None:
        asyncio.run(self.main)


def _asyncio_raise_if_failed(handle: _AsyncioBaseHandle) -> None:
    if handle.exception is not None:
        raise handle.exception


async def _asyncio_advance(handle: _AsyncioBaseHandle, through: IntEnum) -> None:
    if handle.done:
        _asyncio_raise_if_failed(handle)
        return

    handle.target = through
    handle.changed.set()
    while True:
        _asyncio_raise_if_failed(handle)
        if handle.done:
            return
        if handle.wait_phase is not None and handle.wait_phase > through:
            return
        await handle.progress.wait()
        handle.progress.clear()


async def _asyncio_enter_phase(phase: IntEnum) -> tuple[object, object]:
    handle = _asyncio_active_handle.get()
    storage = _asyncio_active_storage.get()
    if handle is None or storage is None:
        raise RuntimeError("enter_phase() called outside an asyncio batch/concern task")
    if handle.last_phase is not None and phase <= handle.last_phase:
        raise RuntimeError(
            f"enter_phase({phase.name}) does not strictly advance from {handle.last_phase.name}"
        )

    handle.last_phase = phase
    handle.wait_phase = phase
    handle.progress.set()

    while handle.target is None or phase > handle.target:
        await handle.changed.wait()
        handle.changed.clear()

    return storage, storage


@asynccontextmanager
async def _asyncio_batch_phase(phase: IntEnum) -> Any:
    token = _asyncio_active_phase.set(phase)
    try:
        yield await _asyncio_enter_phase(phase)
    finally:
        _asyncio_active_phase.reset(token)


async def _asyncio_again() -> None:
    raise RuntimeError("again() is not used by this single-rank overlap microbenchmark")


async def _asyncio_resume(child: Optional[_AsyncioConcern]) -> None:
    if child is None:
        return
    phase = _asyncio_active_phase.get()
    if phase is None:
        raise RuntimeError("resume() called outside batch_phase()")
    await _asyncio_advance(child, phase)


async def _asyncio_try_resume(child: Optional[_AsyncioConcern]) -> bool:
    if child is None:
        return True
    await _asyncio_resume(child)
    return True


async def _asyncio_step(
    handle: Optional[_AsyncioBatch], *, through: IntEnum
) -> tuple[object, Optional[object]]:
    if handle is None:
        return None, None
    if handle.done:
        _asyncio_raise_if_failed(handle)
        return handle.storage, None

    await _asyncio_advance(handle, through)
    _asyncio_raise_if_failed(handle)
    if handle.done:
        return handle.storage, None
    return handle.storage, handle.storage


_DRIVER_RUNTIME: Optional[dict[str, object]] = None


def _install_driver_runtime() -> None:
    global _DRIVER_RUNTIME
    global Batch, Concern, Driver, again, batch_phase, enter_phase, resume, step, try_resume

    if _DRIVER_RUNTIME is None:
        from tensorrt_llm._torch.pyexecutor import coroutines

        _DRIVER_RUNTIME = {
            "Batch": coroutines.Batch,
            "Concern": coroutines.Concern,
            "Driver": coroutines.Driver,
            "again": coroutines.again,
            "batch_phase": coroutines.batch_phase,
            "enter_phase": coroutines.enter_phase,
            "resume": coroutines.resume,
            "step": coroutines.step,
            "try_resume": coroutines.try_resume,
        }

    Batch = _DRIVER_RUNTIME["Batch"]
    Concern = _DRIVER_RUNTIME["Concern"]
    Driver = _DRIVER_RUNTIME["Driver"]
    again = _DRIVER_RUNTIME["again"]
    batch_phase = _DRIVER_RUNTIME["batch_phase"]
    enter_phase = _DRIVER_RUNTIME["enter_phase"]
    resume = _DRIVER_RUNTIME["resume"]
    step = _DRIVER_RUNTIME["step"]
    try_resume = _DRIVER_RUNTIME["try_resume"]


def _install_asyncio_runtime() -> None:
    global Batch, Concern, Driver, again, batch_phase, enter_phase, resume, step, try_resume

    Batch = _AsyncioBatch
    Concern = _AsyncioConcern
    Driver = _AsyncioDriver
    again = _asyncio_again
    batch_phase = _asyncio_batch_phase
    enter_phase = _asyncio_enter_phase
    resume = _asyncio_resume
    step = _asyncio_step
    try_resume = _asyncio_try_resume


def _install_runtime(runtime: str) -> None:
    if runtime == "driver":
        _install_driver_runtime()
    elif runtime == "asyncio":
        _install_asyncio_runtime()
    else:
        raise ValueError(f"unknown runtime: {runtime}")


def run_once(iters: int, mode: str, runtime: str) -> int:
    _install_runtime(runtime)
    ctx = Context(iters=iters, skipping=mode != "phase")
    crn = Concerns()
    start = time.perf_counter_ns()
    Driver(scheduler_iter_overlap(ctx, crn)).run()
    return time.perf_counter_ns() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=100_000)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--mode", choices=("phase", "empty"), default="phase")
    parser.add_argument("--runtime", choices=("driver", "asyncio"), default="driver")
    args = parser.parse_args()

    gc.disable()
    try:
        for _ in range(args.warmup):
            run_once(args.iters, args.mode, args.runtime)
        samples = [
            run_once(args.iters, args.mode, args.runtime)
            for _ in range(args.repeat)
        ]
    finally:
        gc.enable()

    best = min(samples)
    median = statistics.median(samples)
    print(f"runtime={args.runtime} mode={args.mode} iters={args.iters} repeat={args.repeat}")
    print(f"best_ns={best} median_ns={median}")
    print(f"best_iter_per_sec={args.iters / best * 1e9:,.0f}")
    print(f"median_iter_per_sec={args.iters / median * 1e9:,.0f}")
    print(f"best_ns_per_iter={best / args.iters:.0f}")
    print(f"median_ns_per_iter={median / args.iters:.0f}")


if __name__ == "__main__":
    main()
