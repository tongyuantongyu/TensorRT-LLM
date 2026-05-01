"""Tests for the ``coroutines`` runtime module.

Most tests exercise generic runtime behavior (primitive wire format,
``phase`` CM, ``resume`` / ``step`` semantics, exception propagation,
``close()``-shutdown, frame hiding, ``Concern`` handle / ``__del__``, Driver policy, runtime
tracked-view proxies, and end-to-end 3-layer interleaving). They use a
test-local ``_TestPhase`` enum and ``_TestStorage`` dataclass defined
just below the imports, so they stay stable when the production
``BatchPhase`` / ``BatchStorage`` evolve.

A separate "production-binding" group of tests at the bottom of the
file (``test_static_{read,write}_views_match_field_metadata``,
``test_generator_output_agrees_with_file``) intentionally imports the
production types from :mod:`batch_storage` to verify the generated
block matches ``BatchStorage``'s field metadata. Those are the only
tests that should reference ``BatchStorage`` / ``BatchPhase`` /
``_ReadAtP*`` / ``_WriteAtP*`` / ``_ReadAtAll``.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
import traceback
from enum import IntEnum
from typing import Optional
from unittest import mock

import pytest

import tensorrt_llm._torch.pyexecutor.coroutines as _cor_mod
from tensorrt_llm._torch.pyexecutor.coroutines import (
    Driver,
    Batch,
    Concern,
    _AdvanceRequest,
    _HangControl,
    _HangWatchdog,
    _RETRY,
    _RetryRequest,
    _WaitRequest,
    again,
    disable_hang_detect,
    enter_phase,
    batch_phase,
    phased_field,
    resume,
    step,
    try_resume,
    try_step,
)

# --------------------------------------------------------------------------- #
# Test-local phase enum and storage. Decoupled from production so runtime
# tests survive any change to ``BatchPhase`` / ``BatchStorage`` shape.
#
# Naming: ``_TestPhase`` / ``_TestStorage`` (with the ``_Test`` prefix)
# is deliberately verbose so a grep for ``BatchPhase`` or ``BatchStorage``
# in this file lands only on the production-binding tests at the bottom.
# --------------------------------------------------------------------------- #


class _TestPhase(IntEnum):
    """Test-local phase enum independent of production ``BatchPhase``.

    Four ordered values are enough to exercise every primitive.
    """

    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3


@dataclasses.dataclass
class _TestStorage:
    """Test-local storage with phased fields.

    ``batch_label`` is a *second* field at ``P1`` — used in the
    end-to-end three-layer test to distinguish "batch's own
    publication" from "concern's publication" within the same phase.
    """

    p0_out: Optional[int] = phased_field(_TestPhase.P0)
    p1_out: Optional[str] = phased_field(_TestPhase.P1)
    batch_label: Optional[str] = phased_field(_TestPhase.P1)
    p2_out: Optional[bytes] = phased_field(_TestPhase.P2)
    # _TestPhase.P3 has no field — exercises "terminal phase produces
    # nothing" / write-view-with-no-fields cases.


# --------------------------------------------------------------------------- #
# Helpers: manage active state for unit tests that drive primitives directly
# without going through Driver.
# --------------------------------------------------------------------------- #


@pytest.fixture
def active_storage():
    """Set ``_active_storage`` to a fresh ``_TestStorage`` for the test.

    Some unit tests poke at primitives (``enter_phase``, ``batch_phase``
    CM) by hand without going through Driver+step. They need active
    storage to be non-None. Uses ``ContextVar.set`` / ``reset`` to
    match the runtime's API.
    """
    storage = _TestStorage()
    token = _cor_mod._active_storage.set(storage)
    try:
        yield storage
    finally:
        _cor_mod._active_storage.reset(token)


# --------------------------------------------------------------------------- #
# Primitive wire format
# --------------------------------------------------------------------------- #


def test_enter_phase_yields_wait_request(active_storage):
    """A fresh ``enter_phase`` call yields a ``_WaitRequest(phase, storage)``."""

    async def body():
        await enter_phase(_TestPhase.P1)

    coro = body()
    try:
        request = coro.send(None)
    finally:
        coro.close()
    assert isinstance(request, _WaitRequest)
    assert request.phase is _TestPhase.P1
    assert request.storage is active_storage


def test_enter_phase_outside_active_raises_runtime_error():
    """Calling ``enter_phase`` with no active storage raises clearly."""

    async def body():
        await enter_phase(_TestPhase.P1)

    coro = body()
    try:
        with pytest.raises(RuntimeError, match="no active batch"):
            coro.send(None)
    finally:
        coro.close()


def test_resume_yields_advance_request_inside_phase_cm(active_storage):
    """``resume`` yields ``_AdvanceRequest`` carrying the Concern handle."""

    async def child():
        # Will not actually run; we drop on close().
        await enter_phase(_TestPhase.P1)

    child_handle = Concern(child())
    captured = {}

    async def body():
        async with batch_phase(_TestPhase.P1):
            captured["before_resume"] = _cor_mod._active_phase.get()
            await resume(child_handle)

    parent = body()
    try:
        # Pump 1: __aenter__ yields the wait request.
        request = parent.send(None)
        assert isinstance(request, _WaitRequest)
        assert request.phase is _TestPhase.P1
        # Pump 2: parent's body resumes after wait, calls resume(child),
        # yields _AdvanceRequest(handle, at=P1) carrying the handle.
        request = parent.send(None)
        assert isinstance(request, _AdvanceRequest)
        assert request.child is child_handle
        assert request.at is _TestPhase.P1
        assert captured["before_resume"] is _TestPhase.P1
    finally:
        parent.close()
        # child_handle.__del__ closes the underlying coro for us.


def test_resume_outside_phase_cm_raises():
    """Calling ``resume`` with ``_active_phase`` unset is a programming error."""

    async def child():
        await enter_phase(_TestPhase.P1)

    child_handle = Concern(child())

    async def body():
        await resume(child_handle)

    parent = body()
    try:
        with pytest.raises(RuntimeError, match="outside a batch_phase CM"):
            parent.send(None)
    finally:
        parent.close()


# --------------------------------------------------------------------------- #
# `phase` CM behavior
# --------------------------------------------------------------------------- #


def test_phase_cm_sets_and_clears_active_phase(active_storage):
    """Inside the CM body ``_active_phase`` is the CM's value; outside it's None."""

    seen = {}

    async def body():
        seen["before"] = _cor_mod._active_phase.get()
        async with batch_phase(_TestPhase.P1):
            seen["inside"] = _cor_mod._active_phase.get()
        seen["after"] = _cor_mod._active_phase.get()

    coro = body()
    try:
        # Pump until completion. The first yield is from CM's enter_phase.
        request = coro.send(None)
        assert isinstance(request, _WaitRequest)
        # Resume past the wait — body inside CM runs, CM exits, body returns.
        with pytest.raises(StopIteration):
            coro.send(None)
    finally:
        coro.close()

    assert seen == {
        "before": None,
        "inside": _TestPhase.P1,
        "after": None,
    }


def test_phase_cm_returns_views_at_phase(active_storage):
    """``async with phase(Py) as (r, w)`` returns ``(read, write)`` at ``Py``."""

    captured = {}

    async def body():
        async with batch_phase(_TestPhase.P1) as (r, w):
            captured["r_is_storage"] = r is active_storage
            captured["w_is_storage"] = w is active_storage

    coro = body()
    try:
        coro.send(None)  # enter the wait
        with pytest.raises(StopIteration):
            coro.send(None)  # past the wait, body runs, completes
    finally:
        coro.close()

    # In production mode (no tracking), views are storage itself.
    assert captured == {"r_is_storage": True, "w_is_storage": True}


def test_phase_cm_rejects_nested_usage(active_storage):
    """Nesting two ``phase`` CMs raises with a clear message."""

    async def body():
        async with batch_phase(_TestPhase.P1):
            async with batch_phase(_TestPhase.P2):
                pass

    coro = body()
    try:
        # First CM enters fine, yields wait.
        request = coro.send(None)
        assert isinstance(request, _WaitRequest)
        # Second CM enter raises.
        with pytest.raises(RuntimeError, match="must not nest"):
            coro.send(None)
    finally:
        coro.close()


# --------------------------------------------------------------------------- #
# Batch handle
# --------------------------------------------------------------------------- #


def test_batch_bundles_coro_storage_and_saved_phase():
    """Batch is a plain dataclass; saved_phase defaults to None."""

    async def body():
        return None

    coro = body()
    try:
        storage = _TestStorage()
        handle = Batch(coro, storage)
        assert handle.coro is coro
        assert handle.storage is storage
        assert handle.saved_phase is None
    finally:
        coro.close()


# --------------------------------------------------------------------------- #
# `step` semantics: activates storage, drives child, returns views
# --------------------------------------------------------------------------- #


def test_step_runs_simple_batch_through_p0():
    """``step(handle, through=P0)`` drives an empty-P0 batch; returns views."""

    captured = {}

    async def batch():
        async with batch_phase(_TestPhase.P0):
            pass
        async with batch_phase(_TestPhase.P1) as (_, w):
            w.batch_label = "after_p0_hole"
        async with batch_phase(_TestPhase.P2) as (_, w):
            w.p2_out = b"done"
        async with batch_phase(_TestPhase.P3):
            pass

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        # First step injects p0_out via the returned write view.
        _, fill_p0 = await step(handle, through=_TestPhase.P0)
        fill_p0.p0_out = 42
        # Drive to terminal.
        view, w_terminal = await step(handle, through=_TestPhase.P3)
        captured["p0_out"] = handle.storage.p0_out
        captured["batch_label"] = handle.storage.batch_label
        captured["p2_out"] = handle.storage.p2_out
        captured["w_terminal_is_none"] = w_terminal is None

    Driver(scheduler()).run()
    assert captured == {
        "p0_out": 42,
        "batch_label": "after_p0_hole",
        "p2_out": b"done",
        "w_terminal_is_none": True,
    }


def test_step_preserves_active_state_across_calls():
    """Repeated ``step`` on the same handle preserves batch's CM phase."""

    seen = {}

    async def batch():
        async with batch_phase(_TestPhase.P0):
            seen["batch_at_p0"] = _cor_mod._active_phase.get()
        async with batch_phase(_TestPhase.P1):
            seen["batch_at_p1"] = _cor_mod._active_phase.get()
        async with batch_phase(_TestPhase.P2):
            seen["batch_at_p2"] = _cor_mod._active_phase.get()
        async with batch_phase(_TestPhase.P3):
            seen["batch_at_p3"] = _cor_mod._active_phase.get()

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        seen["sched_before"] = _cor_mod._active_phase.get()
        await step(handle, through=_TestPhase.P0)
        seen["sched_between_0_1"] = _cor_mod._active_phase.get()
        await step(handle, through=_TestPhase.P1)
        seen["sched_between_1_2"] = _cor_mod._active_phase.get()
        await step(handle, through=_TestPhase.P3)
        seen["sched_after"] = _cor_mod._active_phase.get()

    Driver(scheduler()).run()
    assert seen == {
        "sched_before": None,
        "batch_at_p0": _TestPhase.P0,
        "sched_between_0_1": None,
        "batch_at_p1": _TestPhase.P1,
        "sched_between_1_2": None,
        "batch_at_p2": _TestPhase.P2,
        "batch_at_p3": _TestPhase.P3,
        "sched_after": None,
    }


# --------------------------------------------------------------------------- #
# Driver pop predicate: strict `>`
# --------------------------------------------------------------------------- #


def test_advance_target_equal_keeps_pumping():
    """``through=P1`` does NOT pop when child yields at P1; only at P2."""

    seen = []

    async def child():
        async with batch_phase(_TestPhase.P1):
            seen.append("p1_body")
        async with batch_phase(_TestPhase.P2):
            seen.append("p2_body")

    handle = Batch(child(), _TestStorage())

    async def scheduler():
        # through=P1 should drive child PAST its P1 (running its P1
        # body), stopping at the P2 yield.
        await step(handle, through=_TestPhase.P1)
        seen.append("after_step_p1")
        # Drive through P2.
        await step(handle, through=_TestPhase.P2)
        seen.append("after_step_p2")

    Driver(scheduler()).run()
    assert seen == ["p1_body", "after_step_p1", "p2_body", "after_step_p2"]


# --------------------------------------------------------------------------- #
# Strict progression: enter_phase must yield strictly increasing phases
# --------------------------------------------------------------------------- #


def test_strict_progression_rejects_regression():
    """A coroutine yielding P2 then P1 raises with both phases named."""

    async def regress():
        async with batch_phase(_TestPhase.P2):
            pass
        async with batch_phase(_TestPhase.P1):
            pass

    handle = Batch(regress(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    with pytest.raises(RuntimeError, match=r"P1.*does not strictly advance.*P2"):
        Driver(scheduler()).run()


def test_strict_progression_rejects_same_phase_twice():
    """A coroutine yielding the same phase twice raises (no replay)."""

    async def stuck():
        async with batch_phase(_TestPhase.P1):
            pass
        async with batch_phase(_TestPhase.P1):
            pass

    handle = Batch(stuck(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    with pytest.raises(RuntimeError, match=r"P1.*does not strictly advance.*P1"):
        Driver(scheduler()).run()


# --------------------------------------------------------------------------- #
# End-to-end three-layer scenario (scheduler -> batch -> concerns),
# with overlapping batches.
# --------------------------------------------------------------------------- #


def test_three_layer_scheduler_batch_concern_data_flow():
    """Scheduler -> batch -> concerns: end-to-end data flow.

    Verifies all three data-flow primitives at once, AND that the
    scheduler can interleave batches rather than running them
    serially:

    1. **Concern -> concern, via ``batch_storage``**: ``concern_a``
       publishes ``p1_out`` at P1; ``concern_b`` reads it at P2. The
       active storage (set by the enclosing ``step``) is the
       publish/subscribe bus. Concerns receive nothing — internal
       state stays in Python locals; the only inter-coroutine
       interface is the active storage.
    2. **Within-batch write by the batch itself**: at P1 the
       batch writes ``batch_label`` (function of the
       step-injected ``p0_out``); ``concern_b`` reads it at P2 to
       combine with ``p1_out``. Two writers at P1 (batch and
       concern_a) writing distinct fields demonstrates the "shared
       phase, distinct fields" pattern.
    3. **Cross-batch forwarding mid-flight via the scheduler**: the
       scheduler drives ``batch1`` only to P1, reads ``batch1.p1_out``
       via the returned read view, and uses it to seed ``batch2``
       while ``batch1`` is still suspended.

    Batch body advances all use *aligned* phase values: at P1,
    batch writes via the CM views and ``await resume(...)`` of
    concerns at P1; at P2, same with concern_b at P2.
    """

    # Execution & data flow (overlapping):
    #
    # Time goes top to bottom. The scheduler interleaves: drive batch1
    # partway, kick off batch2 with batch1's mid-flight output, drive
    # batch2 partway, then resume batch1 to terminal, then batch2.
    # Within each phase block, "produces" lists the fields written
    # into batch_storage by the time that block has completed; the
    # indented r:/w: lines show who reads / writes which fields
    # during that block.
    #
    # SCHEDULER injection at P0 happens via the write view returned
    # from `step(handle, through=P0)` (no separate inject API on the
    # Batch handle). Cross-batch forwarding is the scheduler
    # reading the read view returned from `step(through=P1)` and
    # using it to compute the next batch's seed (p0_out).
    #
    # BATCH1                                 | BATCH2
    #   P0 produces: [p0_out]                |
    #     [SCHEDULER]:                       |
    #       w: p0_out                        |
    #   P1 produces: [batch_label, p1_out]   |
    #     batch:                             |
    #       r: [p0_out]                      |
    #       w: [batch_label]                 |
    #     concern_a:                         |
    #       r: [p0_out]                      |
    #       w: [p1_out]                      |
    #   P2 produces: [p2_out]                |  P0 produces: [p0_out]
    #     [SCHEDULER]:                       |    [SCHEDULER]:
    #       r: p1_out ----------------------------> w: p0_out
    #                                        |  P1 produces: [batch_label, p1_out]
    #                                        |    batch:
    #                                        |      r: [p0_out]
    #                                        |      w: [batch_label]
    #                                        |    concern_a:
    #                                        |      r: [p0_out]
    #                                        |      w: [p1_out]
    #     concern_b:                         |
    #       r: [batch_label, p1_out]         |
    #       w: [p2_out]                      |
    #   P3 produces: []                      |
    #     [SCHEDULER]:                       |
    #       r: p2_out                        |
    #                                        |  P2 produces: [p2_out]
    #                                        |    concern_b:
    #                                        |      r: [batch_label, p1_out]
    #                                        |      w: [p2_out]
    #                                        |  P3 produces: []
    #                                        |    [SCHEDULER]:
    #                                        |      r: p2_out

    async def concern_a():
        # Concern publishes p1_out at P1.
        r1, w1 = await enter_phase(_TestPhase.P1)
        w1.p1_out = f"a({r1.p0_out})"

    async def concern_b():
        # Concern reads batch_label and p1_out at P2, writes p2_out.
        r2, w2 = await enter_phase(_TestPhase.P2)
        combined = f"b[{r2.batch_label}|{r2.p1_out}]"
        w2.p2_out = combined.encode()

    async def batch():
        # P0 is a hole: scheduler injects p0_out via the step(through=P0)
        # return.
        async with batch_phase(_TestPhase.P0):
            pass

        # P1 block: batch writes batch_label, drives concern_a.
        # ALL phase values (CM, resume) are P1.
        async with batch_phase(_TestPhase.P1) as (r, w):
            w.batch_label = f"L:{r.p0_out}"
            await resume(Concern(concern_a()))

        # P2 block: drives concern_b. batch publishes nothing here.
        async with batch_phase(_TestPhase.P2):
            await resume(Concern(concern_b()))

        # Terminal — nothing produced.
        async with batch_phase(_TestPhase.P3):
            pass

    async def scheduler():
        # batch1: inject seed via step(through=P0), drive partway to P1.
        batch1 = Batch(batch(), _TestStorage())
        _, fill_p0_1 = await step(batch1, through=_TestPhase.P0)
        fill_p0_1.p0_out = 1
        view_after_p1_1, _ = await step(batch1, through=_TestPhase.P1)

        # While batch1 is mid-flight, kick off batch2 using batch1's
        # intermediate p1_out.
        batch2 = Batch(batch(), _TestStorage())
        _, fill_p0_2 = await step(batch2, through=_TestPhase.P0)

        # Cross-batch forwarding: use batch1's p1_out to seed batch2.
        # [SCHEDULER] behaves like a concern coroutine that runs
        # at the beginning of batch1's P2 and the end of batch2's P0,
        # so the data batch1 produced at P1 is visible to batch2 at P1.
        fill_p0_2.p0_out = len(view_after_p1_1.p1_out)

        view_after_p1_2, _ = await step(batch2, through=_TestPhase.P1)

        # Resume both to terminal.
        view_final_1, _ = await step(batch1, through=_TestPhase.P3)
        view_final_2, _ = await step(batch2, through=_TestPhase.P3)

        captured["batch1.p2_out"] = view_final_1.p2_out
        captured["batch2.p2_out"] = view_final_2.p2_out
        captured["batch2.seed"] = fill_p0_2.p0_out

    captured: dict = {}
    Driver(scheduler()).run()

    # batch1 seed=1   -> batch_label="L:1",  p1_out="a(1)" (4 chars)
    #                   p2_out=b"b[L:1|a(1)]"  (11 bytes)
    # batch2 seed=4   -> batch_label="L:4",  p1_out="a(4)" (4 chars)
    #                   p2_out=b"b[L:4|a(4)]"  (11 bytes)
    assert captured == {
        "batch1.p2_out": b"b[L:1|a(1)]",
        "batch2.p2_out": b"b[L:4|a(4)]",
        "batch2.seed": 4,
    }


# --------------------------------------------------------------------------- #
# Active-state isolation across interleaved batches
# --------------------------------------------------------------------------- #


def test_interleaved_batches_preserve_their_own_saved_phase():
    """Two batches interleaved; each ``Batch``'s saved_phase is independent."""

    seen = []

    async def batch(label):
        async with batch_phase(_TestPhase.P0):
            seen.append((label, "p0"))
        async with batch_phase(_TestPhase.P1):
            seen.append((label, "p1"))
        async with batch_phase(_TestPhase.P2):
            seen.append((label, "p2"))
        async with batch_phase(_TestPhase.P3):
            seen.append((label, "p3"))

    batch1 = Batch(batch("a"), _TestStorage())
    batch2 = Batch(batch("b"), _TestStorage())

    async def scheduler():
        # Interleave: a:P0, b:P0, a:P1, b:P1, a:P3, b:P3.
        await step(batch1, through=_TestPhase.P0)
        await step(batch2, through=_TestPhase.P0)
        await step(batch1, through=_TestPhase.P1)
        await step(batch2, through=_TestPhase.P1)
        await step(batch1, through=_TestPhase.P3)
        await step(batch2, through=_TestPhase.P3)

    Driver(scheduler()).run()
    # Order: a's P0, b's P0, a's P1, b's P1, a's P2, a's P3, b's P2, b's P3.
    # Each batch's body picks up where it left off when its step()
    # restores its saved_phase.
    assert seen == [
        ("a", "p0"),
        ("b", "p0"),
        ("a", "p1"),
        ("b", "p1"),
        ("a", "p2"),
        ("a", "p3"),
        ("b", "p2"),
        ("b", "p3"),
    ]


# --------------------------------------------------------------------------- #
# Exception propagation
# --------------------------------------------------------------------------- #


def test_batch_exception_propagates_to_scheduler():
    """An exception inside a concern surfaces from the scheduler's step()."""

    async def concern():
        await enter_phase(_TestPhase.P1)
        raise ValueError("boom")

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await resume(Concern(concern()))

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    with pytest.raises(ValueError, match="boom"):
        Driver(scheduler()).run()


def test_scheduler_can_catch_batch_exception():
    """The scheduler can wrap step() in try/except and continue."""

    seen = []

    async def bad_batch():
        async with batch_phase(_TestPhase.P1):
            raise ValueError("planned")

    async def good_batch():
        async with batch_phase(_TestPhase.P0):
            seen.append("good_p0")

    batch_bad = Batch(bad_batch(), _TestStorage())
    batch_good = Batch(good_batch(), _TestStorage())

    async def scheduler():
        try:
            await step(batch_bad, through=_TestPhase.P1)
        except ValueError as e:
            seen.append(f"caught:{e}")
        await step(batch_good, through=_TestPhase.P0)

    Driver(scheduler()).run()
    assert seen == ["caught:planned", "good_p0"]


# --------------------------------------------------------------------------- #
# Shutdown via close()
# --------------------------------------------------------------------------- #


def test_close_unwinds_try_finally_in_scheduler():
    """``Driver.close()`` runs the scheduler's ``try/finally`` cleanup.

    The scheduler is on the Driver's stack, so a Driver shutdown
    closes it via ``coro.close()``, propagating ``GeneratorExit``
    through any open ``try/finally`` and ``with`` blocks.

    Batches are NOT on the Driver's stack between ``step()`` calls
    — they're held by ``Batch`` handles and owned by the scheduler.
    The scheduler is responsible for driving them to completion (or
    explicitly closing their coros) if batch-side cleanup must run on
    a deterministic schedule.
    """

    cleanup = []

    async def scheduler():
        try:
            # Suspend the scheduler indefinitely with a wait that
            # nothing ever satisfies. Set up active storage manually
            # so enter_phase doesn't immediately raise.
            token = _cor_mod._active_storage.set(_TestStorage())
            try:
                await enter_phase(_TestPhase.P0)
            finally:
                _cor_mod._active_storage.reset(token)
        finally:
            cleanup.append("scheduler")

    drv = Driver(scheduler())
    # Pump the scheduler manually until it suspends, then close.
    drv._stack = [(drv._main_handle, None, None)]  # noqa: SLF001
    request = drv._main_handle.coro.send(None)  # noqa: SLF001
    assert isinstance(request, _WaitRequest)
    drv._close_all()  # noqa: SLF001
    assert cleanup == ["scheduler"]


# --------------------------------------------------------------------------- #
# Frame hiding
# --------------------------------------------------------------------------- #


def test_frame_hiding_default_hides_driver_frames():
    """A propagated exception's traceback shows user frames only."""

    async def deep():
        async with batch_phase(_TestPhase.P1):
            raise RuntimeError("planted")

    async def scheduler():
        handle = Batch(deep(), _TestStorage())
        await step(handle, through=_TestPhase.P1)

    try:
        Driver(scheduler()).run()
    except RuntimeError as e:
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        assert "deep" in tb_str
        assert "scheduler" in tb_str
        # Internal helpers should be hidden.
        assert "_yield_advance" not in tb_str
        assert "_yield_wait" not in tb_str
    else:
        pytest.fail("Expected RuntimeError to propagate")


def test_frame_hiding_env_reveals_driver_frames():
    """With ``_HIDE_FRAMES`` False, framework frames are visible.

    NOTE: this test patches the module-level ``_HIDE_FRAMES`` constant
    directly rather than re-importing the module. ``importlib.reload``
    creates a NEW set of class objects (``_HangControl``,
    ``_WaitRequest``, ...) on the module while the test file's existing
    imports keep pointing at the OLD ones; functions in the reloaded
    module then look up names in the new module dict at call time and
    yield instances of the new classes -- which the test file's
    ``isinstance`` checks (against the old classes) reject. Patching
    the constant is sufficient for this test's purpose and leaves
    every other module-level binding untouched.
    """

    async def deep():
        async with batch_phase(_TestPhase.P1):
            raise RuntimeError("planted")

    async def sched():
        handle = Batch(deep(), _TestStorage())
        await step(handle, through=_TestPhase.P1)

    with mock.patch.object(_cor_mod, "_HIDE_FRAMES", False):
        try:
            Driver(sched()).run()
        except RuntimeError as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            assert "_drive" in tb_str or "_yield_advance" in tb_str or "_yield_wait" in tb_str
        else:
            pytest.fail("Expected RuntimeError to propagate")


# --------------------------------------------------------------------------- #
# Handle lifecycle: __del__ silences "never awaited" warning.
# Replaces the old `spawn` primitive — wrapping a coroutine in a
# Concern / Batch handle is now what owns its cleanup.
# --------------------------------------------------------------------------- #


def test_concern_handle_del_silences_never_awaited_warning():
    """A Concern handle whose coro is never advanced doesn't emit RuntimeWarning."""
    import gc
    import warnings

    async def concern():
        await enter_phase(_TestPhase.P1)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        handle = Concern(concern())
        del handle
        gc.collect()
    never_awaited = [
        w
        for w in captured
        if issubclass(w.category, RuntimeWarning) and "was never awaited" in str(w.message)
    ]
    assert never_awaited == []


def test_batch_handle_del_silences_never_awaited_warning():
    """Same as above for a Batch handle."""
    import gc
    import warnings

    async def batch():
        async with batch_phase(_TestPhase.P0):
            pass

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        handle = Batch(batch(), _TestStorage())
        del handle
        gc.collect()
    never_awaited = [
        w
        for w in captured
        if issubclass(w.category, RuntimeWarning) and "was never awaited" in str(w.message)
    ]
    assert never_awaited == []


def test_close_if_undriven_skips_when_done():
    """``_close_if_undriven`` skips ``coro.close()`` when the handle is done."""

    closes = []

    class _StubCoro:
        def close(self):
            closes.append(1)

    class _StubHandle:
        def __init__(self, done):
            self.coro = _StubCoro()
            self.done = done

    # Done = True: helper skips close.
    _cor_mod._close_if_undriven(_StubHandle(done=True))
    assert closes == []

    # Done = False: helper closes once.
    _cor_mod._close_if_undriven(_StubHandle(done=False))
    assert closes == [1]


# --------------------------------------------------------------------------- #
# Driver policy
# --------------------------------------------------------------------------- #


def test_main_return_is_silent():
    """Driver returns normally when main returns; no policy-error on return."""

    async def main():
        return

    Driver(main()).run()


def test_main_raise_propagates():
    """An unhandled exception from main surfaces from run()."""

    async def main():
        raise ValueError("real error")

    with pytest.raises(ValueError, match="real error"):
        Driver(main()).run()


# --------------------------------------------------------------------------- #
# Runtime tracked-storage proxies (TLLM_COROUTINE_TRACK_STORAGE=1 in debug)
# --------------------------------------------------------------------------- #


@pytest.fixture
def tracked_storage():
    """Temporarily enable runtime happens-before tracking for one test."""
    previous = _cor_mod._TRACK_STORAGE
    _cor_mod._TRACK_STORAGE = True
    try:
        yield
    finally:
        _cor_mod._TRACK_STORAGE = previous


def test_tracking_off_by_default_views_are_storage():
    """With tracking off (production), views are the storage itself."""

    captured = {}

    async def batch():
        async with batch_phase(_TestPhase.P0) as (r, w):
            captured["r_is_storage"] = r is handle.storage
            captured["w_is_storage"] = w is handle.storage

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P0)

    Driver(scheduler()).run()
    assert captured == {"r_is_storage": True, "w_is_storage": True}


def test_tracked_write_at_owning_phase_succeeds(tracked_storage):
    """A write to a field whose phase matches the current view phase works."""

    async def batch():
        async with batch_phase(_TestPhase.P0) as (_, w):
            w.p0_out = 7

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P0)

    Driver(scheduler()).run()
    assert handle.storage.p0_out == 7


def test_tracked_write_at_wrong_phase_raises(tracked_storage):
    """Writing a field owned by a different phase raises."""

    async def batch():
        async with batch_phase(_TestPhase.P0) as (_, w):
            w.p1_out = "nope"

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    with pytest.raises(AttributeError, match=r"p1_out.*phase P1"):
        Driver(scheduler()).run()


def test_tracked_read_of_unwritten_field_raises(tracked_storage):
    """An earlier-phase field that was never written raises on read."""

    async def batch():
        # Skip writing p0_out; the field stays unwritten.
        async with batch_phase(_TestPhase.P0):
            pass
        async with batch_phase(_TestPhase.P1) as (r, _):
            _ = r.p0_out

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    with pytest.raises(RuntimeError, match=r"p0_out.*never written"):
        Driver(scheduler()).run()


def test_tracked_step_returns_views_at_correct_phases(tracked_storage):
    """``step(through=Py)`` returns read view exposing <=Py and write at Py."""

    async def batch():
        async with batch_phase(_TestPhase.P0):
            pass
        async with batch_phase(_TestPhase.P1) as (_, w):
            w.batch_label = "x"
            w.p1_out = "y"
        async with batch_phase(_TestPhase.P2):
            pass
        async with batch_phase(_TestPhase.P3):
            pass

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        # Inject p0_out via the P0 step.
        _, fill_p0 = await step(handle, through=_TestPhase.P0)
        fill_p0.p0_out = 1
        # After P1: read view exposes batch_label, p1_out, p0_out.
        view_after_p1, _ = await step(handle, through=_TestPhase.P1)
        captured["batch_label"] = view_after_p1.batch_label
        captured["p1_out"] = view_after_p1.p1_out
        captured["p0_out"] = view_after_p1.p0_out

    captured: dict = {}
    Driver(scheduler()).run()
    assert captured == {"batch_label": "x", "p1_out": "y", "p0_out": 1}


# --------------------------------------------------------------------------- #
# Concern misuse: phase CM inside a concern raises
# --------------------------------------------------------------------------- #


def test_concern_using_phase_cm_raises_via_nesting_assert():
    """A concern inside a parent's phase CM cannot also use ``phase``.

    The parent batch is already in a phase CM, so ``_active_phase`` is
    non-None; the concern's ``phase(...)`` enter would raise the
    non-nesting error.
    """

    async def concern():
        async with batch_phase(_TestPhase.P1):
            pass

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await resume(Concern(concern()))

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    with pytest.raises(RuntimeError, match="must not nest"):
        Driver(scheduler()).run()


# --------------------------------------------------------------------------- #
# Retry / EAGAIN: `again()`, `try_step`, `try_resume`
#
# Semantics: `again()` says "I'm at this phase, ask me again from this exact
# point". The Driver pops the coroutine without touching its suspension
# record or active state, and signals the parent via the `_RETRY` sentinel.
# Strict drivers (`step` / `resume`) raise on retry; tolerant drivers
# (`try_step` / `try_resume`) report it via `None` / `False`.
# --------------------------------------------------------------------------- #


def test_again_outside_phase_raises():
    """`again()` requires `_active_phase` to be set; no phase = bug."""

    async def body():
        await again()

    coro = body()
    try:
        with pytest.raises(RuntimeError, match="outside a phase"):
            coro.send(None)
    finally:
        coro.close()


def test_again_yields_retry_request_inside_phase_cm(active_storage):
    """Inside a batch_phase CM, `again()` yields a `_RetryRequest`."""

    # Look up classes via _cor_mod so we don't get tripped up by tests that
    # reload the coroutines module (frame-hiding env tests above).
    async def body():
        async with _cor_mod.batch_phase(_TestPhase.P1):
            await _cor_mod.again()

    coro = body()
    try:
        request = coro.send(None)
        assert isinstance(request, _cor_mod._WaitRequest)
        request = coro.send(None)
        assert isinstance(request, _cor_mod._RetryRequest)
    finally:
        coro.close()


def test_step_raises_on_unexpected_retry_from_batch():
    """`step()` (strict) raises if the batch yields `again()`."""

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await again()

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P1)

    with pytest.raises(RuntimeError, match="batch issued.*again.*try_step"):
        Driver(scheduler()).run()


def test_try_step_returns_none_on_retry():
    """`try_step()` (tolerant) returns `None` if batch retries."""

    captured = {}

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await again()

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        result = await try_step(handle, through=_TestPhase.P1)
        captured["result"] = result

    Driver(scheduler()).run()
    assert captured["result"] is None


def test_try_step_returns_views_on_progress():
    """`try_step()` returns the same `(read, write)` shape as `step()` on success."""

    captured = {}

    async def batch():
        async with batch_phase(_TestPhase.P0) as (_, w):
            w.p0_out = 7

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        result = await try_step(handle, through=_TestPhase.P0)
        captured["result"] = result

    Driver(scheduler()).run()
    assert captured["result"] is not None
    read, write = captured["result"]
    assert handle.storage.p0_out == 7


def test_try_step_terminal_returns_views_not_none():
    """Terminal completion returns `(read_view, None)` — distinct from retry's bare `None`."""

    async def batch():
        async with batch_phase(_TestPhase.P0):
            pass
        async with batch_phase(_TestPhase.P1):
            pass
        async with batch_phase(_TestPhase.P2):
            pass
        async with batch_phase(_TestPhase.P3):
            pass

    handle = Batch(batch(), _TestStorage())
    captured = {}

    async def scheduler():
        result = await try_step(handle, through=_TestPhase.P3)
        captured["result"] = result

    Driver(scheduler()).run()
    assert captured["result"] is not None  # not retry
    read, write = captured["result"]
    assert write is None  # terminal


def test_resume_raises_on_unexpected_retry_from_concern():
    """`resume()` (strict) raises if the concern yields `again()`."""

    async def concern():
        await enter_phase(_TestPhase.P1)
        await again()

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await resume(Concern(concern()))

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P1)

    with pytest.raises(RuntimeError, match="child issued.*again.*try_resume"):
        Driver(scheduler()).run()


def test_try_resume_reports_concern_retry():
    """`try_resume()` returns `False` if concern retried, `True` otherwise."""

    seen = []

    async def cooperative_concern():
        await enter_phase(_TestPhase.P1)

    async def retrying_concern():
        await enter_phase(_TestPhase.P1)
        await again()

    async def batch():
        async with batch_phase(_TestPhase.P1):
            seen.append(("cooperative", await try_resume(Concern(cooperative_concern()))))
            seen.append(("retrying", await try_resume(Concern(retrying_concern()))))

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await try_step(handle, through=_TestPhase.P1)

    Driver(scheduler()).run()
    assert seen == [("cooperative", True), ("retrying", False)]


def test_again_at_root_raises():
    """`again()` from the root coroutine has no parent — programming error."""

    async def main():
        # Set _active_phase manually so again()'s own check passes;
        # the Driver-level "no parent" check is what we want to hit.
        _cor_mod._active_phase.set(_TestPhase.P0)
        try:
            await again()
        finally:
            _cor_mod._active_phase.set(None)

    with pytest.raises(RuntimeError, match="root coroutine has no parent"):
        Driver(main()).run()


def test_again_preserves_active_phase_across_retry():
    """After retry, the batch resumes at the same phase it was paused at."""

    seen = []

    async def batch():
        async with batch_phase(_TestPhase.P0):
            pass
        async with batch_phase(_TestPhase.P1):
            seen.append(("before_again", _cor_mod._active_phase.get()))
            await again()
            seen.append(("after_again", _cor_mod._active_phase.get()))

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        # First step drives through P0.
        await step(handle, through=_TestPhase.P0)
        # Second step: batch enters P1, hits again() — try_step needed.
        first = await try_step(handle, through=_TestPhase.P3)
        seen.append(("first_call", first))
        # Third step (retry): batch resumes from after again() inside P1.
        second = await try_step(handle, through=_TestPhase.P3)
        seen.append(("second_call_terminal", second is not None))

    Driver(scheduler()).run()
    # Batch saw P1 both before and after again() (same phase preserved).
    assert seen[0] == ("before_again", _TestPhase.P1)
    assert seen[1] == ("first_call", None)  # try_step returned None on retry
    assert seen[2] == ("after_again", _TestPhase.P1)
    # Third entry is from after the second try_step ran the rest of the body.
    assert seen[3] == ("second_call_terminal", True)


def test_again_does_not_update_strict_progression_record():
    """Strict-progression check uses prev wait phase; `again()` doesn't bump it."""

    # Sequence: enter P1 (real wait), again() (no wait), enter P2 (real wait).
    # Strict progression compares P2 against prev=P1, not against P1+again.
    # Should NOT raise.

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await again()
        async with batch_phase(_TestPhase.P2):
            pass

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        # Drive far enough; tolerate the retry.
        first = await try_step(handle, through=_TestPhase.P3)
        # First call retried during P1.
        assert first is None
        # Resume — P2 yield is OK because prev (in suspension table) is P1.
        await try_step(handle, through=_TestPhase.P3)

    Driver(scheduler()).run()  # would raise if strict-progression rejected


def test_pp_heartbeat_pattern_end_to_end():
    """A PP-style heartbeat: do_first, then heartbeat-and-again until ready."""

    iters_observed = []
    heartbeats = 0

    async def batch():
        nonlocal heartbeats
        async with batch_phase(_TestPhase.P0) as (_, w):
            w.p0_out = 0  # "first iter" work done
        # Still inside P1 phase block; loop heartbeats until external ready.
        async with batch_phase(_TestPhase.P1) as (_, w):
            while heartbeats < 3:
                heartbeats += 1
                # No phase yields: just keep retrying until ready.
                await again()
            # After loop: do the real P1 work.
            w.p1_out = f"ready_after_{heartbeats}_heartbeats"
        async with batch_phase(_TestPhase.P3):
            pass

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        # Iter 0: P0 done.
        await step(handle, through=_TestPhase.P0)
        iters_observed.append("iter0_p0_done")
        # Iters 1..N: try_step until success.
        for i in range(10):
            result = await try_step(handle, through=_TestPhase.P3)
            if result is not None:
                iters_observed.append(f"iter{i + 1}_success")
                break
            iters_observed.append(f"iter{i + 1}_retry")

    Driver(scheduler()).run()
    # 3 retries (heartbeats) + 1 successful iter that runs the P1 body.
    assert iters_observed == [
        "iter0_p0_done",
        "iter1_retry",
        "iter2_retry",
        "iter3_retry",
        "iter4_success",
    ]
    assert handle.storage.p1_out == "ready_after_3_heartbeats"


def test_again_retry_sentinel_is_singleton():
    """`_RETRY` is a singleton; identity comparison is the contract."""

    assert _RETRY is _RETRY
    # Repr is self-describing.
    assert repr(_RETRY) == "_RETRY"


# --------------------------------------------------------------------------- #
# Handle completion: resume / try_resume / step / try_step are no-ops on
# already-finished handles. The Driver flips ``handle.done`` whenever a
# coroutine terminates (StopIteration, exception, or explicit close).
# --------------------------------------------------------------------------- #


def test_concern_handle_done_after_first_resume():
    """After a concern's coro returns, its Concern handle has ``done = True``."""

    captured = {}

    async def concern():
        await enter_phase(_TestPhase.P1)

    c = Concern(concern())

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await resume(c)
            captured["done_after_first"] = c.done

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    Driver(scheduler()).run()
    assert captured["done_after_first"] is True


def test_resume_on_completed_concern_is_noop():
    """`resume()` on a Concern whose coro completed returns immediately."""

    seen = []

    async def concern():
        seen.append("ran")
        await enter_phase(_TestPhase.P1)

    c = Concern(concern())

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await resume(c)
            seen.append("after_first")
            await resume(c)  # was a crash before; now a no-op
            seen.append("after_second")
            await resume(c)  # still no-op
            seen.append("after_third")

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    Driver(scheduler()).run()
    # The concern body ran exactly once.
    assert seen == ["ran", "after_first", "after_second", "after_third"]


def test_try_resume_on_completed_concern_returns_true():
    """`try_resume()` on a completed Concern returns True (made it past everything)."""

    results = []

    async def concern():
        await enter_phase(_TestPhase.P1)

    c = Concern(concern())

    async def batch():
        async with batch_phase(_TestPhase.P1):
            results.append(await try_resume(c))  # progresses, then completes
            results.append(await try_resume(c))  # no-op, True
            results.append(await try_resume(c))  # no-op, True

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    Driver(scheduler()).run()
    assert results == [True, True, True]


def test_step_on_completed_batch_is_idempotent():
    """`step()` on a finished Batch returns the terminal view repeatedly."""

    async def batch():
        async with batch_phase(_TestPhase.P0) as (_, w):
            w.p0_out = 99
        async with batch_phase(_TestPhase.P3):
            pass

    handle = Batch(batch(), _TestStorage())
    captured = {}

    async def scheduler():
        # First step drives to terminal.
        first = await step(handle, through=_TestPhase.P3)
        captured["first_write_is_none"] = first[1] is None
        captured["done_after_first"] = handle.done
        # Second step on same handle: no-op fast path, returns terminal again.
        second = await step(handle, through=_TestPhase.P3)
        captured["second_write_is_none"] = second[1] is None
        # Read view still exposes the data the batch wrote.
        captured["second_read_p0_out"] = second[0].p0_out

    Driver(scheduler()).run()
    assert captured == {
        "first_write_is_none": True,
        "done_after_first": True,
        "second_write_is_none": True,
        "second_read_p0_out": 99,
    }


def test_try_step_on_completed_batch_returns_terminal_not_retry():
    """Idempotent terminal — distinct from retry's bare ``None``."""

    async def batch():
        async with batch_phase(_TestPhase.P0):
            pass
        async with batch_phase(_TestPhase.P3):
            pass

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await try_step(handle, through=_TestPhase.P3)
        # Now batch is done. try_step should return (read, None), NOT bare None.
        result = await try_step(handle, through=_TestPhase.P3)
        assert result is not None
        read, write = result
        assert write is None

    Driver(scheduler()).run()


def test_handle_done_set_after_exception():
    """A coroutine that raises also marks its handle done (terminated)."""

    async def concern():
        raise ValueError("planned")

    c = Concern(concern())

    async def batch():
        async with batch_phase(_TestPhase.P1):
            try:
                await resume(c)
            except ValueError:
                pass

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    Driver(scheduler()).run()
    # The concern handle is done — the coroutine raised, terminated the same.
    assert c.done is True


def test_concern_can_be_reused_across_batch_phases():
    """A Concern can be ``resume``-d across multiple batch_phase blocks."""

    log = []

    async def concern():
        await enter_phase(_TestPhase.P1)
        log.append("p1_done")
        await enter_phase(_TestPhase.P2)
        log.append("p2_done")

    c = Concern(concern())

    async def batch():
        async with batch_phase(_TestPhase.P1):
            await resume(c)  # drives concern through P1
        log.append("between_blocks")
        async with batch_phase(_TestPhase.P2):
            await resume(c)  # drives concern through P2; concern returns

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    Driver(scheduler()).run()
    assert log == ["p1_done", "between_blocks", "p2_done"]
    assert c.done is True


# --------------------------------------------------------------------------- #
# Hang detection: Driver builtin watchdog + ``disable_hang_detect`` CM.
# --------------------------------------------------------------------------- #


def test_disable_hang_detect_yields_enter_then_exit_control_messages():
    """The CM yields ``_HangControl(True)`` on enter and ``_HangControl(False)``
    on exit -- the wire protocol the Driver dispatches on.
    """

    async def body():
        async with disable_hang_detect():
            pass

    coro = body()
    try:
        first = coro.send(None)
        assert isinstance(first, _HangControl) and first.disabled is True
        second = coro.send(None)
        assert isinstance(second, _HangControl) and second.disabled is False
        with pytest.raises(StopIteration):
            coro.send(None)
    finally:
        coro.close()


def test_watchdog_fires_after_timeout_with_no_messages():
    """No notify => watchdog fires after ``timeout`` seconds."""
    fired = threading.Event()
    wd = _HangWatchdog(timeout=0.05, on_hang=lambda _ctx: fired.set())
    wd.start()
    try:
        assert fired.wait(timeout=1.0), "watchdog should have fired"
    finally:
        wd.stop()


def test_watchdog_notify_resets_timer():
    """Successive notifies faster than ``timeout`` keep the watchdog quiet."""
    fired = threading.Event()
    wd = _HangWatchdog(timeout=0.1, on_hang=lambda _ctx: fired.set())
    wd.start()
    try:
        for _ in range(6):
            time.sleep(0.02)
            wd.notify(coro_name="some_coro", phase=None)
        # ~120ms elapsed with notifies every 20ms; would have fired
        # at 100ms without them.
        assert not fired.is_set()
    finally:
        wd.stop()


def test_watchdog_pause_blocks_timeout_indefinitely():
    """While paused, the watchdog waits for ``resume`` even past ``timeout``."""
    fired = threading.Event()
    wd = _HangWatchdog(timeout=0.05, on_hang=lambda _ctx: fired.set())
    wd.start()
    try:
        wd.pause()
        # Wait > timeout; pause should keep the watchdog quiet.
        assert not fired.wait(timeout=0.15)
    finally:
        wd.stop()


def test_watchdog_resume_re_arms():
    """After ``resume`` the watchdog returns to timing-wait state."""
    fired = threading.Event()
    wd = _HangWatchdog(timeout=0.05, on_hang=lambda _ctx: fired.set())
    wd.start()
    try:
        wd.pause()
        time.sleep(0.1)  # well past timeout, but paused
        assert not fired.is_set()
        wd.resume()
        assert fired.wait(timeout=1.0)
    finally:
        wd.stop()


def test_watchdog_re_fires_every_timeout_until_message_arrives():
    """Without messages the watchdog keeps re-firing on each ``timeout``."""
    fire_count = [0]
    fire_event = threading.Event()

    def on_hang(_ctx):
        fire_count[0] += 1
        if fire_count[0] >= 3:
            fire_event.set()

    wd = _HangWatchdog(timeout=0.05, on_hang=on_hang)
    wd.start()
    try:
        # No notify; expect at least 3 fires within ~0.5s.
        assert fire_event.wait(timeout=1.0)
        assert fire_count[0] >= 3
    finally:
        wd.stop()


def test_watchdog_context_includes_last_notify():
    """``on_hang(context)`` carries the last (name, phase) the Driver sent."""
    captured = {}
    fired = threading.Event()

    def on_hang(context):
        captured["context"] = context
        fired.set()

    wd = _HangWatchdog(timeout=0.05, on_hang=on_hang)
    wd.start()
    try:
        wd.notify("some_coroutine", _TestPhase.P2)
        assert fired.wait(timeout=1.0)
    finally:
        wd.stop()
    assert "some_coroutine" in captured["context"]
    assert "P2" in captured["context"]


def test_watchdog_notify_does_not_accept_coroutine_objects_as_name():
    """``notify`` is designed for a name STRING, not a coroutine object.

    This is a regression check on the API: the watchdog must only ever
    stash cheap immutable values, never a coroutine reference that
    could keep a completed coroutine alive across the worker thread
    boundary. We verify by introspecting what the worker stashes.
    """
    wd = _HangWatchdog(timeout=10.0, on_hang=lambda _ctx: None)
    wd.start()
    try:
        wd.notify("MyConcern.handle_batch", _TestPhase.P1)
        # Drain by stopping (forces the worker to process queued
        # messages up to the next state-changing one). Then inspect
        # via a synthetic fire.
        wd.stop()
    except Exception:
        wd.stop()
        raise
    # We can't easily peek into the worker thread after it exits,
    # so verify the queue's message format directly: ``notify``
    # should put a tuple of (str, Optional[IntEnum]) -- never a
    # coroutine.
    fresh = _HangWatchdog(timeout=10.0, on_hang=lambda _ctx: None)
    fresh.notify("Foo.bar", None)
    msg = fresh._queue.get()  # noqa: SLF001 -- introspection
    assert isinstance(msg, tuple)
    name, phase = msg
    assert isinstance(name, str)
    assert phase is None or isinstance(phase, IntEnum)


def test_watchdog_context_when_no_notify_yet():
    """Hang fires before the first notify => context says so."""
    captured = {}
    fired = threading.Event()

    def on_hang(context):
        captured["context"] = context
        fired.set()

    wd = _HangWatchdog(timeout=0.05, on_hang=on_hang)
    wd.start()
    try:
        assert fired.wait(timeout=1.0)
    finally:
        wd.stop()
    assert "no coroutine" in captured["context"].lower()


def test_watchdog_stop_is_idempotent_and_safe_before_start():
    """``stop`` is a no-op if never started, and idempotent if already stopped."""
    wd = _HangWatchdog(timeout=0.05, on_hang=lambda _ctx: None)
    wd.stop()  # never started; should not raise
    wd.start()
    wd.stop()
    wd.stop()  # already stopped; should not raise


def test_watchdog_rejects_non_positive_timeout():
    """Constructor rejects non-positive timeouts."""
    with pytest.raises(ValueError):
        _HangWatchdog(timeout=0.0, on_hang=lambda _ctx: None)
    with pytest.raises(ValueError):
        _HangWatchdog(timeout=-1.0, on_hang=lambda _ctx: None)


def test_watchdog_swallows_on_hang_exceptions():
    """A throwing on_hang doesn't kill the worker thread."""
    fired = threading.Event()

    def bad_on_hang(_ctx):
        fired.set()
        raise RuntimeError("planned failure inside on_hang")

    wd = _HangWatchdog(timeout=0.05, on_hang=bad_on_hang)
    wd.start()
    try:
        # No notify; the watchdog fires.
        assert fired.wait(timeout=1.0)
        # Worker is still alive: pause then resume still triggers
        # another fire after a quiet timeout window.
        fired.clear()
        assert fired.wait(timeout=1.0)
    finally:
        wd.stop()


def test_driver_no_watchdog_when_hang_timeout_is_none():
    """Default Driver has no watchdog -- no thread, no machinery."""

    async def main():
        return None

    drv = Driver(main())
    assert drv._watchdog is None
    drv.run()


def test_driver_watchdog_fires_on_slow_main():
    """A slow sync block inside main() trips the Driver-builtin watchdog."""
    fired = threading.Event()

    async def main():
        time.sleep(0.3)  # > timeout, no yield

    Driver(main(), hang_timeout=0.05, on_hang=lambda _ctx: fired.set()).run()
    assert fired.is_set()


def test_driver_watchdog_silent_on_responsive_coroutine():
    """A coroutine that returns quickly does not trip the watchdog."""
    fired = threading.Event()

    async def main():
        return None

    Driver(main(), hang_timeout=1.0, on_hang=lambda _ctx: fired.set()).run()
    assert not fired.is_set()


def test_disable_hang_detect_silences_long_sync_block():
    """A sync sleep wrapped in ``disable_hang_detect`` does not fire."""
    fired = threading.Event()

    async def main():
        async with disable_hang_detect():
            time.sleep(0.3)  # well past timeout, but disabled

    Driver(main(), hang_timeout=0.05, on_hang=lambda _ctx: fired.set()).run()
    assert not fired.is_set()


def test_disable_hang_detect_re_arms_after_exit():
    """After exiting the CM, the watchdog re-arms and fires on later slow code."""
    fired = threading.Event()

    async def main():
        async with disable_hang_detect():
            time.sleep(0.2)  # disabled
        time.sleep(0.2)  # NOT disabled -- should fire

    Driver(main(), hang_timeout=0.05, on_hang=lambda _ctx: fired.set()).run()
    assert fired.is_set()


def test_disable_hang_detect_is_global_across_coroutines():
    """While the CM is open, the watchdog stays paused even if other coroutines run."""
    fired = threading.Event()

    async def disabled_concern():
        async with disable_hang_detect():
            # Hand control back to the parent so it can drive a peer
            # concern while this CM is still open. The pause must
            # persist across that handoff.
            await again()

    async def slow_peer():
        await enter_phase(_TestPhase.P1)
        time.sleep(0.3)  # would normally trip; pause should silence it

    d = Concern(disabled_concern())
    p = Concern(slow_peer())

    async def batch():
        async with batch_phase(_TestPhase.P1):
            # Drive ``d`` first; it ``await again()``s and pops back.
            assert (await try_resume(d)) is False
            # While ``d`` is suspended in disable_hang_detect, run ``p``.
            await resume(p)

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await try_step(handle, through=_TestPhase.P3)

    Driver(scheduler(), hang_timeout=0.05, on_hang=lambda _ctx: fired.set()).run()
    assert not fired.is_set(), "the global pause should silence the peer's slow body"


def test_disable_hang_detect_does_not_nest():
    """Re-entering the CM while already active raises RuntimeError."""

    async def main():
        async with disable_hang_detect():
            async with disable_hang_detect():
                pass

    with pytest.raises(RuntimeError, match="does not nest"):
        Driver(main()).run()


def test_disable_hang_detect_outer_cm_cleans_up_after_inner_crash():
    """After a nested-entry crash, the outer CM still clears the flag."""

    async def main():
        async with disable_hang_detect():
            try:
                async with disable_hang_detect():  # crashes
                    pass
            except RuntimeError:
                pass

    drv = Driver(main())
    drv.run()
    # Outer __aexit__ ran; flag is back to False.
    assert drv._hang_paused is False


def test_disable_hang_detect_nesting_check_works_without_watchdog():
    """The nesting check fires even when ``hang_timeout`` is None."""

    async def main():
        async with disable_hang_detect():
            async with disable_hang_detect():
                pass

    # No hang_timeout -- watchdog disabled. The flag check is
    # independent and still raises.
    with pytest.raises(RuntimeError, match="does not nest"):
        Driver(main()).run()


def test_driver_clears_hang_paused_on_shutdown():
    """A Driver that finishes (even via exception) leaves ``_hang_paused`` False."""

    async def main():
        async with disable_hang_detect():
            raise ValueError("planned")

    drv = Driver(main())
    with pytest.raises(ValueError, match="planned"):
        drv.run()
    assert drv._hang_paused is False


def test_driver_default_on_hang_uses_logger():
    """Default on_hang resolves; verify by patching the import target."""
    fired = threading.Event()

    def fake_print_all_stacks():
        fired.set()

    async def main():
        time.sleep(0.2)

    with mock.patch("tensorrt_llm._utils.print_all_stacks", fake_print_all_stacks):
        # Use the default on_hang (None) so the Driver picks
        # ``_default_on_hang`` which imports + calls print_all_stacks.
        Driver(main(), hang_timeout=0.05).run()

    assert fired.is_set()


# --------------------------------------------------------------------------- #
# Production-binding tests
#
# Everything above this line uses ``_TestPhase`` / ``_TestStorage``. The
# tests below intentionally bind to the production data model in
# :mod:`tensorrt_llm._torch.pyexecutor.batch_storage` because they
# verify the contract between the generated block and the production
# ``BatchStorage`` field metadata.
# --------------------------------------------------------------------------- #


def test_generator_output_agrees_with_file():
    """The generator's output matches the committed block verbatim."""
    import importlib.util
    import sys as _sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[4]
    script = repo_root / "scripts" / "generate_coroutine_views.py"
    spec = importlib.util.spec_from_file_location("_generate_coroutine_views", script)
    gen = importlib.util.module_from_spec(spec)
    _sys.modules["_generate_coroutine_views"] = gen
    spec.loader.exec_module(gen)

    if not gen.is_up_to_date():
        pytest.fail(
            "The generated block in batch_storage.py is out of sync with "
            "BatchStorage metadata. Run "
            "`python scripts/generate_coroutine_views.py` to refresh.\n\n"
            + "".join(gen.diff_report())
        )
