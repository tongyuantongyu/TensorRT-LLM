"""Tests for the ``coroutines`` runtime module.

Most tests exercise generic runtime behavior (primitive wire format,
``phase`` CM, ``resume`` / ``step`` semantics, exception propagation,
``close()``-shutdown, frame hiding, ``spawn``, Driver policy, runtime
tracked-view proxies, and end-to-end 3-layer interleaving). They use a
test-local ``_TestPhase`` enum and ``_TestStorage`` dataclass defined
just below the imports, so they stay stable when the production
``LoopPhase`` / ``BatchStorage`` evolve.

A separate "production-binding" group of tests at the bottom of the
file (``test_static_{read,write}_views_match_field_metadata``,
``test_generator_output_agrees_with_file``) intentionally imports the
production types from :mod:`batch_storage` to verify the generated
block matches ``BatchStorage``'s field metadata. Those are the only
tests that should reference ``BatchStorage`` / ``LoopPhase`` /
``_ReadAtP*`` / ``_WriteAtP*`` / ``_ReadAtAll``.
"""

from __future__ import annotations

import dataclasses
import os
import traceback
from enum import IntEnum
from typing import Optional
from unittest import mock

import pytest

import tensorrt_llm._torch.pyexecutor.coroutines as _cor_mod
from tensorrt_llm._torch.pyexecutor.coroutines import (
    Driver,
    Batch,
    _AdvanceRequest,
    _WaitRequest,
    enter_phase,
    batch_phase,
    phased_field,
    resume,
    spawn,
    step,
)

# --------------------------------------------------------------------------- #
# Test-local phase enum and storage. Decoupled from production so runtime
# tests survive any change to ``LoopPhase`` / ``BatchStorage`` shape.
#
# Naming: ``_TestPhase`` / ``_TestStorage`` (with the ``_Test`` prefix)
# is deliberately verbose so a grep for ``LoopPhase`` or ``BatchStorage``
# in this file lands only on the production-binding tests at the bottom.
# --------------------------------------------------------------------------- #


class _TestPhase(IntEnum):
    """Test-local phase enum independent of production ``LoopPhase``.

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
    """``resume`` yields ``_AdvanceRequest`` with ``at`` from the phase CM."""

    async def child():
        # Will not actually run; we drop on close().
        await enter_phase(_TestPhase.P1)

    child_coro = child()
    captured = {}

    async def body():
        async with batch_phase(_TestPhase.P1):
            captured["before_resume"] = _cor_mod._active_phase.get()
            await resume(child_coro)

    parent = body()
    try:
        # Pump 1: __aenter__ yields the wait request.
        request = parent.send(None)
        assert isinstance(request, _WaitRequest)
        assert request.phase is _TestPhase.P1
        # Pump 2: parent's body resumes after wait, calls resume(child),
        # yields _AdvanceRequest(child, at=P1).
        request = parent.send(None)
        assert isinstance(request, _AdvanceRequest)
        assert request.child is child_coro
        assert request.at is _TestPhase.P1
        assert captured["before_resume"] is _TestPhase.P1
    finally:
        parent.close()
        child_coro.close()


def test_resume_outside_phase_cm_raises():
    """Calling ``resume`` with ``_active_phase`` unset is a programming error."""

    async def child():
        await enter_phase(_TestPhase.P1)

    child_coro = child()
    try:

        async def body():
            await resume(child_coro)

        parent = body()
        try:
            with pytest.raises(RuntimeError, match="outside a batch_phase CM"):
                parent.send(None)
        finally:
            parent.close()
    finally:
        child_coro.close()


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
            await resume(concern_a())

        # P2 block: drives concern_b. batch publishes nothing here.
        async with batch_phase(_TestPhase.P2):
            await resume(concern_b())

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
            await resume(concern())

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
    drv._stack = [(drv._main, None, None)]  # noqa: SLF001
    request = drv._main.send(None)  # noqa: SLF001
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


def _reload_coroutines_with_env(show_frames: str):
    """Re-import the coroutines module with TLLM_COROUTINE_SHOW_FRAMES set."""
    import importlib

    import tensorrt_llm._torch.pyexecutor.coroutines as mod

    with mock.patch.dict(os.environ, {"TLLM_COROUTINE_SHOW_FRAMES": show_frames}):
        importlib.reload(mod)
    return mod


def test_frame_hiding_env_reveals_driver_frames():
    """With TLLM_COROUTINE_SHOW_FRAMES=1, framework frames are visible."""
    mod = _reload_coroutines_with_env("1")

    async def deep():
        async with mod.batch_phase(_TestPhase.P1):
            raise RuntimeError("planted")

    async def sched():
        handle = mod.Batch(deep(), _TestStorage())
        await mod.step(handle, through=_TestPhase.P1)

    try:
        mod.Driver(sched()).run()
    except RuntimeError as e:
        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        assert "_drive" in tb_str or "_yield_advance" in tb_str or "_yield_wait" in tb_str
    finally:
        _reload_coroutines_with_env("0")


# --------------------------------------------------------------------------- #
# spawn helper
# --------------------------------------------------------------------------- #


def test_spawn_primes_via_send_none():
    """spawn() runs the coroutine up to its first yield."""

    started: list[str] = []

    async def concern():
        started.append("started")
        # Pretend to yield — but spawn needs an active state. Use a
        # mock active state to pre-prime without going through the
        # Driver.
        await enter_phase(_TestPhase.P1)

    storage = _TestStorage()
    token = _cor_mod._active_storage.set(storage)
    try:
        coro = concern()
        result = spawn(coro)
        assert result is coro
        assert started == ["started"]
    finally:
        _cor_mod._active_storage.reset(token)
        coro.close()


def test_spawn_silences_never_awaited_warning():
    """A spawned coroutine that's never advanced doesn't emit RuntimeWarning."""
    import warnings

    storage = _TestStorage()
    token = _cor_mod._active_storage.set(storage)
    try:

        async def concern():
            await enter_phase(_TestPhase.P1)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            coro = concern()
            spawn(coro)
            del coro  # let GC see it
            import gc

            gc.collect()
        never_awaited = [
            w
            for w in captured
            if issubclass(w.category, RuntimeWarning) and "was never awaited" in str(w.message)
        ]
        assert never_awaited == []
    finally:
        _cor_mod._active_storage.reset(token)


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
            await resume(concern())

    handle = Batch(batch(), _TestStorage())

    async def scheduler():
        await step(handle, through=_TestPhase.P3)

    with pytest.raises(RuntimeError, match="must not nest"):
        Driver(scheduler()).run()


# --------------------------------------------------------------------------- #
# Production-binding tests
#
# Everything above this line uses ``_TestPhase`` / ``_TestStorage``. The
# tests below intentionally bind to the production data model in
# :mod:`tensorrt_llm._torch.pyexecutor.batch_storage` because they
# verify the contract between the generated block and the production
# ``BatchStorage`` field metadata.
# --------------------------------------------------------------------------- #


from tensorrt_llm._torch.pyexecutor.coroutines import _field_phases  # noqa: E402
from tensorrt_llm._torch.pyexecutor.batch_storage import (  # noqa: E402
    BatchStorage,
    LoopPhase,
    _ReadAtAll,
    _ReadAtP0,
    _ReadAtP1,
    _ReadAtP2,
    _ReadAtP3,
    _WriteAtP0,
    _WriteAtP1,
    _WriteAtP2,
    _WriteAtP3,
)


def _read_protocol_fields(proto_cls: type) -> set[str]:
    """Field names declared on a read Protocol (as ``@property`` members)."""
    exposed: set[str] = set()
    for base in proto_cls.__mro__:
        for name, val in vars(base).items():
            if isinstance(val, property):
                exposed.add(name)
    return exposed


def _write_dataclass_fields(dc_cls: type) -> set[str]:
    """Field names declared on a write dataclass."""
    return {f.name for f in dataclasses.fields(dc_cls)}


def test_static_read_views_match_field_metadata():
    """Each ``_ReadAtP*`` exposes exactly the fields produced at phases < P.

    Derived from the ``phase`` metadata on ``BatchStorage`` fields. If a
    field is added / moved, this test shows the Protocol update that's
    still needed (independent of running the generator).
    """
    phase_of_field = _field_phases(BatchStorage)
    read_protos = {
        LoopPhase.P0: _ReadAtP0,
        LoopPhase.P1: _ReadAtP1,
        LoopPhase.P2: _ReadAtP2,
        LoopPhase.P3: _ReadAtP3,
    }
    for phase_value, proto in read_protos.items():
        expected = {name for name, p in phase_of_field.items() if p < phase_value}
        declared = _read_protocol_fields(proto)
        assert declared == expected, (
            f"{proto.__name__} declares {declared}; "
            f"BatchStorage metadata says fields readable at {phase_value.name} "
            f"are {expected}"
        )


def test_static_read_at_all_includes_every_field():
    """``_ReadAtAll`` exposes every field across every phase."""
    phase_of_field = _field_phases(BatchStorage)
    expected = set(phase_of_field.keys())
    declared = _read_protocol_fields(_ReadAtAll)
    assert declared == expected


def test_static_write_views_match_field_metadata():
    """Each ``_WriteAtP*`` exposes exactly the fields produced at phase P."""
    phase_of_field = _field_phases(BatchStorage)
    write_dcs = {
        LoopPhase.P0: _WriteAtP0,
        LoopPhase.P1: _WriteAtP1,
        LoopPhase.P2: _WriteAtP2,
        LoopPhase.P3: _WriteAtP3,
    }
    for phase_value, dc in write_dcs.items():
        expected = {name for name, p in phase_of_field.items() if p == phase_value}
        declared = _write_dataclass_fields(dc)
        assert declared == expected, (
            f"{dc.__name__} declares {declared}; "
            f"BatchStorage metadata says fields produced at {phase_value.name} "
            f"are {expected}"
        )


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
