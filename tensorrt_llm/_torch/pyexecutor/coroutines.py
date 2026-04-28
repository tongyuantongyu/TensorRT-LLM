"""Deterministic, generic coroutine runtime for the PyExecutor forward loop.

Three layers, three primitives
==============================

The runtime is shaped around a strict three-layer hierarchy:

| Layer | Who | Primitive |
|---|---|---|
| Concern body | non-batch coroutines wrapped in a ``Concern`` handle | ``r, w = await enter_phase(Py)`` (no storage arg) |
| Batch body | batch coroutine only | ``async with batch_phase(Py): ...`` + ``await resume(c)`` where ``c: Concern`` |
| Scheduler | top-level driver of batches | ``Batch(coro, storage)`` + ``await step(handle, through=Py)`` |

Coroutines in the lower two layers are always wrapped in a *handle*
(:class:`Batch` for batches, :class:`Concern` for concerns). The handle
owns its coroutine, tracks completion via ``handle.done``, and closes
the coroutine in ``__del__`` if it was never advanced — replacing the
old ``spawn`` primitive. After ``handle.done`` becomes ``True``, all
further ``resume`` / ``try_resume`` (or ``step`` / ``try_step``) calls
on that handle are no-ops.

The single happens-before rule
==============================

At any layer, code that operates at a phase ``P``:

- **Reads** data produced at phases ``< P`` through a *read view* exposing
  only those fields.
- **Writes** data that becomes visible to downstream readers at phases
  ``> P`` through a *write view* exposing only the fields produced at
  ``P``.

The phase value used by ``enter_phase(Py)``, ``async with batch_phase(Py):``,
and ``step(handle, through=Py)`` is consistent: "Py is now". Inside the
batch's ``batch_phase(Py)`` block the batch is at Py, doing Py work;
calling ``await resume(child)`` drives the child through *the same* Py
work (its own ``enter_phase(Py)``) and returns when the child has
moved past Py. The pop predicate is strict ``>``: a child is popped
when its current wait phase is strictly greater than the parent's
target.

Active state
============

The runtime keeps two ``ContextVar``s: ``_active_storage`` and
``_active_phase``. ``Driver.run`` runs its body inside a fresh
``contextvars.Context`` so multiple Drivers don't see each other's
state.

- ``step()`` asserts no active batch, then sets ``_active_storage``
  from the batch handle, ``_active_phase`` from the handle's
  ``saved_phase``. On exit it captures the batch's current phase
  back into ``handle.saved_phase`` and clears both. ``step()`` does
  not nest.
- ``batch_phase`` CM sets ``_active_phase`` on enter and clears it on
  exit; the CM never nests (asserts on enter when ``_active_phase``
  is already set).
- Concerns inherit the active state from their batch via the
  ``ContextVar``s; ``enter_phase(Py)`` reads ``_active_storage`` and
  yields a wait at ``Py``. Concerns never touch ``_active_phase``.

Strict progression
==================

The Driver enforces that successive ``enter_phase`` calls from the same
coroutine yield strictly greater phases — both regression (P2 then P1)
and stay-at-same-phase (P1 then P1) raise. This catches misuse at the
offending ``enter_phase`` call rather than waiting for an attribute
access on a tracked-storage proxy to surface a confusing error.

Retry: ``await again()`` and the ``try_*`` drivers
==================================================

Sometimes a coroutine is "at" a phase but cannot make further progress
yet because of external state (e.g., a PP rank is still waiting on its
upstream). Looking sideways at this: ``again()`` is the coroutine
saying *EAGAIN* — "ask me again later, from this exact point". The
Driver pops the coroutine without touching its ``_active_phase``,
``_active_storage``, suspension record, or tracked-write log. When the
parent next drives it, execution resumes from right after ``await
again()`` with the same view of storage it had before; the data
barrier (read at Py sees writes at < Py) is unchanged.

Because retry isn't always expected, drivers come in pairs:

- :func:`step` / :func:`resume`: strict. Raise ``RuntimeError`` if the
  child issued ``again()``. Use when ``again()`` would be a logic bug.
- :func:`try_step` / :func:`try_resume`: retry-aware. ``try_step``
  returns ``None`` instead of ``(read, write)``; ``try_resume``
  returns ``False`` instead of ``True``. Use only at call sites
  that genuinely expect EAGAIN.

The contract: ``again()`` *must* be paired with a ``try_*`` driver
upstream. The strict variants exist so accidental retries surface
as exceptions at the immediate boundary rather than silently looping.

Shutdown and cleanup
====================

Shutdown uses Python-native ``close()`` / ``GeneratorExit``. No custom
Shutdown exception class. Cleanup inside coroutine bodies uses ordinary
``with`` / ``try`` / ``finally`` blocks.

Diagnostics
===========

Driver / primitive frames are stripped from tracebacks by default.
Set ``TLLM_COROUTINE_SHOW_FRAMES=1`` to keep them visible when
debugging the runtime itself. Set ``TLLM_COROUTINE_TRACK_STORAGE=1``
to have ``_views_at`` return tracked proxies that enforce
happens-before at attribute access — useful in correctness-focused
CI; default off in production for zero overhead.

Generic over phase enum and storage
===================================

This module is the generic runtime — it dispatches requests, manages
the active state, and pumps coroutines, but it does not know what
phases or what storage fields exist. The production data model
(``BatchPhase`` enum and ``BatchStorage`` dataclass) lives in
:mod:`batch_storage`, which also provides the typed ``@overload`` chain
on ``step``. Tests build their own tiny storage classes locally.
"""

from __future__ import annotations

import contextvars
import dataclasses
import os
import types
import weakref
from contextlib import asynccontextmanager
from enum import IntEnum
from typing import Any, AsyncIterator, Coroutine, Generator, List, Optional, Protocol, Tuple

__all__ = [
    "Driver",
    "Batch",
    "Concern",
    "again",
    "enter_phase",
    "batch_phase",
    "phased_field",
    "resume",
    "step",
    "try_resume",
    "try_step",
]


# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #

# Any coroutine object — what ``async def foo(...)`` calls produce. The
# yield / send / return types are intentionally unconstrained: the
# runtime's contract is encoded by which primitives the coroutine uses.
_CoroutineLike = Coroutine[Any, Any, Any]


# --------------------------------------------------------------------------- #
# Frame hiding
# --------------------------------------------------------------------------- #

_HIDE_FRAMES = os.environ.get("TLLM_COROUTINE_SHOW_FRAMES", "0") != "1"
"""Hide Driver / primitive frames from exception tracebacks by default.

Set ``TLLM_COROUTINE_SHOW_FRAMES=1`` in the environment to keep them
visible when debugging the runtime internals.
"""


def _hide_framework_frames(exc: BaseException) -> None:
    """Rebuild ``exc.__traceback__`` without frames from this module."""
    if not _HIDE_FRAMES:
        return
    module_file = __file__
    tb = exc.__traceback__
    kept: List[Tuple[types.FrameType, int, int]] = []
    while tb is not None:
        if tb.tb_frame.f_code.co_filename != module_file:
            kept.append((tb.tb_frame, tb.tb_lasti, tb.tb_lineno))
        tb = tb.tb_next
    new_tb: Optional[types.TracebackType] = None
    for frame, lasti, lineno in reversed(kept):
        new_tb = types.TracebackType(new_tb, frame, lasti, lineno)
    exc.__traceback__ = new_tb


# --------------------------------------------------------------------------- #
# Storage tagging API (generic — used by BatchStorage and any test storage)
# --------------------------------------------------------------------------- #


def phased_field(phase: IntEnum, default: Any = None) -> Any:
    """Dataclass field tagged with the phase that produces it.

    The phase is stashed in the field's metadata so both the runtime
    (``_TrackedReadView`` / ``_TrackedWriteView``) and offline tooling
    (``scripts/generate_coroutine_views.py``) can recover the
    field-to-phase mapping programmatically — no second source of
    truth to maintain.

    Generic over the phase enum: any ``IntEnum`` whose values induce a
    consistent ordering works. Production uses ``batch_storage.BatchPhase``;
    tests use a local ``_TestPhase``.
    """
    return dataclasses.field(
        default=default,
        metadata={"phase": phase},
    )


def _field_phases(storage_cls: Any) -> dict[str, IntEnum]:
    """Extract the field-name -> phase map from a storage dataclass."""
    return {
        f.name: f.metadata["phase"]
        for f in dataclasses.fields(storage_cls)
        if "phase" in f.metadata
    }


# --------------------------------------------------------------------------- #
# Runtime-enforced views (TLLM_COROUTINE_TRACK_STORAGE=1 in debug mode)
# --------------------------------------------------------------------------- #

_TRACK_STORAGE = os.environ.get("TLLM_COROUTINE_TRACK_STORAGE", "0") == "1"
"""Default off: production trusts the static ``@overload`` / Protocol types.

Set ``TLLM_COROUTINE_TRACK_STORAGE=1`` in the environment to enable the
runtime proxies below. The proxies raise ``AttributeError`` /
``RuntimeError`` on happens-before violations that the type checker
would otherwise only flag as warnings — useful on CI for correctness
tests, or locally when chasing a suspected dataflow bug.
"""


_STORAGE_WRITES: dict[int, set[str]] = {}
"""Per-storage set of field names that have been assigned through a tracked
write view.

Populated only in debug mode, as the only caller is
``_TrackedWriteView.__setattr__``. Keyed by ``id(storage)`` (storage
dataclasses have a generated ``__eq__`` and aren't hashable for
``WeakKeyDictionary``). A ``weakref.finalize`` callback pops the entry
when the storage is garbage-collected, so debug runs don't leak
tracking sets.

Membership — not ``value is None`` — is what the read view uses to tell
"never touched" apart from "touched, value happens to be ``None``".
"""


def _record_write(storage: object, name: str) -> None:
    """Remember that ``name`` has been assigned on ``storage``."""
    key = id(storage)
    written = _STORAGE_WRITES.get(key)
    if written is None:
        written = set()
        _STORAGE_WRITES[key] = written
        weakref.finalize(storage, _STORAGE_WRITES.pop, key, None)
    written.add(name)


def _was_written(storage: object, name: str) -> bool:
    """Return whether ``name`` was recorded as assigned on ``storage``."""
    written = _STORAGE_WRITES.get(id(storage))
    return written is not None and name in written


class _TrackedReadView:
    """Runtime proxy that exposes only fields produced before ``phase``.

    Created on-demand by :func:`_views_at` when ``_TRACK_STORAGE`` is on.
    Reads gated on:

    - The field exists on the underlying storage (else ``AttributeError``).
    - The field's declared phase is strictly less than ``phase`` (else
      ``AttributeError`` — that field is not readable yet).
    - The field has been assigned via a tracked write view (else
      ``RuntimeError`` — read before write).

    Writes raise unconditionally — a read view is not a write handle.
    """

    __slots__ = ("_storage", "_phase", "_field_phase")

    _storage: object
    _phase: IntEnum
    _field_phase: dict[str, IntEnum]

    def __init__(self, storage: object, phase: IntEnum) -> None:
        object.__setattr__(self, "_storage", storage)
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "_field_phase", _field_phases(type(storage)))

    def __getattr__(self, name: str) -> object:
        field_phase = self._field_phase.get(name)
        if field_phase is None:
            raise AttributeError(f"{type(self._storage).__name__!r} has no field {name!r}")
        if field_phase >= self._phase:
            raise AttributeError(
                f"field {name!r} is produced at phase "
                f"{field_phase.name}; read view at phase "
                f"{self._phase.name} does not expose it"
            )
        if not _was_written(self._storage, name):
            raise RuntimeError(
                f"field {name!r} (produced at phase {field_phase.name}) "
                f"was never written but read at phase {self._phase.name}"
            )
        return getattr(self._storage, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"read view at phase "
            f"{object.__getattribute__(self, '_phase').name} "
            f"is read-only; cannot assign {name!r}"
        )


class _TrackedWriteView:
    """Runtime proxy that accepts writes only for fields owned by ``phase``.

    Created on-demand by :func:`_views_at` when ``_TRACK_STORAGE`` is on.
    Writes gated on:

    - The field exists on the underlying storage (else ``AttributeError``).
    - The field's declared phase equals ``phase`` (else ``AttributeError``
      — that field is owned by a different phase).

    Every successful write is recorded in ``_STORAGE_WRITES`` so the
    read view can later distinguish "nobody assigned this" from "someone
    assigned ``None`` on purpose".

    Reads raise unconditionally — a write view is not a read handle.
    """

    __slots__ = ("_storage", "_phase", "_field_phase")

    _storage: object
    _phase: IntEnum
    _field_phase: dict[str, IntEnum]

    def __init__(self, storage: object, phase: IntEnum) -> None:
        object.__setattr__(self, "_storage", storage)
        object.__setattr__(self, "_phase", phase)
        object.__setattr__(self, "_field_phase", _field_phases(type(storage)))

    def __getattr__(self, name: str) -> object:
        raise AttributeError(
            f"write view at phase "
            f"{object.__getattribute__(self, '_phase').name} "
            f"is write-only; cannot read {name!r}"
        )

    def __setattr__(self, name: str, value: object) -> None:
        field_phase = self._field_phase.get(name)
        if field_phase is None:
            raise AttributeError(f"{type(self._storage).__name__!r} has no field {name!r}")
        if field_phase != self._phase:
            raise AttributeError(
                f"field {name!r} belongs to phase "
                f"{field_phase.name}; write view at phase "
                f"{self._phase.name} does not accept it"
            )
        setattr(self._storage, name, value)
        _record_write(self._storage, name)


class _TrackedAllReadView:
    """Read view exposing every field on ``storage`` regardless of phase.

    Returned by :func:`_all_read_view` for the terminal ``step()`` case
    — once a batch has completed, every field its body could have
    produced is in scope. Read enforcement is "the field exists and has
    been written"; there's no phase ceiling to compare against.
    """

    __slots__ = ("_storage", "_field_phase")

    _storage: object
    _field_phase: dict[str, IntEnum]

    def __init__(self, storage: object) -> None:
        object.__setattr__(self, "_storage", storage)
        object.__setattr__(self, "_field_phase", _field_phases(type(storage)))

    def __getattr__(self, name: str) -> object:
        field_phase = self._field_phase.get(name)
        if field_phase is None:
            raise AttributeError(f"{type(self._storage).__name__!r} has no field {name!r}")
        if not _was_written(self._storage, name):
            raise RuntimeError(
                f"field {name!r} (produced at phase {field_phase.name}) was never written"
            )
        return getattr(self._storage, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"all-read view is read-only; cannot assign {name!r}")


def _views_at(storage: object, phase: IntEnum) -> Tuple[object, object]:
    """Return ``(read_view, write_view)`` of ``storage`` at ``phase``.

    Production mode (default): both views are ``storage`` itself. The
    type checker narrows them via the ``@overload`` chains in the
    storage-specific module (e.g. :mod:`batch_storage`'s ``step``).

    Debug mode (``TLLM_COROUTINE_TRACK_STORAGE=1``): proxies that
    enforce happens-before at attribute-access time.
    """
    if _TRACK_STORAGE:
        return (
            _TrackedReadView(storage, phase),
            _TrackedWriteView(storage, phase),
        )
    return storage, storage


def _all_read_view(storage: object) -> object:
    """Return a read view exposing every field on ``storage``.

    Used by ``step()`` at the terminal phase — once the batch has
    completed, the scheduler can read every field it produced.
    """
    if _TRACK_STORAGE:
        return _TrackedAllReadView(storage)
    return storage


# --------------------------------------------------------------------------- #
# Active state — ContextVars scoped to the current Driver run
# --------------------------------------------------------------------------- #

_active_storage: contextvars.ContextVar[Optional[object]] = contextvars.ContextVar(
    "_active_storage", default=None
)
"""The BatchStorage of the batch currently being driven.

Set by :func:`step` (which asserts it's not already set; ``step`` does
not nest), cleared on exit. Read by :func:`enter_phase` so concerns
never need a storage parameter, and by the :class:`batch_phase` CM
which delegates to ``enter_phase``.

This is a ``ContextVar`` rather than a module global so each
``Driver.run`` call is isolated: ``Driver.run`` runs ``_drive`` inside
``contextvars.copy_context().run(...)``, so ContextVar mutations made
during the run don't leak out and don't collide with concurrent
Drivers.
"""

_active_phase: contextvars.ContextVar[Optional[IntEnum]] = contextvars.ContextVar(
    "_active_phase", default=None
)
"""The phase of the batch's current ``batch_phase(Py)`` block, or
``None`` if not inside one.

Set by :func:`batch_phase`'s ``__aenter__`` and cleared on
``__aexit__`` (using ``ContextVar``'s ``set`` / ``reset`` token
protocol). Read by :func:`resume` to fill in the advance target.

``step()`` brackets each batch's pump with set/reset, using the
``Batch`` handle's ``saved_phase`` slot as the per-batch
stash so the batch's CM-state survives across step boundaries.
"""


# --------------------------------------------------------------------------- #
# Handle protocol + request sentinels (internal to the Driver protocol)
# --------------------------------------------------------------------------- #


class _Handle(Protocol):
    """Structural type matched by :class:`Batch`, :class:`Concern`, and the
    private :class:`_RootHandle`.

    All three carry the same minimal shape: a ``coro`` they wrap and a
    ``done`` flag the Driver flips when that coro terminates. This
    Protocol lets the Driver's internals annotate stack entries and
    advance-request payloads precisely without forward-referencing
    every concrete handle class — and stops the type checker from
    warning about ``handle.coro`` / ``handle.done`` accesses on
    otherwise-``object``-typed values.

    Public ``step`` / ``resume`` keep the concrete (``Batch`` /
    ``Concern``) parameter types so call-site documentation and the
    typed ``@overload`` chains in :mod:`batch_storage` stay specific.
    """

    coro: _CoroutineLike
    done: bool


@dataclasses.dataclass(frozen=True)
class _WaitRequest:
    """A coroutine has yielded ``await enter_phase(phase)``.

    The Driver compares ``phase`` against the parent's advance target
    to decide whether to keep pumping (resume the child) or pop and
    return control to the parent.
    """

    phase: IntEnum
    storage: object


@dataclasses.dataclass(frozen=True)
class _AdvanceRequest:
    """A coroutine has yielded ``await resume(child)`` (or ``step``).

    ``child`` is the handle (:class:`Batch` for ``step``,
    :class:`Concern` for ``resume``) wrapping the coroutine to drive.
    ``at`` is the target phase: drive ``child.coro`` until it has
    yielded a ``_WaitRequest`` at ``phase > at``, completed, or raised.

    Carrying the handle (not the raw coroutine) lets the Driver flip
    ``child.done`` directly when the wrapped coroutine completes, so
    the user-side fast path in ``resume`` / ``step`` can short-circuit
    by reading ``child.done`` without consulting any Driver-internal
    state.
    """

    child: _Handle
    at: IntEnum


@dataclasses.dataclass(frozen=True)
class _RetryRequest:
    """A coroutine has yielded ``await again()``.

    The Driver pops the coroutine to its parent without updating its
    suspension record or active state. The parent's
    :func:`_yield_advance` returns :data:`_RETRY`; ``step`` / ``resume``
    raise on it, ``try_step`` / ``try_resume`` translate it to the
    user-facing retry signal.
    """


class _RetryToken:
    """Singleton type for the :data:`_RETRY` sentinel.

    A bespoke type so ``isinstance``-style checks via ``is _RETRY`` are
    unambiguous, and the repr is self-describing in tracebacks.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "_RETRY"


_RETRY: object = _RetryToken()
"""Sentinel sent into a parent coroutine after its child yielded
``await again()``.

The parent's :func:`_yield_advance` returns this value, and ``step`` /
``try_step`` / ``resume`` / ``try_resume`` decide how to surface it.
``is _RETRY`` is the only way to test for it; equality and identity
are the same here because :class:`_RetryToken` is a singleton.
"""


# --------------------------------------------------------------------------- #
# Internal yield helpers
# --------------------------------------------------------------------------- #


@types.coroutine
def _yield_wait(
    phase: IntEnum,
    storage: object,
) -> Generator[_WaitRequest, None, None]:
    """Yield a ``_WaitRequest`` and resume on send."""
    yield _WaitRequest(phase, storage)


@types.coroutine
def _yield_advance(
    child: _Handle,
    at: IntEnum,
) -> Generator[_AdvanceRequest, Any, Any]:
    """Yield an ``_AdvanceRequest`` and return the value the Driver sends back.

    The Driver sends ``None`` on a normal pop (child reached / passed
    its target phase or returned) and :data:`_RETRY` if the child
    yielded ``await again()``. ``step`` / ``resume`` treat the latter
    as a programming error; ``try_step`` / ``try_resume`` thread it
    through to the user.
    """
    return (yield _AdvanceRequest(child, at))


@types.coroutine
def _yield_retry() -> Generator[_RetryRequest, None, None]:
    """Yield a ``_RetryRequest``. Resume value is always ``None``.

    The :data:`_RETRY` sentinel travels *outward* (from the Driver back
    to the popped child's parent), not inward to the retrying child.
    The retrying child just resumes after :func:`again` like any other
    awaited primitive.
    """
    yield _RetryRequest()


# --------------------------------------------------------------------------- #
# Public primitives
# --------------------------------------------------------------------------- #


async def enter_phase(p: IntEnum) -> Tuple[object, object]:
    """Concern-level primitive: wait until phase ``p``, return ``(read, write)``.

    Used by every non-batch coroutine inside a batch (concerns).
    The active storage is read from the ``ContextVar`` set by the
    enclosing :func:`step`. There is no storage parameter — concerns
    never see raw ``BatchStorage``.

    The batch coroutine uses :func:`batch_phase` instead, which wraps
    this primitive and additionally manages ``_active_phase`` for
    ``resume``.

    Raises if no batch is active (``_active_storage`` is unset),
    typically because the coroutine wasn't pumped via ``step()``.
    """
    storage = _active_storage.get()
    if storage is None:
        raise RuntimeError(
            "enter_phase called with no active batch; "
            "concerns must be pumped from inside step()"
        )
    await _yield_wait(p, storage)
    return _views_at(storage, p)


@asynccontextmanager
async def batch_phase(p: IntEnum) -> AsyncIterator[Tuple[object, object]]:
    """Batch-only async CM. Sugar for ``enter_phase`` plus ``_active_phase``.

    On enter:

    1. Asserts ``_active_phase`` is unset (the CM does NOT nest).
    2. Sets ``_active_phase = p`` so :func:`resume` calls inside the
       block can read the current phase.
    3. Calls ``enter_phase(p)`` and yields the ``(read, write)`` views.

    On exit, resets ``_active_phase`` to its previous (None) state.

    Use only in batch-level coroutines. Concerns get the
    non-nesting assertion against them automatically: while a concern
    is pumped, its parent batch's ``batch_phase`` CM is active, so
    ``_active_phase`` is non-None and the concern's
    ``async with batch_phase(...)`` would raise here.

    The function is decorated with ``@asynccontextmanager`` so that
    storage-specific wrappers (e.g. in :mod:`batch_storage`) can
    declare ``@overload`` signatures returning
    ``AbstractAsyncContextManager[Tuple[_ReadAtPy, _WriteAtPy]]`` and
    have the type checker narrow ``r``/``w`` per ``p`` literal.

    Cleanup uses ``set(None)`` (not ``reset(token)``) because GC may
    close a half-finished batch coroutine *after* its enclosing
    ``Driver.run`` has returned, in which case the cleanup runs in a
    different ``contextvars.Context`` than the one where the token
    was created. ``ContextVar.reset`` rejects cross-Context tokens
    with ``ValueError``; plain ``set(None)`` is portable across
    contexts and matches the non-nesting invariant (the outer value
    is always ``None``).
    """
    current = _active_phase.get()
    if current is not None:
        raise RuntimeError(
            f"batch_phase({p.name}): another phase {current.name} is "
            f"already active. The batch_phase CM is batch-only and must "
            f"not nest."
        )
    _active_phase.set(p)
    try:
        yield await enter_phase(p)
    finally:
        _active_phase.set(None)


async def again() -> None:
    """Yield to scheduler asking to resume from this exact point later.

    Like POSIX ``EAGAIN``: the coroutine cannot make further progress
    at its current phase yet (e.g., upstream PP rank not ready) and
    asks the scheduler to try again. The Driver pops this coroutine
    to its parent and sends :data:`_RETRY` to the parent's
    :func:`_yield_advance`. The coroutine's ``_active_phase``,
    ``_active_storage``, suspension record, and tracked-write log are
    all unchanged: when resumed, it sees the same world it saw before
    ``again()``. The data barrier (read at Py exposes writes at
    ``< Py``) holds.

    ``again()`` must be paired with a ``try_*`` driver upstream:

    - From a batch coroutine: the scheduler must use :func:`try_step`,
      not :func:`step`. The latter raises on retry.
    - From a concern: the parent batch must use :func:`try_resume`,
      not :func:`resume`. Cascading is fine — the batch can itself
      ``await again()`` after a child concern retries.

    ``again()`` requires an active phase (``_active_phase`` non-``None``):
    inside a ``batch_phase`` CM (for batches) or after ``enter_phase``
    has returned (for concerns). Calling it between phases is a
    programming error and raises.
    """
    p = _active_phase.get()
    if p is None:
        raise RuntimeError(
            "again() called outside a phase. The semantics is 'I'm at "
            "phase Py and want to be at phase Py when resumed', so "
            "_active_phase must be set: inside `async with "
            "batch_phase(Py): ...` for batches, or after `await "
            "enter_phase(Py)` returned in a concern."
        )
    await _yield_retry()


async def resume(child: "Concern") -> None:
    """Batch-only primitive: drive ``child`` through the current phase's work.

    ``child`` is a :class:`Concern` handle wrapping the concern coroutine.
    Reads the target phase from ``_active_phase`` (set by the
    enclosing :func:`batch_phase` CM). Drives ``child.coro`` until it
    yields at a phase strictly greater than the current phase,
    completes, or raises. Returns nothing — the batch's view of any
    data the child published goes through the batch's own
    ``batch_phase`` CM views at the next phase, not through this
    primitive's return.

    Fast path: if ``child.done`` (the concern coroutine already
    completed on a previous ``resume`` / ``try_resume``), this is a
    no-op and returns immediately without entering the Driver.

    Strict on retry: if ``child`` yields ``await again()``, ``resume``
    raises ``RuntimeError`` — the call site declared retry to be a
    programming error by choosing ``resume``. Use :func:`try_resume`
    when the child legitimately retries.

    Raises if called outside a ``batch_phase`` CM (``_active_phase``
    is ``None``).
    """
    if child.done:
        return
    p = _active_phase.get()
    if p is None:
        raise RuntimeError(
            "resume() called outside a batch_phase CM; resume must be "
            "used inside `async with batch_phase(Py): ...`"
        )
    sent = await _yield_advance(child, p)
    if sent is _RETRY:
        raise RuntimeError(
            f"resume({child!r}): child issued `await again()` but the "
            f"caller used resume(), which forbids retry. Use "
            f"try_resume() if retry is part of the protocol."
        )


async def try_resume(child: "Concern") -> bool:
    """Like :func:`resume`, but report retry instead of raising.

    Returns ``True`` if ``child`` progressed past the current phase
    (or completed); ``False`` if ``child`` yielded ``await again()``.
    The caller decides what to do on ``False`` — typically either
    cascade (``await again()`` itself) or skip the child for this
    pass.

    Fast path: if ``child.done`` (the concern coroutine already
    completed on a previous call), returns ``True`` immediately
    without entering the Driver — semantically "the concern made it
    past every phase by way of finishing".

    Like :func:`resume`, must be called inside a ``batch_phase`` CM.
    """
    if child.done:
        return True
    p = _active_phase.get()
    if p is None:
        raise RuntimeError(
            "try_resume() called outside a batch_phase CM; try_resume "
            "must be used inside `async with batch_phase(Py): ...`"
        )
    sent = await _yield_advance(child, p)
    return sent is not _RETRY


def _close_if_undriven(handle: object) -> None:
    """``__del__`` body shared by :class:`Batch` and :class:`Concern`.

    If the handle's coroutine never reached completion, close it now so
    Python doesn't emit ``RuntimeWarning: coroutine '...' was never
    awaited`` when the coroutine itself is GC-ed afterwards. This
    replaces the old ``spawn`` primitive: instead of priming the
    coroutine eagerly to silence the warning, we let the handle's
    lifetime own the cleanup — undriven handles GC themselves cleanly.

    Defensive lookups (``getattr``) make this safe during interpreter
    shutdown when module attrs may already be gone.
    """
    if getattr(handle, "done", True):
        return
    coro = getattr(handle, "coro", None)
    if coro is None:
        return
    try:
        coro.close()
    except BaseException:
        # ``__del__`` swallows everything; otherwise Python prints
        # to stderr and keeps the GC moving. We do the same explicitly.
        pass


@dataclasses.dataclass
class Batch:
    """Handle wrapping a batch coroutine. Used by schedulers via :func:`step`.

    Bundles the coroutine, its :class:`BatchStorage`-shaped storage,
    saved CM state across step boundaries, and a ``done`` flag the
    Driver flips when the coroutine completes. The scheduler hands
    ``Batch`` instances to ``step`` / ``try_step`` to drive forward;
    storage and CM state are intentionally encapsulated — there are
    no read/write helpers on this class. Data exchange with the batch
    happens only through the views returned by ``step``.

    After ``done`` is set, ``step`` / ``try_step`` short-circuit to
    the terminal view ``(_all_read_view, None)`` without re-entering
    the Driver — calling them on a finished batch is a no-op.

    The handle's ``__del__`` closes the wrapped coroutine if it was
    never advanced, which silences Python's ``RuntimeWarning:
    coroutine '...' was never awaited``. This means a scheduler can
    construct ``Batch`` lazily and discard handles whose work isn't
    needed (e.g., conditional batches) without manual priming.
    """

    coro: _CoroutineLike
    storage: object
    saved_phase: Optional[IntEnum] = None
    done: bool = False

    def __del__(self) -> None:
        _close_if_undriven(self)


@dataclasses.dataclass
class Concern:
    """Handle wrapping a concern coroutine. Used by batches via :func:`resume`.

    Mirror of :class:`Batch` for the layer below: a batch creates a
    ``Concern`` around a freshly-instantiated concern coroutine and
    drives it with ``await resume(concern_handle)`` (or
    ``await try_resume(concern_handle)`` when the concern may issue
    ``again()``).

    The Driver flips ``done`` when the wrapped coroutine completes,
    after which ``resume`` / ``try_resume`` on the same handle are
    cheap no-ops (``resume`` returns immediately, ``try_resume``
    returns ``True`` — "the concern made it past every phase, by
    way of finishing"). This lets a batch's body reuse the same
    handle across multiple ``batch_phase`` blocks without worrying
    about whether the concern still has work to do.

    The handle's ``__del__`` closes the wrapped coroutine if it was
    never advanced — replacing the old ``spawn`` primitive. A
    conditionally-needed concern can be wrapped early and discarded
    without manual priming if the condition turns out false.
    """

    coro: _CoroutineLike
    done: bool = False

    def __del__(self) -> None:
        _close_if_undriven(self)


class _RootHandle:
    """Private handle wrapping the Driver's root coroutine.

    Exists purely so the Driver's stack can hold one shape — handles —
    rather than mixing handles with raw coroutines. Carries the same
    ``coro`` / ``done`` shape as :class:`Batch` / :class:`Concern`,
    plus the ``__del__`` closer (in case a Driver is constructed and
    GC-ed without ever calling ``run``).
    """

    __slots__ = ("coro", "done")

    def __init__(self, coro: _CoroutineLike) -> None:
        self.coro: _CoroutineLike = coro
        self.done: bool = False

    def __del__(self) -> None:
        _close_if_undriven(self)


async def step(handle: Batch, *, through: IntEnum) -> Tuple[object, Optional[object]]:
    """Scheduler-level primitive: drive ``handle`` through phase ``through``.

    Activates ``handle``'s storage on the current ``ContextVar``,
    drives ``handle.coro`` until it has finished its ``through`` phase
    work (i.e., the batch's body has either yielded a wait at
    ``phase > through`` or returned), then returns:

    - ``(read_view_after_through, write_view_at_through)`` for
      non-terminal ``through``: read exposes everything produced
      through ``through`` (cumulative); write is the slot for
      injecting Py-fields the batch left as holes.
    - ``(full_read_view, None)`` if the batch completed during
      this step.

    Fast path: if ``handle.done`` (the batch already completed on a
    previous ``step`` / ``try_step``), this returns
    ``(_all_read_view, None)`` immediately without entering the
    Driver — re-stepping a finished batch is idempotent.

    Strict on retry: if the batch yields ``await again()``, ``step``
    raises ``RuntimeError`` — the call site declared retry to be a
    programming error by choosing ``step``. Use :func:`try_step` if
    the batch may legitimately retry.

    ``step`` does not nest. The scheduler drives batches
    sequentially via successive ``step`` calls, never two
    simultaneously. The non-nesting assert at entry catches any
    accidental misuse early.

    The batch's own ``_active_phase`` at the moment of pop is
    captured into ``handle.saved_phase`` so the next ``step()`` /
    ``try_step()`` can restore it.
    """
    result = await _do_step(handle, through, allow_retry=False)
    # _do_step never returns None when allow_retry=False (it raises
    # instead). The cast makes the type narrow for callers using the
    # untyped (generic) overload.
    assert result is not None
    return result


async def try_step(
    handle: Batch, *, through: IntEnum
) -> Optional[Tuple[object, Optional[object]]]:
    """Like :func:`step`, but report retry instead of raising.

    Returns ``None`` if the batch yielded ``await again()`` instead of
    progressing past ``through``. The caller decides what to do —
    typically retry on a later iter, or skip this batch for now.

    Otherwise returns the same ``(read, write)`` shape as :func:`step`,
    including ``(full_read_view, None)`` on terminal completion.
    Distinguishing retry from terminal: retry returns a bare ``None``;
    terminal returns ``(read_view, None)``.

    Fast path: if ``handle.done``, returns ``(_all_read_view, None)``
    immediately — same idempotent behavior as :func:`step` on a
    finished batch.

    Same non-nesting semantics as :func:`step`.
    """
    return await _do_step(handle, through, allow_retry=True)


async def _do_step(
    handle: Batch,
    through: IntEnum,
    *,
    allow_retry: bool,
) -> Optional[Tuple[object, Optional[object]]]:
    """Common implementation for :func:`step` and :func:`try_step`.

    Differs only in the retry policy: ``allow_retry=False`` raises on
    ``_RETRY``, ``allow_retry=True`` returns ``None``.
    """
    if handle.done:
        # Batch coroutine has already completed. Skip the Driver
        # round-trip and return the terminal view directly. Both
        # ``step`` and ``try_step`` agree on this: completion is
        # not retry, and is observable through the storage.
        return _all_read_view(handle.storage), None
    if _active_storage.get() is not None:
        raise RuntimeError(
            "step()/try_step() does not nest; another batch is currently active"
        )
    _active_storage.set(handle.storage)
    _active_phase.set(handle.saved_phase)
    try:
        sent = await _yield_advance(handle, through)
        if sent is _RETRY:
            if allow_retry:
                return None
            raise RuntimeError(
                f"step({handle.coro!r}, through={through!r}): batch "
                f"issued `await again()` but the scheduler used step(), "
                f"which forbids retry. Use try_step() if the batch may "
                f"retry."
            )
        # ``_active_phase`` now reflects where the batch is paused:
        # - inside a batch_phase CM at the wait that triggered the pop, or
        # - ``None`` if the batch completed (last CM cleared it).
        next_phase = _active_phase.get()
        if next_phase is None:
            return _all_read_view(handle.storage), None
        read_view = _views_at(handle.storage, next_phase)[0]
        write_view = _views_at(handle.storage, through)[1]
        return read_view, write_view
    finally:
        # Capture the batch's CM state for the next step()/try_step()
        # to restore. ``set(None)`` rather than ``reset(token)`` for
        # cross-Context cleanup robustness; see :func:`batch_phase`.
        # On retry the captured value reflects the batch's *unchanged*
        # phase, since `again()` doesn't touch ``_active_phase``.
        handle.saved_phase = _active_phase.get()
        _active_phase.set(None)
        _active_storage.set(None)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


class Driver:
    """Pumps a top-level coroutine, interpreting its yielded requests.

    The Driver decides nothing about scheduling order; the top-level
    coroutine's sequence of ``step`` / ``resume`` calls IS the
    schedule. The Driver interprets three request sentinels —
    :class:`_WaitRequest`, :class:`_AdvanceRequest`, and
    :class:`_RetryRequest` — and manages a stack for nested advance.

    Stack uniformity: every entry on the stack is a *handle*
    (:class:`Batch`, :class:`Concern`, or the private
    :class:`_RootHandle` wrapping the scheduler coroutine). The
    Driver flips ``handle.done`` when a coroutine completes (normally
    or by exception) and when the Driver closes the coroutine during
    shutdown. User-side fast paths in ``step`` / ``resume`` /
    ``try_step`` / ``try_resume`` consult ``handle.done`` to skip
    re-entering the Driver for an already-finished handle.

    Pop predicate is strict ``>``: a child is popped when its current
    wait phase is strictly greater than the parent's advance target
    — i.e., the child has finished the requested phase's work.

    Strict-progression check on ``_WaitRequest``: a coroutine yielding
    a wait at phase ``P`` after a previous wait at ``Pprev`` raises
    if ``P <= Pprev`` (regression or stay-at-same-phase).

    Retry on ``_RetryRequest``: the Driver pops the coroutine and
    sends :data:`_RETRY` to the parent's ``_yield_advance``. The
    coroutine's suspension record and the active state's
    ``_active_phase`` / ``_active_storage`` are intentionally left
    unchanged — when the parent next drives this coroutine, it picks
    up exactly where it left off, with the same view of storage. A
    retry from the root (no parent) is a programming error and is
    raised back into the offending coroutine.

    The Driver is policy-free about main completion: it returns when
    main returns and raises what main raises. Higher layers decide
    whether a normal return is legitimate.

    Shutdown is Python-native: ``close()`` throws ``GeneratorExit``
    into the main coroutine. Children held on the stack are closed
    in LIFO order so deep unwinds run their cleanup before parents.
    """

    def __init__(self, main: _CoroutineLike) -> None:
        self._main_handle: _RootHandle = _RootHandle(main)
        # Stack of currently-active handles:
        #   (handle, target_phase_or_None, last_storage_or_None)
        # ``target_phase`` is the phase the parent asked the child to
        # reach via ``step`` / ``resume``; ``None`` marks the root.
        self._stack: List[Tuple[_Handle, Optional[IntEnum], Optional[object]]] = []
        # Handles whose coroutine has yielded a ``_WaitRequest`` and
        # are suspended at a known phase. Keyed by ``id(handle.coro)``.
        # Used for the "child already past target" short-circuit and
        # for the strict-progression check.
        self._suspensions: dict[int, Tuple[IntEnum, object]] = {}

    # --------------------------------------------------------------- run #

    def run(self) -> None:
        """Pump the main coroutine until it completes or raises.

        Runs ``_drive`` and ``_close_all`` inside a fresh
        ``contextvars.Context`` copied from the caller's context. Any
        ``ContextVar`` mutation made during the run (most notably
        :data:`_active_storage` / :data:`_active_phase` flipped by
        ``step`` and ``batch_phase``) is isolated to this Driver and
        does not leak back to the caller. Multiple ``Driver``
        instances in the same thread are independent.

        Normal completion returns; exceptions propagate. Neither is
        interpreted as policy.
        """
        ctx = contextvars.copy_context()
        ctx.run(self._run_in_context)

    def _run_in_context(self) -> None:
        """Inner of :meth:`run`. Runs in a fresh contextvars.Context."""
        try:
            self._drive()
        finally:
            self._close_all()

    def _drive(self) -> None:
        """Main loop: pump the top of stack, dispatch on yielded requests."""
        self._stack = [(self._main_handle, None, None)]
        send_value: object = None
        send_exc: Optional[BaseException] = None

        while self._stack:
            handle, target, last_storage = self._stack[-1]
            coro = handle.coro

            try:
                if send_exc is not None:
                    exc = send_exc
                    send_exc = None
                    request = coro.throw(exc)
                else:
                    request = coro.send(send_value)
                    send_value = None
            except StopIteration:
                # Coroutine returned. Mark the handle done so user-side
                # fast paths short-circuit on the next call; pop the
                # stack and resume the parent (if any).
                handle.done = True
                self._stack.pop()
                self._suspensions.pop(id(coro), None)
                # Parent's resume value is None — under the new design,
                # advance/resume returns are unused (data exchange is
                # via active state, not yield-return).
                send_value = None
                continue
            except BaseException as e:
                # Coroutine raised. Mark handle done (terminated by
                # exception is still terminated) and propagate to
                # parent (if any), or out of run() at the root.
                handle.done = True
                self._stack.pop()
                self._suspensions.pop(id(coro), None)
                if self._stack:
                    send_exc = e
                    continue
                _hide_framework_frames(e)
                raise

            # Dispatch on the yielded request.
            if isinstance(request, _AdvanceRequest):
                child_handle = request.child
                new_target = request.at
                if child_handle.done:
                    # Child already completed (e.g., a previous
                    # ``resume`` consumed it). The user-side ``resume``
                    # / ``step`` fast path should normally catch this,
                    # but a defensive short-circuit here handles any
                    # path that builds an ``_AdvanceRequest`` without
                    # going through the typed helpers.
                    send_value = None
                    continue
                child_coro = child_handle.coro
                suspension = self._suspensions.get(id(child_coro))
                if suspension is not None and suspension[0] > new_target:
                    # Child is already past the target — no pumping
                    # needed. Resume the parent immediately.
                    send_value = None
                else:
                    # Push child handle for pumping.
                    self._stack.append((child_handle, new_target, None))
                    send_value = None
            elif isinstance(request, _WaitRequest):
                phase_value = request.phase
                storage = request.storage
                # Strict-progression check: a coroutine's successive
                # waits must yield strictly greater phases.
                prev = self._suspensions.get(id(coro))
                if prev is not None and phase_value <= prev[0]:
                    send_exc = RuntimeError(
                        f"enter_phase({phase_value.name}) does not strictly "
                        f"advance from the coroutine's previous phase "
                        f"{prev[0].name}"
                    )
                    continue
                self._suspensions[id(coro)] = (phase_value, storage)
                # Update the current stack entry's last_storage.
                self._stack[-1] = (handle, target, storage)
                if target is None:
                    # Root main coroutine yielded a wait. Treat as a
                    # no-op and keep pumping main.
                    send_value = None
                elif phase_value > target:
                    # Child has moved past the target. Pop; parent
                    # resumes (with no view return — data exchange is
                    # via active state).
                    self._stack.pop()
                    send_value = None
                else:
                    # Child still needs more pumping to reach target.
                    send_value = None
            elif isinstance(request, _RetryRequest):
                # `await again()`: the coroutine is at some phase and
                # cannot move forward yet. Pop to its parent and signal
                # via :data:`_RETRY`. The suspension record and active
                # state (``_active_phase`` / ``_active_storage``) are
                # left as-is so the next ``try_step`` / ``try_resume``
                # restores them and the coroutine resumes from right
                # after ``await again()`` with the same world view.
                # ``handle.done`` is also unchanged: the coroutine is
                # paused, not finished.
                if target is None:
                    send_exc = RuntimeError(
                        "again() at the root coroutine has no parent "
                        "to retry against; root must complete or raise"
                    )
                    continue
                self._stack.pop()
                send_value = _RETRY
            else:
                raise RuntimeError(f"Unknown request yielded by coroutine: {request!r}")

    # --------------------------------------------------------------- close #

    def close(self) -> None:
        """Close the main coroutine and any children still on the stack."""
        self._close_all()

    def _close_all(self) -> None:
        """Close stack LIFO then main. Swallow per-coro ``Exception``s.

        ``BaseException`` (``KeyboardInterrupt``, ``SystemExit``) is
        intentionally allowed to propagate so a Ctrl-C during shutdown
        actually stops shutdown. Each closed handle is marked
        ``done`` so its ``__del__`` skips the redundant ``close()``.
        """
        main_closed = False
        while self._stack:
            handle, _, _ = self._stack.pop()
            if handle is self._main_handle:
                main_closed = True
            try:
                handle.coro.close()
            except Exception:
                pass
            handle.done = True
        self._suspensions.clear()
        if not main_closed:
            try:
                self._main_handle.coro.close()
            except Exception:
                pass
            self._main_handle.done = True
