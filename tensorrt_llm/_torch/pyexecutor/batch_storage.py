"""Batch data model for the PyExecutor coroutine runtime.

The runtime in :mod:`tensorrt_llm._torch.pyexecutor.coroutines` is
generic over phase enum and storage class. This module pins down the
*specific* ``LoopPhase`` and ``BatchStorage`` used by TensorRT-LLM's
forward loop, plus the typed ``@overload`` chain that narrows
``step`` for users of ``BatchStorage``.

Layering
========

- :mod:`coroutines` exposes the generic primitives (``enter_phase``,
  ``batch_phase`` CM, ``resume``, ``step``, ``Batch``, ``Driver``,
  ``spawn``, ``phased_field`` and the runtime proxies). It never
  references ``LoopPhase`` or ``BatchStorage``.
- This module imports those primitives and provides the production
  data model on top: phase enum, storage dataclass, and the typed
  views generated from the storage's ``phase`` metadata.

Production code that creates / drives batches imports from here;
runtime tests that exercise the generic mechanics import directly from
:mod:`coroutines` against their own test-local storage definitions.

Adding a new field
==================

1. Add ``Optional[T] = phased_field(LoopPhase.X)`` to ``BatchStorage``.
2. Run ``python scripts/generate_coroutine_views.py`` to refresh the
   ``# ===== BEGIN GENERATED =====`` … ``# ===== END GENERATED =====``
   region below. Never edit that region by hand.

Two tests guard the contract: the static-vs-runtime sync tests in
``tests/unittest/_torch/executor/test_coroutines.py`` and the
generator-output check, which runs the generator's ``--check`` mode
against the committed file.
"""

from __future__ import annotations

import dataclasses
from contextlib import asynccontextmanager
from enum import IntEnum
from typing import AsyncIterator, Literal, Optional, Protocol, Tuple, overload, runtime_checkable

from tensorrt_llm._torch.pyexecutor.coroutines import (
    Driver,
    Batch,
    again,
    enter_phase,
    phased_field,
    resume,
    spawn,
    try_resume,
)
from tensorrt_llm._torch.pyexecutor.coroutines import batch_phase as _generic_batch_phase
from tensorrt_llm._torch.pyexecutor.coroutines import step as _generic_step
from tensorrt_llm._torch.pyexecutor.coroutines import try_step as _generic_try_step

__all__ = [
    "Driver",
    "BatchStorage",
    "Batch",
    "LoopPhase",
    "again",
    "enter_phase",
    "batch_phase",
    "phased_field",
    "resume",
    "spawn",
    "step",
    "try_resume",
    "try_step",
]


# --------------------------------------------------------------------------- #
# Phase enum (placeholder values — real batch phases are a followup)
# --------------------------------------------------------------------------- #


class LoopPhase(IntEnum):
    """Placeholder phases used to exercise the runtime.

    Real phase values (the batch's actual lifecycle) are a followup.
    These four values are sufficient to drive every concrete coroutine
    the executor needs to write today.
    """

    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3


# --------------------------------------------------------------------------- #
# Storage (hand-written — edit here, then regenerate the block below)
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class BatchStorage:
    """Mutable storage holding all batch-level fields across phases.

    Held by an ``Batch`` handle. ``step()`` hands out typed views
    derived from the field metadata. Fields are ``Optional`` and
    default to ``None``. Each field carries a ``{"phase": LoopPhase.P_k}``
    metadata entry (via :func:`phased_field`) declaring which phase
    produces it.

    The field metadata is the single source of truth. It drives:

    - ``_ReadAtP*`` / ``_WriteAtP*`` Protocols and the ``step``
      ``@overload`` signatures in the generated block below; refresh
      them with ``scripts/generate_coroutine_views.py``.
    - The runtime ``_TrackedReadView`` / ``_TrackedWriteView`` proxies
      (``TLLM_COROUTINE_TRACK_STORAGE=1``) — recompute the map on each
      view construction, no codegen needed.
    """

    p0_out: Optional[int] = phased_field(LoopPhase.P0)
    p1_out: Optional[str] = phased_field(LoopPhase.P1)
    p2_out: Optional[bytes] = phased_field(LoopPhase.P2)


# --------------------------------------------------------------------------- #
# Generated phase-scoped views + overload signatures.
#
# The region between the BEGIN / END markers is produced from
# ``BatchStorage``'s ``phase`` metadata by
# ``scripts/generate_coroutine_views.py``. Edit that script (or the
# metadata), do not edit the block by hand.
#
# A CI test runs the generator in --check mode and fails if the block
# has drifted from ``BatchStorage``'s field declarations.
# --------------------------------------------------------------------------- #


# ===== BEGIN GENERATED from BatchStorage =====


@runtime_checkable
class _ReadAtP0(Protocol):
    """Readable fields at phase P0 — no field is produced yet."""


@runtime_checkable
class _ReadAtP1(_ReadAtP0, Protocol):
    """Readable fields at phase P1 — cumulative through P0."""

    @property
    def p0_out(self) -> int: ...


@runtime_checkable
class _ReadAtP2(_ReadAtP1, Protocol):
    """Readable fields at phase P2 — cumulative through P1."""

    @property
    def p1_out(self) -> str: ...


@runtime_checkable
class _ReadAtP3(_ReadAtP2, Protocol):
    """Readable fields at phase P3 — cumulative through P2."""

    @property
    def p2_out(self) -> bytes: ...


@runtime_checkable
class _ReadAtAll(_ReadAtP3, Protocol):
    """Readable after all phases — cumulative through P3."""


@dataclasses.dataclass
class _WriteAtP0:
    """Fields producible at phase P0."""

    p0_out: Optional[int] = None


@dataclasses.dataclass
class _WriteAtP1:
    """Fields producible at phase P1."""

    p1_out: Optional[str] = None


@dataclasses.dataclass
class _WriteAtP2:
    """Fields producible at phase P2."""

    p2_out: Optional[bytes] = None


@dataclasses.dataclass
class _WriteAtP3:
    """Fields producible at phase P3 — no field is produced here."""


@overload
async def step(
    handle: Batch,
    *,
    through: Literal[LoopPhase.P0],
) -> Tuple[_ReadAtP1, _WriteAtP0]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[LoopPhase.P1],
) -> Tuple[_ReadAtP2, _WriteAtP1]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[LoopPhase.P2],
) -> Tuple[_ReadAtP3, _WriteAtP2]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[LoopPhase.P3],
) -> Tuple[_ReadAtAll, None]: ...


@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[LoopPhase.P0],
) -> Optional[Tuple[_ReadAtP1, _WriteAtP0]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[LoopPhase.P1],
) -> Optional[Tuple[_ReadAtP2, _WriteAtP1]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[LoopPhase.P2],
) -> Optional[Tuple[_ReadAtP3, _WriteAtP2]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[LoopPhase.P3],
) -> Optional[Tuple[_ReadAtAll, None]]: ...


@overload
@asynccontextmanager
def batch_phase(
    p: Literal[LoopPhase.P0],
) -> AsyncIterator[Tuple[None, _WriteAtP0]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[LoopPhase.P1],
) -> AsyncIterator[Tuple[_ReadAtP1, _WriteAtP1]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[LoopPhase.P2],
) -> AsyncIterator[Tuple[_ReadAtP2, _WriteAtP2]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[LoopPhase.P3],
) -> AsyncIterator[Tuple[_ReadAtP3, _WriteAtP3]]: ...


# ===== END GENERATED =====


# --------------------------------------------------------------------------- #
# Implementations. The ``@overload`` chains above describe the typed
# signatures; these impls just delegate to the generic primitives in
# ``coroutines``.
# --------------------------------------------------------------------------- #


async def step(handle, *, through):  # type: ignore[misc]
    """``step`` narrowed for :class:`BatchStorage` — see overloads above."""
    return await _generic_step(handle, through=through)


async def try_step(handle, *, through):  # type: ignore[misc]
    """``try_step`` narrowed for :class:`BatchStorage` — see overloads above."""
    return await _generic_try_step(handle, through=through)


def batch_phase(p):  # type: ignore[misc]
    """``batch_phase`` narrowed for :class:`BatchStorage` — see overloads above.

    Sync function returning an ``AbstractAsyncContextManager`` (the
    runtime decorates the underlying generator with
    ``@asynccontextmanager``). The narrowed return type makes
    ``async with batch_phase(LoopPhase.Py) as (r, w):`` give ``r`` and
    ``w`` the proper ``_ReadAtPy`` / ``_WriteAtPy`` types at every
    call site.
    """
    return _generic_batch_phase(p)
