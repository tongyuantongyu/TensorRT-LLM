"""Regenerate the phase-scoped view / overload block in batch_storage.py.

The batch data model in
``tensorrt_llm/_torch/pyexecutor/batch_storage.py`` uses a single source
of truth — the ``phase`` metadata on ``BatchStorage`` fields — to define:

- ``_ReadAtP*`` Protocols    : cumulative read views up to each phase.
- ``_ReadAtAll`` Protocol     : the cumulative view at the terminal
  ``step`` (i.e., everything produced through the last phase).
- ``_WriteAtP*`` dataclasses  : phase-local write views.
- ``@overload`` signatures for ``step`` : pair ``Literal[LoopPhase.Py]``
  with ``(_ReadAtP{y+1}, _WriteAtPy)`` for non-terminal phases and
  with ``(_ReadAtAll, None)`` for the terminal phase.
- ``@overload`` signatures for ``try_step`` : same pairing wrapped in
  ``Optional[...]`` to surface the retry-on-EAGAIN ``None`` return.

This script reads the metadata and rewrites the generated block in
batch_storage.py, between the ``# ===== BEGIN GENERATED`` and
``# ===== END GENERATED`` marker lines.

Usage
-----
Refresh the block in place::

    python scripts/generate_coroutine_views.py

CI / pre-commit check (exit 1 on drift, no writes)::

    python scripts/generate_coroutine_views.py --check

When to run
-----------
After adding, removing, or re-phasing any field on ``BatchStorage``, or
after adding a new ``LoopPhase`` member. The tests under
``tests/unittest/_torch/executor/test_coroutines.py`` include a
``--check``-equivalent test that fails if the block has drifted.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

# The script imports the batch data model via the canonical package
# path. In practice this works in a standard TRT-LLM dev environment;
# if the heavy top-level ``tensorrt_llm`` package init fails on a given
# host (e.g. without a GPU driver for NVML), running this script is
# pointless on that host anyway — the generator isn't a test, it's a
# dev tool.
from tensorrt_llm._torch.pyexecutor.batch_storage import BatchStorage, LoopPhase

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "batch_storage.py"

BEGIN_MARKER = "# ===== BEGIN GENERATED from BatchStorage ====="
END_MARKER = "# ===== END GENERATED ====="


# --------------------------------------------------------------------------- #
# Block generation
# --------------------------------------------------------------------------- #


def _fields_by_phase(
    storage_cls: type,
) -> dict[LoopPhase, List[dataclasses.Field]]:
    """Group a storage's fields by the phase that produces them."""
    result: dict[LoopPhase, List[dataclasses.Field]] = {p: [] for p in LoopPhase}
    for f in dataclasses.fields(storage_cls):
        phase = f.metadata.get("phase")
        if phase is not None:
            result[phase].append(f)
    return result


def _unwrap_optional(annotation: str) -> str:
    """Try strip one ``Optional[X]`` wrapper.

    Fields are declared ``Optional`` so that "unset" is representable as
    ``None``. The read-view property annotations drop the wrapper
    because, by the happens-before contract, a field is guaranteed to
    be set (non-``None``) once a reader at a later phase observes it.
    """
    stripped = annotation.strip()
    prefix, suffix = "Optional[", "]"
    if stripped.startswith(prefix) and stripped.endswith(suffix):
        return stripped[len(prefix) : -len(suffix)]
    return stripped


def _sorted_phases() -> List[LoopPhase]:
    return sorted(LoopPhase, key=lambda p: p.value)


def _render_read_protocol(
    phase: LoopPhase,
    new_fields: List[dataclasses.Field],
    is_first: bool,
) -> List[str]:
    """Emit a cumulative read-view Protocol for `phase`."""
    name = f"_ReadAtP{phase.value}"
    lines = ["@runtime_checkable"]
    if is_first:
        lines.append(f"class {name}(Protocol):")
        lines.append(
            f'    """Readable fields at phase P{phase.value} — no field is produced yet."""'
        )
        return lines

    prev = f"_ReadAtP{phase.value - 1}"
    lines.append(f"class {name}({prev}, Protocol):")
    lines.append(
        f'    """Readable fields at phase P{phase.value} — '
        f'cumulative through P{phase.value - 1}."""'
    )
    for f in new_fields:
        type_str = _unwrap_optional(f.type)
        lines.append("")
        lines.append("    @property")
        lines.append(f"    def {f.name}(self) -> {type_str}: ...")
    return lines


def _render_write_dataclass(
    phase: LoopPhase,
    fields: List[dataclasses.Field],
) -> List[str]:
    """Emit a phase-local write-view dataclass for `phase`."""
    name = f"_WriteAtP{phase.value}"
    lines = ["@dataclasses.dataclass", f"class {name}:"]
    if not fields:
        lines.append(
            f'    """Fields producible at phase P{phase.value} — no field is produced here."""'
        )
        return lines
    lines.append(f'    """Fields producible at phase P{phase.value}."""')
    lines.append("")
    for f in fields:
        lines.append(f"    {f.name}: {f.type} = None")
    return lines


def _render_read_at_all(
    last_phase: LoopPhase,
    last_phase_fields: List[dataclasses.Field],
) -> List[str]:
    """Emit the cumulative read-view Protocol used by terminal ``step``.

    Inherits from the per-phase ``_ReadAtP{LAST}`` Protocol and adds
    properties for whatever fields the last phase produces (so reading
    via ``step(through=P_LAST)``'s return exposes those too).
    """
    prev = f"_ReadAtP{last_phase.value}"
    lines = [
        "@runtime_checkable",  #
        f"class _ReadAtAll({prev}, Protocol):",
        f'    """Readable after all phases — cumulative through P{last_phase.value}."""',
    ]
    for f in last_phase_fields:
        type_str = _unwrap_optional(f.type)
        lines.append("")
        lines.append("    @property")
        lines.append(f"    def {f.name}(self) -> {type_str}: ...")
    return lines


def _render_step_overload(phase: LoopPhase, is_terminal: bool) -> List[str]:
    """Emit one ``@overload`` signature for ``step``.

    Non-terminal: ``(_ReadAtP{y+1}, _WriteAtPy)`` — the read view
    exposes everything produced through ``Py`` and the write view is
    the slot for filling Py-fields the batch left as holes.

    Terminal: ``(_ReadAtAll, None)`` — the batch has completed,
    everything is visible, no further write slot exists.
    """
    if is_terminal:
        read = "_ReadAtAll"
        write = "None"
    else:
        read = f"_ReadAtP{phase.value + 1}"
        write = f"_WriteAtP{phase.value}"
    params = [
        "    handle: Batch,",
        "    *,",
        f"    through: Literal[LoopPhase.P{phase.value}],",
    ]
    return [
        "@overload",
        "async def step(",
        *params,
        f") -> Tuple[{read}, {write}]: ...",
    ]


def _render_try_step_overload(phase: LoopPhase, is_terminal: bool) -> List[str]:
    """Emit one ``@overload`` signature for ``try_step``.

    Mirrors :func:`_render_step_overload` but wraps the return type in
    ``Optional[...]``. ``None`` from ``try_step`` means "the batch
    issued ``await again()``" (retry); otherwise the return shape is
    identical to :func:`step`.
    """
    if is_terminal:
        read = "_ReadAtAll"
        write = "None"
    else:
        read = f"_ReadAtP{phase.value + 1}"
        write = f"_WriteAtP{phase.value}"
    params = [
        "    handle: Batch,",
        "    *,",
        f"    through: Literal[LoopPhase.P{phase.value}],",
    ]
    return [
        "@overload",
        "async def try_step(",
        *params,
        f") -> Optional[Tuple[{read}, {write}]]: ...",
    ]


def _render_batch_phase_overload(phase: LoopPhase) -> List[str]:
    """Emit one ``@overload`` signature for ``batch_phase``.

    The overload uses the pattern::

        @overload
        @asynccontextmanager
        def batch_phase(
            p: Literal[LoopPhase.Py],
        ) -> AsyncIterator[Tuple[_ReadAtPy, _WriteAtPy]]: ...

    Stacking ``@overload`` outside ``@asynccontextmanager`` and using
    ``AsyncIterator`` as the return is what most IDEs / type
    checkers (PyCharm, mypy, pyright) need to correctly narrow
    ``async with batch_phase(LoopPhase.Py) as (r, w):`` to the right
    view types. Returning ``AbstractAsyncContextManager`` directly
    works for some checkers but is missed by others, so we mirror
    the pattern stdlib's own ``asynccontextmanager``-typed overloads
    use.
    """
    read = "None" if phase.value == 0 else f"_ReadAtP{phase.value}"
    write = f"_WriteAtP{phase.value}"
    return [
        "@overload",
        "@asynccontextmanager",
        "def batch_phase(",
        f"    p: Literal[LoopPhase.P{phase.value}],",
        f") -> AsyncIterator[Tuple[{read}, {write}]]: ...",
    ]


def generate_block() -> str:
    """Return the full generated block content (no BEGIN/END markers).

    The returned text is the exact body that should sit between the
    marker lines in ``batch_storage.py``. Trailing newline not included.
    """
    groups = _fields_by_phase(BatchStorage)
    phases = _sorted_phases()

    out: List[str] = []

    # Per-phase read Protocols.
    for i, phase in enumerate(phases):
        if i == 0:
            new_fields: List[dataclasses.Field] = []
        else:
            # Each `_ReadAtP{k}` adds exactly the fields produced at
            # the immediately previous phase; earlier ones come in via
            # Protocol inheritance.
            new_fields = groups[phases[i - 1]]
        out.extend(_render_read_protocol(phase, new_fields, i == 0))
        out.append("")
        out.append("")

    # Terminal-phase read Protocol: cumulative through the LAST phase
    # (i.e., includes the LAST phase's own fields, if any).
    last_phase = phases[-1]
    out.extend(_render_read_at_all(last_phase, groups[last_phase]))
    out.append("")
    out.append("")

    # Write dataclasses.
    for phase in phases:
        out.extend(_render_write_dataclass(phase, groups[phase]))
        out.append("")
        out.append("")

    # step overloads.
    for i, phase in enumerate(phases):
        is_terminal = i == len(phases) - 1
        out.extend(_render_step_overload(phase, is_terminal))
    out.append("")
    out.append("")

    # try_step overloads (same shape, Optional-wrapped return).
    for i, phase in enumerate(phases):
        is_terminal = i == len(phases) - 1
        out.extend(_render_try_step_overload(phase, is_terminal))
    out.append("")
    out.append("")

    # batch_phase overloads.
    for phase in phases:
        out.extend(_render_batch_phase_overload(phase))

    # Strip any trailing blank lines to keep the output stable.
    while out and out[-1] == "":
        out.pop()

    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# File update / drift check
# --------------------------------------------------------------------------- #


def _split_on_markers(text: str) -> Tuple[str, str, str]:
    """Return ``(prefix, current_body, suffix)`` for a marker-bracketed region."""
    begin = text.index(BEGIN_MARKER)
    after_begin = text.index("\n", begin) + 1  # consume the begin marker line
    end = text.index(END_MARKER, after_begin)
    prefix = text[:after_begin]
    body = text[after_begin:end]
    suffix = text[end:]
    return prefix, body, suffix


def _assemble(prefix: str, body: str, suffix: str) -> str:
    """Re-join a marker-bracketed file after replacing ``body``.

    Two blank lines surround the generated block on each side — that's
    the project formatter's spacing for "top-level declarations around
    a stretch of comments". Matching it here keeps the formatter from
    having to reshape the region after every regeneration.
    """
    core = body.strip("\n")
    return prefix + "\n\n" + core + "\n\n\n" + suffix


def update_in_place(target: Path = TARGET) -> bool:
    """Refresh the generated block. Returns ``True`` if content changed."""
    text = target.read_text()
    prefix, current_body, suffix = _split_on_markers(text)
    new_body = generate_block()
    new_text = _assemble(prefix, new_body, suffix)
    if new_text == text:
        return False
    target.write_text(new_text)
    return True


def diff_report(target: Path = TARGET) -> Iterable[str]:
    """Yield unified-diff lines for the meaningful content drift.

    Boundary blanks around the block are normalized away so the diff
    highlights only the declarations that differ, not the two-blank-line
    padding that ``_assemble`` manages. The authoritative "is the file
    correct?" call is :func:`is_up_to_date`, which is strict.
    """
    text = target.read_text()
    _, current_body, _ = _split_on_markers(text)
    current_core = current_body.strip("\n") + "\n"
    expected_core = generate_block().rstrip("\n") + "\n"
    yield from difflib.unified_diff(
        current_core.splitlines(keepends=True),
        expected_core.splitlines(keepends=True),
        fromfile=str(target) + " (current generated block)",
        tofile=str(target) + " (expected generator output)",
        lineterm="",
    )


def is_up_to_date(target: Path = TARGET) -> bool:
    """Return ``True`` iff running ``--check`` would be a no-op.

    Strict byte-for-byte comparison of the full assembled file against
    the current contents: any drift in the generated region — including
    the two blank lines the formatter enforces at each end — is
    treated as a failure. Drift outside the region is impossible to
    trigger here because ``_assemble`` copies ``prefix`` / ``suffix``
    verbatim.
    """
    text = target.read_text()
    prefix, _, suffix = _split_on_markers(text)
    expected = _assemble(prefix, generate_block(), suffix)
    return text == expected


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if the generated block is out of date; do not write.",
    )
    args = parser.parse_args(argv)

    if args.check:
        if is_up_to_date():
            return 0
        sys.stderr.write(
            f"Generated block in {TARGET} is out of date. "
            "Run `python scripts/generate_coroutine_views.py` to refresh.\n\n"
        )
        sys.stderr.writelines(diff_report())
        return 1

    if update_in_place():
        print(f"Updated {TARGET}")
    else:
        print(f"{TARGET} already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
