"""Regenerate the phase-scoped view / overload block in batch_storage.py.

The batch data model in
``tensorrt_llm/_torch/pyexecutor/batch_storage.py`` uses a single source
of truth — the ``phase`` metadata on ``BatchStorage`` fields — to define:

- ``_ReadAt<Name>`` Protocols : cumulative read views up to each phase
  (one per ``BatchPhase`` member, named after the member with the
  enum's UPPER_SNAKE alpha segments converted to PascalCase and the
  trailing numeric segment preserved with its underscore; e.g.
  ``_ReadAtSchedule_0``, ``_ReadAtForward_1``, ``_ReadAtStateUpd_3``).
- ``_ReadAtAll`` Protocol     : the cumulative view at the terminal
  ``step`` (i.e., everything produced through the last phase).
- ``_WriteAt<Name>`` dataclasses : phase-local write views (one per
  ``BatchPhase`` member, same naming convention).
- ``@overload`` signatures for ``step`` : pair
  ``Literal[BatchPhase.<NAME>]`` with
  ``(_ReadAt<Next>, _WriteAt<Name>)`` for non-terminal phases and
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
after adding a new ``BatchPhase`` member. The tests under
``tests/unittest/_torch/executor/test_coroutines.py`` include a
``--check``-equivalent test that fails if the block has drifted.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import difflib
import inspect
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# The script imports the batch data model via the canonical package
# path. In practice this works in a standard TRT-LLM dev environment;
# if the heavy top-level ``tensorrt_llm`` package init fails on a given
# host (e.g. without a GPU driver for NVML), running this script is
# pointless on that host anyway — the generator isn't a test, it's a
# dev tool.
from tensorrt_llm._torch.pyexecutor.batch_storage import BatchStorage, BatchPhase

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET = REPO_ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "batch_storage.py"

BEGIN_MARKER = "# ===== BEGIN GENERATED from BatchStorage ====="
END_MARKER = "# ===== END GENERATED ====="


# --------------------------------------------------------------------------- #
# Block generation
# --------------------------------------------------------------------------- #


def _fields_by_phase(
    storage_cls: type,
) -> dict[BatchPhase, List[dataclasses.Field]]:
    """Group a storage's fields by the phase that produces them."""
    result: dict[BatchPhase, List[dataclasses.Field]] = {p: [] for p in BatchPhase}
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


def _sorted_phases() -> List[BatchPhase]:
    return sorted(BatchPhase, key=lambda p: p.value)


def _extract_field_docstrings(
    source_path: Path,
    class_name: str,
) -> Dict[str, str]:
    """Walk the source AST, find ``class_name``, return ``{field_name: docstring}``.

    A "field docstring" here means a triple-quoted string literal that
    appears IMMEDIATELY AFTER an annotated assignment in the class
    body — the trailing-string convention many type checkers and IDEs
    treat as per-attribute documentation::

        scheduled_batch: Optional[ScheduledRequests] = phased_field(...)
        \"\"\"The scheduled set of requests for this batch.\"\"\"

    The generator copies these docstrings into the generated read /
    write views so a user hovering on ``r.scheduled_batch`` (or
    ``w.scheduled_batch``) sees the same one-paragraph "what is it"
    explanation that BatchStorage itself carries — without the
    Producer/Consumer lifecycle comments (those live as ``# `` lines
    ABOVE the field in BatchStorage and are intentionally not
    propagated here; they bloat the IDE quick-doc and are not what a
    caller needs at the use site).
    """
    tree = ast.parse(source_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return _docs_from_class_body(node.body)
    return {}


def _docs_from_class_body(body: List[ast.stmt]) -> Dict[str, str]:
    """Pair each ``AnnAssign`` field with its immediately-following docstring."""
    result: Dict[str, str] = {}
    pending_field: Optional[str] = None
    for stmt in body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            pending_field = stmt.target.id
            continue
        if (
            pending_field is not None
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            result[pending_field] = stmt.value.value
        pending_field = None
    return result


def _format_docstring(raw: str, indent: str) -> List[str]:
    """Render a docstring at ``indent``; single- vs multi-line aware.

    Uses :func:`inspect.cleandoc` to strip the source's baked-in
    indentation (a triple-quoted block in a class body carries the
    class's indent on every line except the first), then re-indents
    for the target placement.

    Single-line docstrings collapse to one triple-quoted line at
    ``indent``. Multi-line docstrings emit an opening triple-quote
    + first line, the body lines re-indented to ``indent``, and a
    closing triple-quote on its own line at ``indent`` -- matching
    the convention the rest of the file uses.
    """
    cleaned = inspect.cleandoc(raw)
    if not cleaned:
        return []
    if "\n" not in cleaned:
        return [f'{indent}"""{cleaned}"""']
    lines = cleaned.split("\n")
    out = [f'{indent}"""{lines[0]}']
    for line in lines[1:]:
        out.append(f"{indent}{line}" if line else "")
    out.append(f'{indent}"""')
    return out


def _phase_pascal(phase: BatchPhase) -> str:
    """Render a phase as ``<PascalCaseAlpha>_<number>``.

    The enum names are UPPER_SNAKE_CASE with a trailing ``_<int>`` segment
    that mirrors the enum value (e.g. ``SCHEDULE_0``, ``STATE_UPD_3``,
    ``FINALIZE_8``). For type-name purposes — ``_ReadAt<...>`` /
    ``_WriteAt<...>`` — UPPER_SNAKE looks shouty and the inner
    underscores hurt readability. Convert the alpha segments to
    PascalCase, concatenate them, and re-attach ``_<number>`` so the
    relative phase order remains visible at a glance:

    - ``SCHEDULE_0``  -> ``Schedule_0``
    - ``STATE_UPD_3`` -> ``StateUpd_3``
    - ``SYNC_EVT_4``  -> ``SyncEvt_4``
    - ``FINALIZE_8``  -> ``Finalize_8``

    The trailing ``_<number>`` is kept (rather than dropping the
    underscore to make a single PascalCase token) to make the order
    cue visually distinct from the alpha part. ``BatchPhase.<NAME>``
    references inside generated code keep the canonical UPPER_SNAKE
    enum member names — this transform applies only to
    ``_ReadAt<...>`` / ``_WriteAt<...>`` type names.
    """
    *alpha, num = phase.name.split("_")
    pascal = "".join(seg.title() for seg in alpha)
    return f"{pascal}_{num}"


def _render_read_protocol(
    phase: BatchPhase,
    new_fields: List[dataclasses.Field],
    prev_phase: Optional[BatchPhase],
    field_docs: Dict[str, str],
) -> List[str]:
    """Emit a cumulative read-view Protocol for `phase`."""
    name = f"_ReadAt{_phase_pascal(phase)}"
    lines = ["@runtime_checkable"]
    if prev_phase is None:
        lines.append(f"class {name}(Protocol):")
        lines.append(
            f'    """Readable fields at phase {phase.name} — no field is produced yet."""'
        )
        return lines

    prev = f"_ReadAt{_phase_pascal(prev_phase)}"
    lines.append(f"class {name}({prev}, Protocol):")
    lines.append(
        f'    """Readable fields at phase {phase.name} — '
        f'cumulative through {prev_phase.name}."""'
    )
    for f in new_fields:
        type_str = _unwrap_optional(f.type)
        lines.append("")
        lines.append("    @property")
        doc = field_docs.get(f.name)
        if doc:
            # Multi-line: spell the property body explicitly so the
            # docstring sits where IDEs / Sphinx pick it up.
            lines.append(f"    def {f.name}(self) -> {type_str}:")
            lines.extend(_format_docstring(doc, indent="        "))
            lines.append("        ...")
        else:
            lines.append(f"    def {f.name}(self) -> {type_str}: ...")
    return lines


def _render_write_dataclass(
    phase: BatchPhase,
    fields: List[dataclasses.Field],
    field_docs: Dict[str, str],
) -> List[str]:
    """Emit a phase-local write-view dataclass for `phase`."""
    name = f"_WriteAt{_phase_pascal(phase)}"
    lines = ["@dataclasses.dataclass", f"class {name}:"]
    if not fields:
        lines.append(
            f'    """Fields producible at phase {phase.name} — no field is produced here."""'
        )
        return lines
    lines.append(f'    """Fields producible at phase {phase.name}."""')
    lines.append("")
    for f in fields:
        lines.append(f"    {f.name}: {f.type} = None")
        doc = field_docs.get(f.name)
        if doc:
            # Trailing-string convention: dataclass field "docstring".
            lines.extend(_format_docstring(doc, indent="    "))
    return lines


def _render_read_at_all(
    last_phase: BatchPhase,
    last_phase_fields: List[dataclasses.Field],
    field_docs: Dict[str, str],
) -> List[str]:
    """Emit the cumulative read-view Protocol used by terminal ``step``.

    Inherits from the per-phase ``_ReadAt<Last>`` Protocol and adds
    properties for whatever fields the last phase produces (so reading
    via ``step(through=BatchPhase.<LAST>)``'s return exposes those too).
    """
    prev = f"_ReadAt{_phase_pascal(last_phase)}"
    lines = [
        "@runtime_checkable",  #
        f"class _ReadAtAll({prev}, Protocol):",
        f'    """Readable after all phases — cumulative through {last_phase.name}."""',
    ]
    for f in last_phase_fields:
        type_str = _unwrap_optional(f.type)
        lines.append("")
        lines.append("    @property")
        doc = field_docs.get(f.name)
        if doc:
            lines.append(f"    def {f.name}(self) -> {type_str}:")
            lines.extend(_format_docstring(doc, indent="        "))
            lines.append("        ...")
        else:
            lines.append(f"    def {f.name}(self) -> {type_str}: ...")
    return lines


def _render_step_overload(
    phase: BatchPhase,
    next_phase: Optional[BatchPhase],
) -> List[str]:
    """Emit one ``@overload`` signature for ``step``.

    Non-terminal: ``(_ReadAt<Next>, _WriteAt<Phase>)`` — the read view
    exposes everything produced through ``<Phase>`` and the write view
    is the slot for filling <Phase>-fields the batch left as holes.

    Terminal: ``(_ReadAtAll, None)`` — the batch has completed,
    everything is visible, no further write slot exists.
    """
    if next_phase is None:
        read = "_ReadAtAll"
        write = "None"
    else:
        read = f"_ReadAt{_phase_pascal(next_phase)}"
        write = f"_WriteAt{_phase_pascal(phase)}"
    params = [
        "    handle: Batch,",
        "    *,",
        f"    through: Literal[BatchPhase.{phase.name}],",
    ]
    return [
        "@overload",
        "async def step(",
        *params,
        f") -> Tuple[{read}, {write}]: ...",
    ]


def _render_try_step_overload(
    phase: BatchPhase,
    next_phase: Optional[BatchPhase],
) -> List[str]:
    """Emit one ``@overload`` signature for ``try_step``.

    Mirrors :func:`_render_step_overload` but wraps the return type in
    ``Optional[...]``. ``None`` from ``try_step`` means "the batch
    issued ``await again()``" (retry); otherwise the return shape is
    identical to :func:`step`.
    """
    if next_phase is None:
        read = "_ReadAtAll"
        write = "None"
    else:
        read = f"_ReadAt{_phase_pascal(next_phase)}"
        write = f"_WriteAt{_phase_pascal(phase)}"
    params = [
        "    handle: Batch,",
        "    *,",
        f"    through: Literal[BatchPhase.{phase.name}],",
    ]
    return [
        "@overload",
        "async def try_step(",
        *params,
        f") -> Optional[Tuple[{read}, {write}]]: ...",
    ]


def _render_batch_phase_overload(
    phase: BatchPhase,
    is_first: bool,
) -> List[str]:
    """Emit one ``@overload`` signature for ``batch_phase``.

    The overload uses the pattern::

        @overload
        @asynccontextmanager
        def batch_phase(
            p: Literal[BatchPhase.<NAME>],
        ) -> AsyncIterator[Tuple[_ReadAt<Name>, _WriteAt<Name>]]: ...

    Stacking ``@overload`` outside ``@asynccontextmanager`` and using
    ``AsyncIterator`` as the return is what most IDEs / type
    checkers (PyCharm, mypy, pyright) need to correctly narrow
    ``async with batch_phase(BatchPhase.<NAME>) as (r, w):`` to the
    right view types. Returning ``AbstractAsyncContextManager``
    directly works for some checkers but is missed by others, so we
    mirror the pattern stdlib's own ``asynccontextmanager``-typed
    overloads use.

    For the FIRST phase the read view is ``None`` because no field has
    been produced yet (``BATCH`` writing at the first phase has no
    earlier producer to read from -- the scheduler injects inputs via
    that phase's write view).
    """
    read = "None" if is_first else f"_ReadAt{_phase_pascal(phase)}"
    write = f"_WriteAt{_phase_pascal(phase)}"
    return [
        "@overload",
        "@asynccontextmanager",
        "def batch_phase(",
        f"    p: Literal[BatchPhase.{phase.name}],",
        f") -> AsyncIterator[Tuple[{read}, {write}]]: ...",
    ]


def generate_block(target: Path = None) -> str:
    """Return the full generated block content (no BEGIN/END markers).

    The returned text is the exact body that should sit between the
    marker lines in ``batch_storage.py``. Trailing newline not included.
    """
    groups = _fields_by_phase(BatchStorage)
    phases = _sorted_phases()
    field_docs = _extract_field_docstrings(target or TARGET, "BatchStorage")

    out: List[str] = []

    # Per-phase read Protocols.
    for i, phase in enumerate(phases):
        prev_phase = phases[i - 1] if i > 0 else None
        # Each `_ReadAt<PHASE>` adds exactly the fields produced at
        # the immediately previous phase; earlier ones come in via
        # Protocol inheritance.
        new_fields = groups[prev_phase] if prev_phase is not None else []
        out.extend(_render_read_protocol(phase, new_fields, prev_phase, field_docs))
        out.append("")
        out.append("")

    # Terminal-phase read Protocol: cumulative through the LAST phase
    # (i.e., includes the LAST phase's own fields, if any).
    last_phase = phases[-1]
    out.extend(_render_read_at_all(last_phase, groups[last_phase], field_docs))
    out.append("")
    out.append("")

    # Write dataclasses.
    for phase in phases:
        out.extend(_render_write_dataclass(phase, groups[phase], field_docs))
        out.append("")
        out.append("")

    # step overloads.
    for i, phase in enumerate(phases):
        next_phase = phases[i + 1] if i < len(phases) - 1 else None
        out.extend(_render_step_overload(phase, next_phase))
    out.append("")
    out.append("")

    # try_step overloads (same shape, Optional-wrapped return).
    for i, phase in enumerate(phases):
        next_phase = phases[i + 1] if i < len(phases) - 1 else None
        out.extend(_render_try_step_overload(phase, next_phase))
    out.append("")
    out.append("")

    # batch_phase overloads.
    for i, phase in enumerate(phases):
        out.extend(_render_batch_phase_overload(phase, is_first=(i == 0)))

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
    new_body = generate_block(target)
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
    expected_core = generate_block(target).rstrip("\n") + "\n"
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
    expected = _assemble(prefix, generate_block(target), suffix)
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
