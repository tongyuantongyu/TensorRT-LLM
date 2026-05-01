"""State-advance concern: per-request state transitions.

A single-phase concern (plain method, not a coroutine) that runs
at the BATCH body's ``STATE_UPD_4``. Merges two legacy helpers
that the original code kept apart only because they were called
from different sites (``_update_request_states_tp`` and
``_update_generation_requests_that_will_complete_next_iteration``);
both label themselves ``Concern: sample`` in the legacy code but
neither has a real data dep on ``sample_state`` -- only
``scheduled_batch`` is needed.

What :meth:`advance` does, by request bucket:

* **Context requests** -- chunk-position advance + GENERATION_*
  transitions, including the overlap-only ``GENERATION_TO_COMPLETE``
  jump on the same-iter ctx->gen promotion.
* **Generation requests** (overlap only) -- mark ongoing gen
  requests as ``GENERATION_TO_COMPLETE`` so the scheduler's
  ``no_schedule_after_state=GENERATION_TO_COMPLETE`` filter drops
  them on the next iter (avoiding one wasted sample iter and the
  resulting +1 token in the API surface).

Pulling them out of ``SampleConcern`` makes the cut honest --
``SampleConcern.APPLY_7`` is now strictly sampler.update_requests
(block-on-event + write tokens).

Phase / call-site rationale
---------------------------

Runs from the BATCH body at ``STATE_UPD_4``. Shares two cross-batch
ordering constraints, both encoded by the OVERLAP scheduler driving
curr through its phases in pieces:

1. **Reads fresh ``num_tokens``.** Both the gen
   ``will_complete_next_iteration()`` check and the same-iter
   ctx->gen ``will_complete_next_iteration()`` check need prev's
   apply to have already added prev's sampled token onto the shared
   LlmRequest. The scheduler runs ``step(prev, APPLY_7)`` before
   ``step(curr, STATE_UPD_4)``.
2. **Flips ``set_exclude_last_generation_logits(False)`` only after
   prev's RESPOND.** Streaming-mode responses for shared-by-id
   requests in prev would otherwise compute wrong indices for
   generation logits. The scheduler runs ``step(prev, FINALIZE_9)``
   before ``step(curr, STATE_UPD_4)``.

PLAIN mode has no prev so both constraints are trivially satisfied;
the gen-half is also a no-op there because the no-skip-next-iter
optimization is moot when each batch fully drains within its iter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensorrt_llm.inputs.multimodal import strip_mm_data_for_generation

from ..llm_request import LlmRequest, LlmRequestState

if TYPE_CHECKING:
    from ..context import Context
    from ..scheduler import ScheduledRequests


def _strip_multimodal_post_prefill(request: LlmRequest) -> None:
    """Drop pinned encoder cache + raw pre-encoder tensors after prefill.

    Inlined from the legacy ``_strip_py_multimodal_data_post_prefill``
    (in ``py_executor.py``). Wraps :func:`strip_mm_data_for_generation`
    and mutates the shared ``request.py_multimodal_data`` in-place
    so the request's multimodal tensors actually get freed.
    """
    mm_data = getattr(request, "py_multimodal_data", None)
    if not mm_data:
        return
    strip_mm_data_for_generation(mm_data)


class StateAdvanceConcern:
    """Plain-loop request state advance driver."""

    def __init__(self, *, disable_overlap_scheduler: bool = True) -> None:
        # Plain loop = overlap disabled. The
        # ``will_complete_next_iteration`` branch in the legacy body
        # is overlap-only and stays unreachable here. Storing the
        # flag lets the same concern serve overlap once that loop is
        # wired up (the branch becomes live).
        self._disable_overlap_scheduler = disable_overlap_scheduler

    def advance(
        self,
        ctx: "Context",
        scheduled_batch: "ScheduledRequests",
    ) -> None:
        """Advance per-request state for ``scheduled_batch``.

        Two halves, both at the same phase, both with the same
        cross-batch ordering constraints (see module docstring):

        Context requests
            * Skip already-terminated (Mode B fail-fast set state to
              ``GENERATION_COMPLETE`` without going through the
              normal flow). Schedule's iter_live discipline keeps
              these out of the active pool, but they may still be in
              ``scheduled_batch`` because that snapshot is immutable.
            * Record the current chunk's bounds in
              ``py_last_context_chunk`` (consumed by sampler
              bookkeeping).
            * Advance ``context_current_position`` to the next chunk.
            * On the LAST chunk: strip pinned multimodal tensors
              (otherwise encoder outputs stay on GPU through the full
              decode lifetime and cause OOMs at high concurrency);
              then transition to ``GENERATION_IN_PROGRESS`` (plain)
              or, on overlap with ``will_complete_next_iteration``,
              jump to ``GENERATION_TO_COMPLETE`` and turn off the
              last-generation-logits exclusion.

        Generation requests (overlap only)
            * For each ongoing gen request that
              ``will_complete_next_iteration()`` returns True for,
              transition state to ``GENERATION_TO_COMPLETE`` and
              turn off the last-generation-logits exclusion. The
              ``MicroBatchScheduler``'s default
              ``no_schedule_after_state=GENERATION_TO_COMPLETE``
              filter then drops it on the next iter, so no further
              sampling happens for the request -- avoiding the
              "+1 token" the overlap pipeline would otherwise
              produce (sample one iter ahead of apply).
            * Plain mode skips this half (the no-skip-next-iter
              optimization is moot when each batch fully drains in
              its own iter).
        """
        del ctx  # this concern doesn't read ctx; signature kept for parity.
        for req in scheduled_batch.context_requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                # Skip failed requests (Mode B fail-fast set this).
                continue
            req.py_last_context_chunk = (
                req.context_current_position,
                req.context_current_position + req.context_chunk_size,
            )
            req.move_to_next_context_chunk()
            if req.context_remaining_length == 0:
                _strip_multimodal_post_prefill(req)
                if (not self._disable_overlap_scheduler
                        and req.will_complete_next_iteration()):
                    req.set_exclude_last_generation_logits(False)
                    req.state = LlmRequestState.GENERATION_TO_COMPLETE
                else:
                    req.state = LlmRequestState.GENERATION_IN_PROGRESS

        if self._disable_overlap_scheduler:
            return
        for req in scheduled_batch.generation_requests:
            if (req.state != LlmRequestState.GENERATION_COMPLETE
                    and req.will_complete_next_iteration()):
                req.set_exclude_last_generation_logits(False)
                req.state = LlmRequestState.GENERATION_TO_COMPLETE


__all__ = ["StateAdvanceConcern"]
