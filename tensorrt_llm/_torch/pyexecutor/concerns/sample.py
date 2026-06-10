"""Sample concern: queue sampling kernel + apply tokens.

Loop- and rank-agnostic (no ADP / no CP):

* SAMPLE_3
  - **Last PP rank (or single-rank)**: compute the per-context-
    request logits-prefix sum, run :class:`HandleLogits` +
    :class:`HandleAdditionalOutputs` on ``batch_outputs``, call
    ``sampler.sample_async`` (queues sampling kernel, queues D2H
    copy, records ``sampler_event``).
  - **Non-last PP rank**: ``batch_outputs`` is ``None`` because
    :class:`ForwardConcern` discards the unusable dict on non-last
    ranks (PP forward only sends activations via NCCL p2p there).
    To keep the slot-ring shape uniform, this concern produces a
    PLACEHOLDER ``sample_state`` whose ``sampler_event`` is a
    no-op event and whose ``host`` is left for the
    :class:`RingBroadcastSampleConcern` to fill in via the
    cross-rank receive.
  - Publish ``sample_state`` (real or placeholder) to the
    SAMPLE_3 write view.

* APPLY_7
  - Call ``sampler.update_requests(sample_state, resource_manager)``
    -- BLOCKS on ``sampler_event`` (or, on non-last PP, the
    no-op event already returns immediately and the ring-broadcast
    receive has populated ``host``), then applies sampled tokens
    to each request. In the overlap loop, the SCHEDULER suspends
    this batch at STATE_UPD_4 (between SAMPLE_3 and APPLY_7) so
    the NEXT batch can queue its forward in parallel; APPLY_7
    then runs one iter later (when this batch is the SCHEDULER's
    ``previous``). In plain, APPLY_7 runs in the same iter as
    SAMPLE_3. In PP, APPLY_7 runs the iter prev's HC10 hop
    completes -- see :func:`scheduler_iter_pp`.
  - Per-request CONTEXT-CHUNK ADVANCE / state transitions are NOT
    here; that work is independent of ``sample_state`` (only needs
    ``scheduled_batch``) and lives in :class:`StateAdvanceConcern`,
    which the BATCH body invokes at STATE_UPD_4 BEFORE this
    concern's APPLY_7 yield.

Catastrophic exceptions during sampling propagate as exceptions
(Mode A failure handling). The legacy ``_handle_errors`` wrapping
is intentionally NOT replicated -- the SCHEDULER iter catches at
its top.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from ..batch_storage import BatchPhase, enter_phase
from ..handle_additional_outputs import HandleAdditionalOutputs
from ..handle_logits import HandleLogits

if TYPE_CHECKING:
    from ..context import Context
    from ..resource_manager import ResourceManager
    from ..sampler import Sampler, SampleState


class SampleConcern:
    """Plain-loop sampler driver."""

    def __init__(
        self,
        *,
        sampler: "Sampler",
        resource_manager: "ResourceManager",
    ) -> None:
        self._sampler = sampler
        # Passed into ``update_requests`` so spec-decode resource
        # managers can react to applied tokens.
        self._resource_manager = resource_manager

    async def handle_batch(self, ctx: "Context") -> None:
        # SAMPLE_3 -- queue the sampling kernel, record sampler_event.
        r3, w3 = await enter_phase(BatchPhase.SAMPLE_3)
        scheduled_batch = r3.scheduled_batch

        sample_state: Optional["SampleState"] = None
        if r3.can_queue:
            if ctx.svc.dist.is_last_pp_rank:
                # Real sampling: last PP rank or single-rank.
                batch_outputs = r3.batch_outputs
                num_context_logits_prefix_sum = [0]
                prefix_sum = 0
                num_context_tokens = 0
                for req in scheduled_batch.context_requests:
                    ctx_chunk = req.context_chunk_size
                    prefix_sum += ctx_chunk if req.py_return_context_logits else 1
                    num_context_logits_prefix_sum.append(prefix_sum)
                    num_context_tokens += ctx_chunk

                beam_width = self._sampler.beam_width(
                    scheduled_batch.all_requests())
                HandleLogits()(
                    scheduled_batch.context_requests,
                    scheduled_batch.generation_requests,
                    batch_outputs["logits"],
                    beam_width,
                    num_context_logits_prefix_sum,
                    self._sampler.is_generation_model(),
                )
                HandleAdditionalOutputs()(
                    scheduled_batch.context_requests,
                    scheduled_batch.generation_requests,
                    batch_outputs,
                    beam_width,
                    num_context_tokens,
                )
                sample_state = self._sampler.sample_async(
                    scheduled_batch,
                    batch_outputs,
                    num_context_logits_prefix_sum,
                )
            else:
                # Non-last PP rank: produce a placeholder
                # sample_state so the slot-ring carries the same
                # shape across ranks. Mirrors the legacy
                # ``_forward_step_inter_pp`` placeholder. The
                # ``host`` field stays None here -- the
                # :class:`RingBroadcastSampleConcern` fills it via
                # the cross-rank receive before APPLY_7 reads it.
                from ..sampler import SamplerEvent
                sampler_event = torch.cuda.Event()
                sampler_event.record()
                sampling_requests = (
                    scheduled_batch.context_requests_last_chunk
                    + scheduled_batch.generation_requests)
                sample_state = self._sampler.SampleState(
                    requests=sampling_requests,
                    sampler_event=SamplerEvent(cuda_event=sampler_event),
                )
        w3.sample_state = sample_state

        # APPLY_7 -- BLOCK on sampler_event, then apply sampled tokens.
        # The state-advance step (``_update_request_states_tp`` body)
        # ran at STATE_UPD_4 via the ``state_advance`` concern's
        # ``handle_batch`` coroutine -- it has no data dependency on
        # ``sample_state`` so it legitimately belongs to its own
        # concern at the earlier phase.
        r7, _ = await enter_phase(BatchPhase.APPLY_7)
        if not r7.can_queue:
            return
        sample_state = r7.sample_state
        if sample_state is not None and sample_state.host is not None:
            # On non-last PP rank, ``host`` is populated by the
            # ring-broadcast receive (see
            # :class:`RingBroadcastSampleConcern`). If the host
            # block hasn't arrived yet (e.g., scheduler bug) we
            # skip apply rather than dereferencing None.
            self._sampler.update_requests(sample_state,
                                          self._resource_manager)


__all__ = ["SampleConcern"]
