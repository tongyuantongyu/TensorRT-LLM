"""Forward concern: model engine forward at FORWARD_2.

Single-phase per batch -> exposed as a plain method, not a coroutine.
The BATCH body calls :meth:`run` directly inside its
``async with batch_phase(BatchPhase.FORWARD_2)`` block and writes
the returned outputs into the write view.

Scope: single rank, no draft model, no kv_connector
``wait_for_save`` (no kv_connector wired yet). The distributed
``forward`` concern (PP NCCL p2p, HC7) lands later as a separate
``DistributedForwardConcern`` or as additional phases on this
class.

``new_tensors_device`` and ``num_accepted_tokens_device`` are
overlap-loop inputs (see :func:`scheduler_iter_overlap` --
SCHEDULER bridges them from the prior batch's ``sample_state``
into the current batch's SCHEDULE_0 write view). The plain loop
always passes ``None`` for both; this concern is loop-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

if TYPE_CHECKING:
    from ..context import Context
    from ..model_engine import ModelEngine
    from ..resource_manager import ResourceManager
    from ..sampler import Sampler, SampleStateTensors
    from ..scheduler import ScheduledRequests


class ForwardConcern:
    """Plain / overlap forward driver."""

    def __init__(
        self,
        *,
        model_engine: "ModelEngine",
        resource_manager: "ResourceManager",
        sampler: "Sampler",
        execution_stream: torch.cuda.Stream,
    ) -> None:
        self._model_engine = model_engine
        self._resource_manager = resource_manager
        # Sampler is only consulted for ``get_cache_indirection`` and
        # ``beam_width``; its actual sampling work happens in
        # ``SampleConcern``. Owned here too because the dependency is
        # READ-ONLY.
        self._sampler = sampler
        self._execution_stream = execution_stream

    def run(
        self,
        ctx: "Context",
        scheduled_batch: "ScheduledRequests",
        *,
        new_tensors_device: Optional["SampleStateTensors"] = None,
        num_accepted_tokens_device: Optional[torch.Tensor] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run the model forward on ``scheduled_batch``.

        Returns the model's output dict (``{"logits": ..., ...}``)
        on success or ``None`` if the batch was empty (the BATCH
        body short-circuits forward when ``can_queue`` is False, so
        in practice this method is always called with non-empty
        input).

        Args:
            new_tensors_device: overlap-loop input -- the previous
                batch's ``sample_state.device`` tokens, bridged in
                by the SCHEDULER. ``None`` for plain loop OR for
                the first iter of the overlap loop (no prior batch
                yet).
            num_accepted_tokens_device: overlap + spec-decode input
                produced by the SCHEDULER's HC1 in-bridge draft-
                model run. ``None`` until SpecDecodeConcern is wired.
        """
        # Mirror the legacy ``_forward_step`` body sans:
        # * spec-decode draft propagation (no spec_decode concern),
        # * kv_connector ``wait_for_save`` (no connector concern),
        # * try/except + _handle_errors wrapping (catastrophic
        #   failures propagate as exceptions per the new design;
        #   per-request fail-fast is owned by the concern that
        #   detected the bad request, not by forward).
        gather_context_logits = any(
            req.py_return_context_logits
            for req in scheduled_batch.context_requests)
        cache_indirection_buffer = self._sampler.get_cache_indirection()

        # Run forward on the execution stream so it overlaps with
        # main-stream KVCacheTransferManager onboard / offload work.
        self._execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._execution_stream):
            outputs = self._model_engine.forward(
                scheduled_batch,
                self._resource_manager,
                new_tensors_device,
                gather_context_logits=gather_context_logits,
                cache_indirection_buffer=cache_indirection_buffer,
                num_accepted_tokens_device=num_accepted_tokens_device,
            )
        torch.cuda.current_stream().wait_stream(self._execution_stream)
        return outputs


__all__ = ["ForwardConcern"]
