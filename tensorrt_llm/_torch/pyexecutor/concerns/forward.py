"""Forward concern: model engine forward at FORWARD_2.

Single-phase per batch -> exposed as a plain method, not a coroutine.
The BATCH body calls :meth:`run` directly inside its
``async with batch_phase(BatchPhase.FORWARD_2)`` block and unpacks
the returned ``(outputs, attn_metadata)`` pair into the write view.

Scope: single rank OR multi-PP-rank. PP NCCL p2p (HC7) lives
inside ``model_engine.forward`` so this concern only needs to
discriminate "this rank produces useful logits to sample" -- only
the last PP rank does. On non-last PP ranks, ``forward`` is still
called (its NCCL sends activations to the next rank) but the
returned dict carries no usable logits, so this method returns
``None`` and the SAMPLE_3 phase produces a placeholder
``sample_state`` for slot-ring shape parity (see
:class:`SampleConcern`).

``new_tensors_device`` and ``num_accepted_tokens_device`` are
overlap-loop inputs (see :func:`scheduler_iter_overlap` --
SCHEDULER bridges them from the prior batch's ``sample_state``
into the current batch's SCHEDULE_0 write view). The plain loop
always passes ``None`` for both; this concern is loop-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import torch

if TYPE_CHECKING:
    from ...attention_backend.interface import AttentionMetadata
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
    ) -> Tuple[Optional[Dict[str, Any]], Optional["AttentionMetadata"]]:
        """Run the model forward on ``scheduled_batch``.

        Returns ``(outputs, attn_metadata)``. ``outputs`` is the
        model's output dict (``{"logits": ..., ...}``) on the last PP
        rank, or ``None`` on non-last PP ranks (forward still ran but
        produced no logits this rank can sample from).
        ``attn_metadata`` is ``model_engine.attn_metadata`` after the
        forward pass -- threaded through ``BatchStorage`` so the
        resource concern at RESPOND_8 can update KV-cache draft-token
        slots without reaching into the model engine.

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
        #
        # Reflect the cross-thread warmup signal onto the engine
        # so the model-internal gates that read ``is_warmup``
        # (``torch.compile`` bootstrap path, MoE load-balancer
        # skip, attn-metadata beam-width override, etc.) see the
        # same value the SCHEDULER does. Main thread writes
        # ``ctx.port.is_warmup`` via :attr:`PyExecutorCoro.is_warmup`
        # (e.g. ``_util.py``'s KV-cache memory estimation pass);
        # this concern is the bridge into the engine. Refreshed
        # once per batch -- earlier writes from main thread
        # propagate at the next FORWARD_2.
        self._model_engine.is_warmup = ctx.port.is_warmup
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
        # ``attn_metadata`` is the same mutable object across iters;
        # we hand its current handle back so the BATCH body can stash
        # it in the FORWARD_2 write view for RESPOND_8 readers.
        attn_metadata = getattr(self._model_engine, "attn_metadata", None)
        if not ctx.svc.dist.is_last_pp_rank:
            # Non-last PP rank: ``model_engine.forward`` still ran
            # (its NCCL p2p sent activations to the next rank), but
            # the returned dict carries no logits this rank can
            # sample from. Drop it; SAMPLE_3 will produce a
            # placeholder ``sample_state`` so the slot ring sees a
            # uniform shape. The KV cache and ``attn_metadata`` are
            # rank-local, so we still surface ``attn_metadata`` for
            # the resource concern.
            return None, attn_metadata
        return outputs, attn_metadata


__all__ = ["ForwardConcern"]
