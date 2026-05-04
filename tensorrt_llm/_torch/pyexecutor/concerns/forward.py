"""Forward concern: model engine forward at FORWARD_2.

Single-phase per batch, but still shaped as an ``async def
handle_batch`` coroutine so the runtime gives it the same automatic
NVTX annotation as every other concern (``CForwardConcern.handle_batch``
on the timeline). The BATCH body wraps it with ``Concern(...)`` and
drives it via ``await resume(forward)`` inside the
``async with batch_phase(BatchPhase.FORWARD_2)`` block; the coroutine
itself reads its inputs from the FORWARD_2 read view and writes
``batch_outputs`` / ``attn_metadata`` straight into the write view.

Scope: single rank OR multi-PP-rank. PP NCCL p2p (HC7) lives
inside ``model_engine.forward`` so this concern only needs to
discriminate "this rank produces useful logits to sample" -- only
the last PP rank does. On non-last PP ranks, ``forward`` is still
called (its NCCL sends activations to the next rank) but the
returned dict carries no usable logits, so ``batch_outputs`` is
left ``None`` and the SAMPLE_3 phase produces a placeholder
``sample_state`` for slot-ring shape parity (see
:class:`SampleConcern`). ``attn_metadata`` is rank-local and is
written on every rank so the resource concern's RESPOND_8 update
path can see it.

``previous_tensors_device`` and ``num_accepted_tokens_device`` are
overlap-loop inputs (see :func:`scheduler_iter_overlap` --
SCHEDULER bridges them from the prior batch's ``sample_state``
into the current batch's SCHEDULE_0 write view). The plain loop
leaves both ``None`` in the read view; this concern is loop-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ..batch_storage import BatchPhase, enter_phase

if TYPE_CHECKING:
    from ..context import Context
    from ..model_engine import ModelEngine
    from ..resource_manager import ResourceManager
    from ..sampler import Sampler


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

    async def handle_batch(self, ctx: "Context") -> None:
        """Run the model forward on the FORWARD_2 read view's batch.

        Mirrors the legacy ``_forward_step`` body sans:

        * spec-decode draft propagation (no spec_decode concern),
        * kv_connector ``wait_for_save`` (no connector concern),
        * try/except + _handle_errors wrapping (catastrophic
          failures propagate as exceptions per the new design;
          per-request fail-fast is owned by the concern that
          detected the bad request, not by forward).
        """
        r, w = await enter_phase(BatchPhase.FORWARD_2)
        if not r.can_queue:
            return

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

        scheduled_batch = r.scheduled_batch
        new_tensors_device = r.previous_tensors_device
        num_accepted_tokens_device = r.num_accepted_tokens_device

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
        # we surface its current handle in the FORWARD_2 write view
        # so the resource concern at RESPOND_8 can hand it to
        # ``ResourceManager.update_resources`` without reaching into
        # the model engine. Always written -- it's rank-local on PP.
        w.attn_metadata = getattr(self._model_engine, "attn_metadata",
                                  None)
        if not ctx.svc.dist.is_last_pp_rank:
            # Non-last PP rank: ``model_engine.forward`` still ran
            # (its NCCL p2p sent activations to the next rank), but
            # the returned dict carries no logits this rank can
            # sample from. Leave ``batch_outputs`` unset; SAMPLE_3
            # will produce a placeholder ``sample_state`` so the
            # slot ring sees a uniform shape.
            return
        w.batch_outputs = outputs


__all__ = ["ForwardConcern"]
