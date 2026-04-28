"""Batch data model for the PyExecutor coroutine runtime.

The runtime in :mod:`tensorrt_llm._torch.pyexecutor.coroutines` is
generic over phase enum and storage class. This module pins down the
*specific* ``BatchPhase`` and ``BatchStorage`` used by TensorRT-LLM's
three executor loops -- the field set is the union of what each loop
needs, with loop-specific fields left ``None`` on the loops that
don't use them.

Layering
========

- :mod:`coroutines` exposes the generic primitives (``enter_phase``,
  ``batch_phase`` CM, ``resume``, ``step``, ``Batch``, ``Driver``,
  ``spawn``, ``phased_field`` and the runtime proxies). It never
  references ``BatchPhase`` or any storage class.
- This module imports those primitives and provides the production
  data model on top: phase enum + storage dataclass + the typed
  ``@overload`` chain that narrows ``step`` / ``try_step`` /
  ``batch_phase`` per phase.

Production code that creates / drives batches imports from here;
runtime tests that exercise the generic mechanics import directly from
:mod:`coroutines` against their own test-local storage definitions.

Adding a new field
==================

1. Add ``Optional[T] = phased_field(BatchPhase.X)`` to ``BatchStorage``.
2. Run ``python scripts/generate_coroutine_views.py`` to refresh the
   ``# ===== BEGIN GENERATED =====`` ... ``# ===== END GENERATED =====``
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
from typing import AsyncIterator, Dict, List, Literal, Optional, Protocol, Tuple, overload, runtime_checkable

import torch

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
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
from tensorrt_llm._torch.pyexecutor.sampler import SampleState, SampleStateTensors
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

__all__ = [
    "Driver",
    "BatchStorage",
    "Batch",
    "BatchPhase",
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
# Phase enum (final shape; FINALIZE_8 added to resolve the
# response.finished_requests -> iter_stats.process_iter_stats
# cross-concern same-phase conflict).
# --------------------------------------------------------------------------- #


class BatchPhase(IntEnum):
    """Lifecycle phases of a per-batch coroutine in the executor loops.

    All three executor loops in
    :mod:`tensorrt_llm._torch.pyexecutor.py_executor`
    (``_executor_loop``, ``_executor_loop_overlap``, ``_executor_loop_pp``)
    drive a per-batch coroutine through the SAME ordered phase set
    below; each loop just yields at a *subset* (skipping phases that
    don't apply). The strict-progression invariant
    (``enter_phase`` must yield a strictly greater phase than the
    last) holds for every loop because each only *skips* phases --
    never repeats or regresses.

    Naming convention
    =================

    No ``P_`` prefix: values are always referenced via
    ``BatchPhase.XXX`` so the prefix is redundant. The trailing
    numeric suffix matches the enum value, making the relative order
    visible at the call site without consulting this file --
    e.g., a reader can see immediately that
    ``BatchPhase.STATE_UPD_3 < BatchPhase.HANDOFF_5``.

    Phase semantics
    ===============

    SCHEDULE_0
        Per-iter scheduling decision + resource prep. All loops; on
        PP it also includes the cross-rank schedule chain (HC8),
        which is the ``schedule`` distributed concern's internal
        coordination -- intra-batch.
    FORWARD_1
        Model forward queues GPU work. All loops. PP: also
        ``pp_recv`` / ``pp_send`` via NCCL p2p (HC7), internal to
        the distributed ``forward`` concern.
    SAMPLE_2
        Sampling kernel queued + ``sampler_event`` recorded.
        All loops. PP: last rank only -- non-last ranks emit a
        placeholder event in ``_forward_step_inter_pp`` so
        ``BatchStorage`` has a unified shape across ranks.
    STATE_UPD_3
        Advance request states (context_chunk position,
        ``GENERATION_*``, drop ADP dummy). Used by overlap + PP.
        Plain folds it into ``APPLY_6`` (no separate yield needed
        because there's no other batch interleaving in plain).
    SYNC_EVT_4
        Explicit ``sampler_event.synchronize()``. PP only -- last
        rank waits at iter ``N+1`` so the ring-broadcast can read
        host tokens. Plain + overlap fold into ``APPLY_6`` because
        ``sampler.update_requests`` blocks on the event implicitly.
    HANDOFF_5
        Hand off the batch to the long-running
        ``ring_broadcast_sample`` distributed concern (HC10). PP only.
    APPLY_6
        Apply sampled tokens via ``_update_requests``. All loops.
        Plain: same iter as ``SCHEDULE_0``. Overlap: NEXT iter
        (this batch is now ``previous_batch``). PP: iter
        ``N+(n-1)``, inside ``_handle_executed_batch``.
    RESPOND_7
        Build responses, free resources, etc. All loops.
        Produces ``finished_requests``. Plain: same iter.
        Overlap: NEXT iter, after ``APPLY_6`` for this batch as
        prev. PP: iter ``N+(n-1)``, inside ``_handle_executed_batch``.
    FINALIZE_8
        End-of-iter stats / cleanup that depends on ``RESPOND_7``'s
        output (specifically ``finished_requests``). All loops.
        Currently: ``iter_stats.process_iter_stats`` lives here.
        This phase exists because ``response.handle_responses``
        produces ``finished_requests`` at RESPOND_7 and
        ``iter_stats`` (a different concern) consumes it -- the
        rule "cross-concern producer-consumer pair must be on
        different phases" forces the split.

    Per-loop subsets:

    +-------------+--------+---------+-----+
    | phase       | plain  | overlap | pp  |
    +=============+========+=========+=====+
    | SCHEDULE_0  | yes    | yes     | yes |
    | FORWARD_1   | yes    | yes     | yes |
    | SAMPLE_2    | yes    | yes     | yes |
    | STATE_UPD_3 | folded | yes     | yes |
    | SYNC_EVT_4  | folded | folded  | yes |
    | HANDOFF_5   | n/a    | n/a     | yes |
    | APPLY_6     | yes    | yes     | yes |
    | RESPOND_7   | yes    | yes     | yes |
    | FINALIZE_8  | yes    | yes     | yes |
    +-------------+--------+---------+-----+

    "folded" = the work happens inside the next yielded phase's
    block (e.g., plain's ``STATE_UPD_3`` work runs inside its
    ``APPLY_6`` block); "n/a" = the work doesn't exist on that
    loop variant.

    NOT BatchPhase values
    =====================

    The following stay above the per-batch coroutines as pure
    scheduler-direct code (no ``step`` / ``Batch`` / ``enter_phase``
    machinery wraps them):

    * HC9 ``retire_vote`` (PP only) -- cross-rank consensus on how
      many batches retire this iter; payload is a count, spans
      multiple batches, no single request set.
    * PP-only termination ballot
      (``_disagg_pp_termination_handler.terminate_pending_requests``)
      -- same shape as HC9.

    Cross-rank distributed concern coroutines (intra-batch -- same
    request set runs on every rank) are NOT new enum entries either,
    but their per-rank bodies happen to use the same per-batch phase
    names internally -- e.g., the distributed ``forward`` concern's
    body yields at ``FORWARD_1`` on every rank simultaneously. Same
    for ``schedule`` (``SCHEDULE_0``) and ``ring_broadcast_sample``
    (spans ``HANDOFF_5`` through ``APPLY_6`` across ``(n-1)`` iters).
    """

    SCHEDULE_0 = 0
    FORWARD_1 = 1
    SAMPLE_2 = 2
    STATE_UPD_3 = 3
    SYNC_EVT_4 = 4
    HANDOFF_5 = 5
    APPLY_6 = 6
    RESPOND_7 = 7
    FINALIZE_8 = 8


# --------------------------------------------------------------------------- #
# Unified BatchStorage
#
# All three executor loops (plain / overlap / PP) share this field set:
#  * Each loop's per-batch coroutine writes / reads the SUBSET of fields
#    that apply to it; loop-specific fields stay ``None`` on loops that
#    don't use them.
#  * Producer/consumer phases agree across loops -- the phase a field is
#    written at is the same on every loop, and every consumer's phase is
#    strictly greater than the producer's phase (the runtime invariant).
#  * The loop-specific fields are clearly marked in their docstrings
#    (``OVERLAP-ONLY``, ``PP-ONLY``).
#
# Documentation conventions for each field's docstring:
#   Producer : the concern (or BATCH / SCHEDULER) that writes the field
#              at the field's declared phase.
#   Consumers: the concerns (or BATCH / SCHEDULER) that read the field,
#              along with the phase at which they read it. The phase
#              must be strictly greater than the producer's phase.
#
# Intra-concern data (data produced and consumed by ONE concern across
# different phases) is NOT in this dataclass -- it lives as Python
# locals inside that concern's coroutine body. Examples:
#   * ``iter_stats`` record + ``iter_start_time`` (iter_stats concern)
#   * gpu_*_start/_end CUDA events, ``fwd_timing``, ``sample_timing``
#     (perf_metric concern)
#   * ``guided_decoder_failed_requests`` (guided_decoder concern)
#   * ``has_draft_batch``, ``use_previous_draft_tokens``, ``target_inputs``,
#     ``previous_tensors`` (overlap's HC1 bridge -- BUT see
#     ``previous_tensors_device`` and ``num_accepted_tokens_device``
#     for the cross-concern pieces).
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class BatchStorage:
    """Per-batch fields shared by the plain, overlap, and PP executor loops.

    The field set is the union of what each loop needs; loop-specific
    fields stay ``None`` on loops that don't use them.

    Fields are grouped by the phase that produces them. Within a group
    the field is written at that phase and visible to readers at any
    later phase. Each field carries:

    - A leading ``#`` comment block describing its lifecycle (Producer,
      Consumers, loop-specificity, phase-ordering rationale). This is
      reference material for the executor refactor and does NOT bloat
      the IDE quick-doc when callers hover on ``r.<field>`` /
      ``w.<field>``.
    - A trailing one-paragraph docstring that explains *what the field
      represents*. The generator copies this docstring into the
      generated read / write views, so hovering on either side gets
      the same brief explanation.
    """

    # ===================== SCHEDULE_0 =====================

    # Producer: ``schedule`` concern (via ``_prepare_and_schedule_batch``
    # which calls ``_schedule()``). On PP this is a *distributed*
    # concern -- rk0's body schedules + serializes, other ranks' bodies
    # recv from prev PP rank and isend to next (HC8 PP schedule chain
    # inside ``_pp_schedule_and_propagate``).
    #
    # Consumers (all at FORWARD_1+ -- same-phase reads of SCHEDULE_0
    # productions are forbidden by the runtime's read-view rule):
    #
    #   - BATCH at FORWARD_1: gates whether forward+sample+respond run.
    #   - ``disagg`` at FORWARD_1: ``_prepare_disagg_gen_transmission_complete``.
    #   - ``response`` at FORWARD_1: ``_handle_first_token_response``.
    #   - ``spec_decode`` at FORWARD_1: ``_handle_dynamic_draft_len``,
    #     ``drafter.prepare_draft_tokens`` / ``pad_draft_tokens_for_cuda_graph``.
    #   - ``resource`` at FORWARD_1: ``prepare_resources``.
    #   - ``kv_connector`` at FORWARD_1: ``_kv_connector_start_batch``.
    #   - ``guided_decoder`` at FORWARD_1: ``add_batch`` / ``init_disagg_gen_requests``.
    #   - ``forward`` at FORWARD_1.
    #   - ``guided_decoder`` at SAMPLE_2: ``execute(batch_outputs.logits)``.
    #   - ``sample`` at SAMPLE_2: ``sample_async``.
    #   - ``perf_metric`` at SAMPLE_2: ``save_timing_to_requests``.
    #   - ``sample`` at APPLY_6: ``update_request_states`` / ``update_requests``.
    #   - ``disagg`` at RESPOND_7: ``_send_kv_async``.
    #   - ``response`` at RESPOND_7: ``_handle_canceled_requests`` /
    #     ``_handle_responses``.
    #   - ``perf_metric`` at RESPOND_7: ``compute_batch_gpu_times``.
    #   - ``resource`` at RESPOND_7: ``update_resources``.
    #   - ``ring_broadcast_sample`` distributed concern at HANDOFF_5+ (PP):
    #     iterates ``context_requests_last_chunk`` + ``generation_requests``
    #     during the ring traversal.
    #   - ``iter_stats`` at FINALIZE_8: ``_process_iter_stats``.
    scheduled_batch: Optional[ScheduledRequests] = phased_field(BatchPhase.SCHEDULE_0)
    """The scheduled set of requests this batch will run."""

    # Producer: ``schedule`` concern (``_can_queue`` collective).
    # Consumer: BATCH at FORWARD_1 -- decides whether to spawn the
    # forward / sample / respond coroutines (skipped when ``False``).
    can_queue: Optional[bool] = phased_field(BatchPhase.SCHEDULE_0)
    """Whether every TP rank has a non-empty scheduled batch this iter."""

    # OVERLAP-ONLY. Plain and PP discard the second return value of
    # ``_can_queue`` (no consumer needs it).
    #
    # Producer: ``schedule`` concern.
    # Consumer: SCHEDULER at the next iter's bridge code -- used to
    # compute ``should_process_previous_batch = can_queue or not
    # can_queue_this_rank`` and to set ``self.previous_batch = None``
    # on empty-rank (HC2 cleanup variant).
    can_queue_this_rank: Optional[bool] = phased_field(BatchPhase.SCHEDULE_0)
    """Whether THIS specific rank has a non-empty scheduled batch."""

    # Producer: ``schedule`` concern (return value of ``_schedule()``).
    # Consumer: ``resource`` + ``disagg`` at FORWARD_1 -- runs
    # ``_prepare_disagg_gen_init`` (preps KV resources for these
    # requests + submits the async KV recv to the ctx worker).
    fitting_disagg_gen_init_requests: Optional[List[LlmRequest]] = phased_field(BatchPhase.SCHEDULE_0)
    """Disagg gen-init requests that fit this iter."""

    # Producer: ``schedule`` concern.
    # Consumer: ``disagg`` at FORWARD_1 -- backpressure logic in
    # ``_prepare_and_schedule_batch``: when ``num_fitting_reqs == 0``
    # AND no disagg-gen-init either, decides whether to block on
    # at-least-one ctx KV transfer or just opportunistically clean up.
    num_fitting_reqs: Optional[int] = phased_field(BatchPhase.SCHEDULE_0)
    """Number of regular (non-disagg-gen-init) requests that fit."""

    # PP-ONLY.
    #
    # Producer: SCHEDULER (assigned at batch creation -- the SCHEDULER
    # iterates ``microbatch_id = (microbatch_id + 1) % num_micro_batches``
    # every iter and stamps the new batch with that index).
    #
    # Consumers:
    #
    #   - ``ring_broadcast_sample`` distributed concern at HANDOFF_5+:
    #     uses ``self.send_handles[microbatch_id]`` to slot the per-batch
    #     isend handle. ``self.send_expected_batch_num_handles`` and
    #     ``self.send_schedule_handles`` follow the same per-slot pattern
    #     (HC5 / HC5a / HC5b inside their respective distributed concerns
    #     / scheduler-direct ``retire_vote``).
    #   - ``iter_stats.process_iter_stats`` at FINALIZE_8 (passes
    #     ``microbatch_id % pp_size`` to the stats aggregator).
    microbatch_id: Optional[int] = phased_field(BatchPhase.SCHEDULE_0)
    """The PP slot ring index for this batch (``iter_counter mod n``)."""

    # OVERLAP-ONLY. The half-concern HC1 produce-half on this batch.
    #
    # Producer: SCHEDULER (HC1 cross-iter bridge code that runs
    # *between* step calls in the scheduler iter). Reads
    # ``prev.sample_state.device`` (or, when the draft model ran
    # in-bridge, ``target_inputs`` from ``_handle_speculative_decoding``)
    # and writes the value into curr's storage at SCHEDULE_0.
    #
    # Consumer: ``forward`` at FORWARD_1 -- passed as the
    # ``new_tensors_device`` arg to ``_forward_step``.
    previous_tensors_device: Optional[SampleStateTensors] = phased_field(BatchPhase.SCHEDULE_0)
    """Forward-input device tensors threaded in from the previous batch."""

    # OVERLAP-ONLY. Produced only when the SCHEDULER decides
    # ``has_draft_batch`` for this iter.
    #
    # Producer: SCHEDULER (HC1 in-bridge ``_handle_speculative_decoding``
    # runs the draft model on prev's sample tensors and computes the
    # accepted-token tensor).
    # Consumer: ``forward`` at FORWARD_1 -- passed as the
    # ``num_accepted_tokens_device`` arg to ``_forward_step``.
    num_accepted_tokens_device: Optional[torch.Tensor] = phased_field(BatchPhase.SCHEDULE_0)
    """Per-request accepted-token counts from the in-bridge draft model."""

    # ===================== FORWARD_1 =====================

    # Producer: ``forward`` concern (``_forward_step``).
    # PP: only populated on the *last* PP rank -- non-last ranks return
    # ``None`` because ``_forward_step_inter_pp`` doesn't expose a
    # batch_outputs (their forward just runs the layer slice and
    # pp_sends to the next rank).
    #
    # Consumers (at SAMPLE_2; PP: last rank only):
    #
    #   - ``guided_decoder.execute(batch_outputs['logits'])``: masks
    #     logits per the constraint matcher; produces
    #     ``guided_decoder_failed_requests`` as a Python local in the
    #     guided_decoder coroutine.
    #   - ``sample.sample_async(scheduled_batch, batch_outputs)``.
    batch_outputs: Optional[Dict[str, torch.Tensor]] = phased_field(BatchPhase.FORWARD_1)
    """Forward pass outputs: logits, hidden_states, additional outputs."""

    # ===================== SAMPLE_2 =====================

    # Producer (loop / rank dependent):
    #
    #   - Plain / overlap: ``sample`` concern (``sample_async``).
    #   - PP last rank: ``sample`` concern (real ``sample_async``).
    #   - PP non-last ranks: ``forward`` concern emits a *placeholder* via
    #     ``_forward_step_inter_pp`` (just a recorded ``cuda.Event`` with
    #     empty ``.host``) so BatchStorage has a unified shape across ranks.
    #
    # Consumers:
    #
    #   - ``sample.update_requests`` at APPLY_6 (every loop / rank, after
    #     sampler_event is synced and host data is available). Plain: same
    #     iter as SAMPLE_2. Overlap: NEXT iter (this batch becomes
    #     ``previous_batch``). PP: iter ``N+(n-1)`` inside
    #     ``_handle_executed_batch``, after host data has arrived via the
    #     ring.
    #   - SCHEDULER at next iter's HC1 cross-iter bridge (OVERLAP-ONLY):
    #     reads ``prev.sample_state.device`` to feed curr's
    #     ``previous_tensors_device`` (and, when ``has_draft_batch``, feeds
    #     ``_handle_speculative_decoding`` which produces ``target_inputs``
    #     and ``num_accepted_tokens_device``).
    #   - SCHEDULER at SYNC_EVT_4 (PP last rank only): reads
    #     ``sample_state.sampler_event`` and synchronizes -- the explicit
    #     D2H sync that fills ``sample_state.host`` with real tokens before
    #     the ring broadcast.
    #   - ``ring_broadcast_sample`` distributed concern at HANDOFF_5+ (PP
    #     only, cross-rank): on the last rank reads ``sample_state.host``
    #     to send to rk0; on non-last ranks receives ``sample_state.host``
    #     from prev rank and writes back to its own ``sample_state.host``
    #     as the ring traverses ((n-1) iters of receive-then-send).
    #
    # NOTE: in the plain loop ``sample_state`` is intra-concern in isolation
    # (``sample`` produces and consumes); it could legally be a Python local
    # there. It's kept as a storage field for uniformity with overlap and PP.
    sample_state: Optional[SampleState] = phased_field(BatchPhase.SAMPLE_2)
    """Sampler state: ``sampler_event``, host tokens, device tensors."""

    # ===================== RESPOND_7 =====================

    # Producer: ``response`` concern (return value of ``_handle_responses``).
    # Consumer: ``iter_stats.process_iter_stats`` at FINALIZE_8.
    #
    # PHASE NOTE: this producer / consumer pair is the reason FINALIZE_8
    # exists in BatchPhase. ``_process_iter_stats(finished_requests, ...)``
    # cannot run at RESPOND_7 because cross-concern reads at the same phase
    # as production are forbidden (read view at phase P sees writes at < P
    # only). FINALIZE_8 is the smallest phase split that resolves the
    # conflict.
    finished_requests: Optional[List[LlmRequest]] = phased_field(BatchPhase.RESPOND_7)
    """Requests that finished (terminated) during this batch's response."""


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
class _ReadAtSchedule_0(Protocol):
    """Readable fields at phase SCHEDULE_0 — no field is produced yet."""


@runtime_checkable
class _ReadAtForward_1(_ReadAtSchedule_0, Protocol):
    """Readable fields at phase FORWARD_1 — cumulative through SCHEDULE_0."""

    @property
    def scheduled_batch(self) -> ScheduledRequests:
        """The scheduled set of requests this batch will run."""
        ...

    @property
    def can_queue(self) -> bool:
        """Whether every TP rank has a non-empty scheduled batch this iter."""
        ...

    @property
    def can_queue_this_rank(self) -> bool:
        """Whether THIS specific rank has a non-empty scheduled batch."""
        ...

    @property
    def fitting_disagg_gen_init_requests(self) -> List[LlmRequest]:
        """Disagg gen-init requests that fit this iter."""
        ...

    @property
    def num_fitting_reqs(self) -> int:
        """Number of regular (non-disagg-gen-init) requests that fit."""
        ...

    @property
    def microbatch_id(self) -> int:
        """The PP slot ring index for this batch (``iter_counter mod n``)."""
        ...

    @property
    def previous_tensors_device(self) -> SampleStateTensors:
        """Forward-input device tensors threaded in from the previous batch."""
        ...

    @property
    def num_accepted_tokens_device(self) -> torch.Tensor:
        """Per-request accepted-token counts from the in-bridge draft model."""
        ...


@runtime_checkable
class _ReadAtSample_2(_ReadAtForward_1, Protocol):
    """Readable fields at phase SAMPLE_2 — cumulative through FORWARD_1."""

    @property
    def batch_outputs(self) -> Dict[str, torch.Tensor]:
        """Forward pass outputs: logits, hidden_states, additional outputs."""
        ...


@runtime_checkable
class _ReadAtStateUpd_3(_ReadAtSample_2, Protocol):
    """Readable fields at phase STATE_UPD_3 — cumulative through SAMPLE_2."""

    @property
    def sample_state(self) -> SampleState:
        """Sampler state: ``sampler_event``, host tokens, device tensors."""
        ...


@runtime_checkable
class _ReadAtSyncEvt_4(_ReadAtStateUpd_3, Protocol):
    """Readable fields at phase SYNC_EVT_4 — cumulative through STATE_UPD_3."""


@runtime_checkable
class _ReadAtHandoff_5(_ReadAtSyncEvt_4, Protocol):
    """Readable fields at phase HANDOFF_5 — cumulative through SYNC_EVT_4."""


@runtime_checkable
class _ReadAtApply_6(_ReadAtHandoff_5, Protocol):
    """Readable fields at phase APPLY_6 — cumulative through HANDOFF_5."""


@runtime_checkable
class _ReadAtRespond_7(_ReadAtApply_6, Protocol):
    """Readable fields at phase RESPOND_7 — cumulative through APPLY_6."""


@runtime_checkable
class _ReadAtFinalize_8(_ReadAtRespond_7, Protocol):
    """Readable fields at phase FINALIZE_8 — cumulative through RESPOND_7."""

    @property
    def finished_requests(self) -> List[LlmRequest]:
        """Requests that finished (terminated) during this batch's response."""
        ...


@runtime_checkable
class _ReadAtAll(_ReadAtFinalize_8, Protocol):
    """Readable after all phases — cumulative through FINALIZE_8."""


@dataclasses.dataclass
class _WriteAtSchedule_0:
    """Fields producible at phase SCHEDULE_0."""

    scheduled_batch: Optional[ScheduledRequests] = None
    """The scheduled set of requests this batch will run."""
    can_queue: Optional[bool] = None
    """Whether every TP rank has a non-empty scheduled batch this iter."""
    can_queue_this_rank: Optional[bool] = None
    """Whether THIS specific rank has a non-empty scheduled batch."""
    fitting_disagg_gen_init_requests: Optional[List[LlmRequest]] = None
    """Disagg gen-init requests that fit this iter."""
    num_fitting_reqs: Optional[int] = None
    """Number of regular (non-disagg-gen-init) requests that fit."""
    microbatch_id: Optional[int] = None
    """The PP slot ring index for this batch (``iter_counter mod n``)."""
    previous_tensors_device: Optional[SampleStateTensors] = None
    """Forward-input device tensors threaded in from the previous batch."""
    num_accepted_tokens_device: Optional[torch.Tensor] = None
    """Per-request accepted-token counts from the in-bridge draft model."""


@dataclasses.dataclass
class _WriteAtForward_1:
    """Fields producible at phase FORWARD_1."""

    batch_outputs: Optional[Dict[str, torch.Tensor]] = None
    """Forward pass outputs: logits, hidden_states, additional outputs."""


@dataclasses.dataclass
class _WriteAtSample_2:
    """Fields producible at phase SAMPLE_2."""

    sample_state: Optional[SampleState] = None
    """Sampler state: ``sampler_event``, host tokens, device tensors."""


@dataclasses.dataclass
class _WriteAtStateUpd_3:
    """Fields producible at phase STATE_UPD_3 — no field is produced here."""


@dataclasses.dataclass
class _WriteAtSyncEvt_4:
    """Fields producible at phase SYNC_EVT_4 — no field is produced here."""


@dataclasses.dataclass
class _WriteAtHandoff_5:
    """Fields producible at phase HANDOFF_5 — no field is produced here."""


@dataclasses.dataclass
class _WriteAtApply_6:
    """Fields producible at phase APPLY_6 — no field is produced here."""


@dataclasses.dataclass
class _WriteAtRespond_7:
    """Fields producible at phase RESPOND_7."""

    finished_requests: Optional[List[LlmRequest]] = None
    """Requests that finished (terminated) during this batch's response."""


@dataclasses.dataclass
class _WriteAtFinalize_8:
    """Fields producible at phase FINALIZE_8 — no field is produced here."""


@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.SCHEDULE_0],
) -> Tuple[_ReadAtForward_1, _WriteAtSchedule_0]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.FORWARD_1],
) -> Tuple[_ReadAtSample_2, _WriteAtForward_1]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.SAMPLE_2],
) -> Tuple[_ReadAtStateUpd_3, _WriteAtSample_2]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.STATE_UPD_3],
) -> Tuple[_ReadAtSyncEvt_4, _WriteAtStateUpd_3]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.SYNC_EVT_4],
) -> Tuple[_ReadAtHandoff_5, _WriteAtSyncEvt_4]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.HANDOFF_5],
) -> Tuple[_ReadAtApply_6, _WriteAtHandoff_5]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.APPLY_6],
) -> Tuple[_ReadAtRespond_7, _WriteAtApply_6]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.RESPOND_7],
) -> Tuple[_ReadAtFinalize_8, _WriteAtRespond_7]: ...
@overload
async def step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.FINALIZE_8],
) -> Tuple[_ReadAtAll, None]: ...


@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.SCHEDULE_0],
) -> Optional[Tuple[_ReadAtForward_1, _WriteAtSchedule_0]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.FORWARD_1],
) -> Optional[Tuple[_ReadAtSample_2, _WriteAtForward_1]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.SAMPLE_2],
) -> Optional[Tuple[_ReadAtStateUpd_3, _WriteAtSample_2]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.STATE_UPD_3],
) -> Optional[Tuple[_ReadAtSyncEvt_4, _WriteAtStateUpd_3]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.SYNC_EVT_4],
) -> Optional[Tuple[_ReadAtHandoff_5, _WriteAtSyncEvt_4]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.HANDOFF_5],
) -> Optional[Tuple[_ReadAtApply_6, _WriteAtHandoff_5]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.APPLY_6],
) -> Optional[Tuple[_ReadAtRespond_7, _WriteAtApply_6]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.RESPOND_7],
) -> Optional[Tuple[_ReadAtFinalize_8, _WriteAtRespond_7]]: ...
@overload
async def try_step(
    handle: Batch,
    *,
    through: Literal[BatchPhase.FINALIZE_8],
) -> Optional[Tuple[_ReadAtAll, None]]: ...


@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.SCHEDULE_0],
) -> AsyncIterator[Tuple[None, _WriteAtSchedule_0]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.FORWARD_1],
) -> AsyncIterator[Tuple[_ReadAtForward_1, _WriteAtForward_1]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.SAMPLE_2],
) -> AsyncIterator[Tuple[_ReadAtSample_2, _WriteAtSample_2]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.STATE_UPD_3],
) -> AsyncIterator[Tuple[_ReadAtStateUpd_3, _WriteAtStateUpd_3]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.SYNC_EVT_4],
) -> AsyncIterator[Tuple[_ReadAtSyncEvt_4, _WriteAtSyncEvt_4]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.HANDOFF_5],
) -> AsyncIterator[Tuple[_ReadAtHandoff_5, _WriteAtHandoff_5]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.APPLY_6],
) -> AsyncIterator[Tuple[_ReadAtApply_6, _WriteAtApply_6]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.RESPOND_7],
) -> AsyncIterator[Tuple[_ReadAtRespond_7, _WriteAtRespond_7]]: ...
@overload
@asynccontextmanager
def batch_phase(
    p: Literal[BatchPhase.FINALIZE_8],
) -> AsyncIterator[Tuple[_ReadAtFinalize_8, _WriteAtFinalize_8]]: ...


# ===== END GENERATED =====


# --------------------------------------------------------------------------- #
# Implementations. The ``@overload`` chains above describe the typed
# signatures; these impls just delegate to the generic primitives in
# ``coroutines``.
# --------------------------------------------------------------------------- #


async def step(handle, *, through):  # type: ignore[misc]
    """``step`` narrowed for :class:`BatchStorage` -- see overloads above."""
    return await _generic_step(handle, through=through)


async def try_step(handle, *, through):  # type: ignore[misc]
    """``try_step`` narrowed for :class:`BatchStorage` -- see overloads above."""
    return await _generic_try_step(handle, through=through)


def batch_phase(p):  # type: ignore[misc]
    """``batch_phase`` narrowed for :class:`BatchStorage` -- see overloads above.

    Sync function returning an ``AbstractAsyncContextManager`` (the
    runtime decorates the underlying generator with
    ``@asynccontextmanager``). The narrowed return type makes
    ``async with batch_phase(BatchPhase.<NAME>) as (r, w):`` give ``r``
    and ``w`` the proper ``_ReadAt<Name>`` / ``_WriteAt<Name>`` types at
    every call site (PascalCase form, e.g. ``_ReadAtSchedule_0``).
    """
    return _generic_batch_phase(p)
