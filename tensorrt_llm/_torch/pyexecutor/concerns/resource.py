"""Resource concern: KV cache + per-resource-manager prep / update.

Plain-loop scope:

* RESOURCE_PREP_1
  - Run ``resource_manager.prepare_resources(scheduled_batch)`` for
    the main scheduled batch (gates KV block allocation + per-
    resource-manager bookkeeping for each request).
  - Disagg-gen-init prep is the OTHER reason RESOURCE_PREP_1
    exists; not present here because the plain-loop bring-up has
    no disagg path. When DisaggConcern lands, it writes
    ``disagg_gen_init_to_prepare`` at SCHEDULE_0 and reads it back
    at RESOURCE_PREP_1 to call its own prep loop -- this concern
    only handles the main scheduled_batch.
* RESPOND_8
  - Free per-request resources for finished requests via
    ``resource_manager.update_resources``.

The legacy executor also calls ``revert_gen_alloc`` when can_queue
flips False; that path is V2-scheduler-only and tied to its own
KV growth model. Skipped here -- once V2 path is exercised, the
revert lands as either an additional RESOURCE_PREP_1 branch or a
separate concern method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..batch_storage import BatchPhase, enter_phase

if TYPE_CHECKING:
    from ..context import Context
    from ..resource_manager import ResourceManager


class ResourceConcern:
    """Plain-loop KV / resource-manager driver."""

    def __init__(
        self,
        *,
        resource_manager: "ResourceManager",
    ) -> None:
        self._resource_manager = resource_manager

    async def handle_batch(self, ctx: "Context") -> None:
        # RESOURCE_PREP_1 -- prepare resources for the main batch.
        r1, _ = await enter_phase(BatchPhase.RESOURCE_PREP_1)
        scheduled_batch = r1.scheduled_batch
        # Skip when ScheduleConcern produced an empty batch (nothing
        # to forward / no resources to prep). The ``can_queue`` check
        # in the BATCH body short-circuits forward+sample on the
        # same condition.
        if not r1.can_queue:
            # Still need to participate at RESPOND_8 (no-op for
            # update_resources because there's nothing to free).
            r8, _ = await enter_phase(BatchPhase.RESPOND_8)
            return
        self._resource_manager.prepare_resources(scheduled_batch)

        # RESPOND_8 -- update resources (free finished, rebalance
        # attention metadata bookkeeping). The legacy code passes
        # ``model_engine.attn_metadata`` and a ``kv_cache_dtype_byte_size``
        # to handle the perf-metric / KV-events bookkeeping; for the
        # plain-loop minimum we forward whatever ``ForwardConcern``
        # has stashed via the read view's downstream slots. Since we
        # don't yet plumb attn_metadata through BatchStorage, pass
        # ``None`` -- ``update_resources`` accepts that and the
        # downstream consumers (perf metrics etc.) are also gated
        # on optional attribute presence.
        r8, _ = await enter_phase(BatchPhase.RESPOND_8)
        self._resource_manager.update_resources(
            scheduled_batch,
            None,
            None,
        )


__all__ = ["ResourceConcern"]
