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
    ``resource_manager.update_resources``. ``attn_metadata`` is
    read from the FORWARD_2 storage slot (produced by
    ``ForwardConcern``) and forwarded to ``update_resources`` so
    the KV cache manager can shift draft-token slots when
    speculative decoding is enabled. ``kv_cache_dtype_byte_size``
    is a static model property -- captured at construction so we
    don't need to expose it through ``BatchStorage`` per iter.

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
        kv_cache_dtype_byte_size: float,
    ) -> None:
        self._resource_manager = resource_manager
        # Static for the lifetime of the executor (set once on
        # ``model_engine``); we cache it here instead of threading it
        # through ``BatchStorage`` each iter.
        self._kv_cache_dtype_byte_size = kv_cache_dtype_byte_size

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
        # attention metadata bookkeeping). ``attn_metadata`` is
        # threaded in by ``ForwardConcern`` via the FORWARD_2 write
        # view; ``update_resources`` uses it for spec-decode KV
        # draft-token shifting. ``update_resources`` accepts ``None``
        # for both args (downstream consumers are gated on optional
        # attribute presence), so empty-rank / non-spec paths still
        # work cleanly.
        r8, _ = await enter_phase(BatchPhase.RESPOND_8)
        self._resource_manager.update_resources(
            scheduled_batch,
            r8.attn_metadata,
            self._kv_cache_dtype_byte_size,
        )


__all__ = ["ResourceConcern"]
