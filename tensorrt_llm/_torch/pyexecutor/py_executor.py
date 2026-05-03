import dataclasses
import datetime
import functools
import os
import threading
import time
import traceback
from contextlib import contextmanager
from enum import IntEnum
from queue import Queue
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch

from tensorrt_llm.llmapi import DisaggScheduleStyle
from tensorrt_llm.serve.responses_utils import get_steady_clock_now_in_seconds

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._utils import (customized_gc_thresholds, is_trace_enabled,
                                 mpi_comm, mpi_disabled, nvtx_range,
                                 set_thread_local_mpi_comm, trace_func)
from tensorrt_llm.bindings.executor import (DisServingRequestStats,
                                            FinishReason, InflightBatchingStats,
                                            IterationStats, KvCacheStats,
                                            RequestStage, RequestStats,
                                            RequestType, SpecDecodingStats,
                                            StaticBatchingStats)
from tensorrt_llm.bindings.internal.batch_manager import (LlmRequestType,
                                                          ReqIdsSet)
from tensorrt_llm.inputs.multimodal import strip_mm_data_for_generation
from tensorrt_llm.llmapi.llm_args import PeftCacheConfig, WaitingQueuePolicy
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import CpType
from tensorrt_llm.runtime.generation import CUASSERT
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator
from tensorrt_llm.tools.profiler.host_profile_tools.host_profiler import \
    host_profiler_context

from ..distributed import Distributed
from ..expert_statistic import ExpertStatistic
from ..models.modeling_llama import Llama4ForConditionalGeneration
from ..models.modeling_utils import DecoderModelForCausalLM
from ..modules.decoder_layer import DecoderLayer
from ..speculative.drafter import Drafter
from ..speculative.spec_sampler_base import SampleStateTensorsSpec
from ..speculative.speculation_gate import SpeculationGate
from .connectors.kv_cache_connector import KvCacheConnectorManager
from .dwdp import DwdpManager
from .error_classification import ErrorBudget
from .executor_request_queue import ExecutorRequestQueue, RequestQueueItem
from .guided_decoder import GuidedDecoder
from .handle_additional_outputs import HandleAdditionalOutputs
from .handle_logits import HandleLogits
from .hang_detector import HangDetector
from .kv_cache_transceiver import KvCacheTransceiver
from .llm_request import (ExecutorRequest, LlmRequest, LlmRequestState,
                          LlmResponse, get_draft_token_length)
from .model_engine import ModelEngine
from .perf_metrics_manager import PerfMetricsManager
from .request_utils import (RequestBroadcaster, attach_py_objects_to_requests,
                            get_from_waiting_queue, merge_requests)
from .resource_manager import (KVCacheManagerV2, ResourceManager,
                               ResourceManagerType, request_context)
from .sampler import (AsyncWorkerMixin, Sampler, SamplerEvent, SampleState,
                      SampleStateTensors, TRTLLMSampler)
from .scheduler import (RequestScheduler, ScheduledRequests,
                        SerializableSchedulerOutput, WaitingQueue,
                        create_waiting_queue)
from .scheduler.adp_router import ADPRouter

# Environment variable to specify iteration ranges for profiling start/stop.
# Format: "start1-stop1,start2-stop2,..." or single iterations "iter1,iter2,..."
PROFILE_START_STOP_ENV_VAR_NAME = "TLLM_PROFILE_START_STOP"

# Environment variable to enable PyTorch profiler tracing.
# Set to a path to save detailed tracing of PyTorch operations.
PROFILE_TRACE_ENV_VAR_NAME = "TLLM_TORCH_PROFILE_TRACE"

# Environment variable to control which ranks print step logging.
# Format: comma-separated rank IDs, e.g. "0,1,3", or "all" for all ranks.
# Default: "0" (only rank 0 prints, matching existing behavior).
PROFILE_LOG_RANKS_ENV_VAR_NAME = "TLLM_PROFILE_LOG_RANKS"


class PPCommTag(IntEnum):
    """
    Unique tags for pipeline parallelism communication.
    """
    TERMINATION = 20000
    SCHEDULE_RESULT = 20001
    EXECUTED_BATCH_NUM = 20002
    SAMPLE_STATE = 20003


@functools.cache
def _load_iteration_indexes(env_var: str):
    spans = os.environ.get(env_var, None)
    starts, stops = [], []

    if spans:
        spans = spans.split(',')

        for span in spans:
            try:
                if '-' in span:
                    start, stop = span.strip().split('-')
                    starts.append(int(start))
                    stops.append(int(stop))
                else:
                    it = int(span.strip())
                    starts.append(it)
                    stops.append(it)
            except ValueError as e:
                raise ValueError(
                    f"Cannot parse span in environment variable `{env_var}`: {e}"
                ) from None

    return frozenset(starts), frozenset(stops)


def _strip_py_multimodal_data_post_prefill(request: LlmRequest) -> None:
    """Drop pinned encoder cache and raw pre-encoder tensors after prefill completes.

    Wraps `strip_mm_data_for_generation` and mutates the shared `request.py_multimodal_data`
    in-place so the `LlmRequest`'s multimodal tensors actually get freed (unlike
    `MultimodalParams.strip_for_generation`, which rebinds a per-forward-call wrapper's attribute
    and leaves the request's dict untouched).
    """
    mm_data = getattr(request, "py_multimodal_data", None)
    if not mm_data:
        return
    strip_mm_data_for_generation(mm_data)


@dataclasses.dataclass
class BatchState:
    scheduled_requests: ScheduledRequests
    sample_state: SampleState

    iter_start_time: float = 0
    iter_stats: IterationStats = None


@dataclasses.dataclass
class BatchStatePP(BatchState):
    microbatch_id: int = -1


class AsyncTransferManager:
    """
    Handle asynchronous transfer of KV cache after a request has completed.
    When running with both the KV cache transceiver and the KV cache connector, we must ensure that BOTH transfers (if any) are completed before we can release the KV cache blocks.
    The AsyncTransferManager has a few key responsibilities:
    1. Track requests in transfer.
    2. Pin blocks for reuse while blocks are in transfer.
    3. Unpin blocks after all transfers are complete.

    TODO(jthomson04): This only handles async send/saving, not loading. Loading kv cache is handled through a separate codepath. Eventually, we'll want to merge these two paths.
    """

    class RequestTransferMetadata:

        def __init__(self, block_id: Optional[int]):
            self.block_id = block_id
            self.counter = 0

        def start_transfer(self):
            self.counter += 1

        def end_transfer(self) -> bool:
            """
            Returns:
                bool: True if there are no more transfers for this request
            """
            self.counter -= 1
            return self.counter == 0

    def __init__(self,
                 resource_manager: "ResourceManager",
                 should_store_blocks: bool = True):
        self.resource_manager = resource_manager
        self.kv_cache_manager = resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)

        self.should_store_blocks = should_store_blocks

        # Mapping of request id to the LlmRequest
        self._requests_in_transfer: Dict[int, LlmRequest] = dict()

        # Mapping of request id to the request metadata
        self._request_transfer_metadata: Dict[
            int, self.RequestTransferMetadata] = dict()

    def requests_in_transfer(self) -> Dict[int, LlmRequest]:
        return self._requests_in_transfer

    def start_transfer(self, request: LlmRequest):
        """
        Called when a Cache transceiver or connector transfer is started.
        1. Increment the counter for the request.
        2. Releases all resources except for the KV cache, if not already released.
        3. Store KV cache blocks for reuse.
        """

        req_id = request.py_request_id

        if req_id not in self._requests_in_transfer:
            for resource_mgr_type in (
                    ResourceManagerType.SEQ_SLOT_MANAGER,
                    ResourceManagerType.SPEC_RESOURCE_MANAGER):
                if resource_mgr_type in self.resource_manager.resource_managers and self.resource_manager.resource_managers[
                        resource_mgr_type] is not None:
                    self.resource_manager.resource_managers[
                        resource_mgr_type].free_resources(request)

            request.state = LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

            if self.should_store_blocks:
                block_id = self.kv_cache_manager.store_blocks_for_reuse(
                    request, True)
            else:
                block_id = None

            self._requests_in_transfer[req_id] = request
            self._request_transfer_metadata[
                req_id] = self.RequestTransferMetadata(block_id)

        self._request_transfer_metadata[req_id].start_transfer()

    def end_transfer(self, request: LlmRequest) -> bool:
        """
        Called after a send of KV cache is complete.
        1. Decrements counter for request.
        2. If there are no more inflight transfers for this request, unpin the blocks and mark the request as complete.

        Returns:
            bool: True if the request should be terminated after call to end_transfer
        """
        try:
            transfer_metadata = self._request_transfer_metadata[
                request.py_request_id]
        except KeyError:
            logger.warning(
                f"Request {request.py_request_id} not found in transfer manager"
            )
            return False

        if transfer_metadata.end_transfer():
            self._requests_in_transfer.pop(request.py_request_id)
            self._request_transfer_metadata.pop(request.py_request_id)

            if self.should_store_blocks:
                self.kv_cache_manager.unpin_blocks_by_id(
                    transfer_metadata.block_id)

            # We don't want to overwrite any error state.
            if request.state != LlmRequestState.DISAGG_TRANS_ERROR:
                request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE

            return True

        return False

    def has_any_inflight_requests(self) -> bool:
        return len(self._requests_in_transfer) > 0


class PyExecutor:
    """Legacy executor with per-method refactor categorization comments.

    Refactor categorization scheme (read this before scanning the per-method
    comments below)
    ===========================================================================

    Each method / property below carries a one-line ``# REFACTOR:`` comment
    above its definition (above any decorators). The categories mirror the
    plan in ``concerns.py`` and ``context.py``:

    Category 1 (NOT in the new coroutine loop)
    ------------------------------------------

    - ``1a (public API)``: called by user code from the main thread.
      Includes lifecycle (``__init__`` / ``start_worker`` / ``shutdown`` /
      ``__enter__`` / ``__exit__``) and the data-plane methods
      (``enqueue_request*`` / ``await_responses`` / ``cancel_request`` /
      ``get_latest_*`` / etc.).

    - ``1b (legacy loop runtime)``: the legacy executor loops themselves
      (``_executor_loop`` / ``_executor_loop_overlap`` / ``_executor_loop_pp``)
      and immediate runtime helpers (``_event_loop_wrapper``,
      ``_executor_loop_cleanup``). Also marks methods that disappear
      in the new design with their work distributed elsewhere:

      * Multi-concern fan-outs annotated in-line in the loop body
        (split into the respective concerns' phase blocks):
        ``_prepare_and_schedule_batch``, ``_handle_executed_batch``,
        ``_process_previous_batch``, ``_send_kv_async``,
        ``_handle_responses``.
      * Loop-side dispatchers: ``_handle_special_queue_items``.
      * Inter-concern sequencing helpers identified by review (split
        with data passed through ``BatchStorage`` across at least 2
        phases): ``_prepare_disagg_gen_init`` (disagg -> resource ->
        disagg), ``_end_transfer_and_maybe_terminate`` (response ->
        disagg -> schedule -> termination).
      * Failing-path helper that conflates two distinct modes:
        ``_handle_errors``. Splits in the new design into
        - Mode A (catastrophic, no ``requests=`` arg): concerns
          just ``raise``; the SCHEDULER catches at the top of its
          loop, calls the Mode B utility on the active set, and
          sets the shutdown latch.
        - Mode B (per-request fail-fast, with ``requests=`` arg):
          becomes a 2b utility ``fail_requests(reqs, msg)``.
        Per-concern ``finally:`` blocks handle concern-LOCAL
        cleanup (CUDA event handles, etc.); the failing of
        REQUESTS is the SCHEDULER's / ``fail_requests``'s job, not
        per-concern work.
      * Legacy machinery eliminated by coroutine semantics (no
        replacement needed): ``wait_on_pp_send_handles`` (slot rings
        of ``isend`` handles become coroutine locals).

    Category 2 (will live INSIDE the new coroutine loop)
    ----------------------------------------------------

    - ``2a (concern: <name>)``: used by exactly ONE concern in the new
      design. Will move INTO that concern's class. The ``<name>`` matches
      the concern name in ``concerns.py`` -- e.g., ``schedule``, ``forward``,
      ``sample``, ``response``, ``disagg``, ``kv_connector``, ``spec_decode``,
      ``guided_decoder``, ``perf_metric``, ``iter_stats``, ``profile``,
      ``control``, ``benchmark_disagg_gate``, ``kv_cache_events``,
      ``ring_broadcast_sample``.

    - ``2b (shared helper, NON-interleaving)``: GENUINELY general-purpose
      code that multiple concerns call individually. Bodies do NOT
      interleave concerns -- that's the criterion that distinguishes
      2b from the "interleaves, needs splitting" 1b sub-set above.
      Bodies MAY touch multiple concerns' state when they represent
      a coherent atomic operation (e.g., "really terminate this
      request"); the distinguishing test is "would each concern
      individually want to do this exact unit, vs being a fan-out
      of distinct steps?".

      The surviving 2b set lands on three NEW services exposed via
      ``ctx.svc.*`` (see ``concerns.py`` "SHARED SERVICES" + the
      ``Service`` dataclass docstring in ``context.py``):

        * ``_terminate_request`` / ``_do_terminate_request`` ->
          ``ctx.svc.termination.terminate(req)``. Called from
          normal-path concern code (response RESPOND_8, schedule
          paused-request handling, disagg / kv_connector terminate
          sweeps), the SCHEDULER's catastrophic-error handler, and
          the per-request ``fail_requests`` utility.
        * ``fail_requests(ctx, reqs, msg)`` (NEW free function in
          ``concerns/_shared.py``; surviving body of
          ``_handle_errors``'s per-request fail-fast mode -- see
          1b notes above). Composes three services:
          ``ctx.svc.pool.remove_active(reqs)``,
          ``ctx.svc.client.enqueue(error_responses)``,
          ``ctx.svc.termination.terminate(req)``. Called by
          concerns that detect a specific bad subset (validation
          failure, timeout, guided-decoder grammar failure, ...)
          and by the SCHEDULER's catastrophic-error handler with
          the active set.

      Two more services in the same shape exist alongside
      ``termination`` -- ``RequestPool`` (owns ``active_requests``
      + ``inflight_req_ids``) and ``ClientChannel`` (owns response
      output / TP gather / cross-thread put). The methods that
      land on those services are categorized 2a-on-the-service
      below (see per-method REFACTOR comments).

    Special / scheduler-direct
    --------------------------

    - ``SPECIAL``: used by neither a concern nor the legacy loop directly.
      Typically scheduler-iter-direct code (the SCHEDULER is not a concern
      -- see ``concerns.py``'s "THE SCHEDULER SIDE" section), or a
      cross-thread state flag with non-trivial reader/writer split.

    Threading note
    ==============

    ``PyExecutor`` is a single class today, but the refactor splits its
    surface into:

    - a main-thread side: ``PyExecutorCoro.__init__`` builds everything,
      then exposes a small public API (data-plane methods, lifecycle,
      shutdown). 1a + 1b methods stay here (1b methods get DELETED once
      the new loop replaces them).

    - a loop-thread side: a ``run_loop(ctx)`` entry point owned by the
      new coroutine SCHEDULER. 2a methods move into their respective
      concern classes; 2b methods become standalone helpers callable
      from inside the loop layer; SPECIAL methods become inlined into
      the SCHEDULER iter or the loop's bootstrap code.

    The ``# REFACTOR:`` line on each method below records where it
    will land.
    """

    # Minimum number of async micro batches for async PP execution.
    # This is a trade-off between memory usage and performance.
    # If the number of micro batches is too small, the executor will spend too much time in synchronization.
    # If the number of micro batches is too large, the executor will spend too much host memory (No additional GPU memory is required).
    # 1024 in-flight micro batches can avoid synchronization in most cases and keep host memory usage low.
    MIN_ASYNC_MICRO_BATCH_NUM = 1024

    # REFACTOR: 1a (public API) -- main-thread lifecycle (constructor).
    def __init__(
            self,
            resource_manager,
            scheduler: RequestScheduler,
            model_engine: ModelEngine,
            sampler: Sampler,
            dist: Distributed,
            max_num_sequences: int,
            drafter: Optional[Drafter] = None,
            disable_overlap_scheduler: bool = False,
            max_input_len: int = 0x7fffffff,
            max_batch_size: int = 8,
            max_beam_width: int = 1,
            max_draft_len: int = 0,
            max_total_draft_tokens: int = 0,
            kv_cache_transceiver: Optional[KvCacheTransceiver] = None,
            guided_decoder: Optional[GuidedDecoder] = None,
            garbage_collection_gen0_threshold: Optional[int] = None,
            start_worker: bool = True,
            kv_connector_manager: Optional[KvCacheConnectorManager] = None,
            max_seq_len: Optional[int] = None,
            peft_cache_config: Optional[PeftCacheConfig] = None,
            virtual_memory_pools: Optional[dict] = None,
            hang_detection_timeout: Optional[int] = None,
            execution_stream: Optional[torch.cuda.Stream] = None,
            waiting_queue_policy: WaitingQueuePolicy = WaitingQueuePolicy.FCFS,
            adp_router: Optional[ADPRouter] = None,
            dwdp_manager: Optional[DwdpManager] = None):
        super(PyExecutor, self).__init__()
        self.device_id = torch.cuda.current_device()
        self.global_rank = dist.rank
        # Store the execution stream for model forward operations.
        # This stream is used for proper synchronization with KVCacheTransferManager.
        # execution_stream can be provided by create_py_executor
        # Create a new stream if none provided
        self.execution_stream = execution_stream if execution_stream is not None else torch.cuda.Stream(
        )
        logger.info(
            f"[PyExecutor] execution_stream initialized: {self.execution_stream}. "
        )

        self.peft_cache_config = peft_cache_config

        self.iter_counter = 0
        # profile config
        self.profile_start_iters, self.profile_stop_iters = _load_iteration_indexes(
            PROFILE_START_STOP_ENV_VAR_NAME)

        # related modules
        self.resource_manager = resource_manager
        self.scheduler = scheduler
        self.model_engine = model_engine
        self.enable_attention_dp = model_engine.enable_attention_dp
        self.dist = dist
        self.sampler = sampler
        self.drafter = drafter
        self.draft_model_engine = getattr(self.drafter, "draft_model_engine",
                                          None)
        self.guided_decoder = guided_decoder
        self.disable_overlap_scheduler = disable_overlap_scheduler
        self.virtual_memory_pools = virtual_memory_pools

        # enqueue and _fetch_new_requests used data
        self.active = True
        self.max_beam_width = max_beam_width
        self.max_draft_len = max_draft_len
        self.max_total_draft_tokens = max_total_draft_tokens
        self.llm_args = self.model_engine.llm_args
        self.max_stats_len = max(self.llm_args.max_stats_len, 1)
        self.max_num_tokens = self.llm_args.max_num_tokens
        self.print_log = self.llm_args.print_iter_log
        self.enable_iter_perf_stats = self.llm_args.enable_iter_perf_stats
        self.enable_iter_req_stats = self.llm_args.enable_iter_req_stats
        self.stream_interval = self.llm_args.stream_interval
        self.perf_manager = PerfMetricsManager(
            enabled=getattr(self.llm_args, 'return_perf_metrics', False))
        self.attention_dp_enable_balance = (
            self.llm_args.attention_dp_config is not None
            and self.llm_args.attention_dp_config.enable_balance)
        if self.attention_dp_enable_balance:
            self.attention_dp_time_out_iters = self.llm_args.attention_dp_config.timeout_iters
            self.attention_dp_batching_wait_iters = self.llm_args.attention_dp_config.batching_wait_iters
        self.batch_wait_timeout_ms = self.llm_args.batch_wait_timeout_ms
        self.batch_wait_timeout_iters = self.llm_args.batch_wait_timeout_iters
        self.batch_wait_max_tokens_ratio = self.llm_args.batch_wait_max_tokens_ratio
        self.enable_batch_waiting = self.batch_wait_timeout_iters > 0 or self.batch_wait_max_tokens_ratio > 0

        self.num_fetch_requests_cur_rank = 0
        self.num_fetch_requests = 0
        self.shutdown_event = threading.Event()

        # Rolling acceptance tracking for spec decode (disable speculation if rolling acceptance is below threshold)
        spec_config = getattr(self.model_engine, 'spec_config', None)
        self.acceptance_window = getattr(
            spec_config, 'acceptance_window',
            None) if spec_config is not None else None
        self.acceptance_length_threshold = getattr(
            spec_config, 'acceptance_length_threshold',
            None) if spec_config is not None else None
        self.speculation_permanently_disabled = False
        self.speculation_gate = None
        if self.acceptance_window and self.acceptance_length_threshold is not None:
            self.speculation_gate = SpeculationGate(
                self.acceptance_window, self.acceptance_length_threshold)

        # response used data
        self.response_lock = threading.Lock()
        self.response_cv = threading.Condition(self.response_lock)
        self.responses = {}
        self.result_wait_queues = {}

        # kv cache events
        self.kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        # V2 scheduler calls suspend_request() during scheduling, which
        # offloads GPU pages while preserving the radix tree.  The executor
        # does not need to call _terminate_requests (GPU resources are already
        # freed by suspend) or _pause_requests (V2's prepare_context handles
        # resume internally, so resetting to CONTEXT_INIT is unnecessary).
        self._scheduler_manages_kv_suspend = isinstance(self.kv_cache_manager,
                                                        KVCacheManagerV2)
        self.enable_kv_cache_events = self.kv_cache_manager is not None and self.kv_cache_manager.event_buffer_max_size > 0
        self.enable_kv_cache_reuse = self.kv_cache_manager is not None and self.kv_cache_manager.enable_block_reuse
        self.enable_partial_reuse_for_disagg = (
            self.enable_kv_cache_reuse
            and self.kv_cache_manager.enable_partial_reuse)

        self.max_input_len = max_input_len
        # _executor_loop private data
        self.max_num_active_requests = model_engine.get_max_num_sequences()
        self.active_requests: List[LlmRequest] = []
        self.expected_num_active_requests = 0
        # TODO: Remove the condition on the PP size once disagg support from KVCache reuse
        # path is fixed.
        # Buffer for responses generated inside _end_transfer_and_maybe_terminate.
        # With ADP, _enqueue_responses does a tp_gather collective.  When called
        # from _send_kv_async the owning DP rank has a response but the other
        # rank does not, causing a collective mismatch deadlock.  Buffering the
        # responses and flushing them at a synchronised point in the executor
        # loop avoids the mismatch.
        self._pending_transfer_responses: List[Tuple[int, LlmResponse]] = []

        self.async_transfer_manager = AsyncTransferManager(
            self.resource_manager,
            should_store_blocks=self.enable_partial_reuse_for_disagg
            and not self.kv_cache_manager.is_vswa and self.dist.pp_size == 1)

        # Router is built after async_transfer_manager so KVCacheAwareADPRouter
        # can receive the transfer-manager reference at construction time.
        self.adp_router: ADPRouter = ADPRouter.create(
            dist=self.dist,
            kv_cache_manager=self.kv_cache_manager,
            attention_dp_config=self.llm_args.attention_dp_config,
            async_transfer_manager=self.async_transfer_manager,
        )

        self.previous_batch: Optional[BatchState] = None
        self.has_previous_draft_tokens = False
        self.num_scheduled_requests: int = 0
        self.benchmark_req_queues_size = int(
            os.environ.get("TLLM_BENCHMARK_REQ_QUEUES_SIZE", 0))

        # list of requests in each PP micro batch
        self.num_micro_batches = max(self.dist.pp_size,
                                     self.MIN_ASYNC_MICRO_BATCH_NUM)
        self.micro_batches: List[BatchStatePP
                                 | None] = [None] * self.num_micro_batches
        self.send_handles = [None] * self.num_micro_batches
        # schedule handle for PP to propagate the first PP rank's schedule result
        self.send_schedule_handles = [None] * self.num_micro_batches
        self.send_expected_batch_num_handles = [None] * self.num_micro_batches
        self.unhandled_batch_counter = 0
        self.pp_scheduler_max_retry_count = int(
            os.environ.get("TLLM_PP_SCHEDULER_MAX_RETRY_COUNT", 10))
        self.pp_multi_stream_sample = os.environ.get(
            "TRTLLM_PP_MULTI_STREAM_SAMPLE", "1") == "1"
        self.sample_stream = torch.cuda.Stream()
        self.finish_sample_event = torch.cuda.Event()
        if (self.dist.pp_size > 1 and self.pp_multi_stream_sample
                and isinstance(self.sampler, TRTLLMSampler)):
            # TRTLLM sampler uses default stream for store and algorithms.
            # To enable multi-stream sampling, we need to re-initialize
            # the sampler store and algorithms on the sample stream.
            with torch.cuda.stream(self.sample_stream):
                self.sampler._initialize_store()
                self.sampler._instantiate_algorithms()

        # Set of request IDs that are currently in flight across all micro batches.
        # The scheduler will avoid scheduling requests that are already in flight.
        self.inflight_req_ids = ReqIdsSet()

        # During warmup, we don't enable the profiler
        # Run warmup on the execution_stream for proper synchronization with
        # KVCacheTransferManager's onboard/offload operations.
        self.is_warmup = True

        self.execution_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.execution_stream):
            self.model_engine.warmup(self.resource_manager)
            if self.draft_model_engine is not None:
                self.draft_model_engine.warmup(self.resource_manager)

        # Ensure the default stream waits for execution_stream to complete
        # before subsequent operations.
        torch.cuda.current_stream().wait_stream(self.execution_stream)
        self.is_warmup = False

        # Snapshot some cumulative KV cache counters so that stats reported to
        # users exclude blocks reused and missed during warmup dummy requests.
        if hasattr(self.kv_cache_manager, 'snapshot_warmup_baseline'):
            self.kv_cache_manager.snapshot_warmup_baseline()

        self.is_shutdown = False
        self._fatal_error: Optional[BaseException] = None
        self._error_budget = ErrorBudget()
        self.max_batch_size = max_batch_size
        self.adp_ctx_waiting_iters_count = 0
        self.adp_ctx_batching_wait_iters_count = 0
        self.batch_wait_iters_count = 0

        def on_detected():
            self._handle_errors(
                f"Hang detected on rank {self.global_rank} in PyExecutor.")
            self.shutdown_event.set()
            self.is_shutdown = True

        self.hang_detector = HangDetector(timeout=hang_detection_timeout,
                                          on_detected=on_detected)

        # request fetcher initialization
        self._set_global_steady_clock_offset()
        self.executor_request_queue = ExecutorRequestQueue(
            dist=self.dist,
            max_batch_size=max_batch_size,
            enable_iter_perf_stats=self.enable_iter_perf_stats,
            batch_wait_timeout_ms=self.batch_wait_timeout_ms,
        )
        # When overlap scheduler is enabled then when starting to handle a new prompt,
        # _sample_async is called twice before the first call to update_requests:
        # - 1st time as a context request that operates on the 1st generated token
        # - 2nd time as a generation request that operates on the 2nd generated token.
        # and only after these two calls the sampler's update_request method is called.
        # So in a sampler that works by the expected flow of handling the logits in
        # _sample_async, every update_request doesn't handle the newest token, but one
        # before it. Since all these calls work on the same request object, then its
        # logits storage contains the logits of both the token update_requests should work
        # on, and also its next token. Thus, excluding the last generation logits from any
        # getter is required.
        self.should_exclude_last_generation_logits = (
            not self.disable_overlap_scheduler and self.dist.pp_size == 1)

        # Request processing state (managed by executor)
        self.canceled_req_ids: List[int] = []
        self.control_requests: List[RequestQueueItem] = []
        self.request_accumulated: List[RequestQueueItem] = []
        self.new_active_requests_queue_latency_ms = 0.0
        self._disable_mpi = mpi_disabled()
        self.request_broadcaster = RequestBroadcaster(self.dist,
                                                      self.hang_detector)

        # Waiting queue for requests that have been fetched but not yet scheduled
        self.waiting_queue: WaitingQueue = create_waiting_queue(
            waiting_queue_policy)

        self.control_request_barrier = threading.Event()
        self.control_action_done = threading.Event()

        self.stats_lock = threading.Lock()
        self.stats = []
        self._latest_kv_iter_stats = None
        self._last_kv_iter_stats_fetch_iter = None
        self._kv_iter_stats_interval = getattr(
            getattr(self.llm_args, 'kv_cache_config', None),
            'iteration_stats_interval', 1)
        self.gather_all_responses = False

        self.kv_cache_transceiver = kv_cache_transceiver
        self.is_benchmark_disagg = (self.benchmark_req_queues_size > 0
                                    and self.kv_cache_transceiver is not None)
        # True while the benchmark disagg fill phase is in progress (waiting
        # for all benchmark requests to complete KV transfer before the first
        # forward pass).  Cleared by _check_benchmark_disagg_gate when the
        # can_forward gate opens.  Used by _should_skip_dummy_for_benchmark_disagg
        # to prevent permanent dummy insertion during fill; once False, the
        # normal dummy add-forward-terminate lifecycle handles taper-down.
        # Only relevant in benchmark disagg mode; False otherwise.
        self._benchmark_fill_phase_active = self.is_benchmark_disagg

        # Initialize disagg PP termination handler if needed
        self._disagg_pp_termination_handler = None
        if self.dist.pp_size > 1 and self.enable_kv_cache_reuse and self.kv_cache_transceiver:
            self._disagg_pp_termination_handler = DisaggPPTerminationHandler(
                self.dist, self._do_terminate_request)

        if self.dist.pp_size > 1:
            self.event_loop = self._executor_loop_pp
            # `TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE` controls whether to broadcast the sample state asynchronously.
            # If true, the executor loop can broadcast and handle sample states asynchronously to achieve best perf.
            # If false, the executor loop can only broadcast and handle each sample state in a pre-defined iteration.
            # It is only for debugging purposes.
            # Some tests can disable it to get a deterministic behavior.
            self.pp_async_broadcast_sample_state = os.environ.get(
                "TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE", "1") == "1"
        else:
            self.event_loop = self._executor_loop if self.disable_overlap_scheduler else self._executor_loop_overlap
        if is_trace_enabled("TLLM_TRACE_EXECUTOR_LOOP"):
            self.event_loop = trace_func(self.event_loop)

        if dwdp_manager is not None and not self.disable_overlap_scheduler:
            raise ValueError(
                "DWDP requires disable_overlap_scheduler=True. "
                "Overlap scheduler is not yet supported with DWDP.")

        if self.drafter is not None:
            if self.event_loop.__name__ == self._executor_loop_pp.__name__:
                raise NotImplementedError(
                    "Drafting is not supported for selected executor loop. "
                    "Please disable disagg/pipeline parallelism scheduler.")
        self.garbage_collection_gen0_threshold = garbage_collection_gen0_threshold
        self.max_seq_len = max_seq_len

        self.worker_started = False
        self.worker_lock = threading.Lock()

        self.kv_connector_manager = kv_connector_manager

        self._maybe_init_kv_connector_manager()

        self.dwdp_manager = dwdp_manager

        if start_worker:
            self.start_worker()

    # REFACTOR: 1a (public API) -- ``__init__`` helper, runs on main thread.
    def _maybe_init_kv_connector_manager(self):
        if self.kv_connector_manager is not None:
            if self.kv_cache_transceiver is not None:
                logger.warning(
                    "Both KV Cache Connector and KV Cache Transceiver are enabled. Are you sure you want to do this?"
                )

            if self.dist.pp_size > 1:
                raise NotImplementedError(
                    "KV Cache Connector is not supported with pipeline parallelism."
                )

            if self.kv_cache_manager is None:
                raise ValueError(
                    "KV Cache Connector requires a KV Cache Manager.")

            kv_tensor = self.kv_cache_manager.get_unique_primary_pool()
            self.kv_connector_manager.worker.register_kv_caches(kv_tensor)

            # For each of our layers, we need to register the pre/post hooks.
            # These are used for methods like `wait_for_layer_load` and `save_kv_layer`.
            for _name, module in self.model_engine.model.named_modules():
                if isinstance(module, DecoderLayer):
                    module.register_forward_pre_hook(
                        self.kv_connector_manager.layer_pre_hook)
                    module.register_forward_hook(
                        self.kv_connector_manager.layer_post_hook)

            self.kv_connector_manager.wait_for_initialization()

    # REFACTOR: 1b (multi-concern, inlined+split) -- body interleaves four
    # services / concerns per request: ``response`` (fast-path create
    # + ``ctx.svc.client.enqueue`` when the request is still active),
    # ``disagg`` (``end_transfer``), ``ctx.svc.pool.remove_active``,
    # and ``ctx.svc.termination.terminate``. Both callers (disagg's
    # ctx-cache probe and kv_connector's terminate sweep) iterate
    # completed transfers and call this per request.
    #
    # Refactor direction: split per concern, using the services for
    # the cross-cutting parts:
    #   (a) Inline the per-request 4-step dance into each caller
    #       (disagg / kv_connector concerns), calling
    #       ``ctx.svc.client.enqueue``, ``ctx.svc.pool.remove_active``,
    #       and ``ctx.svc.termination.terminate`` directly. Each call
    #       site gets ~4 lines but the dependencies are visible.
    #   (b) Move the "fast-path response for completed transfer" piece
    #       into a method on the response concern (or on
    #       ``ClientChannel`` if it's purely a response build) that
    #       disagg / kv_connector invoke; the rest stays in disagg.
    # Pick (b) once the response concern is wired; until then, do not
    # treat this method as a destination for new logic.
    def _end_transfer_and_maybe_terminate(self, request: LlmRequest):
        if self.kv_cache_transceiver and request in self.active_requests:
            # Fast-transfer: KV transfer completed in the same iteration
            # before _handle_responses could run. Create the response now
            # while state is still TRANS_IN_PROGRESS (required by C++
            # createResult). Then proceed with end_transfer + termination.
            response = request.create_response(False, self.dist.rank)
            if response:
                response.result.cached_tokens = request.cached_tokens
                # Buffer the response instead of enqueueing immediately.
                # With ADP, _enqueue_responses does a tp_gather collective.
                # Calling it here would deadlock because only the owning DP
                # rank reaches this point; the other DP rank never enters
                # the matching collective.  The buffer is flushed later at
                # _flush_pending_transfer_responses where all ranks
                # participate.
                self._pending_transfer_responses.append(
                    (request.py_request_id, response))
            if self.async_transfer_manager.end_transfer(request):
                self.active_requests.remove(request)
                self._terminate_request(request)
            return
        if self.async_transfer_manager.end_transfer(request):
            # When should_store_blocks is True, _handle_responses already
            # terminated this request via the early-termination path
            # (enable_partial_reuse_for_disagg branch). Skip the redundant
            # termination to avoid double free_resources calls.
            if not self.async_transfer_manager.should_store_blocks:
                self._terminate_request(request)

    def _flush_pending_transfer_responses(self):
        """Enqueue buffered transfer-completion responses.

        Must be called at a point where ALL DP ranks execute in lockstep so
        that the tp_gather inside _enqueue_responses does not deadlock.
        """
        responses = self._pending_transfer_responses
        self._pending_transfer_responses = []
        if responses or self.enable_attention_dp:
            # Even when this rank has no responses we must participate in the
            # collective when ADP is enabled so that the other rank's gather
            # can complete.
            self._enqueue_responses(responses)

    # Performance metrics methods are in PerfMetricsManager (self.perf_manager)

    # REFACTOR: 1b (legacy loop runtime) -- threading-target wrapper that
    # runs on ``worker_thread`` and dispatches to one of the legacy
    # ``_executor_loop*`` variants. Disappears with the new loop.
    def _event_loop_wrapper(self):
        try:
            # Skip line profiler during warmup/memory estimation phase to avoid
            # saving incomplete results that would be overwritten anyway
            enable_profiler = bool(os.environ.get(
                "TLLM_LINE_PROFILER_PATH")) and not self.is_warmup
            with host_profiler_context(enable=enable_profiler), \
                 customized_gc_thresholds(self.garbage_collection_gen0_threshold):
                self.event_loop()
        except Exception as e:
            logger.error(f"Error in event loop: {e}")
            logger.error(traceback.format_exc())
            raise e
        finally:
            self._executor_loop_cleanup()

    # REFACTOR: SPECIAL -- one-way main-thread setter (warmup driver) read
    # by multiple loop-thread sites (profiler, benchmark gate, several
    # concerns). After refactor, lives in ``Configuration.is_warmup``
    # (effectively immutable after warmup completes).
    @property
    def is_warmup(self) -> bool:
        return getattr(self, "_is_warmup", False)

    # REFACTOR: SPECIAL -- see ``is_warmup`` getter above.
    @is_warmup.setter
    def is_warmup(self, value: bool):
        self._is_warmup = value
        # Set warmup flag in model engine to trigger torch compile and avoid moe load balancer statistics update
        self.model_engine.is_warmup = value
        if self.draft_model_engine is not None:
            self.draft_model_engine.is_warmup = value

    # REFACTOR: 1a (public API) -- main-thread lifecycle; spawns the
    # legacy worker_thread (and PP broadcast thread). New design starts
    # ``run_loop`` on its own loop thread instead.
    def start_worker(self):
        with self.worker_lock:
            if not self.worker_started:
                if self.dist.pp_size > 1:
                    self.executed_batch_queue: Queue[BatchStatePP] = Queue(
                        maxsize=self.num_micro_batches)
                    self.executed_batch_response_queue: Queue[
                        BatchStatePP] = Queue(maxsize=-1)
                    broadcast_sample_state_loop = self._broadcast_sample_state_loop
                    if is_trace_enabled("TLLM_TRACE_EXECUTOR_LOOP"):
                        broadcast_sample_state_loop = trace_func(
                            broadcast_sample_state_loop)
                    self.broadcast_sample_state_handler = threading.Thread(
                        target=broadcast_sample_state_loop,
                        daemon=True,
                        name="broadcast_sample_state_handler",
                    )
                    self.broadcast_sample_state_handler.start()
                self.worker_thread = threading.Thread(
                    target=self._event_loop_wrapper, daemon=True)
                self.worker_thread.start()
                self.worker_started = True
            # Start the sampler's async worker, if it is enabled
            if (isinstance(self.sampler, AsyncWorkerMixin)
                    and self.sampler.async_worker_enabled()):
                logger.info("Starting the async worker for sampler D2H copies")
                self.sampler.async_worker_start()

    # REFACTOR: 1a (public API) -- ``__init__`` helper, runs on main thread.
    def _set_global_steady_clock_offset(self):
        assert self.global_rank >= 0, "rank should be >= 0"

        # Sync all ranks
        self.dist.barrier()
        # Immediately take the local steady clock timestamp
        local_timestamp = get_steady_clock_now_in_seconds()
        all_rank_timestamps = self.dist.allgather(local_timestamp)
        if self.global_rank == 0:
            logger.info(
                f"global_steady_clock_offset at each rank: {[local_timestamp - ts for ts in all_rank_timestamps]}"
            )
        # Compute the steady clock offset between rank 0 and current rank
        global_steady_clock_offset = all_rank_timestamps[0] - local_timestamp
        LlmRequest.global_steady_clock_offset = global_steady_clock_offset
        logger.info(
            f"Setting global_steady_clock_offset: {global_steady_clock_offset} seconds for rank {self.global_rank}"
        )

    # REFACTOR: 1a (public API) -- context manager entry.
    def __enter__(self):
        return self

    # REFACTOR: 1a (public API) -- context manager exit; calls ``shutdown``.
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    # REFACTOR: 1a (public API) -- enqueue user requests.
    def enqueue_requests(
        self,
        requests: List[ExecutorRequest],
        result_wait_queue: "Optional[ray.actor.ActorHandle]" = None
    ) -> List[int]:
        """
        Enqueue new requests
        """
        req_ids = self.executor_request_queue.enqueue_requests(requests)
        if result_wait_queue is not None:
            with self.response_cv:
                for req_id in req_ids:
                    self.result_wait_queues[req_id] = result_wait_queue
        return req_ids

    # REFACTOR: 1a (public API) -- block until responses are available.
    def await_responses(
        self,
        id: Optional[Union[List[int], int]] = None,
        timeout: Optional[datetime.timedelta] = None,
    ) -> Union[List[List[LlmResponse]], List[LlmResponse]]:
        """
        Await ready responses
        Args:
            id (Optional[Union[List[int], int]]): Request id
            timeout (Optional[datetime.timedelta]): The maximum time to wait for new responses
        Returns:
            Union[List[LlmResponse], List[List[LlmResponse]]]: Responses
        """
        timeout = timeout.total_seconds() if timeout is not None else None
        if id is None:
            return self._await_any_response(timeout=timeout)
        if isinstance(id, int):
            return self._await_single_response(id=id, timeout=timeout)
        responses = []
        for req_id in id:
            responses.append(
                self._await_single_response(id=req_id, timeout=timeout))

        return responses

    # REFACTOR: 1a (public API) -- request cancellation entry point.
    def cancel_request(self, id: int):
        """
        Cancel the request with provided request id
        Args:
            id (int): The request id for which to cancel the response
        """
        self.executor_request_queue.enqueue_cancel_request(id)

    # REFACTOR: 1a (public API) -- main-thread shutdown entry point.
    def shutdown(self):
        """
        Signals the server to shutdown.
        """
        self.executor_request_queue.enqueue_shutdown_request()
        self.shutdown_event.wait()
        if self.hang_detector.detected():
            # Early return here to avoid waiting for hanging threads.
            # Since `on_detected` has sent the error message as response,
            # this worker will be asked to shutdown immediately.
            # Since the whole process will shutdown after this `shutdown` call,
            # All threads and memory pools will be freed properly.
            logger.error("Hang detected, shutting down immediately.")
            return
        self.worker_thread.join()
        if self.dist.pp_size > 1:
            self.executed_batch_queue.put(None)
            self.broadcast_sample_state_handler.join()
        self.worker_started = False
        # Release CUDA graphs before resource managers free their GPU memory.
        # Resource managers (e.g. SuffixAutomatonManager) allocate GPU workspace
        # that is referenced by raw pointers inside captured CUDA graphs.  If
        # the workspace is freed first (and returned to the driver via
        # empty_cache), the subsequent CUDA graph teardown can trigger a
        # device-wide cudaErrorIllegalAddress when the driver touches metadata
        # for the now-freed memory regions.
        for engine in (self.model_engine, self.draft_model_engine):
            if engine is not None and hasattr(engine, '_release_cuda_graphs'):
                engine._release_cuda_graphs()
        # Ensure graph destruction has fully completed on device before
        # resource managers start freeing GPU-backed workspaces.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for manager in self.resource_manager.resource_managers.values():
            if manager:
                manager.shutdown()
        del self.model_engine
        if self.draft_model_engine is not None:
            del self.draft_model_engine
        if self.virtual_memory_pools is not None:
            keys = list(self.virtual_memory_pools.keys())
            for key in keys:
                del self.virtual_memory_pools[key]
        # Stop the sampler's async worker, if it was used
        if (isinstance(self.sampler, AsyncWorkerMixin)
                and self.sampler.async_worker_enabled()):
            self.sampler.async_worker_stop()
        if self.dwdp_manager is not None:
            self.dwdp_manager.__exit__(None, None, None)
            self.dwdp_manager = None

    # REFACTOR: 1a (public API).
    def can_enqueue_requests(self) -> bool:
        """
        Indicates if the current process is allowed to enqueue requests
        """
        return self.executor_request_queue.can_enqueue_request()

    # REFACTOR: 1a (public API).
    def get_latest_iteration_stats(self):
        """
        Returns the per-iterations statistics computed since last call to this method.
        Contains at most iter_stats_max_iterations iterations.
        """
        if self.enable_iter_perf_stats == False:
            return []

        latest_stats = (IterationStats(), None)
        with self.stats_lock:
            latest_stats = self.stats
            self.stats = []
        return latest_stats

    # REFACTOR: 1a (public API).
    def get_latest_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if not kv_cache_manager or not self.enable_kv_cache_events:
            return []

        events = kv_cache_manager.get_latest_events(0)
        return events

    # REFACTOR: 1a (public API) -- main-thread blocking wait for shutdown.
    def wait_shutdown(self):
        self.shutdown_event.wait()

    # REFACTOR: 1a (public API) -- single-request enqueue variant.
    def enqueue_request(
            self,
            request: ExecutorRequest,
            query: Optional[List] = None,
            result_wait_queue: "Optional[ray.actor.ActorHandle]" = None) -> int:
        """
        Enqueue a new request, query is only used in `StarAttention`.
        """
        req_id = self.executor_request_queue.enqueue_request(request, query)
        if result_wait_queue is not None:
            with self.response_cv:
                self.result_wait_queues[req_id] = result_wait_queue
        return req_id

    # REFACTOR: 1a (public API).
    def set_gather_responses(self, gather_all_responses):
        self.gather_all_responses = gather_all_responses

    # REFACTOR: SPECIAL -- scheduler-direct shutdown gate. Read by the
    # scheduler iter (loop_control), not by any per-batch concern. New
    # design: inline into the scheduler iter body as
    # ``ctx.port.is_shutdown and ctx.svc.pool.is_drained() and
    # ctx.port.waiting_queue.empty()``.
    @property
    def should_stop_processing(self):
        return self.is_shutdown and len(self.active_requests) == 0 and \
            len(self.waiting_queue) == 0

    # REFACTOR: 2a (concern: profile) -- per-iter profiler tick CM. New
    # design: ``ProfileConcern.tick(ctx)`` method called from
    # ``batch_body`` at SCHEDULE_0.
    @contextmanager
    def _profiler(self):
        it = -1
        enabled = False
        start_time = None

        # These events are used to record the time of the previous batch.
        # We need two set of the start-end events to record the time through
        # a ping-pong way so that it works with overlap scheduler.
        start_event_1 = None
        end_event_1 = torch.cuda.Event(enable_timing=True)
        start_event_2 = None
        end_event_2 = torch.cuda.Event(enable_timing=True)
        prev_device_step_time = None

        torch_trace_path = os.environ.get(PROFILE_TRACE_ENV_VAR_NAME, None)
        if torch_trace_path is not None:
            # Append the rank so each rank writes to its own file. Without
            # this, TP/PP/DP > 1 runs have every rank calling
            # torch_profiler.export_chrome_trace() on the same path
            # concurrently, producing interleaved output that fails to
            # parse in Chrome tracing / Perfetto.
            trace_base, trace_ext = os.path.splitext(torch_trace_path)
            torch_trace_path = f"{trace_base}-rank-{self.global_rank}{trace_ext}"
        profile_start_stop = os.environ.get(PROFILE_START_STOP_ENV_VAR_NAME,
                                            None)
        enable_torch_trace = bool(torch_trace_path and profile_start_stop)
        if torch_trace_path and profile_start_stop is None:
            logger.warning(
                f"{PROFILE_START_STOP_ENV_VAR_NAME} environment variable "
                "needs to be set to enable the torch trace. Example to profile "
                f"iteration 10-20: export {PROFILE_START_STOP_ENV_VAR_NAME}=10-20"
            )

        if enable_torch_trace:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                torch.profiler.ProfilerActivity.XPU,
            ]
            torch_profiler = torch.profiler.profile(activities=activities,
                                                    record_shapes=True,
                                                    with_modules=True)

        log_ranks_str = os.environ.get(PROFILE_LOG_RANKS_ENV_VAR_NAME, "0")
        if log_ranks_str.strip().lower() == "all":
            log_all_ranks = True
            log_ranks = set()
        else:
            log_all_ranks = False
            log_ranks = {int(r) for r in log_ranks_str.split(",")}

        calibrator = get_calibrator()

        def profile_step():
            nonlocal it, enabled, start_time, start_event_1, end_event_1, start_event_2, end_event_2, prev_device_step_time
            calibrator.post_step(it)
            if it in self.profile_stop_iters and not self.is_warmup:
                assert enabled, "Inconsistent CUDA profiling state"
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(f"Profiling stopped at iteration {it}, "
                                f"trace saved to {torch_trace_path}")
                torch.cuda.cudart().cudaProfilerStop()
                calibrator.stop()
                enabled = False

            if start_time is not None and self.print_log and (
                    log_all_ranks or self.dist.rank in log_ranks):
                end_time = time.time()
                if it % 2 == 0:
                    end_event_1.record()
                    if start_event_2 is not None:
                        end_event_2.synchronize()
                        prev_device_step_time = start_event_2.elapsed_time(
                            end_event_2)
                else:
                    end_event_2.record()
                    if start_event_1 is not None:
                        end_event_1.synchronize()
                        prev_device_step_time = start_event_1.elapsed_time(
                            end_event_1)

                if prev_device_step_time is None:
                    prev_device_step_time = "N/A"  # Handle first iteration
                else:
                    prev_device_step_time = f"{prev_device_step_time}ms"
                host_step_time = (end_time - start_time) * 1000  # milliseconds
                formatted_timestamp = datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")
                logger.info(
                    f"iter = {self.iter_counter}, "
                    f"global_rank = {self.global_rank}, "
                    f"rank = {self.dist.rank}, "
                    f"currank_total_requests = {self.num_fetch_requests_cur_rank}/"
                    f"{self.num_fetch_requests}, "
                    f"host_step_time = {host_step_time}ms, "
                    f"prev_device_step_time = {prev_device_step_time}, "
                    f"timestamp = {formatted_timestamp}, "
                    f"num_scheduled_requests: {self.num_scheduled_requests}, "
                    f"states = {self.model_engine.iter_states}")

            it += 1

            if it in self.profile_start_iters and not self.is_warmup:
                assert not enabled, "Inconsistent CUDA profiling state"
                calibrator.start()
                torch.cuda.cudart().cudaProfilerStart()
                if enable_torch_trace:
                    torch_profiler.start()
                logger.info(f"Profiling started at iteration {it}.")
                enabled = True
            calibrator.pre_step(it)
            start_time = time.time()
            if it % 2 == 0:
                if start_event_1 is None:
                    start_event_1 = torch.cuda.Event(enable_timing=True)
                start_event_1.record()
            else:
                if start_event_2 is None:
                    start_event_2 = torch.cuda.Event(enable_timing=True)
                start_event_2.record()

        try:
            yield profile_step
        finally:
            if enabled:
                # Stop on early exit / exception
                if enable_torch_trace:
                    torch_profiler.stop()
                    torch_profiler.export_chrome_trace(torch_trace_path)
                    logger.info(f"Profiling stopped at iteration {it}, "
                                f"trace saved to {torch_trace_path}")
                torch.cuda.cudart().cudaProfilerStop()
                calibrator.stop()

    # REFACTOR: 2a (concern: iter_stats) -- SCHEDULE_0 init step.
    def _get_init_iter_stats(self, num_new_active_requests,
                             new_active_requests_queue_latency_ms):
        stats = IterationStats()
        stats.timestamp = datetime.datetime.now().strftime(
            "%m-%d-%Y %H:%M:%S.%f")

        stats.num_new_active_requests = num_new_active_requests
        stats.num_active_requests = len(self.active_requests)
        stats.new_active_requests_queue_latency_ms = new_active_requests_queue_latency_ms
        stats.inflight_batching_stats = InflightBatchingStats()
        # staticBatchingStats is not used in pytorch path
        stats.static_batching_stats = StaticBatchingStats()

        # Create specdec_stats if speculative decoding is enabled
        # Either via spec_resource_manager (two-model mode) or spec_config (one-model mode)
        spec_resource_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.SPEC_RESOURCE_MANAGER)
        has_spec_config = self.model_engine.spec_config is not None

        if spec_resource_manager is not None or has_spec_config:
            stats.specdec_stats = SpecDecodingStats()
            # Reset draft latency at the start of each iteration to prevent stale values
            # from previous iterations when speculation is disabled
            if self.drafter is not None and hasattr(self.drafter,
                                                    'last_draft_latency_ms'):
                self.drafter.last_draft_latency_ms = 0.0

        return stats

    # REFACTOR: 2a (concern: iter_stats) -- helper for ``_process_iter_stats``.
    def _populate_req_stats(
            self, finished_requests: List[LlmRequest],
            active_requests: List[LlmRequest],
            scheduled_requests: ScheduledRequests
    ) -> Optional[List[RequestStats]]:

        def get_req_stats(req: LlmRequest) -> RequestStats:
            req_stat = RequestStats()
            req_stat.id = req.request_id
            req_stat.context_prefill_position = req.context_current_position
            req_stat.num_generated_tokens = req.max_beam_num_tokens - req.orig_prompt_len
            req_stat.avg_num_decoded_tokens_per_iter = req.avg_decoded_tokens_per_iter
            req_stat.alloc_total_blocks_per_request = req.alloc_total_blocks
            req_stat.alloc_new_blocks_per_request = req.alloc_new_blocks
            req_stat.reused_blocks_per_request = req.reused_blocks
            req_stat.missed_blocks_per_request = req.missed_blocks
            req_stat.kv_cache_hit_rate_per_request = req.kv_cache_hit_rate
            req_stat.scheduled = req in scheduled_requests.context_requests or req in scheduled_requests.generation_requests
            if req.llm_request_type == LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY or req.llm_request_type == LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY:
                req_stat.dis_serving_stats = DisServingRequestStats()
                req_stat.dis_serving_stats.kv_cache_transfer_ms = req.kv_cache_transfer_time_ms
                req_stat.dis_serving_stats.kv_cache_size = req.kv_cache_size
            return req_stat

        def get_queued_req_stats(request_id: int) -> RequestStats:
            req_stat = RequestStats()
            req_stat.id = request_id
            req_stat.context_prefill_position = 0
            req_stat.num_generated_tokens = 0
            req_stat.avg_num_decoded_tokens_per_iter = 0
            req_stat.alloc_total_blocks_per_request = 0
            req_stat.alloc_new_blocks_per_request = 0
            req_stat.reused_blocks_per_request = 0
            req_stat.missed_blocks_per_request = 0
            req_stat.kv_cache_hit_rate_per_request = 0
            return req_stat

        req_stats = []
        for req in active_requests:
            req_stat = get_req_stats(req)
            req_stat.stage = req.stage
            req_stats.append(req_stat)

        for req in list(self.executor_request_queue.get_request_queue().queue):
            if isinstance(req, RequestQueueItem):
                req_stat = get_queued_req_stats(req.id)
                req_stat.stage = RequestStage.QUEUED
                req_stats.append(req_stat)

        for req in finished_requests:
            req_stat = get_req_stats(req)
            req_stat.stage = RequestStage.GENERATION_COMPLETE
            req_stats.append(req_stat)

        return req_stats

    # REFACTOR: 2a (concern: iter_stats) -- helper for ``_process_iter_stats``.
    def _update_iter_stats(self, stats, iter_latency_ms, num_completed_requests,
                           scheduled_batch, micro_batch_id) -> IterationStats:
        stats.iter_latency_ms = iter_latency_ms

        stats.num_queued_requests = self.executor_request_queue.get_request_queue_size(
        )
        stats.num_completed_requests = num_completed_requests
        stats.max_num_active_requests = self.max_num_active_requests

        end, total_gpu_memory = torch.cuda.mem_get_info()
        stats.gpu_mem_usage = total_gpu_memory - end
        stats.cpu_mem_usage = 0
        stats.pinned_mem_usage = 0

        stats.iter = self.iter_counter

        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if kv_cache_manager is not None:
            kv_stats = kv_cache_manager.get_kv_cache_stats()
            kv_stats_to_save = KvCacheStats()
            kv_stats_to_save.max_num_blocks = kv_stats.max_num_blocks
            kv_stats_to_save.free_num_blocks = kv_stats.free_num_blocks
            kv_stats_to_save.used_num_blocks = kv_stats.used_num_blocks
            kv_stats_to_save.tokens_per_block = kv_stats.tokens_per_block
            kv_stats_to_save.alloc_total_blocks = kv_stats.alloc_total_blocks
            kv_stats_to_save.alloc_new_blocks = kv_stats.alloc_new_blocks
            kv_stats_to_save.reused_blocks = kv_stats.reused_blocks
            kv_stats_to_save.missed_blocks = kv_stats.missed_blocks
            kv_stats_to_save.cache_hit_rate = kv_stats.cache_hit_rate
            stats.kv_cache_stats = kv_stats_to_save

            # Collect per-iteration stats (with deltas) at configured interval.
            # Between calls, C++ deltas accumulate so the reported values cover multiple iterations.
            # Guard: only fetch once per iter_counter to avoid draining deltas in PP multi-batch.
            if (self.iter_counter % self._kv_iter_stats_interval == 0 and
                    self._last_kv_iter_stats_fetch_iter != self.iter_counter):
                self._latest_kv_iter_stats = kv_cache_manager.get_iteration_stats(
                )
                self._last_kv_iter_stats_fetch_iter = self.iter_counter
            else:
                self._latest_kv_iter_stats = None

        stats.inflight_batching_stats.num_context_requests = scheduled_batch.num_context_requests
        stats.inflight_batching_stats.num_gen_requests = scheduled_batch.num_generation_requests
        stats.inflight_batching_stats.num_scheduled_requests = stats.inflight_batching_stats.num_context_requests + stats.inflight_batching_stats.num_gen_requests
        stats.inflight_batching_stats.num_paused_requests = len(
            scheduled_batch.paused_requests)
        stats.inflight_batching_stats.avg_num_decoded_tokens_per_iter = 0
        stats.inflight_batching_stats.micro_batch_id = micro_batch_id

        if stats.specdec_stats is not None:
            total_draft_tokens = 0
            total_accepted_tokens = 0
            num_requests_with_draft = 0

            # Aggregate stats from all generation requests
            for req in scheduled_batch.generation_requests:
                draft_len = getattr(req, 'num_draft_tokens', 0)
                py_draft_tokens = getattr(req, 'py_draft_tokens', None)
                py_num_accepted = getattr(req, 'py_num_accepted_draft_tokens',
                                          None)

                # Use py_draft_tokens length if num_draft_tokens is 0
                if draft_len == 0 and py_draft_tokens is not None:
                    # Count non-zero draft tokens
                    draft_len = sum(1 for t in py_draft_tokens if t != 0)

                if draft_len > 0:
                    total_draft_tokens += draft_len
                    accepted_tokens = py_num_accepted if py_num_accepted is not None else 0
                    total_accepted_tokens += accepted_tokens
                    num_requests_with_draft += 1

            stats.specdec_stats.num_draft_tokens = total_draft_tokens
            stats.specdec_stats.num_accepted_tokens = total_accepted_tokens
            stats.specdec_stats.num_requests_with_draft_tokens = num_requests_with_draft

            # Calculate acceptance length: average tokens produced per step for requests with draft tokens
            if num_requests_with_draft > 0:
                # acceptance_length = (total_accepted_tokens + num_requests_with_draft) / num_requests_with_draft
                # Each request produces 1 target token + accepted draft tokens per iteration
                stats.specdec_stats.acceptance_length = float(
                    total_accepted_tokens +
                    num_requests_with_draft) / float(num_requests_with_draft)
            else:
                stats.specdec_stats.acceptance_length = 0.0

            # Get draft latency from drafter if available (only for two-model mode)
            # Only use draft latency if there were actually draft tokens in this iteration
            draft_latency_ms = 0.0
            if total_draft_tokens > 0 and self.drafter is not None and hasattr(
                    self.drafter, 'last_draft_latency_ms'):
                draft_latency_ms = getattr(self.drafter,
                                           'last_draft_latency_ms', 0.0)

            stats.specdec_stats.iter_latency_ms = draft_latency_ms

            # Calculate draft overhead
            stats.specdec_stats.draft_overhead = 0.0 if iter_latency_ms <= 0.0 else float(
                draft_latency_ms) / float(iter_latency_ms)

        # Extra per-iteration request-aggregate counters attached to
        # inflight_batching_stats. These complement the existing
        # num_context_requests / num_gen_requests / num_ctx_tokens /
        # num_paused_requests members with token-weighted counts and
        # queue/paused KV accounting.

        # Tokens read from prior state (prefix-cache hits and
        # previously-chunked tokens) summed across scheduled context
        # requests; complements num_ctx_tokens (tokens computed this
        # iteration). Read from py_last_context_chunk, a Python-side
        # cache set by _update_request_states before state mutation — it
        # stays valid after the request transitions to
        # GENERATION_IN_PROGRESS, unlike the C++ getContextChunkSize() /
        # getContextCurrentPosition() accessors that would raise
        # RuntimeError on a mutated request.
        num_ctx_kv_tokens = 0
        for req in scheduled_batch.context_requests:
            if getattr(req, "is_attention_dp_dummy", False):
                continue
            last_chunk = getattr(req, "py_last_context_chunk", None)
            if last_chunk is not None and last_chunk[0] is not None:
                start, _end = last_chunk
                num_ctx_kv_tokens += start
            else:
                try:
                    num_ctx_kv_tokens += \
                        req.context_current_position
                except RuntimeError:
                    pass

        # Total KV context length (prompt + tokens generated so far)
        # summed across scheduled generation requests.
        num_gen_kv_tokens = 0
        for req in scheduled_batch.generation_requests:
            if getattr(req, "is_attention_dp_dummy", False):
                continue
            try:
                num_gen_kv_tokens += req.get_num_tokens(0)
            except RuntimeError:
                pass

        # Normal requests waiting in the executor_request_queue that have
        # never been scheduled. Excludes non-normal control items
        # (shutdown/cancel) and items with a missing payload. Each queued
        # item is a RequestQueueItem wrapping an ExecutorRequest
        # (tle::Request). Requests are routed by request_type:
        #   - CONTEXT_AND_GENERATION (default) and CONTEXT_ONLY
        #     (disagg-prefill side) -> queued-context counters.
        #   - GENERATION_ONLY (disagg-decode side, awaiting KV transfer
        #     before they can start decoding) -> queued-gen counters.
        # On a non-disagg engine all items land in the context counters;
        # on a disagg-decode engine all items land in the gen counters.
        num_queued_context_requests = 0
        num_queued_ctx_tokens = 0
        num_queued_gen_requests = 0
        num_queued_gen_kv_tokens = 0
        for item in list(self.executor_request_queue.get_request_queue().queue):
            if not item.is_normal_request:
                continue
            if item.request is None:
                continue
            try:
                token_count = len(item.request.input_token_ids)
            except (AttributeError, TypeError) as e:
                # Unusual request shape with no usable token payload;
                # exclude from all queued counters so downstream consumers
                # see consistent per-request averages. Not expected on the
                # current API (ExecutorRequest construction requires a
                # non-empty input_token_ids), logged so future API drift
                # surfaces instead of being silently dropped.
                logger.warning(f"Excluding queued item {item.id} from queued "
                               f"counters: input_token_ids not readable "
                               f"({type(e).__name__})")
                continue
            if item.request.request_type == RequestType.REQUEST_TYPE_GENERATION_ONLY:
                num_queued_gen_requests += 1
                num_queued_gen_kv_tokens += token_count
            else:
                num_queued_context_requests += 1
                num_queued_ctx_tokens += token_count

        # Total KV context length summed across paused (preempted-decode)
        # requests — were decoding but got evicted back to the waiting
        # pool for this iteration.
        num_paused_kv_tokens = 0
        for req in scheduled_batch.paused_requests:
            if getattr(req, "is_attention_dp_dummy", False):
                continue
            try:
                num_paused_kv_tokens += req.get_num_tokens(0)
            except RuntimeError:
                pass

        stats.inflight_batching_stats.num_ctx_kv_tokens = num_ctx_kv_tokens
        stats.inflight_batching_stats.num_gen_kv_tokens = num_gen_kv_tokens
        stats.inflight_batching_stats.num_queued_context_requests = num_queued_context_requests
        stats.inflight_batching_stats.num_queued_ctx_tokens = num_queued_ctx_tokens
        stats.inflight_batching_stats.num_queued_gen_requests = num_queued_gen_requests
        stats.inflight_batching_stats.num_queued_gen_kv_tokens = num_queued_gen_kv_tokens
        stats.inflight_batching_stats.num_paused_kv_tokens = num_paused_kv_tokens

        return stats

    # REFACTOR: 2a (concern: iter_stats) -- helper for ``_process_iter_stats``.
    def _append_iter_stats(self,
                           stats: IterationStats,
                           req_stats: Optional[List[RequestStats]] = None):

        with self.stats_lock:
            if len(self.stats) > self.max_stats_len:
                self.stats.pop(0)
            self.stats.append((stats, req_stats, self._latest_kv_iter_stats))

    # REFACTOR: 2a (concern: iter_stats) -- FINALIZE_9 step.
    def _process_iter_stats(
        self,
        finished_requests: list[LlmRequest],
        active_requests: List[LlmRequest],
        batch_state: BatchState,
        micro_batch_id: int = 0,
    ):
        iter_end_time = time.time()
        iter_latency_ms = (iter_end_time - batch_state.iter_start_time) * 1e3
        if batch_state.iter_stats is None:
            return

        req_stats = self._populate_req_stats(
            finished_requests, active_requests,
            batch_state.scheduled_requests) if (
                self.enable_iter_req_stats
                and self.enable_iter_perf_stats) else None

        self._append_iter_stats(
            self._update_iter_stats(batch_state.iter_stats, iter_latency_ms,
                                    len(finished_requests),
                                    batch_state.scheduled_requests,
                                    micro_batch_id), req_stats)

    # REFACTOR: 1b (legacy loop runtime) -- post-loop teardown for the
    # legacy loops; called from ``_event_loop_wrapper``'s finally.
    def _executor_loop_cleanup(self):

        for i in range(self.num_micro_batches):
            self.wait_on_pp_send_handles(self.send_handles, i)
            self.wait_on_pp_send_handles(self.send_schedule_handles, i)
            self.wait_on_pp_send_handles(self.send_expected_batch_num_handles,
                                         i)

        with self.response_cv:
            self.is_shutdown = True
            self.response_cv.notify_all()
        self.shutdown_event.set()

    # REFACTOR: 2a (concern: schedule) -- PP variant of the schedule
    # concern's SCHEDULE_0 work. Implements HC8 (PP schedule chain): rk0
    # schedules + serializes; other ranks recv + isend along the PP
    # forward chain, then deserialize.
    def _pp_schedule_and_propagate(self, microbatch_id: int):
        """The first PP rank schedules the requests and propagates the result to all other PP ranks."""

        # For TP/CP cases, the first rank schedules the requests.
        # For DP cases, the first PP rank schedules the requests.
        scheduled_batch = None
        serializable_schedule = None
        is_dp_broadcast = self.dist.tp_size > 1 and self.enable_attention_dp
        if self.dist.rank == 0 or (self.dist.is_first_pp_rank
                                   and is_dp_broadcast):
            scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
            )
            serializable_schedule = SerializableSchedulerOutput.from_scheduler_result(
                scheduled_batch, fitting_disagg_gen_init_requests,
                num_fitting_reqs)

        # Broadcast within first tp+cp group before send/recv chain to other tp+cp groups
        if self.dist.is_first_pp_rank:
            if self.dist.tp_size > 1 and not self.enable_attention_dp:
                with nvtx_range("tp_broadcast_schedule"):
                    serializable_schedule = self.dist.tp_broadcast(
                        serializable_schedule, root=0)
            if self.dist.cp_size > 1:
                with nvtx_range("cp_broadcast_schedule"):
                    serializable_schedule = self.dist.cp_broadcast(
                        serializable_schedule, root=0)

        # Other ranks receive the schedule result from the previous PP rank.
        if not self.dist.is_first_pp_rank:
            with nvtx_range("recv_schedule_from_prev_pp"):
                serializable_schedule = self.dist.recv_object(
                    self.dist.prev_pp_rank, PPCommTag.SCHEDULE_RESULT)

        # Propagate the schedule result to the next PP rank except the last PP rank.
        if not self.dist.is_last_pp_rank:
            self.wait_on_pp_send_handles(self.send_schedule_handles,
                                         microbatch_id)
            with nvtx_range("send_schedule_to_next_pp"):
                self.send_schedule_handles[
                    microbatch_id] = self.dist.isend_object(
                        serializable_schedule, self.dist.next_pp_rank,
                        PPCommTag.SCHEDULE_RESULT)

        if scheduled_batch is None:
            scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = serializable_schedule.to_scheduler_result(
                self.active_requests)
        return scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs

    # REFACTOR: 2a (concern: schedule) -- PP variant of the schedule
    # concern; non-rk0 retry loop on KV-cache exhaustion.
    def _pp_retry_until_can_schedule(self, scheduled_batch):
        """
        If current rank cannot run the scheduled batch, it will retry following steps until it has enough KV cache resources or reach maximum retry count:
        1. Wait for cache transceiver to finish at least one cache transmission.
        2. Terminate requests that have finished context cache transmission.
        3. Check if current rank has enough KV cache resources to run the scheduled batch.
        """
        scheduled_batch_requests = scheduled_batch.all_requests()
        if self.scheduler.can_schedule(scheduled_batch_requests):
            return

        logger.warning(
            "Cannot run first PP's schedule result due to limited KV cache resources. This may cause bubbles in the PP pipeline. Please consider increasing the KV cache size by setting `free_gpu_memory_fraction` to a larger value."
        )
        if self.kv_cache_transceiver is None:
            raise RuntimeError(
                "KV cache transceiver is not enabled, but current rank cannot run first PP's schedule result due to limited KV cache resources. This is not expected."
            )
        if not self.async_transfer_manager.has_any_inflight_requests():
            raise RuntimeError(
                "No context cache transmission is in progress, but current rank cannot run first PP's schedule result due to limited KV cache resources. This is not expected."
            )
        if self.enable_kv_cache_reuse and self._disagg_pp_termination_handler is not None:
            raise RuntimeError(
                "Cannot terminate requests in cache transmission and release their KV cache resources when block reuse is enabled. Please consider increasing the KV cache size."
            )

        for retry_count in range(self.pp_scheduler_max_retry_count):
            if self.scheduler.can_schedule(scheduled_batch_requests):
                break
            logger.debug(
                f"Retrying to run first PP's schedule result ({retry_count + 1}/{self.pp_scheduler_max_retry_count})"
            )

            # Let cache transceiver finish at least one cache transmission and release requests' KV cache resources
            self._check_disagg_ctx_cache_transfer_status(1)
            self._check_kv_transfer_timeout()
        else:
            raise RuntimeError(
                f"Reach maximum PP retry count ({self.pp_scheduler_max_retry_count}) but still cannot run first PP's schedule result. Please consider increasing the KV cache size by setting `free_gpu_memory_fraction` to a larger value. Or you can set `TLLM_PP_SCHEDULER_MAX_RETRY_COUNT` to a larger value to allow more retries."
            )

    # REFACTOR: 1b (legacy loop runtime) -- legacy PP executor loop; will
    # be replaced by the new coroutine SCHEDULER's PP variant.
    def _executor_loop_pp(self):
        # ===================================================================
        # 3-LAYER COROUTINE REFACTOR ROADMAP -- PIPELINE-PARALLEL LOOP
        # See `coroutines.py` + `batch_storage.py` for the target runtime.
        # ===================================================================
        # Most complex of the three loops. Each batch lives
        # `n = pp_size = num_micro_batches` iters in iter-INDEX on every
        # rank (~`2n` wall-time steps end-to-end across the cluster, with
        # implicit 1F1B stagger from NCCL p2p inside `_forward_step`).
        # `n - 1` batches alive on every rank simultaneously. Three
        # off-main-thread lanes the scheduler keeps moving:
        #   - GPU streams (`execution_stream` for forward + transfer,
        #     `sample_stream` for the last rank's sampling).
        #   - Slot ring `self.micro_batches[N mod n]` parking suspended
        #     batches between forward and bcast handoff.
        #   - Background `_broadcast_sample_state_loop` thread doing the
        #     reverse ring `(n-1) -> 0 -> 1 -> ... -> (n-2)` of decoded
        #     tokens. In the refactor, this becomes a long-running concern
        #     coroutine that spans `(n-1)` scheduler iterations.
        #
        # Batch perspective -- lifecycle of ONE batch on each rank
        # (in iter-INDEX local to that rank; span = `n` iters)
        # -------------------------------------------------------------------
        #   Iter N (creation):
        #     P_SCHEDULE : `_pp_schedule_and_propagate` (rk0 decides;
        #                  non-rk0 recv+isend along PP forward chain)
        #     P_FORWARD  : `_forward_step` (this rank's layer slice;
        #                  `pp_recv` <- prev, layers, `pp_send` -> next)
        #     P_SAMPLE   : LAST RANK ONLY -- `_sample_async` queued on
        #                  `sample_stream` if `pp_multi_stream_sample` else
        #                  default stream. (batch_state stored in
        #                  `self.micro_batches[N mod n]`.)
        #
        #   Iter N+1 (sample sync; last rank's bcast handoff):
        #     P_SYNC_EVT : `previous_batch.sample_state.sampler_event`
        #                  `.synchronize()`
        #                    - last rank: real GPU->CPU copy of decoded
        #                      tokens (host data feeds the ring isend)
        #                    - other ranks: backpressure-only sync of a
        #                      placeholder event
        #     P_HANDOFF  : LAST RANK ONLY -- `executed_batch_queue.put`
        #                  (offset `-1`); bcast thread `isend_object`s
        #                  tokens -> rk0. (For other ranks, P_HANDOFF
        #                  happens at iter `N+(n-1)` instead.)
        #
        #   Iter N+2 .. N+(n-2) (parked):
        #     Batch sits in its slot (intermediate ranks) or in
        #     `executed_batch_response_queue` (last rank). bcast thread on
        #     rk0/.../(n-2) walks the reverse ring in the background.
        #
        #   Iter N+(n-1) (retirement on every rank):
        #     P_HANDOFF  : NON-LAST RANKS -- `executed_batch_queue.put`
        #                  (offset `-(n-1)`) + bcast thread does its single
        #                  ring hop (recv <- prev, isend -> next, or
        #                  terminus on rk(n-2))
        #     P_VOTE     : rk0 alone fetches its response queue and sets
        #                  `executed_batch_num`; the count propagates along
        #                  the PP forward chain so all ranks retire the
        #                  same number of batches.
        #     P_RETIRE   : `_handle_executed_batch(mb_N)` --
        #                  `_update_requests` (apply tokens),
        #                  `_send_kv_async`, `_handle_responses`
        #                  (rk0 -> user), `update_resources`,
        #                  `_remove_inflight_ids`.
        # User-visible first-token latency = `n` wall-time steps (rk0's
        # P_RETIRE). End-to-end batch lifetime across the cluster =
        # `2n` wall-time steps (rk(n-1)'s P_RETIRE).
        #
        # Scheduler perspective -- per-iter work (3 batches per iter)
        # -------------------------------------------------------------------
        # On every rank, identify each batch by `k = iters since forward`:
        #   k = 0    : `mb_t`            -- current iter's batch
        #   k = 1    : `mb_(t-1)`        -- prev iter's batch (sync, plus
        #                                   last-rank handoff)
        #   k = n-1  : `mb_(t-(n-1))`    -- oldest in flight (handoff +
        #                                   vote + retire)
        # Non-last ranks: cleanly grouped k=0 -> k=1 -> k=n-1.
        # Last rank: one k=1 wait (`finish_sample_event.wait()`, fencing
        # `sample_stream`) is interleaved into the k=0 forward block.
        # Pseudo-refactored body:
        #
        #     async def scheduler_iter():
        #         await step(handle_curr,   through=P_SAMPLE)    # k=0
        #         await step(handle_prev,   through=P_HANDOFF)   # k=1
        #         await step(handle_oldest, through=P_RETIRE)    # k=n-1
        #         ring.append(handle_curr); handle_oldest = ring.popleft()
        #
        # The slot ring + bcast thread + sample_stream all encode lag the
        # refactor must replace with phased `BatchStorage` fields and a
        # ring-broadcast concern coroutine that spans `(n-1)` scheduler
        # iters.
        #
        # Thread-elimination policy (shared with all `_executor_loop*` variants)
        # -------------------------------------------------------------------
        # The refactor removes application-owned auxiliary threads. Their
        # work moves onto the main loop coroutine via a uniform
        # "submit -> opportunistic probe -> wait at deadline" pattern:
        #   submit : non-blocking primitive returning a handle (e.g.,
        #            `MPI.Comm.Irecv` / `Isend`, `cudaEventRecord`,
        #            KV transceiver `start_transfer`).
        #   probe  : non-blocking query at iter top to drive library
        #            progress (`MPI.Request.Test`, `cudaEventQuery`,
        #            `ucp_worker_progress`). Cheap; lets a concern
        #            coroutine handle results early when ready.
        #   wait   : the coroutine block-waits at the deadline phase
        #            (`MPI.Request.Wait`, `cudaEventSynchronize`,
        #            `end_transfer`).
        # Exception: operations that are fundamentally sync-blocking with
        # no async API (rare) are wrapped in one shared
        # `ThreadPoolExecutor` that re-shapes them into submit+wait. From
        # the coroutine layer they look identical to native async ops; the
        # thread pool is an implementation detail of those concerns.
        # Library-internal threads (MPI / NIXL / UCX / NCCL progress) are
        # untouched -- we only eliminate threads we own.
        #
        # Threads removed under this policy on this loop variant:
        #   * `broadcast_sample_state_handler` (the sole PP-only
        #     application thread): the entire reverse ring loop collapses
        #     into a single ring-broadcast concern coroutine on the main
        #     thread. New per-slot `recv_handles[N mod n]` complement the
        #     existing `send_handles` ring. The pattern:
        #       - submit `Irecv` at iter N (right after `_forward_step`
        #         returns on every non-source rank) -- earliest legal
        #         phase to know mb_N will need a receive.
        #       - probe via `MPI.Request.Test` at iter top to drive MPI
        #         progress without an idle thread.
        #       - wait at iter N+1 on the last rank (just before the
        #         host-side `isend_object` of the freshly-sampled tokens)
        #         and at iter N+(n-1) on the other ranks (P_RETIRE) --
        #         the same deadlines the bcast thread enforces today.
        #     Consequences:
        #       - `mpi_comm().Dup()` (and the deadlock workaround it
        #         exists for) goes away; all MPI now runs single-threaded
        #         on one comm.
        #       - `executed_batch_queue` and
        #         `executed_batch_response_queue` go away (the slot ring
        #         plus its handle ring is enough state).
        #       - The `TLLM_PP_ASYNC_BROADCAST_SAMPLE_STATE` env var
        #         (which exists only to disable bcast-thread asynchrony
        #         for deterministic tests) retires too -- the runtime is
        #         deterministic by construction.
        #   * Sampler `_async_worker`: same treatment as in the non-PP
        #     loops -- `cudaEventRecord` (submit at P_SAMPLE) paired with
        #     `cudaEventSynchronize` (wait at P_SYNC_EVT, the iter N+1
        #     sync point) replaces the thread pool. On the last rank the
        #     wait gates the host-side ring isend; on other ranks it is
        #     the placeholder backpressure sync.
        # ===================================================================
        # ===================================================================
        # CONCERN / PHASE / HALF-CONCERN ANNOTATIONS (round 1: comments only)
        # -------------------------------------------------------------------
        # Concerns: same set as the plain / overlap loops, minus
        # `kv_connector` (PP loop currently has no kv_connector hooks)
        # and `dwdp` / explicit `perf_metric` GPU-event blocks.
        #
        # Per-batch phase order (each batch lives `n = pp_size` iters in
        # iter-INDEX local to this rank):
        #   P_SCHEDULE -> P_FORWARD -> P_SAMPLE -> P_STATE_UPD
        #     -> P_SYNC_EVT -> P_HANDOFF -> [parked iters]
        #     -> P_VOTE -> P_RETIRE
        # Phase placement varies by rank because of the implicit 1F1B
        # NCCL stagger in `_forward_step` (when rk0 is at iter N, rk1 is
        # at iter N-1, ... — a single physical batch traverses the
        # cluster in (n-1) wall-time steps after rk0 created it):
        #   Last rank   : P_SCHEDULE/FORWARD/SAMPLE/STATE_UPD at iter N,
        #                 P_SYNC_EVT/HANDOFF at iter N+1,
        #                 P_VOTE/RETIRE at iter N+(n-1).
        #   Other ranks : P_SCHEDULE/FORWARD at iter N (P_SAMPLE collapses
        #                 to a placeholder sampler_event in
        #                 `_forward_step_inter_pp`), P_SYNC_EVT at iter N+1,
        #                 P_HANDOFF at iter N+(n-1), P_VOTE/RETIRE same iter.
        # Iter N+2 .. N+(n-2) the batch is *parked* in slot ring storage
        # (intermediate ranks: `self.micro_batches[N mod n]`; last rank:
        # already in `executed_batch_response_queue` after the bcast hop).
        #
        # Scheduler iter pattern (3 step calls + ring rotation):
        #     async def scheduler_iter():
        #         await step(handle_curr,   through=P_SAMPLE)    # k=0
        #         await step(handle_prev,   through=P_HANDOFF)   # k=1
        #         await step(handle_oldest, through=P_RETIRE)    # k=n-1
        #         ring.append(handle_curr); handle_oldest = ring.popleft()
        #
        # User-visible first-token latency = `n` wall-time steps (rk0's
        # P_RETIRE on the very first batch). End-to-end batch lifetime
        # across the cluster = `2n` wall-time steps (rk(n-1)'s P_RETIRE).
        #
        # "Half concerns" — two flavors in the PP loop:
        # =================================================================
        # (a) SAME-RANK CROSS-ITER half concerns. The scheduler stashes
        # one batch's per-batch data and consumes it in a later iter on
        # the *same* rank. These are the slot ring + handle rings:
        #
        #   HC1 SLOT RING (per rank): mb_t.{scheduled_requests, sample_state,
        #         iter_stats, iter_start_time, microbatch_id} (P_STATE_UPD)
        #         -> mb_t at later iter's HANDOFF (P_HANDOFF).
        #     produce: `self.micro_batches[microbatch_id] = batch_state`
        #              at iter N (P_STATE_UPD end), or `... = None` when
        #              the batch was skipped.
        #     consume: `executed_batch = self.micro_batches[(microbatch_id
        #              + offset) % n]` at iter N+|offset| where
        #              offset=-1 (last rank, k=1) or 1-n (other ranks, k=n-1).
        #     The slot ring is what implements "the batch is parked for
        #     n-2 iters" — a coroutine refactor would replace the slot
        #     by a held `Batch` handle in a deque on the scheduler.
        #
        #   HC2 PREV (last rank only, cross-iter same rank):
        #     mb_t.sample_state.sampler_event (P_SAMPLE)
        #         -> next iter's P_SYNC_EVT for mb_t.
        #     produce: `self.previous_batch = batch_state` at iter N.
        #     consume: `previous_batch = self.previous_batch;
        #              previous_batch.sample_state.sampler_event.synchronize()`
        #              at iter N+1.
        #     The conditional "(is_last_pp_rank or can_queue) and previous_batch
        #     is not None" is the same-iter side: non-last ranks delay
        #     this sync if `not can_queue`, deferring it to a later iter.
        #
        #   HC3 SAMPLE_STREAM (last rank, last-rank multi-stream sampling):
        #     mb_t.finish_sample_event (recorded after _sample_async on
        #     sample_stream) -> next iter's mb_(t+1) sampling.
        #     produce: `self.finish_sample_event.record()` (this iter,
        #              after _sample_async on sample_stream).
        #     consume: `self.finish_sample_event.wait()` (next iter,
        #              before next _sample_async on sample_stream).
        #     Backpressures the sample_stream so two iters of sampling
        #     don't overlap on the same stream resources.
        #
        #   HC4 BCAST QUEUE (per rank, but pairs with the bcast thread's
        #   cross-rank ring HC10 below): mb_t put into bcast queue at
        #   iter N+offset -> retrieved from response queue at the iter
        #   when its ring traversal completes.
        #     produce: `self.executed_batch_queue.put(executed_batch)`.
        #     consume: `self.executed_batch_response_queue.get()` (drained
        #              by `fetch_executed_batches` on rk0 and by
        #              `handle_executed_batches` elsewhere).
        #     In the refactored runtime, this becomes a long-running
        #     "ring broadcast" concern coroutine spanning (n-1) scheduler
        #     iters.
        #
        #   HC5 SEND HANDLES RING (per rank): isend handle for slot N
        #   produced now, waited at the next iter that reuses slot N.
        #     produce: `self.send_handles[mid] = isend_object(...)` (in
        #              bcast thread / rank-broadcast helpers).
        #     consume: `self.wait_on_pp_send_handles(self.send_handles, mid)`
        #              before the next isend into the same slot.
        #     Same pattern applies to `send_schedule_handles` (HC5a) and
        #     `send_expected_batch_num_handles` (HC5b).
        #
        # (b) CROSS-RANK transfers — NOT scheduler-level when intra-batch.
        # A "batch" here means all logic dealing with the *same set of
        # requests* on all ranks. A cross-rank transfer that carries one
        # batch's per-batch data (a request set's activations, schedule
        # decision, or sample tokens) is *intra-batch*: the producer
        # rank and the consumer rank are both inside the SAME logical
        # batch concern, just running its body on different physical
        # ranks. Such transfers belong inside a *distributed concern
        # coroutine* that runs on every rank — the cross-rank send/recv
        # is the concern's own internal coordination, not a scheduler
        # bridge between separate batches.
        #
        # The intra-batch distributed concerns identified:
        #   HC7 PP FORWARD CHAIN — internal to the `forward` distributed
        #     concern (hidden inside `_forward_step`'s NCCL p2p). Each
        #     rank's `forward` body pp_recv <- prev, runs its layer
        #     slice, pp_send -> next. This is what *creates* the 1F1B
        #     stagger that makes the per-rank iter clocks differ.
        #   HC8 PP SCHEDULE CHAIN — internal to the `schedule` distributed
        #     concern (`_pp_schedule_and_propagate`). rk0's body runs
        #     the scheduler and serializes; other ranks' bodies recv
        #     from prev and isend to next, then deserialize the same
        #     decision. The same request set ends up scheduled on every
        #     rank.
        #   HC10 SAMPLE-STATE REVERSE RING — internal to a long-running
        #     `ring_broadcast_sample` distributed concern (currently
        #     `_ring_broadcast_sample_state` running on the bcast
        #     thread `_broadcast_sample_state_loop`). Ring traverses
        #     (n-1) -> 0 -> 1 -> ... -> (n-2) carrying the *same batch's*
        #     decoded tokens to all ranks. In the coroutine refactor
        #     this becomes a per-batch coroutine that spans (n-1)
        #     scheduler iters; the docstring above explicitly calls
        #     this out as the replacement for the bcast thread.
        #
        # The cross-rank transfers that are NOT intra-batch — these
        # belong to NEITHER the per-batch coroutines NOR a distributed
        # concern coroutine. They are pure per-iter scheduler bookkeeping
        # that happens to need cross-rank coordination, so in the
        # refactored model they are *executed directly by the scheduler*
        # (plain Python code between/around `step` calls — no `Batch`
        # handle, no `enter_phase`, no `BatchStorage`):
        #
        #   HC9 EXECUTED-BATCH-NUM CHAIN: rk0 votes how many batches
        #     retire this iter; count propagates rk0 -> rk(n-1) via
        #     the PP forward chain. The count isn't tied to one
        #     request set — multiple retiring batches at this iter
        #     share the vote. Refactored placement: a `dist.recv_object
        #     / isend_object` chain inlined in the scheduler iter,
        #     between the per-batch step calls.
        #
        #   PP-only termination consensus
        #     (`_disagg_pp_termination_handler.terminate_pending_requests`):
        #     ballot of pending request terminations propagated across
        #     PP ranks every iter so KV-reuse paths free a request
        #     symmetrically. Same shape as HC9 — payload is not tied
        #     to one request set. Refactored placement: also scheduler-
        #     direct code, sits in the scheduler iter tail.
        #
        # Anything else (HC7, HC8, HC10, the per-batch HC2/HC3, etc.) is
        # owned by some batch's coroutine or a distributed concern; only
        # the two above are pure scheduler-direct work.
        #
        # The same-rank cross-iter half concerns in (a) above are also
        # NOT scheduler-level when viewed in the refactored model — most
        # become internal state of either a per-batch coroutine (HC2 /
        # HC3 belong to `sample`'s cross-iter flow) or a distributed
        # concern coroutine (HC4 is the local end of HC10's ring; HC5 /
        # HC5a / HC5b are isend-handle accounting inside HC10 / HC8 /
        # HC9 respectively). Only HC1 SLOT RING reflects scheduler
        # bookkeeping proper — and even that collapses to "scheduler
        # holds a deque of Batch handles" in the refactor; the slot is
        # an artifact of the current `BatchStatePP`-record style.
        #
        # No reorders required: the existing line order is a valid
        # topological sort of the per-rank concern DAG.
        # ===================================================================
        logger.debug(f"Starting executor loop for pp_rank {self.dist.pp_rank}")
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        microbatch_id = 0
        with self._profiler() as profile_step, self.hang_detector:
            iter_start_time = time.time()
            iter_stats = None
            while True:
                # =========================================================
                # ===== STEP (1) : curr P_SCHEDULE -> P_SAMPLE ============
                # =========================================================
                # Prepares mb_t (k=0) on this rank. On the last rank this
                # also samples; on other ranks `_forward_step_inter_pp`
                # produces a placeholder sampler_event so HC2's sync at
                # next iter has something to synchronize on.

                # ===== curr PHASE: P_SCHEDULE =====

                # Concern: hang
                # Task: progress watchdog tick.
                self.hang_detector.checkpoint()
                # Concern: profile
                # Task: drive torch / CUDA-event profiler state machine.
                profile_step()
                # Concern: iter_stats (curr)
                # Task: stamp iter_start_time. Read at curr's P_RETIRE
                #       inside `_handle_executed_batch` ->
                #       `_process_iter_stats`. Field carried through the
                #       n-iter lifecycle via the slot ring (HC1).
                # Consume: -
                # Produce: iter_start_time (curr field)
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                # Concern: schedule (request intake)
                # Task: dequeue + validate + activate new requests.
                # Consume: -
                # Produce: new_requests; mutates self.active_requests
                # Fetch new requests from request queue
                new_requests = self._fetch_and_activate_new_requests()
                # Concern: loop_control
                # Task: shutdown gate. NOTE: PP loop breaks BEFORE the
                #       `_handle_control_request` / disagg probes / iter
                #       stats init, unlike the plain loop, because the
                #       PP loop also has the `Stage 5` post-shutdown
                #       drain below.
                if self.should_stop_processing:
                    break

                # Concern: control
                # Task: pause loop if a control request is pending.
                self._handle_control_request()

                if self.kv_cache_transceiver:
                    # Concern: disagg (probes — same as plain loop's
                    #          `_prepare_and_schedule_batch` opening)
                    # Task: opportunistic non-blocking probes that drive
                    #       the disagg KV-transceiver state machine.
                    # Consume: new_requests
                    # Produce: per-request state transitions (out-of-band)
                    self._check_disagg_ctx_schedulable_status(new_requests)
                    self._check_disagg_gen_transfer_status()

                # Concern: iter_stats (init)
                # Task: allocate iter_stats record + queue-latency snapshot.
                # Consume: -
                # Produce: iter_stats (curr field)
                if self.enable_iter_perf_stats:
                    iter_stats = self._get_init_iter_stats(
                        len(new_requests),
                        self._get_new_active_requests_queue_latency())

                # Concern: schedule (ADP dummy padding)
                # Task: pad with a dummy gen request when ADP rank is empty.
                # Consume: -
                # Produce: self.active_requests
                self._pad_attention_dp_dummy_request()

                # Concern: schedule (distributed — HC8 PP SCHEDULE CHAIN
                #          is its internal cross-rank coordination, not
                #          a scheduler half concern)
                # Task: rk0's body runs the scheduler and serializes;
                #       non-rk0 bodies recv from prev PP rank via
                #       recv_object and isend to next (HC5a SEND HANDLES
                #       RING is the schedule concern's own isend
                #       accounting). All ranks deserialize the same
                #       decision so the same request set ends up
                #       scheduled on every rank — that is what makes
                #       this *intra-batch*.
                # Consume: cross-rank: prev rank's serialized schedule
                # Produce: scheduled_batch, fitting_disagg_gen_init_requests,
                #          num_fitting_reqs (curr fields); cross-rank:
                #          isend to next rank
                # Stage 0: first PP rank schedules requests and propagates the result to all other PP ranks.
                scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._pp_schedule_and_propagate(
                    microbatch_id)
                if self.dist.rank != 0:
                    # Concern: schedule (PP retry / local scheduler replay)
                    # Task: non-rk0 ranks may not have enough KV; retry
                    #       by waiting for at-least-one ctx KV transfer
                    #       (this is a `disagg` concern call inside),
                    #       then replay the scheduler locally so request
                    #       state matches what rk0 saw.
                    # Consume: scheduled_batch (PP-propagated)
                    # Produce: scheduler-side request state mutations
                    # Retry until current rank can run first PP's schedule result.
                    self._pp_retry_until_can_schedule(scheduled_batch)
                    # Run scheduler locally because scheduler may change llm requests' state.
                    self.scheduler.schedule_request(self.active_requests,
                                                    self.inflight_req_ids)

                # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
                if self.kv_cache_transceiver:
                    # Concern: resource (+ disagg) — same as plain loop's
                    #          `_prepare_and_schedule_batch`.
                    # Task: prepare KV / resource state for newly fitting
                    #       disagg-gen-init requests; submit async KV
                    #       recv. Then run disagg back-pressure probes.
                    # Consume: fitting_disagg_gen_init_requests,
                    #          scheduled_batch, num_fitting_reqs
                    # Produce: KV blocks, recv handles (out-of-band)
                    self._prepare_disagg_gen_init(
                        fitting_disagg_gen_init_requests)

                    all_gen_first = self.active_requests and all(
                        req.py_disaggregated_params
                        and req.py_disaggregated_params.schedule_style ==
                        DisaggScheduleStyle.GENERATION_FIRST
                        for req in self.active_requests)
                    if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                        if not all_gen_first:
                            logger.warning(
                                "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                            )
                            self._check_disagg_ctx_cache_transfer_status(1)
                        elif self.async_transfer_manager.has_any_inflight_requests(
                        ):
                            # Non-blocking cleanup of completed/timed-out
                            # transfers to free KV blocks (see _executor_loop).
                            self._check_disagg_ctx_cache_transfer_status(0)

                # Concern: schedule (telemetry)
                # Task: stash batch_size for external introspection.
                # Consume: scheduled_batch
                # Produce: self.num_scheduled_requests
                self.num_scheduled_requests = scheduled_batch.batch_size

                logger.debug(
                    f'iteration {self.iter_counter}, microbatch {microbatch_id}, '
                    f'has {len(self.active_requests)} active_requests, '
                    f'scheduled {scheduled_batch.num_context_requests} context requests and '
                    f'{scheduled_batch.num_generation_requests} generation requests'
                )

                # Concern: schedule (collective queue gate)
                # Task: collective check that every TP rank has a non-empty
                #       batch.
                # Consume: scheduled_batch.batch_size
                # Produce: can_queue
                can_queue, _ = self._can_queue(scheduled_batch)
                if not can_queue:
                    # Concern: resource (revert)
                    # Task: undo V2 scheduler's per-gen KV growth when
                    #       skipping the forward.
                    self._revert_gen_alloc(scheduled_batch)
                    logger.debug(
                        f"microbatch {microbatch_id} cannot be queued, skipping"
                    )
                    # Half concern: HC1 SLOT RING (produce — empty slot)
                    # Task: mark slot N as empty so the future drain at
                    #       iter N+offset sees `executed_batch is None`
                    #       and skips HC4 produce.
                    # Consume: -
                    # Produce: self.micro_batches[microbatch_id] (cleared)
                    self.micro_batches[microbatch_id] = None
                else:
                    logger.debug(f"microbatch {microbatch_id} can be queued")

                    # Concern: schedule (inflight tracking)
                    # Task: register curr's request IDs in
                    #       self.inflight_req_ids so the scheduler can
                    #       skip them next iter (until tokens are
                    #       generated). Paired with `_remove_inflight_ids`
                    #       at P_RETIRE inside `_handle_executed_batch`
                    #       — same `schedule` concern, two phases.
                    # Consume: scheduled_batch
                    # Produce: self.inflight_req_ids
                    self._add_inflight_ids(scheduled_batch)

                    if self.kv_cache_transceiver:
                        # Concern: disagg
                        # Task: promote DISAGG_GENERATION_TRANS_COMPLETE
                        #       gen requests to GENERATION_IN_PROGRESS;
                        #       prepend first_gen logits/logprobs.
                        # Consume: scheduled_batch.generation_requests
                        # Produce: per-request state (out-of-band)
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                    # Concern: spec_decode (dynamic draft length)
                    # Task: pad/truncate py_draft_tokens uniform.
                    self._handle_dynamic_draft_len(scheduled_batch)

                    # Concern: resource
                    # Task: allocate KV blocks. Paired with
                    #       `update_resources` at P_RETIRE inside
                    #       `_handle_executed_batch`.
                    self.resource_manager.prepare_resources(scheduled_batch)

                    # Concern: schedule (gen-request reorder)
                    # Task: stable-sort gen requests for disagg layout.
                    # The generation requests that do not have batch_idx
                    # need to be in front of the batch due to the assumptions
                    # made in model_engine.py::_forward_step. This is only important
                    # for disaggregated serving. For non-disaggregated serving,
                    # the generation requests always have batch_idx.
                    scheduled_batch.generation_requests = sorted(  # stable sort
                        scheduled_batch.generation_requests,
                        key=lambda req: int(req.py_batch_idx is not None),
                    )

                    if self.kv_cache_transceiver:
                        # Concern: response (first-token)
                        # Task: emit first-token responses early for the
                        #       disagg client. Paired with the main
                        #       `response` work at P_RETIRE.
                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)

                    # ===== curr PHASE: P_FORWARD =====
                    # Stage 1.1: Async forward (all ranks) and decoding pass (last rank only)
                    if not self.dist.is_last_pp_rank:
                        with torch.cuda.nvtx.range(
                                f"_forward_step_inter_pp pp_rank {self.dist.pp_rank}"
                        ):
                            # Concern: forward (distributed — HC7 PP FORWARD
                            #          CHAIN is its internal cross-rank
                            #          coordination, intra-batch) (+ sample
                            #          placeholder + state advance)
                            # Task: model forward on this rank's layer
                            #       slice. The forward body pp_recv <-
                            #       prev rank and pp_send -> next rank
                            #       internally — same `forward` concern
                            #       running on every rank for the *same
                            #       request set*, which is what creates
                            #       the 1F1B stagger between the per-rank
                            #       iter clocks.
                            #       `_forward_step_inter_pp` ALSO
                            #       constructs a *placeholder*
                            #       sampler_event (just a recorded
                            #       cuda.Event with no data) and runs
                            #       `_update_request_states` so the
                            #       stash at P_STATE_UPD has a unified
                            #       BatchStatePP shape across ranks.
                            #       (i.e., this single function call
                            #       covers what the last rank breaks
                            #       into forward + sample + state-upd.)
                            # Consume: scheduled_batch (cross-rank: prev
                            #          rank's activations via NCCL p2p
                            #          — the consume side of HC7,
                            #          internal to this concern)
                            # Produce: sample_state (placeholder + state
                            #          advance applied), runtime_draft_len
                            #          (via `_update_request_states`
                            #          embedded in this call); cross-rank:
                            #          activations to next rank (HC7
                            #          produce, internal to this concern)
                            sample_state = self._forward_step_inter_pp(
                                scheduled_batch)
                    else:
                        with torch.cuda.nvtx.range(
                                f"_forward_step_last_pp pp_rank {self.dist.pp_rank}"
                        ):
                            # Concern: guided_decoder (curr — pre-forward)
                            # Task: register batch with constraint
                            #       matcher; init disagg-gen request
                            #       matchers. Last-rank only because
                            #       only the last rank produces logits
                            #       to mask.
                            # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                            if self.guided_decoder is not None and self.kv_cache_transceiver:
                                self.guided_decoder.add_batch(scheduled_batch)
                                self.guided_decoder.init_disagg_gen_requests()

                            # Concern: forward (distributed — last rank
                            #          terminus of HC7 PP FORWARD CHAIN)
                            # Task: model forward on the last layer slice;
                            #       internally pp_recv from prev (HC7's
                            #       consume side at this rank, intra-batch),
                            #       no pp_send (terminus of the chain).
                            # Consume: scheduled_batch (cross-rank: prev's
                            #          activations, internal to the
                            #          distributed `forward` concern)
                            # Produce: batch_outputs (logits)
                            batch_outputs = self._forward_step(scheduled_batch)

                            # ===== curr PHASE: P_SAMPLE (last rank only) =====

                            # Concern: guided_decoder (curr — apply masks)
                            # Task: apply matcher to logits, flag failed.
                            # Consume: batch_outputs (logits)
                            # Produce: guided_decoder_failed_requests
                            guided_decoder_failed_requests = None
                            if self.guided_decoder is not None:
                                self.guided_decoder.add_batch(scheduled_batch)
                                guided_decoder_failed_requests = self.guided_decoder.execute(
                                    batch_outputs['logits'])

                            if self.pp_multi_stream_sample:
                                # Half concern: HC3 SAMPLE_STREAM (consume —
                                #          last-rank cross-iter)
                                # Task: wait for last iter's sampling on
                                #       sample_stream to finish (paired
                                #       with last iter's
                                #       `finish_sample_event.record()`
                                #       below).
                                # Consume: self.finish_sample_event
                                #          (cross-iter scheduler state)
                                # Produce: -
                                # Wait for the previous sample to finish.
                                self.finish_sample_event.wait()
                                # Concern: sample (curr — clone for
                                #          stream-isolated sampling)
                                # Task: clone batch_outputs so the next
                                #       iter's forward can overwrite the
                                #       originals on the default stream
                                #       while sampling reads the clones.
                                # Copy the batch outputs as sampler inputs
                                # to avoid next forward step overwriting them.
                                batch_outputs_copy = {
                                    name: tensor.clone()
                                    for name, tensor in batch_outputs.items()
                                }
                                self.sample_stream.wait_stream(
                                    torch.cuda.current_stream())
                                with torch.cuda.stream(self.sample_stream):
                                    # Concern: sample (curr)
                                    # Task: queue sampling kernel + D2H
                                    #       copy of sample-state tensors
                                    #       on sample_stream; record
                                    #       sampler_event for next iter's
                                    #       HC2 consume (P_SYNC_EVT).
                                    # Consume: scheduled_batch,
                                    #          batch_outputs_copy
                                    # Produce: sample_state (with
                                    #          sampler_event + .device
                                    #          tensors), curr field
                                    sample_state = self._sample_async(
                                        scheduled_batch, batch_outputs_copy)
                                    # Half concern: HC3 SAMPLE_STREAM (produce)
                                    # Task: record sample-stream done
                                    #       event for next iter's HC3
                                    #       consume.
                                    # Consume: -
                                    # Produce: self.finish_sample_event
                                    #          (cross-iter scheduler state)
                                    self.finish_sample_event.record()
                            else:
                                # Concern: sample (curr)
                                # Task: same as above but on the default
                                #       stream when multi-stream sampling
                                #       is disabled.
                                sample_state = self._sample_async(
                                    scheduled_batch, batch_outputs)

                            assert sample_state is not None, "Sampling failed"

                            # ===== curr PHASE: P_STATE_UPD (last rank) =====

                            # Concern: guided_decoder (curr — error mark)
                            # Task: mark grammar-failed requests after
                            #       sample (cf. plain loop comment).
                            # Handle guided decoder errors after _sample_async to avoid state conflicts.
                            # If called before, failed requests would be marked as GENERATION_COMPLETE,
                            # causing _sample_async to fail when accessing context_chunk_size property.
                            self._handle_guided_decoder_errors(
                                scheduled_batch, guided_decoder_failed_requests)

                            # Concern: sample (curr — state advance)
                            # Task: advance context_chunk_position; transition
                            #       finished-context to GENERATION_*; drop
                            #       ADP dummy. Last rank does this here;
                            #       non-last ranks fold it into
                            #       `_forward_step_inter_pp`.
                            self._update_request_states(scheduled_batch)
                            if not self.disable_overlap_scheduler:
                                # Concern: spec_decode (curr — late state
                                #          mark; last rank only)
                                # Task: mark gen requests that will finish
                                #       next iter so their excluded-logits
                                #       behavior is correct in the response.
                                self._update_generation_requests_that_will_complete_next_iteration(
                                    scheduled_batch.generation_requests)

                    if self.enable_iter_perf_stats:
                        # Concern: iter_stats (curr — num_ctx_tokens key)
                        # Task: snapshot model_engine.iter_states.num_ctx_tokens
                        #       onto curr's iter_stats. Read at curr's
                        #       P_RETIRE (n-1 iters later) inside
                        #       `_process_iter_stats`.
                        iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                            'num_ctx_tokens']
                    # Concern: iter_stats / sample / schedule (curr — pack
                    #          per-batch fields)
                    # Task: package curr's per-batch fields into a single
                    #       BatchStatePP record so HC1 / HC4 can carry it
                    #       through the slot ring + bcast queue without
                    #       knowing about individual fields.
                    # Consume: scheduled_batch, sample_state,
                    #          iter_start_time, iter_stats, microbatch_id
                    # Produce: batch_state (local; flows into HC1 produce
                    #          and HC2 produce below)
                    batch_state = BatchStatePP(
                        scheduled_requests=scheduled_batch,
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats,
                        microbatch_id=microbatch_id,
                    )

                    # Half concern: HC1 SLOT RING (produce — fill slot)
                    # Bridge: curr's batch_state -> later iter's
                    #          `executed_batch` consume below.
                    # Task: stash curr's BatchStatePP in slot
                    #       `microbatch_id` of the per-rank ring so a
                    #       future iter (offset=-1 last rank, 1-n other
                    #       ranks) can drain it for HC4 produce.
                    # Consume: batch_state
                    # Produce: self.micro_batches[microbatch_id]
                    self.micro_batches[microbatch_id] = batch_state

                # =========================================================
                # ===== STEP (2) : prev P_SYNC_EVT [+ P_HANDOFF on last rank] =
                # =========================================================
                # Drives mb_(t-1) (k=1) through P_SYNC_EVT. On the last
                # rank this is also where mb_(t-1) hits P_HANDOFF (since
                # last rank's HANDOFF deadline is iter N+1, not N+(n-1)).

                # Half concern: HC2 PREV (produce + consume — last rank /
                #          can_queue gating)
                # Bridge: curr.sample_state.sampler_event (P_SAMPLE)
                #          -> next iter's `previous_batch.sample_state.
                #          sampler_event.synchronize()` consume below.
                # Task: roll prev <- curr for next iter's HC2 consume,
                #       AND consume prev (this iter's previous_batch) by
                #       synchronizing its sampler_event. The non-last
                #       rank skips the consume when `not can_queue` —
                #       the sync is then deferred to a later iter when
                #       the rank has another queueable batch.
                # Consume: self.previous_batch (cross-iter scheduler state)
                # Produce: self.previous_batch (cross-iter scheduler state),
                #          prev's sampler_event synchronized (out-of-band)
                # Stage 1.2: Sync sampler for previous microbatch to start new sample state comm chain.
                # For last PP rank, we must synchronize the previous batch
                # since we need to broadcast its sample state soon afterwards in the same iteration.
                # For other PP ranks, we can delay the synchronization if the current batch cannot be queued.
                previous_batch = self.previous_batch
                if can_queue:
                    self.previous_batch = batch_state
                if (self.dist.is_last_pp_rank
                        or can_queue) and previous_batch is not None:
                    with nvtx_range("sync_previous_sampler_event"):
                        previous_batch.sample_state.sampler_event.synchronize()

                # ===== prev PHASE: P_HANDOFF (last rank) =====
                # ===== oldest PHASE: P_HANDOFF (other ranks) =====
                # Stage 2: Enqueue sample state for executed batch to ring broadcast it in background thread asynchronously.
                # send/recv chain: (pp_size - 1) -> 0 -> 1 -> ... -> (pp_size - 2)
                # intermediate ranks: send/recv sample state for next microbatch to allow overlap

                # HC1 SLOT RING (consume) — scheduler bookkeeping.
                # `ring_broadcast_sample` distributed concern (produce
                # end on this rank): hand the freshly-executed batch
                # off to the long-running concern that ring-broadcasts
                # its sample state across all ranks. HC4 BCAST QUEUE
                # is this concern's per-rank input side; the cross-rank
                # ring (HC10) is its internal coordination — same
                # request set seen by every rank, so it's intra-batch.
                # Task (this block): figure out which slot's batch is
                #       reaching its handoff deadline this iter
                #       (`offset = -1` for last rank because last rank's
                #       deadline is iter N+1; `offset = 1 - pp_size` for
                #       other ranks because their deadline is iter N+(n-1)),
                #       drain it from the slot ring, and pass it to the
                #       distributed ring-broadcast concern via its
                #       per-rank input queue. Clear the slot in the
                #       same step so HC1 produce next iter doesn't see
                #       stale data.
                # Consume: self.micro_batches[(microbatch_id+offset) % n]
                # Produce: self.executed_batch_queue (per-rank input of
                #          the `ring_broadcast_sample` concern),
                #          self.unhandled_batch_counter (loop state),
                #          self.micro_batches[...] (cleared)
                offset = -1 if self.dist.is_last_pp_rank else (
                    1 - self.dist.pp_size)
                executed_microbatch_id = (microbatch_id +
                                          offset) % self.num_micro_batches
                executed_batch = self.micro_batches[executed_microbatch_id]
                if executed_batch is not None:
                    self.executed_batch_queue.put(executed_batch)
                    self.unhandled_batch_counter += 1
                self.micro_batches[executed_microbatch_id] = None

                # Nested helpers — one-shot definitions used only in this
                # iter. Each closes over `microbatch_id`, `can_queue`, and
                # locals like `executed_batches`. In a coroutine refactor
                # these would be method calls on the schedule / response
                # coroutines, not nested closures.

                def fetch_executed_batches() -> list[BatchStatePP]:
                    # Concern: `ring_broadcast_sample` distributed concern
                    #          (consume end on this rank, rk0 only) —
                    #          drain batches whose ring traversal has
                    #          reached this rank.
                    # Task: rk0 drains the response queue (the per-rank
                    #       output side of HC10's reverse ring).
                    #       Synchronous (must_get=True) when async-bcast
                    #       disabled, else only blocks when this rank
                    #       has no new work (`not can_queue`) so a
                    #       productive iter isn't gated by the ring.
                    # Consume: self.executed_batch_response_queue
                    # Produce: executed_batches (local)
                    executed_batches = []
                    if self.pp_async_broadcast_sample_state:
                        # Wait for at least one batch to finish if no new request is available.
                        must_get = not can_queue
                    else:
                        must_get = True
                    while not self.executed_batch_response_queue.empty() or (
                            must_get and self.unhandled_batch_counter > 0):
                        with nvtx_range("get_executed_batch"):
                            executed_batches.append(
                                self.executed_batch_response_queue.get())
                        must_get = False
                    return executed_batches

                def ring_broadcast_executed_batch_num(
                        executed_batch_num: int) -> int:
                    # SCHEDULER-DIRECT (HC9): NOT a per-batch coroutine
                    # and NOT a distributed concern coroutine — the
                    # count payload spans all batches retiring this iter,
                    # so it doesn't tie to a single request set. In the
                    # refactored model this stays as plain code in the
                    # scheduler's iter body (no `step`, no `Batch`, no
                    # `enter_phase`) — same behavior, just inlined into
                    # the scheduler instead of wrapped as a coroutine.
                    # Task: ring-broadcast the consensus on how many
                    #       batches retire this iter. rk0 votes;
                    #       count propagates rk0 -> rk(n-1) via the PP
                    #       forward chain. The HC5b SEND HANDLES RING
                    #       (`send_expected_batch_num_handles`) is this
                    #       block's own per-slot isend-handle accounting,
                    #       also scheduler-direct.
                    # Consume: cross-rank: prev rank's vote
                    # Produce: cross-rank: isend to next rank;
                    #          self.send_expected_batch_num_handles[mid]
                    if self.dist.is_first_pp_rank and self.dist.tp_size * self.dist.cp_size > 1:
                        with nvtx_range("tp_cp_broadcast_executed_batch_num"):
                            executed_batch_num = self.dist.tp_cp_broadcast(
                                executed_batch_num,
                                root=0,
                            )
                    if not self.dist.is_first_pp_rank:
                        with nvtx_range("recv_expected_batch_num"):
                            executed_batch_num = self.dist.recv_object(
                                src=self.dist.prev_pp_rank,
                                tag=PPCommTag.EXECUTED_BATCH_NUM,
                            )
                    if not self.dist.is_last_pp_rank:
                        # HC5b consume: wait on prior iter's isend in
                        #              this slot.
                        self.wait_on_pp_send_handles(
                            self.send_expected_batch_num_handles, microbatch_id)
                        with nvtx_range("send_expected_batch_num"):
                            # HC5b produce: stash this iter's isend in
                            #              the slot.
                            self.send_expected_batch_num_handles[
                                microbatch_id] = self.dist.isend_object(
                                    executed_batch_num,
                                    dest=self.dist.next_pp_rank,
                                    tag=PPCommTag.EXECUTED_BATCH_NUM,
                                )
                    return executed_batch_num

                def handle_executed_batches(executed_batch_num: int):
                    # Concern: `ring_broadcast_sample` distributed concern
                    #          (consume end on this rank, non-rk0) +
                    #          per-batch retire dispatch.
                    # Task: non-rk0 ranks drain executed_batch_num
                    #       batches from response_queue and retire each.
                    #       rk0 reuses the already-drained list from
                    #       `fetch_executed_batches`. Each retire calls
                    #       `_handle_executed_batch` which is the
                    #       multi-concern P_RETIRE body (response,
                    #       sample-apply, disagg, resource, kv_cache_events,
                    #       iter_stats — see annotations on
                    #       `_handle_executed_batch`).
                    # Consume: self.executed_batch_response_queue (non-rk0),
                    #          executed_batches (rk0)
                    # Produce: per-batch retire side effects (out-of-band)
                    if self.dist.rank != 0:
                        dequeue_counter = 0
                        while dequeue_counter < executed_batch_num:
                            with nvtx_range("get_executed_batch"):
                                executed_batch = self.executed_batch_response_queue.get(
                                )
                            self._handle_executed_batch(executed_batch)
                            dequeue_counter += 1
                    else:
                        for executed_batch in executed_batches:
                            self._handle_executed_batch(executed_batch)
                    self.unhandled_batch_counter -= executed_batch_num

                executed_batch_num = 0

                # =========================================================
                # ===== STEP (3) : oldest P_VOTE -> P_RETIRE ==============
                # =========================================================
                # Drives mb_(t-(n-1)) (k=n-1) through P_VOTE then P_RETIRE.
                # Non-last ranks: this is also where mb_(t-(n-1)) hits
                # P_HANDOFF, so step (2) above has already enqueued it.

                # ===== oldest PHASE: P_VOTE (rk0 only) =====

                # Stage 3.1: The first rank determines the number of executed batches.
                if self.dist.rank == 0:
                    executed_batches = fetch_executed_batches()
                    executed_batch_num = len(executed_batches)

                # Stage 3.2: Broadcast the number of executed batches to other ranks.
                executed_batch_num = ring_broadcast_executed_batch_num(
                    executed_batch_num)

                # ===== oldest PHASE: P_RETIRE =====
                # Stage 3.3: Handle executed batches.
                handle_executed_batches(executed_batch_num)

                # =========================================================
                # ===== Ring rotation + iter advance ======================
                # =========================================================

                # Concern: loop_control (slot ring rotation)
                # Task: advance the per-rank ring head. In the coroutine
                #       refactor this is `ring.append(handle_curr);
                #       handle_oldest = ring.popleft()`.
                # Consume: -
                # Produce: microbatch_id (loop state)
                # Stage 4: March forward in microbatch slots
                microbatch_id = (microbatch_id + 1) % self.num_micro_batches
                # Concern: loop_control
                # Task: advance global iter counter.
                self.iter_counter += 1

            # =============================================================
            # ===== POST-SHUTDOWN DRAIN ===================================
            # =============================================================
            # When `should_stop_processing` broke us out of the while
            # loop, up to (n-1) batches may still be in flight on this
            # rank — n-1 of them parked between the bcast queue and the
            # response queue. Drain them all here.
            #
            # Concern: `ring_broadcast_sample` distributed concern
            #          (consume end — drain at shutdown) + per-batch
            #          retire dispatch.
            # Task: process the remaining in-flight batches so every
            #       client gets a response.
            # Consume: self.executed_batch_response_queue
            # Produce: per-batch retire side effects (out-of-band)
            # Stage 5: Handle remaining executed batches in the queue.
            while self.unhandled_batch_counter > 0:
                with nvtx_range("get_executed_batch"):
                    executed_batch = self.executed_batch_response_queue.get()
                self._handle_executed_batch(executed_batch)
                self.unhandled_batch_counter -= 1

    # REFACTOR: 2a (concern: ring_broadcast_sample) -- the long-running
    # bcast THREAD's body. New design: replaced by the
    # ``RingBroadcastSampleConcern`` long-running coroutine spanning
    # ``(n-1)`` scheduler iters per batch (no separate thread).
    def _broadcast_sample_state_loop(self):
        logger.debug(
            f"Starting broadcast sample state loop for pp_rank {self.dist.pp_rank}"
        )
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        # Acquiring pkl5.Intracomm's send/recv locks from both executor loop thread
        # and this thread will cause perf drop and even deadlock.
        # We create new MPI comm to avoid these issues.
        logger.info(
            "Create new MPI comm for broadcast sample state thread to avoid deadlock."
        )
        new_mpi_comm = mpi_comm().Dup()
        set_thread_local_mpi_comm(new_mpi_comm)
        while True:
            executed_batch = self.executed_batch_queue.get()
            if executed_batch is None:
                break
            self._ring_broadcast_sample_state(executed_batch)
        set_thread_local_mpi_comm(None)
        new_mpi_comm.Free()

    # REFACTOR: 2a (concern: ring_broadcast_sample) -- per-batch ring hop
    # body: recv from prev rank, push into response queue, isend to next
    # rank. Becomes one phase block inside the new long-running concern.
    def _ring_broadcast_sample_state(
        self,
        executed_batch: Optional[BatchStatePP],
    ) -> None:
        if executed_batch is None:
            return

        tag = PPCommTag.SAMPLE_STATE
        microbatch_id = executed_batch.microbatch_id
        sample_state = executed_batch.sample_state
        requests = sample_state.requests

        if not self.dist.is_last_pp_rank:
            # Receive tokens from previous pp rank (w.r.t model forward direction)
            with nvtx_range("recv_sample_state"):
                sample_state.host, py_result_diffs = self.dist.recv_object(
                    src=self.dist.prev_pp_rank,
                    tag=tag,
                )

            for request, py_result_diff in zip(requests, py_result_diffs):
                request.py_result.apply_diff(py_result_diff)

        self.executed_batch_response_queue.put(executed_batch)

        # Send tokens to next pp rank (w.r.t model forward direction)
        # Second last rank does not need to since last rank has original decoded tokens
        if not self.dist.is_second_last_pp_rank:
            py_result_diffs = []
            for request in requests:
                diff = request.py_result.get_diff()
                py_result_diffs.append(diff)
                request.py_result.reset_diff()
            self.wait_on_pp_send_handles(self.send_handles, microbatch_id)
            with nvtx_range("send_sample_state"):
                self.send_handles[microbatch_id] = self.dist.isend_object(
                    (sample_state.host, py_result_diffs),
                    dest=self.dist.next_pp_rank,
                    tag=tag,
                )

    # REFACTOR: 1b (multi-concern, inlined+split) -- PP P_RETIRE body:
    # sample-apply + disagg send_kv + response (canceled + responses) +
    # disagg ctx-cache probe + resource update + schedule inflight-id
    # remove, plus cross-batch global tail (kv-transfer timeout sweep,
    # PP termination ballot, iter_stats process). Each branch goes to
    # its concern's APPLY_7 / RESPOND_8 / FINALIZE_9 block; scheduler-
    # direct items (termination ballot) move to scheduler iter.
    def _handle_executed_batch(self, executed_batch: Optional[BatchStatePP]):
        # ===================================================================
        # P_RETIRE body for one executed batch on the PP loop. Multi-concern,
        # mirrors the non-overlap loop's "after _update_requests" tail
        # (sample-apply + disagg + response + resource + kv_cache_events +
        # iter_stats), with PP-specific additions:
        #   * `_remove_inflight_ids` — paired with `_add_inflight_ids` at
        #     this batch's P_SCHEDULE on this rank (n-1 iters ago in
        #     iter-INDEX local to this rank).
        #   * `_disagg_pp_termination_handler.terminate_pending_requests`
        #     — PP-only ring-style consensus on which requests get
        #     terminated this iter.
        # All work here is at this batch's P_RETIRE phase. Some sub-blocks
        # are global (run even if `executed_batch is None`) — they are
        # cross-batch sweeps, not part of any single batch's lifecycle.
        # ===================================================================
        finished_requests = []
        if executed_batch is not None:
            with torch.cuda.nvtx.range("_handle_executed_batch_pp"):
                # Concern: sample (apply tokens — paired with this batch's
                #          P_SAMPLE on the last rank, which produced
                #          sample_state.sampler_event consumed at this
                #          rank's P_SYNC_EVT iter and propagated through
                #          HC10 ring broadcast)
                # Task: write sampled tokens onto this batch's requests
                #       via sampler.update_requests. The sampler_event is
                #       already synchronized by step (2) on the last rank,
                #       and on other ranks the host data was filled in by
                #       the bcast thread / HC10 ring broadcast — so this
                #       call doesn't block.
                # Consume: executed_batch.sample_state
                # Produce: per-request new tokens (out-of-band)
                self._update_requests(executed_batch.sample_state)

                scheduled_requests = executed_batch.scheduled_requests
                if self.kv_cache_transceiver:
                    # Concern: disagg
                    # Task: start ctx->gen KV send for finished ctx-only
                    #       requests in this batch (paired with the
                    #       ctx-side recv promotion at this batch's
                    #       P_SCHEDULE inside `_check_disagg_ctx_schedulable_status`).
                    finished_ctx_reqs = scheduled_requests.context_requests_last_chunk
                    self._send_kv_async(finished_ctx_reqs)
                # Concern: response (cancellation)
                # Task: terminate canceled requests if possible.
                self._flush_pending_transfer_responses()
                self._handle_canceled_requests()

                # Concern: response (+ perf_metric step + spec_decode gating)
                # Task: build per-request responses for this batch,
                #       enqueue, terminate finished. Multi-concern function
                #       — see annotations on `_handle_responses`.
                # Consume: self.active_requests, executed_batch.scheduled_requests
                # Produce: finished_requests, responses (out-of-band)
                finished_requests = self._handle_responses()
                # Complete ctx send sessions AFTER responses are created so
                # _handle_responses sees the request before it is terminated.

                # Concern: disagg (opportunistic completion probe)
                # Task: non-blocking sweep for ctx transfers that just
                #       completed; terminates ctx-only requests so their
                #       KV blocks can be reused. Strict ordering AFTER
                #       _handle_responses so the response sees the
                #       request before termination.
                if self.kv_cache_transceiver:
                    self._check_disagg_ctx_cache_transfer_status(0)
                sample_state_scheduled_requests = executed_batch.scheduled_requests
                # Concern: resource
                # Task: free per-request KV / resource state for finished
                #       requests; rebalance attn metadata. Paired with
                #       this batch's `prepare_resources` at P_SCHEDULE.
                attn_metadata = getattr(self.model_engine, 'attn_metadata',
                                        None)
                kv_cache_dtype_byte_size = getattr(self.model_engine,
                                                   'kv_cache_dtype_byte_size',
                                                   None)
                self.resource_manager.update_resources(
                    sample_state_scheduled_requests, attn_metadata,
                    kv_cache_dtype_byte_size)

                # Concern: schedule (inflight tracking — release)
                # Task: remove this batch's request IDs from the inflight
                #       set so the scheduler can pick them up next iter.
                #       Paired with `_add_inflight_ids` at this batch's
                #       P_SCHEDULE on this rank, n-1 iters ago.
                self._remove_inflight_ids(scheduled_requests)

        # NOTE: blocks below run unconditionally (even when executed_batch
        # is None on a rank where no batch retired this iter). They are
        # cross-batch sweeps that belong to a global tail of P_RETIRE.

        if self.kv_cache_transceiver and self.async_transfer_manager.has_any_inflight_requests(
        ):
            # Concern: disagg (cross-batch)
            # Task: end-of-iter timeout sweep over inflight async KV
            #       transfers (regardless of which batch owns them).
            self._check_kv_transfer_timeout()

        if self._disagg_pp_termination_handler is not None:
            # SCHEDULER-DIRECT: NOT a per-batch coroutine and NOT a
            # distributed concern coroutine — same shape as HC9: every
            # rank runs this every iter, payload is a ballot of pending
            # terminations spanning multiple batches, no single request
            # set involved. In the refactored model this call moves
            # out of `_handle_executed_batch` (which becomes per-batch
            # P_RETIRE work) and into the scheduler iter directly,
            # alongside HC9.
            # Task: ring-pass the termination ballot; only requests that
            #       all PP ranks vote YES on get terminated together.
            #       Without this NCCL would hang on KV cache reuse paths
            #       when PP ranks asymmetrically free a request.
            # Consume: cross-rank: prev rank's termination ballot
            # Produce: cross-rank: isend ballot to next rank;
            #          per-request termination (out-of-band)
            self._disagg_pp_termination_handler.terminate_pending_requests()

        if self.enable_iter_perf_stats and executed_batch is not None:
            # Concern: iter_stats (curr — process)
            # Task: end-of-lifecycle stats aggregation. Paired with
            #       `_get_init_iter_stats` at this batch's P_SCHEDULE
            #       (n-1 iters ago) and the `num_ctx_tokens` snapshot
            #       at P_STATE_UPD. Reads `executed_batch.iter_stats` /
            #       `iter_start_time` carried through the slot ring + bcast
            #       queue (HC1 + HC4).
            # Consume: executed_batch (iter_stats, iter_start_time,
            #          scheduled_requests), finished_requests
            # Produce: aggregated iter_stats record (out-of-band)
            self._process_iter_stats(
                finished_requests,
                self.active_requests,
                executed_batch,
                executed_batch.microbatch_id % self.dist.pp_size,
            )

    # REFACTOR: 1b (legacy machinery; ELIMINATED by new design) -- this
    # helper exists only because the legacy loop stashes ``isend``
    # handles in per-microbatch slot rings (``send_handles``,
    # ``send_schedule_handles``, ``send_expected_batch_num_handles``)
    # and waits on the previous slot's handle when reusing it.
    #
    # In the new design every concern that does ``isend_object`` is a
    # coroutine that holds the handle as a local variable and waits on
    # it directly later in its OWN body:
    #
    # - HC10 ``ring_broadcast_sample`` distributed concern (per-batch
    #   coroutine): ``handle = isend_object(...)`` near the end of the
    #   body, ``handle.wait()`` before the body returns. No slot ring.
    # - HC5a schedule's PP variant (per-batch coroutine): same pattern;
    #   schedule's coroutine waits on its own ``isend`` before
    #   completing.
    # - HC5b ``retire_vote`` (per-iter scheduler-direct): no
    #   coroutine; wait immediately after isend (the payload is just
    #   an int so the cost is negligible) or use blocking ``send_object``.
    #
    # Result: the slot rings + this helper all go away.
    @nvtx_range("wait_on_pp_send_handles")
    def wait_on_pp_send_handles(self, send_handles, microbatch_id):
        if send_handles[microbatch_id] is not None:
            send_handles[microbatch_id].wait()
            send_handles[microbatch_id] = None

    # REFACTOR: 2a (concern: spec_decode) -- FORWARD_2 step (per-batch
    # uniform draft-token padding for CUDA-graph compat).
    def _handle_dynamic_draft_len(self,
                                  scheduled_batch: ScheduledRequests) -> None:
        """Handle dynamic draft length for the current batch.

        Must be called BEFORE prepare_resources so that KV cache allocation
        uses the correct draft length.

        Two things happen here:
        1. Determine the runtime draft length from the draft_len_schedule
           based on the current batch size, and store it on model_engine so
           that the rest of the forward path can read it.
        2. Pad / truncate each request's py_draft_tokens to exactly match
           the determined draft length, ensuring uniform draft token counts across the
           batch (required by CUDA graph replay and the attention kernel).

        When dynamic draft length is not enabled, runtime_draft_len is simply
        set to max_draft_len (the static maximum).
        """
        if not hasattr(self.model_engine, 'max_draft_len'):
            return

        if (self.model_engine.spec_config is not None
                and self.model_engine.spec_config.draft_len_schedule is not None
                and self.model_engine.spec_config.spec_dec_mode.
                support_dynamic_draft_len()):
            from tensorrt_llm._torch.speculative.utils import \
                get_draft_len_for_batch_size

            # 1. Resolve runtime draft length from schedule
            runtime_draft_len = get_draft_len_for_batch_size(
                self.model_engine.spec_config.draft_len_schedule,
                scheduled_batch.batch_size, self.model_engine.max_draft_len)

            # 2. Pad or truncate draft tokens to the resolved length
            PADDING_TOKEN = 0
            for request in scheduled_batch.generation_requests:
                current_draft_len = len(request.py_draft_tokens)
                if current_draft_len < runtime_draft_len:
                    padding_needed = runtime_draft_len - current_draft_len
                    request.py_draft_tokens.extend([PADDING_TOKEN] *
                                                   padding_needed)
                elif current_draft_len > runtime_draft_len:
                    request.py_draft_tokens = request.py_draft_tokens[:
                                                                      runtime_draft_len]

            self.model_engine.runtime_draft_len = runtime_draft_len
        else:
            self.model_engine.runtime_draft_len = self.model_engine.max_total_draft_tokens

    # REFACTOR: 2a (concern: schedule) -- collective ``can_queue`` gate
    # at SCHEDULE_0.
    def _can_queue(self, scheduled_batch):

        # can_queue_this_rank is for case that the batch is not empty on this rank, but empty on other ranks
        # For bs == 1, we cannot pad dummy request to make the batch non-empty since it will cause the batch size to be 2.
        # 1 for dummy request, 1 for the yet-to-complete but not-yet-updated request.
        if self.enable_attention_dp:
            tp_batch_sizes = self.dist.tp_allgather(scheduled_batch.batch_size)
            can_queue = 0 not in tp_batch_sizes
            can_queue_this_rank = scheduled_batch.batch_size > 0
        else:
            can_queue = can_queue_this_rank = scheduled_batch.batch_size > 0

        return can_queue, can_queue_this_rank

    # REFACTOR: 2a (concern: resource) -- revert per-gen KV growth when
    # the batch is skipped on this rank (V2 scheduler only).
    def _revert_gen_alloc(self, scheduled_batch):
        """Revert KV cache capacity growth when the batch is skipped.

        With attention DP, can_queue=False means another rank has an empty
        batch so no forward pass will run.  The V2 scheduler already grew
        each generation request's KV cache capacity during scheduling;
        revert that growth so it does not accumulate across skipped
        iterations and overflow the host page-index buffer.

        Only applies to KV cache manager V2 + scheduler V2, because the V2
        scheduler allocates KV cache capacity during scheduling, before the
        can_queue check.  V1 allocates in prepare_resources() after the
        can_queue check, so no revert is needed.
        """
        if self._scheduler_manages_kv_suspend:
            for req in scheduled_batch.generation_requests:
                self.kv_cache_manager.revert_allocate_generation(req)

    # REFACTOR: 1b (multi-concern, inlined+split) -- annotated in-line as
    # a SCHEDULE_0 fan-out across schedule + disagg + iter_stats +
    # spec_decode + resource. Body splits into the respective concerns'
    # SCHEDULE_0 blocks; nothing remains as a monolithic method.
    def _prepare_and_schedule_batch(self):
        # ===================================================================
        # Multi-concern fan-out called once per iter from `_executor_loop*`.
        # All work in this body lives in P_SCHEDULE. The split into
        # sub-concerns is what motivates eventually breaking this function
        # apart into per-coroutine `enter_phase(P_SCHEDULE)` blocks.
        # ===================================================================

        # Concern: schedule (request intake)
        # Task: dequeue new requests from the waiting queue, validate, and
        #       activate. `_handle_errors` may fail invalid ones eagerly.
        # Consume: -
        # Produce: new_requests (local), self.active_requests (loop state)
        new_requests = self._fetch_and_activate_new_requests()
        # Concern: loop_control
        # Task: shutdown gate — no new work AND nothing in flight => stop.
        # Consume: -
        # Produce: -
        if self.should_stop_processing:
            return None, None

        if self.kv_cache_transceiver:
            # Concern: disagg
            # Task: opportunistic probes that drive the disagg KV-transceiver
            #       state machine forward at iter top: promote ctx requests
            #       whose peer info just arrived, completion-poll inflight
            #       gen transfers, and timeout-sweep pending transfers.
            #       All three are non-blocking; the deadline-waits live at
            #       P_RESPOND (`_check_disagg_ctx_cache_transfer_status`,
            #       `_check_kv_transfer_timeout`).
            # Consume: -
            # Produce: per-request state transitions (out-of-band)
            self._check_disagg_ctx_schedulable_status(new_requests)
            self._check_disagg_gen_transfer_status()
            self._check_kv_transfer_timeout()

        # Concern: iter_stats (init)
        # Task: allocate the iter_stats record + record queue latency for
        #       newly arrived requests. Paired with the `_process_iter_stats`
        #       call at P_RESPOND.
        # Consume: -
        # Produce: iter_stats
        iter_stats = None
        if self.enable_iter_perf_stats:
            iter_stats = self._get_init_iter_stats(
                len(new_requests),
                self._get_new_active_requests_queue_latency())

        # Concern: schedule (ADP dummy padding)
        # Task: when attention_dp is on, ensure every rank has a request
        #       to participate in collectives — pad with a dummy gen
        #       request locally if needed (terminated at P_RESPOND).
        # Consume: -
        # Produce: self.active_requests (loop state, possibly +1 dummy)
        self._pad_attention_dp_dummy_request()

        if self.drafter is not None:
            # Concern: spec_decode (gating + draft seed)
            # Task: resolve dynamic max_total_draft_tokens from the
            #       draft_len schedule, decide use_spec_decode for this
            #       iter (draft-len=0 disables; permanent-disable flag
            #       from speculation_gate at P_RESPOND wins; otherwise
            #       drafter heuristic). Then seed every active request's
            #       draft_tokens with dummy zeros so the scheduler can
            #       account for them in KV-block math.
            # Consume: self.active_requests (loop state)
            # Produce: self.use_spec_decode, self.max_total_draft_tokens
            #          (loop state); model_engine.enable_spec_decode;
            #          per-request draft_tokens / py_draft_tokens
            # Honor permanent disable flag based on rolling acceptance first
            if self.drafter.draft_len_schedule is not None:
                batch_size_input = len(self.active_requests)

                self.max_total_draft_tokens = self.drafter.get_draft_len_for_batch_size(
                    batch_size_input)

                self.drafter.update_max_total_draft_tokens(
                    self.max_total_draft_tokens)

            # Check if draft_len=0 → immediately disable
            # self.max_total_draft_tokens==0 is only possible when draft_len_schedule is provided
            # for example, draft_len_schedule = {1:4, 4:2, 8:0}, batch_size >= 8 will set self.max_draft_len = 0
            if self.drafter.draft_len_schedule is not None and self.max_total_draft_tokens == 0:
                self.use_spec_decode = False
            elif getattr(self, 'speculation_permanently_disabled', False):
                self.use_spec_decode = False
            else:
                self.use_spec_decode = self.drafter.should_use_spec_decode(
                    self.active_requests, self.max_batch_size,
                    self.model_engine.llm_args.max_num_tokens,
                    self.max_total_draft_tokens)
            logger.debug(f"Use spec decode: {self.use_spec_decode}")
            self.model_engine.enable_spec_decode = self.use_spec_decode

            # Set up draft_tokens in active_requests, because they could be used in the scheduling stage.
            for request in self.active_requests:
                if request.state not in (
                        LlmRequestState.GENERATION_IN_PROGRESS,
                        LlmRequestState.DISAGG_GENERATION_INIT):
                    continue
                request.draft_tokens = [
                    0
                ] * self.max_total_draft_tokens if self.max_total_draft_tokens > 0 else []

            # If speculation is off, this function sets py_draft_tokens to []
            # for all active requests. If it's on, we initialize py_draft_tokens
            # with dummy draft tokens to make the scheduler aware of the fact
            # that speculation is about to happen.
            self._prepare_draft_requests()

        # Concern: schedule (core)
        # Task: run the scheduler — pick context + generation requests
        #       that fit, balance ADP requests across DP ranks, optional
        #       batch-waiting heuristic. This is the only place that
        #       writes scheduled_batch.
        # Consume: self.active_requests, py_draft_tokens (loop state)
        # Produce: scheduled_batch, fitting_disagg_gen_init_requests,
        #          num_fitting_reqs
        scheduled_batch, fitting_disagg_gen_init_requests, num_fitting_reqs = self._schedule(
        )

        if self.drafter is not None and not self.use_spec_decode:
            # Concern: spec_decode (per-request disable)
            # Task: propagate the use_spec_decode=False decision onto each
            #       scheduled request so the forward / sample paths skip
            #       the speculation branches.
            # Consume: scheduled_batch
            # Produce: per-request py_disable_speculative_decoding (out-of-band)
            for request in scheduled_batch.all_requests():
                request.py_disable_speculative_decoding = True

        if self.kv_cache_transceiver:
            # Concern: resource (+ disagg)
            # Task: for newly-fitting disagg gen-init requests, prepare
            #       KV cache / spec / draft resources, then submit the
            #       async KV recv to the ctx worker. Same `resource`
            #       concern as `prepare_resources` later in P_SCHEDULE
            #       (both allocate KV) but for a different request set.
            # Consume: fitting_disagg_gen_init_requests
            # Produce: KV blocks, resource state, recv handles
            #          (out-of-band)
            # For requests that are fitting disagg gen init, also prepare resources for KV cache manager
            self._prepare_disagg_gen_init(fitting_disagg_gen_init_requests)

            # Concern: disagg (back-pressure probe / benchmark gate)
            # Task: when nothing fit and no gen-init either, decide whether
            #       to block on at-least-1 ctx transfer (gen-first only
            #       requests would deadlock that path), or just opportunistic
            #       cleanup. In benchmark gen-only mode, also check for
            #       stuck disagg-gen-init requests and fail fast.
            # Consume: scheduled_batch, fitting_disagg_gen_init_requests,
            #          num_fitting_reqs
            # Produce: terminated requests / errored responses
            #          (out-of-band); may return (None, None) to break loop.
            all_gen_first = self.active_requests and all(
                req.py_disaggregated_params and req.py_disaggregated_params.
                schedule_style == DisaggScheduleStyle.GENERATION_FIRST
                for req in self.active_requests)
            if num_fitting_reqs == 0 and not fitting_disagg_gen_init_requests:
                if not all_gen_first:
                    logger.warning(
                        "num_fitting_reqs=0 and fitting_disagg_gen_init_requests is empty, may not have enough kvCache"
                    )
                    self._check_disagg_ctx_cache_transfer_status(1)
                elif self.async_transfer_manager.has_any_inflight_requests():
                    # Non-blocking cleanup of completed/timed-out transfers
                    # to free KV blocks. We avoid the blocking check because
                    # gen-first requests may be waiting for peer info (which
                    # would block indefinitely), but completed transfers must
                    # still be reaped so that KV cache can be reclaimed.
                    self._check_disagg_ctx_cache_transfer_status(0)

            # In gen-only benchmark mode, all requests must fit in KV cache
            # simultaneously. If some requests are stuck in INIT state and the
            # scheduler could not allocate KV for any of them, the benchmark
            # will hang forever because in-progress generation requests won't
            # release their KV cache.
            if (self.benchmark_req_queues_size > 0 and not self.is_warmup
                    and not fitting_disagg_gen_init_requests):
                stuck_init_requests = [
                    req for req in self.active_requests
                    if req.is_disagg_generation_init_state
                ]
                # Only fail once all benchmark requests have been fetched
                # so that _handle_errors covers every request and every
                # client receives an error response.
                if (stuck_init_requests and self.num_fetch_requests
                        >= self.benchmark_req_queues_size):
                    error_msg = (
                        f"Insufficient KV cache for gen-only benchmark mode: "
                        f"{len(stuck_init_requests)} request(s) are waiting for "
                        f"KV cache allocation but the scheduler could not fit "
                        f"any of them. Increase free_gpu_memory_fraction or "
                        f"reduce TLLM_BENCHMARK_REQ_QUEUES_SIZE (currently "
                        f"{self.benchmark_req_queues_size}).")
                    logger.error(error_msg)
                    # Fail all active and waiting requests so every
                    # client receives an error instead of hanging.
                    self._handle_errors(error_msg,
                                        requests=self.active_requests)
                    return None, None

        # Concern: schedule (telemetry)
        # Task: stash batch_size for external introspection + log line.
        # Consume: scheduled_batch
        # Produce: self.num_scheduled_requests (loop state)
        self.num_scheduled_requests = scheduled_batch.batch_size
        logger.debug(
            f'has {len(self.active_requests)} active_requests, '
            f'scheduled {scheduled_batch.num_context_requests} context requests and '
            f'{scheduled_batch.num_generation_requests} generation requests')
        return scheduled_batch, iter_stats

    # REFACTOR: 2a (concern: kv_connector) -- start async KV-load at
    # SCHEDULE_0 / FORWARD_2.
    def _kv_connector_start_batch(self, scheduled_batch):
        if self.kv_connector_manager:
            self.kv_connector_manager.take_scheduled_requests_pending_load(
                scheduled_batch)
            self.kv_connector_manager.handle_metadata()
            self.kv_connector_manager.worker.start_load_kv(
                torch.cuda.current_stream())

    # REFACTOR: 2a (concern: kv_connector) -- end-of-iter terminate of
    # connector-flagged finished requests.
    def _kv_connector_terminate_requests(self):
        if self.kv_connector_manager:
            reqs_to_terminate = self.kv_connector_manager.get_finished()
            for req in reqs_to_terminate:
                self._end_transfer_and_maybe_terminate(req)

    # REFACTOR: 2a (concern: kv_connector) -- deadline-wait paired with
    # ``_kv_connector_start_batch``; called from inside ``_forward_step``.
    def _kv_connector_wait_for_save(self):
        if self.kv_connector_manager is not None:
            self.kv_connector_manager.worker.wait_for_save(
                torch.cuda.current_stream())

    # REFACTOR: 2a (concern: benchmark_disagg_gate) -- helper for the gate.
    def _is_benchmark_disagg_fill_complete(
            self, scheduled_batch: ScheduledRequests) -> bool:
        """Check whether all benchmark disagg requests have completed KV transfer.

        With ADP, generation requests are distributed across TP ranks, so an
        allgather is needed to obtain the global count.  Without ADP every
        request is local, and we can compare directly.

        This method must only be called when ``is_benchmark_disagg`` is True.

        Args:
            scheduled_batch: The current iteration's scheduled requests,
                used to count generation requests that have completed
                KV transfer.

        Returns:
            True when the total number of generation-ready requests
            reaches ``benchmark_req_queues_size``.
        """
        if not self.is_benchmark_disagg:
            raise RuntimeError(
                "_is_benchmark_disagg_fill_complete() should not be called outside benchmark "
                "disagg mode.  This is an unexpected error.")
        local_gen_count = sum(1 for req in scheduled_batch.generation_requests
                              if not req.is_attention_dp_dummy)
        if self.enable_attention_dp:
            total_gen_count = sum(self.dist.tp_allgather(local_gen_count))
        else:
            total_gen_count = local_gen_count

        if total_gen_count >= self.benchmark_req_queues_size:
            return True
        if self.dist.rank == 0:
            logger.debug(
                f"Benchmark disagg fill in progress: "
                f"num_fetched={self.num_fetch_requests}, "
                f"total_gen_count={total_gen_count} (local={local_gen_count})")
        return False

    # REFACTOR: 2a (concern: benchmark_disagg_gate) -- SCHEDULE_0 gate.
    def _check_benchmark_disagg_gate(self, scheduled_batch: ScheduledRequests,
                                     can_forward: bool) -> tuple[bool, bool]:
        """Gate the forward pass until all benchmark disagg requests are ready.

        In benchmark disagg mode the GEN executor must defer the forward
        pass until every request has completed KV transfer.  This helper
        consolidates the check used by both ``_executor_loop`` and
        ``_executor_loop_overlap``.

        A short sleep (0.1s) yields the CPU between retries while
        keeping the polling interval short enough to avoid KV transfer
        backpressure on the CTX server.

        Args:
            scheduled_batch: The current scheduled batch.
            can_forward: Current gate state.

        Returns:
            ``(can_forward, should_retry)`` — when *should_retry* is True
            the caller should ``continue`` to the next loop iteration.
        """
        if not self.is_warmup and not can_forward:
            can_forward = self._is_benchmark_disagg_fill_complete(
                scheduled_batch)
            if can_forward:
                self._benchmark_fill_phase_active = False
            else:
                time.sleep(0.1)
                return can_forward, True
        return can_forward, False

    # REFACTOR: 1b (legacy loop runtime) -- legacy plain executor loop;
    # replaced by the new coroutine SCHEDULER's plain variant.
    def _executor_loop(self):
        # ===================================================================
        # 3-LAYER COROUTINE REFACTOR ROADMAP -- PLAIN LOOP
        # See `coroutines.py` + `batch_storage.py` for the target runtime
        # (Concern: `enter_phase` / Batch: `batch_phase` + `resume` /
        #  Scheduler: `step`). After untanglement, every coroutine body
        # is single-topic and linear.
        # ===================================================================
        #
        # Batch perspective -- lifecycle of ONE batch (span = 1 iter)
        # -------------------------------------------------------------------
        #   P_SCHEDULE -> P_FORWARD -> P_SAMPLE -> P_APPLY -> P_RESPOND
        #     P_SCHEDULE : `_prepare_and_schedule_batch`, resource prep,
        #                  first-token response, `_kv_connector_start_batch`
        #     P_FORWARD  : `_forward_step` (queues GPU forward; draft prep
        #                  + guided decoder run within)
        #     P_SAMPLE   : `_sample_async` (records `sampler_event`)
        #     P_APPLY    : `_update_request_states` then `_update_requests`
        #                  -- BLOCKS on sampler_event before applying
        #                  decoded tokens to LlmRequests
        #     P_RESPOND  : `_send_kv_async`, `_handle_canceled_requests`,
        #                  `_handle_responses` (delivers to user),
        #                  `resource_manager.update_resources`,
        #                  `_kv_connector_terminate_requests`
        # The batch is born at P_SCHEDULE and retired at P_RESPOND of the
        # SAME iteration.
        #
        # Scheduler perspective -- per-iter work (1 batch in flight)
        # -------------------------------------------------------------------
        # No inter-batch interleaving. Pseudo-refactored body:
        #
        #     async def scheduler_iter():
        #         handle = Batch(batch_coro(), BatchStorage())
        #         await step(handle, through=P_RESPOND)
        #
        # Thread-elimination policy (shared with all `_executor_loop*` variants)
        # -------------------------------------------------------------------
        # The refactor removes application-owned auxiliary threads. Their
        # work moves onto the main loop coroutine via a uniform
        # "submit -> opportunistic probe -> wait at deadline" pattern:
        #   submit : non-blocking primitive returning a handle (e.g.,
        #            `MPI.Comm.Irecv` / `Isend`, `cudaEventRecord`,
        #            KV transceiver `start_transfer`).
        #   probe  : non-blocking query at iter top to drive library
        #            progress (`MPI.Request.Test`, `cudaEventQuery`,
        #            `ucp_worker_progress`). Cheap; lets a concern
        #            coroutine handle results early when ready.
        #   wait   : the coroutine block-waits at the deadline phase
        #            (`MPI.Request.Wait`, `cudaEventSynchronize`,
        #            `end_transfer`).
        # Exception: operations that are fundamentally sync-blocking with
        # no async API (rare) are wrapped in one shared
        # `ThreadPoolExecutor` that re-shapes them into submit+wait. From
        # the coroutine layer they look identical to native async ops; the
        # thread pool is an implementation detail of those concerns.
        # Library-internal threads (MPI / NIXL / UCX / NCCL progress) are
        # untouched -- we only eliminate threads we own.
        #
        # Threads removed under this policy on this loop variant:
        #   * Sampler `_async_worker` (the `ThreadPoolExecutor` started by
        #     `start_worker` for `AsyncWorkerMixin` samplers): the D2H
        #     copies of sample-state tensors are already CUDA-async, so
        #     pair `cudaEventRecord` (submit at P_SAMPLE) with
        #     `cudaEventSynchronize` (wait at P_APPLY of the SAME iter).
        #     Any genuinely sync CPU bookkeeping that remains lives in the
        #     shared exception thread pool.
        # ===================================================================
        # ===================================================================
        # CONCERN / PHASE ANNOTATIONS (round 1: comments only)
        # -------------------------------------------------------------------
        # Each codeblock below is tagged with:
        #   Concern : domain coroutine the code belongs to (`schedule`,
        #             `disagg`, `kv_connector`, `resource`, `spec_decode`,
        #             `guided_decoder`, `forward`, `sample`, `response`,
        #             `perf_metric`, `iter_stats`, `dwdp`, `kv_cache_events`,
        #             `save_hidden_states`, `hang`, `profile`, `control`,
        #             `benchmark_disagg_gate`, `loop_control`).
        #   Task    : 1-line description of what this block does.
        #   Consume : per-batch (BatchStorage) fields read here. "-" means
        #             the block only touches loop-/global-state, not batch
        #             storage.
        #   Produce : per-batch fields written here. "out-of-band" notes
        #             writes that escape the batch (responses, GPU events,
        #             telemetry, request-state mutations on LlmRequest).
        #
        # Phases (data-dependency DAG; see roadmap above):
        #   P_SCHEDULE -> P_FORWARD -> P_SAMPLE -> P_APPLY -> P_RESPOND
        # The double duty of `BatchPhase` (rendezvous for active-batch
        # switching + happens-before barrier) only matters for the overlap
        # / PP loops; for `_executor_loop` there's a single in-flight batch
        # so phase markers here purely serve as the data-flow barrier.
        # No reorders were required: the existing line order is already a
        # valid topological sort of the concern DAG.
        # ===================================================================
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step, self.hang_detector:
            sample_state = None
            iter_start_time = time.time()
            iter_stats = None
            can_forward = not self.is_benchmark_disagg
            while True:
                # =========================================================
                # ===== PHASE: P_SCHEDULE =================================
                # =========================================================

                # Concern: hang
                # Task: progress watchdog tick (warns / aborts on stall).
                # Consume: -
                # Produce: -
                self.hang_detector.checkpoint()

                # Concern: profile
                # Task: drive torch / CUDA-event profiler state machine.
                # Consume: -
                # Produce: profiler artifacts (out-of-band)
                profile_step()

                # Concern: iter_stats
                # Task: stamp wall-clock start of this iter (used at
                #       P_RESPOND to compute iter_latency_ms).
                # Consume: -
                # Produce: iter_start_time
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                # Concern: schedule (+ disagg + iter_stats + spec_decode + resource)
                # Task: fat fan-out call. See annotations on
                #       `_prepare_and_schedule_batch` for the per-concern
                #       breakdown. High-level effect: fetch new requests,
                #       drive disagg KV-transceiver probes, decide
                #       use_spec_decode, run scheduler, allocate resources
                #       for fitting disagg-gen-init requests, init iter_stats.
                #       Returns (None, None) to signal shutdown.
                # Consume: -
                # Produce: scheduled_batch, iter_stats; mutates
                #          self.active_requests, self.use_spec_decode,
                #          self.max_total_draft_tokens
                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()

                # Concern: control
                # Task: pause loop if a control request is pending so an
                #       external thread can mutate executor state safely.
                # Consume: -
                # Produce: -
                self._handle_control_request()

                # Concern: loop_control
                # Task: shutdown rendezvous — `scheduled_batch is None`
                #       means `_prepare_and_schedule_batch` saw
                #       `should_stop_processing` (or the benchmark-disagg
                #       error path).
                # Consume: scheduled_batch
                # Produce: -
                if scheduled_batch is None:
                    break

                # Concern: benchmark_disagg_gate
                # Task: in disagg-gen benchmark mode, gate the forward
                #       pass until every gen request has finished its KV
                #       transfer. `should_retry` skips the iter (sleep+continue).
                # Consume: scheduled_batch
                # Produce: can_forward (loop state across iters)
                can_forward, should_retry = self._check_benchmark_disagg_gate(
                    scheduled_batch, can_forward)
                if should_retry:
                    continue

                # Concern: schedule (paused-request lifecycle)
                # Task: terminate / re-pause requests preempted by the V1
                #       scheduler. V2 manages KV suspend internally so the
                #       block is gated on `_scheduler_manages_kv_suspend`.
                # Consume: scheduled_batch.paused_requests
                # Produce: terminated/paused requests (out-of-band)
                if not self._scheduler_manages_kv_suspend:
                    self._terminate_requests(scheduled_batch.paused_requests)
                    self._pause_requests(scheduled_batch.paused_requests)

                # Concern: response (init slot)
                # Task: default the per-iter finished-request list so the
                #       trailing iter_stats block can read it whether or
                #       not the can_queue branch ran.
                # Consume: -
                # Produce: finished_requests (initial empty)
                finished_requests = []

                # Concern: schedule (collective queue gate)
                # Task: collective check that every TP rank has a non-empty
                #       batch — mismatched ranks would hang on collectives
                #       inside `_forward_step`.
                # Consume: scheduled_batch.batch_size
                # Produce: can_queue
                can_queue, _ = self._can_queue(scheduled_batch)

                if can_queue:
                    if self.kv_cache_transceiver:
                        # Concern: disagg
                        # Task: promote DISAGG_GENERATION_TRANS_COMPLETE
                        #       gen requests to GENERATION_IN_PROGRESS,
                        #       prepare seq-slot / sampler step, prepend
                        #       first_gen logprobs+logits from prefill.
                        # Consume: scheduled_batch.generation_requests
                        # Produce: per-request state (out-of-band on requests)
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                        # Concern: response (first-token)
                        # Task: emit the *first* token response for newly
                        #       promoted gen requests so the disagg client
                        #       can start streaming before the first
                        #       forward pass on this worker.
                        #       Note: a `response` step at P_SCHEDULE,
                        #       paired with the main `response` work at
                        #       P_RESPOND — the same concern split across
                        #       phases, motivating a single coroutine.
                        # Consume: scheduled_batch.generation_requests
                        # Produce: enqueued first-token responses (out-of-band)
                        self._handle_first_token_response(scheduled_batch)

                    # Concern: spec_decode (dynamic draft length)
                    # Task: pad/truncate every gen request's
                    #       py_draft_tokens to a uniform draft length so
                    #       CUDA-graph capture works; pin
                    #       model_engine.runtime_draft_len.
                    # Consume: scheduled_batch (batch_size, gen requests)
                    # Produce: per-request py_draft_tokens (out-of-band),
                    #          model_engine.runtime_draft_len (engine state)
                    self._handle_dynamic_draft_len(scheduled_batch)

                    # Concern: resource
                    # Task: allocate KV blocks + per-resource-manager state
                    #       for the scheduled batch (paired with
                    #       `update_resources` at P_RESPOND).
                    # Consume: scheduled_batch
                    # Produce: KV blocks, resource-manager state (out-of-band)
                    self.resource_manager.prepare_resources(scheduled_batch)

                if self.kv_connector_manager:
                    # Concern: kv_connector
                    # Task: refresh connector metadata (block layouts,
                    #       remote topology) — must run every iter.
                    # Consume: -
                    # Produce: connector metadata (out-of-band)
                    self.kv_connector_manager.handle_metadata()

                if can_queue:
                    # Concern: kv_connector
                    # Task: queue async KV-load operations on the current
                    #       stream for the scheduled batch. The wait
                    #       (`wait_for_save`) happens implicitly inside
                    #       `_forward_step` at P_FORWARD — same concern,
                    #       split across two phases.
                    # Consume: scheduled_batch
                    # Produce: connector load ops (out-of-band)
                    self._kv_connector_start_batch(scheduled_batch)

                # if using a kv connector, we need to call can_queue again since scheduled_batch might have changed
                if self.kv_connector_manager:
                    # Concern: schedule (collective queue re-check)
                    # Task: re-evaluate cross-rank queue consensus —
                    #       `take_scheduled_requests_pending_load` may have
                    #       removed/added requests, so the previous
                    #       can_queue is potentially stale.
                    # Consume: scheduled_batch.batch_size
                    # Produce: can_queue
                    can_queue, _ = self._can_queue(scheduled_batch)

                if not can_queue:
                    # Concern: resource (revert)
                    # Task: undo V2 scheduler's per-gen KV growth when the
                    #       forward is skipped — without this the host
                    #       page-index buffer overflows after enough
                    #       skipped iters.
                    # Consume: scheduled_batch.generation_requests
                    # Produce: KV cache state (out-of-band)
                    self._revert_gen_alloc(scheduled_batch)

                if can_queue:
                    # =====================================================
                    # ===== PHASE: P_FORWARD ==============================
                    # =====================================================

                    # init_disagg_gen_requests must be before drafter loop, otherwise draft requests do not have initialized matchers.
                    # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                    if self.guided_decoder is not None:
                        # Concern: guided_decoder
                        # Task: register the batch with the constraint
                        #       matcher; init disagg-gen request matchers.
                        #       Must run before the drafter (drafter needs
                        #       valid matchers) and before engine forward
                        #       (forward updates prev_seq_slot).
                        # Consume: scheduled_batch
                        # Produce: matcher per-batch state (out-of-band)
                        self.guided_decoder.add_batch(scheduled_batch)
                        if self.kv_cache_transceiver:
                            self.guided_decoder.init_disagg_gen_requests()

                    if self.drafter is not None and self.use_spec_decode:
                        # Concern: spec_decode (drafter forward)
                        # Task: roll back rejected draft tokens, run the
                        #       *draft* model on execution_stream, pad
                        #       resulting draft tokens for CUDA-graph,
                        #       re-register the batch with guided_decoder
                        #       (draft tokens are now the real ones), and
                        #       roll back draft-token mask. The
                        #       guided_decoder calls are interleaved
                        #       because the matcher state follows the
                        #       drafter's rewriting of py_draft_tokens.
                        # Consume: scheduled_batch
                        # Produce: per-request py_draft_tokens (out-of-band),
                        #          guided_decoder matcher state
                        if self.guided_decoder is not None:
                            self.guided_decoder.rollback_rejected_tokens()
                        with request_context(
                                is_draft=self.draft_model_engine is not None,
                                scheduled_requests=scheduled_batch):
                            self.execution_stream.wait_stream(
                                torch.cuda.current_stream())
                            with torch.cuda.stream(self.execution_stream):
                                self.drafter.prepare_draft_tokens(
                                    scheduled_batch, self.resource_manager)
                                # Pad draft tokens to the max draft length and extend KV cache
                                # capacity to match. This is for CUDA graph compatibility.
                                self.drafter.pad_draft_tokens_for_cuda_graph(
                                    scheduled_batch, self.resource_manager)
                            torch.cuda.current_stream().wait_stream(
                                self.execution_stream)
                        # add_batch must be called again to restore to target requests with updated draft tokens.
                        if self.guided_decoder is not None:
                            self.guided_decoder.add_batch(scheduled_batch)
                            if hasattr(self.drafter, "guided_decoder"):
                                self.guided_decoder.rollback_draft_tokens()

                    # Concern: perf_metric
                    # Task: allocate three CUDA event handles used to
                    #       bracket the forward+sample timing window.
                    #       Same concern as `record_perf_events` /
                    #       `save_timing_to_requests` /
                    #       `compute_batch_gpu_times` — single
                    #       perf_metric coroutine, four phases.
                    # Consume: -
                    # Produce: gpu_forward_start, gpu_forward_end,
                    #          gpu_sample_end (CUDA events)
                    gpu_forward_start, gpu_forward_end, gpu_sample_end = self.perf_manager.create_timing_events(
                    )

                    with self.perf_manager.record_perf_events(
                            gpu_forward_start, gpu_forward_end) as fwd_timing:
                        if self.dwdp_manager is not None:
                            # Concern: dwdp
                            # Task: prefetch the first MoE expert layers'
                            #       weights so they overlap with attention
                            #       compute on the forward critical path.
                            # Consume: -
                            # Produce: H2D weight transfers (out-of-band)
                            self.dwdp_manager.prefetch_first_layers()
                        # Concern: forward (+ kv_connector wait_for_save)
                        # Task: queue model forward on execution_stream;
                        #       internally also calls
                        #       `_kv_connector_wait_for_save` which is the
                        #       deadline-wait counterpart of
                        #       `_kv_connector_start_batch` at P_SCHEDULE.
                        # Consume: scheduled_batch, prepared resources,
                        #          py_draft_tokens, runtime_draft_len
                        # Produce: batch_outputs (logits + extras),
                        #          fwd_timing (CPU ts), forward CUDA events
                        batch_outputs = self._forward_step(scheduled_batch)

                    # =====================================================
                    # ===== PHASE: P_SAMPLE ===============================
                    # =====================================================

                    # Concern: guided_decoder
                    # Task: apply the constraint matcher to logits — masks
                    #       disallowed tokens and flags requests that
                    #       violated their grammar. Must run before
                    #       _sample_async (in same P_SAMPLE) so masking
                    #       affects sampling.
                    # Consume: batch_outputs (logits)
                    # Produce: guided_decoder_failed_requests, masked logits
                    guided_decoder_failed_requests = None
                    if self.guided_decoder is not None:
                        guided_decoder_failed_requests = self.guided_decoder.execute(
                            batch_outputs['logits'])

                    with self.perf_manager.record_perf_events(
                            None, gpu_sample_end) as sample_timing:
                        # Concern: sample
                        # Task: run HandleLogits / HandleAdditionalOutputs,
                        #       queue the sampling kernel, queue the D2H
                        #       copy of sample-state tensors, record
                        #       sampler_event for the P_APPLY sync.
                        # Consume: scheduled_batch, batch_outputs (logits)
                        # Produce: sample_state (with sampler_event +
                        #          .device tensors), sample_timing
                        sample_state = self._sample_async(
                            scheduled_batch, batch_outputs)

                    # Concern: perf_metric
                    # Task: snapshot CUDA event handles + CPU
                    #       start/end timestamps onto each request's perf
                    #       record. Events are still un-synced; the actual
                    #       times are computed at P_RESPOND
                    #       (`compute_batch_gpu_times`).
                    # Consume: scheduled_batch, gpu_*_start/_end (events),
                    #          fwd_timing, sample_timing
                    # Produce: per-request perf records (out-of-band)
                    self.perf_manager.save_timing_to_requests(
                        scheduled_batch.all_requests(), gpu_forward_start,
                        gpu_forward_end, gpu_sample_end, fwd_timing.start_time,
                        fwd_timing.end_time, sample_timing.start_time,
                        sample_timing.end_time)

                    # =====================================================
                    # ===== PHASE: P_APPLY ================================
                    # =====================================================

                    # Handle guided decoder errors after _sample_async to avoid state conflicts.
                    # If called before, failed requests would be marked as GENERATION_COMPLETE,
                    # causing _sample_async to fail when accessing context_chunk_size property.

                    # Concern: guided_decoder
                    # Task: mark grammar-failed requests as errored.
                    #       Strict ordering constraint: must run AFTER
                    #       _sample_async (else GENERATION_COMPLETE
                    #       breaks sample's context_chunk_size access)
                    #       but BEFORE _update_requests writes tokens.
                    # Consume: scheduled_batch,
                    #          guided_decoder_failed_requests
                    # Produce: failed-request error responses (out-of-band)
                    self._handle_guided_decoder_errors(
                        scheduled_batch, guided_decoder_failed_requests)

                    # Handle SaveHiddenStates mode - save hidden states after forward
                    if not self.is_warmup:
                        # Concern: save_hidden_states
                        # Task: persist hidden states captured during
                        #       forward (eagle3 / EAGLE training data
                        #       capture). Reads model_engine.spec_metadata
                        #       which is written by `_forward_step`.
                        # Consume: scheduled_batch,
                        #          model_engine.spec_metadata (engine state)
                        # Produce: hidden-state files (out-of-band)
                        spec_resource_mgr = self.resource_manager.resource_managers.get(
                            ResourceManagerType.SPEC_RESOURCE_MANAGER)
                        if spec_resource_mgr is not None and hasattr(
                                spec_resource_mgr, 'process_and_save'):
                            spec_metadata = getattr(self.model_engine,
                                                    'spec_metadata', None)
                            spec_resource_mgr.process_and_save(
                                scheduled_batch, spec_metadata)

                    # Concern: sample (state advance)
                    # Task: advance context_chunk_position; transition
                    #       finished-context requests to
                    #       GENERATION_IN_PROGRESS / GENERATION_TO_COMPLETE;
                    #       drop the ADP dummy request. Independent of
                    #       sample_state — only needs scheduled_batch —
                    #       so could in principle run at P_SAMPLE.
                    #       Kept at P_APPLY here because the overlap loop
                    #       has a separate P_STATE_UPD slot for it; the
                    #       plain loop folds it into P_APPLY.
                    # Consume: scheduled_batch
                    # Produce: per-request state (out-of-band)
                    self._update_request_states(scheduled_batch)

                    # Concern: sample (apply tokens)
                    # Task: BLOCK on sampler_event, then call
                    #       sampler.update_requests to write the sampled
                    #       tokens onto each request. This is the only
                    #       point in the loop that synchronously waits on
                    #       the sample-state D2H copy.
                    # Consume: sample_state (sampler_event +
                    #          .host tensors)
                    # Produce: per-request new tokens, py_decoding_iter
                    #          (out-of-band)
                    self._update_requests(sample_state, self.resource_manager)

                    # =====================================================
                    # ===== PHASE: P_RESPOND ==============================
                    # =====================================================

                    # Concern: disagg (+ kv_connector)
                    # Task: for finished context-only requests, start the
                    #       async KV send to the gen worker AND respond;
                    #       also lets kv_connector flag-finished requests
                    #       start their async transfer. Probes
                    #       _check_disagg_ctx_cache_transfer_status(0) at
                    #       the end (opportunistic completion sweep).
                    #       Multi-concern function — see annotations on
                    #       `_send_kv_async`.
                    # Consume: scheduled_batch, request finish state
                    # Produce: KV send handles (out-of-band), terminated
                    #          context-only requests
                    self._send_kv_async(scheduled_batch.all_requests())
                    self._flush_pending_transfer_responses()

                    # Concern: response (cancellation)
                    # Task: terminate canceled requests if possible
                    #       (deferred when an in-progress KV transfer
                    #       blocks termination).
                    # Consume: self.canceled_req_ids (external),
                    #          self.active_requests (loop state)
                    # Produce: cancellation responses + termination
                    #          (out-of-band)
                    self._handle_canceled_requests()

                    # Concern: response (+ perf_metric step + spec_decode gating)
                    # Task: build per-request responses, enqueue, terminate
                    #       finished requests. ALSO appends step-level
                    #       perf metrics (perf_metric concern leaking in)
                    #       and updates speculation_gate rolling
                    #       acceptance (spec_decode concern leaking in).
                    #       See annotations inside `_handle_responses`.
                    # Consume: self.active_requests, scheduled_batch,
                    #          per-request py_decoding_iter / draft_tokens
                    # Produce: finished_requests, responses (out-of-band),
                    #          self.speculation_permanently_disabled
                    #          (loop state)
                    finished_requests = self._handle_responses()
                    # Complete ctx send sessions AFTER responses are created so
                    # _handle_responses sees the request before it is terminated.

                    # Concern: disagg
                    # Task: opportunistic probe of ctx-side KV-transfer
                    #       completion; terminates ctx-only requests whose
                    #       KV blocks have finished sending. Strict
                    #       ordering: AFTER _handle_responses so the
                    #       response sees the request before termination.
                    #       Same concern as `_send_kv_async` and the
                    #       `_check_disagg_*` calls at P_SCHEDULE — one
                    #       coroutine, four phases.
                    # Consume: -
                    # Produce: terminated requests (out-of-band)
                    if self.kv_cache_transceiver:
                        self._check_disagg_ctx_cache_transfer_status(0)
                    # Compute GPU times after _handle_responses creates metric entries
                    # (safe in non-overlap mode: no next iteration to overwrite events)

                    # Concern: perf_metric
                    # Task: read forward/sample CUDA events into per-request
                    #       metric records. Only safe in non-overlap mode
                    #       — the overlap loop must defer this to next
                    #       iter to avoid event reuse races.
                    # Consume: scheduled_batch, gpu_*_start/_end (events)
                    # Produce: per-request gpu timings (out-of-band)
                    self.perf_manager.compute_batch_gpu_times(
                        scheduled_batch.all_requests())

                    # Concern: resource
                    # Task: free per-request KV / resource state for
                    #       finished requests; rebalance attention
                    #       metadata bookkeeping. Paired with
                    #       `prepare_resources` at P_SCHEDULE.
                    # Consume: scheduled_batch,
                    #          model_engine.attn_metadata (engine state)
                    # Produce: KV blocks freed, resource-manager state
                    #          (out-of-band)
                    attn_metadata = getattr(self.model_engine, 'attn_metadata',
                                            None)
                    kv_cache_dtype_byte_size = getattr(
                        self.model_engine, 'kv_cache_dtype_byte_size', None)
                    self.resource_manager.update_resources(
                        scheduled_batch, attn_metadata,
                        kv_cache_dtype_byte_size)
                    if self.enable_kv_cache_events:
                        # Concern: kv_cache_events
                        # Task: drain queued KV-cache events into the
                        #       iter-event buffer so user pollers can see
                        #       them after this iter.
                        # Consume: -
                        # Produce: kv-cache event records (out-of-band)
                        self._add_kv_cache_events()

                # NOTE: blocks below run unconditionally (whether or not
                # can_queue was True) — they are still P_RESPOND in the
                # data-flow DAG but live outside the can_queue branch.

                if self.kv_cache_transceiver and self.async_transfer_manager.has_any_inflight_requests(
                ):
                    # Concern: disagg
                    # Task: end-of-iter timeout sweep over inflight async
                    #       KV transfers; flips py_kv_transfer_timed_out
                    #       on requests whose transfer exceeded the
                    #       configured deadline.
                    # Consume: -
                    # Produce: per-request py_kv_transfer_timed_out
                    #          (out-of-band)
                    self._check_kv_transfer_timeout()

                # Concern: kv_connector
                # Task: terminate connector-flagged finished requests —
                #       paired with `_kv_connector_start_batch` at
                #       P_SCHEDULE (start vs terminate, two phases of
                #       one concern).
                # Consume: -
                # Produce: terminated requests (out-of-band)
                self._kv_connector_terminate_requests()

                if self.enable_iter_perf_stats and sample_state is not None:
                    # Concern: iter_stats
                    # Task: snapshot model_engine.iter_states.num_ctx_tokens
                    #       and run end-of-iter stats aggregation. Paired
                    #       with the iter_start_time stamp at P_SCHEDULE
                    #       and the `_get_init_iter_stats` call inside
                    #       `_prepare_and_schedule_batch`.
                    # Consume: scheduled_batch, sample_state,
                    #          finished_requests, iter_start_time,
                    #          iter_stats
                    # Produce: aggregated iter_stats record (out-of-band)
                    iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                        'num_ctx_tokens']
                    self._process_iter_stats(
                        finished_requests, self.active_requests,
                        BatchState(scheduled_requests=scheduled_batch,
                                   sample_state=sample_state,
                                   iter_stats=iter_stats,
                                   iter_start_time=iter_start_time))

                # Concern: loop_control
                # Task: advance the global iteration counter (used by
                #       next iter's profilers, perf metric keys, log
                #       prefixes).
                # Consume: -
                # Produce: iter_counter (loop state)
                self.iter_counter += 1

    # REFACTOR: 2a (concern: spec_decode) -- SCHEDULE_0 helper.
    def _prepare_draft_requests(self):
        try:
            # Set draft tokens here to make the KV cache manager
            # and scheduler aware of them.
            for req in self.active_requests:
                if req.state not in (LlmRequestState.GENERATION_IN_PROGRESS,
                                     LlmRequestState.DISAGG_GENERATION_INIT):
                    continue

                req.py_last_draft_tokens = req.py_draft_tokens

                if self.max_total_draft_tokens > 0 and self.use_spec_decode and not req.py_disable_speculative_decoding:
                    req.py_draft_tokens = [0] * self.max_total_draft_tokens
                    req.py_draft_pages_allocated = self.max_total_draft_tokens
                else:
                    req.py_draft_tokens = []
                    req.py_draft_pages_allocated = 0

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            self._handle_errors(error_msg)

    # REFACTOR: 2a (concern: control) -- SCHEDULE_0 control-request
    # rendezvous (cooperates with main-thread ``control_action`` CM).
    def _handle_control_request(self):
        if len(self.active_requests) == 0 and \
            len(self.waiting_queue) == 0 and \
            len(self.control_requests) > 0:
            assert len(self.control_requests) == 1, (
                f"Expected exactly one control request to be processed at a time, "
                f"but found {len(self.control_requests)} control requests. "
                f"This may indicate a race condition or improper control request handling."
            )
            self.control_requests.pop(0)
            self.control_request_barrier.set()
            self.control_action_done.wait()
            self.control_action_done.clear()

    # REFACTOR: 1a (public API) -- main-thread CM that cooperates with
    # the loop-thread ``_handle_control_request`` (control concern) via
    # ``MessagePort`` queues / events.
    @contextmanager
    def control_action(self):
        """
        Context manager for synchronized control actions.

        Usage:
            with control_action():
                # Eventloop thread has finished all previous requests and paused
                do some actions here
            # Eventloop thread resumes automatically after exiting
        """

        if self.dist.rank == 0:
            self.executor_request_queue.enqueue_control_request()

        # Wait for worker to finish all previous requests
        self.control_request_barrier.wait()

        try:
            # Yield control to the with block
            # Worker is now paused, safe to execute actions
            yield self
        finally:
            # Cleanup: signal worker to resume
            self.control_action_done.set()
            self.control_request_barrier.clear()

    # REFACTOR: 1b (legacy loop runtime) -- legacy overlap executor loop;
    # replaced by the new coroutine SCHEDULER's overlap variant.
    def _executor_loop_overlap(self):
        # ===================================================================
        # 3-LAYER COROUTINE REFACTOR ROADMAP -- OVERLAP LOOP
        # See `coroutines.py` + `batch_storage.py` for the target runtime.
        # ===================================================================
        # Same three layers as the plain loop. This loop deliberately splits
        # each batch's lifecycle across TWO iters so the GPU forward of
        # `mb_t` overlaps the CPU bookkeeping of `mb_(t-1)`. The cross-iter
        # state is `self.previous_batch`; `_forward_step` reads `mb_(t-1)`'s
        # GPU sample tensors directly via `previous_tensors_device`, so
        # nothing on the critical path waits for the previous sampler event.
        #
        # Batch perspective -- lifecycle of ONE batch (span = 2 iters)
        # -------------------------------------------------------------------
        #   Iter t (creation):
        #     P_SCHEDULE  : `_prepare_and_schedule_batch`, resource prep
        #     P_FORWARD   : `_forward_step(scheduled_batch,
        #                                  previous_tensors_device)`
        #     P_SAMPLE    : `_sample_async`
        #     P_STATE_UPD : `_update_request_states`
        #
        #   Iter t+1 (retirement -- the loop body sees this batch as
        #             `self.previous_batch`):
        #     P_APPLY     : `_update_requests(mb_t.sample_state)` -- BLOCKS
        #                   on mb_t's sampler_event, then `_send_kv_async`
        #     P_RESPOND   : `_process_previous_batch` (responses + perf,
        #                   `_kv_connector_terminate_requests`)
        # First iter queues GPU work and stashes the batch as
        # `previous_batch`; second iter syncs and retires it.
        #
        # Scheduler perspective -- per-iter work (2 batches, interleaved 4x)
        # -------------------------------------------------------------------
        # `mb_t` (newly forwarded) and `mb_(t-1)` (about to retire) are
        # interleaved because `_update_requests(prev)` must run AFTER
        # `_forward_step(curr)` is queued (so its sampler_event sync
        # overlaps the GPU forward of mb_t) but BEFORE `_sample_async(curr)`
        # (which mutates the default stream). Pseudo-refactored body:
        #
        #     async def scheduler_iter():
        #         await step(handle_t,    through=P_FORWARD)
        #         await step(handle_prev, through=P_APPLY)
        #         await step(handle_t,    through=P_STATE_UPD)
        #         await step(handle_prev, through=P_RESPOND)   # terminal
        #         handle_prev = handle_t                        # promote
        #
        # Thread-elimination policy (shared with all `_executor_loop*` variants)
        # -------------------------------------------------------------------
        # The refactor removes application-owned auxiliary threads. Their
        # work moves onto the main loop coroutine via a uniform
        # "submit -> opportunistic probe -> wait at deadline" pattern:
        #   submit : non-blocking primitive returning a handle (e.g.,
        #            `MPI.Comm.Irecv` / `Isend`, `cudaEventRecord`,
        #            KV transceiver `start_transfer`).
        #   probe  : non-blocking query at iter top to drive library
        #            progress (`MPI.Request.Test`, `cudaEventQuery`,
        #            `ucp_worker_progress`). Cheap; lets a concern
        #            coroutine handle results early when ready.
        #   wait   : the coroutine block-waits at the deadline phase
        #            (`MPI.Request.Wait`, `cudaEventSynchronize`,
        #            `end_transfer`).
        # Exception: operations that are fundamentally sync-blocking with
        # no async API (rare) are wrapped in one shared
        # `ThreadPoolExecutor` that re-shapes them into submit+wait. From
        # the coroutine layer they look identical to native async ops; the
        # thread pool is an implementation detail of those concerns.
        # Library-internal threads (MPI / NIXL / UCX / NCCL progress) are
        # untouched -- we only eliminate threads we own.
        #
        # Threads removed under this policy on this loop variant:
        #   * Sampler `_async_worker`: the D2H copy of mb_t's sample-state
        #     is already CUDA-async; pair `cudaEventRecord` (submit at
        #     iter t's P_SAMPLE) with `cudaEventSynchronize` (wait at
        #     iter t+1's P_APPLY, where the existing code already
        #     implicitly syncs the sampler event inside `_update_requests`
        #     -- so making the submit + wait explicit is a one-line move,
        #     not a control-flow change).
        # ===================================================================
        # ===================================================================
        # CONCERN / PHASE / HALF-CONCERN ANNOTATIONS (round 1: comments only)
        # -------------------------------------------------------------------
        # Concerns: same set as the plain loop (`schedule`, `disagg`,
        # `kv_connector`, `resource`, `spec_decode`, `guided_decoder`,
        # `forward`, `sample`, `response`, `perf_metric`, `iter_stats`,
        # `dwdp` (not in this loop variant), `kv_cache_events`, `hang`,
        # `profile`, `control`, `benchmark_disagg_gate`, `loop_control`).
        #
        # Per-batch phase order (each batch lives across 2 iters):
        #   P_SCHEDULE -> P_FORWARD -> P_SAMPLE -> P_STATE_UPD
        #     -> P_APPLY -> P_RESPOND
        # The plain loop's P_APPLY/P_STATE_UPD distinction collapses here:
        # in overlap, P_STATE_UPD belongs to iter t (curr's tail) while
        # P_APPLY moves to iter t+1 (prev's head) so the sampler-event
        # sync overlaps the next batch's GPU forward.
        #
        # Scheduler iter pattern (4 step calls + bookkeeping):
        #     async def scheduler_iter():
        #         await step(handle_t,    through=P_FORWARD)    # (1)
        #         await step(handle_prev, through=P_APPLY)      # (2)
        #         await step(handle_t,    through=P_STATE_UPD)  # (3)
        #         await step(handle_prev, through=P_RESPOND)    # (4)
        #         # scheduler-only bookkeeping (perf snapshot,
        #         # iter_stats key, promotion); see HC2 below
        #         handle_prev = handle_t
        # The body below is annotated with the matching `step (N)` slot
        # at every phase boundary.
        #
        # "Half concerns" (scheduler-level cross-batch bridges):
        # The scheduler is the only place that touches BOTH batches'
        # storage; bridge code is split into a *consume half* (reads one
        # batch at some phase) and a *produce half* (writes another
        # batch's input slot at some later phase). One half concern =
        # one logical cross-batch arc.
        #
        #   HC1: prev.P_SAMPLE.sample_state
        #          -> curr.P_FORWARD.previous_tensors_device (+ num_accepted_tokens_device)
        #        Implementation interleaves several lines:
        #          consume:  `previous_tensors = self.previous_batch
        #                     and self.previous_batch.sample_state`
        #          consume:  `use_previous_draft_tokens = self.has_previous_draft_tokens`
        #                    (the cross-iter side-channel set by HC1's draft
        #                    model in the *previous* iter, i.e., the consume
        #                    half of HC3 below)
        #          (in-bridge compute) `_handle_speculative_decoding` runs
        #                    the draft model on prev's sample tensors and
        #                    returns target_inputs / num_accepted_tokens_device
        #          produce:  selects `previous_tensors_device` from
        #                    target_inputs OR prev.sample_state.device
        #          produce:  passes `previous_tensors_device` and
        #                    `num_accepted_tokens_device` into
        #                    `_forward_step(curr)` — write into curr's
        #                    forward-input slot.
        #
        #   HC2 (cross-iter): curr.P_STATE_UPD.{scheduled_requests,
        #          sample_state, iter_stats, iter_start_time}
        #          -> next-iter prev slot (`self.previous_batch`).
        #        consume half (this iter): reads curr's per-batch fields.
        #        produce half (this iter): assigns into self.previous_batch.
        #        The other end of the arc is HC1 in the *next* iter,
        #        which reads self.previous_batch as its consume half.
        #        `self.previous_batch = None` on empty-rank is the
        #        cleanup variant of HC2 (still a produce half — writes
        #        the prev slot, just to None).
        #
        #   HC3 (cross-iter side-channel): curr.P_FORWARD.has_previous_draft_tokens
        #          -> next iter's HC1 `use_previous_draft_tokens` decision.
        #        produce half (this iter, inside `_handle_speculative_decoding`):
        #          sets `self.has_previous_draft_tokens` based on whether
        #          the draft model produced next_draft_tokens.
        #        consume half (next iter): reads `self.has_previous_draft_tokens`
        #          to decide which branch of HC1's selection to take.
        #
        # Cross-batch *ordering* constraints (NOT half concerns — these
        # are placement, not data forwarding through batch storage):
        #   * `_pause_requests(curr.paused_requests)` belongs to curr's
        #     P_SCHEDULE but its execution is deferred until after prev's
        #     P_APPLY (V1 KV handoff). In a coroutine refactor curr's
        #     `schedule` coroutine would yield once between
        #     `_terminate_requests` and `_pause_requests`.
        #   * `_update_generation_requests_that_will_complete_next_iteration`
        #     mutates curr's gen-request state at P_STATE_UPD but must
        #     run AFTER prev's P_RESPOND because prev's `_handle_responses`
        #     reads `exclude_last_generation_logits` while building
        #     responses for shared-by-id requests.
        # ===================================================================
        torch.cuda.set_device(self.device_id)
        # ensure the context is created, otherwise, some MPI calls will fail.
        CUASSERT(cudart.cudaSetDevice(self.device_id))
        with self._profiler() as profile_step, self.hang_detector:
            iter_start_time = time.time()
            iter_stats = None
            target_inputs = None
            previous_tensors_device = None
            can_forward = not self.is_benchmark_disagg
            while True:
                # =========================================================
                # ===== STEP (1) : curr P_SCHEDULE -> P_FORWARD ===========
                # =========================================================

                # ===== curr PHASE: P_SCHEDULE =====

                # Concern: hang
                # Task: progress watchdog tick.
                # Consume: -
                # Produce: -
                self.hang_detector.checkpoint()

                # Concern: profile
                # Task: drive torch / CUDA-event profiler state machine.
                # Consume: -
                # Produce: profiler artifacts (out-of-band)
                profile_step()

                # Concern: iter_stats (curr)
                # Task: stamp wall-clock start of curr's iter t. Read at
                #       prev's P_RESPOND of next iter (inside
                #       `_process_previous_batch` -> `_process_iter_stats`)
                #       — i.e., this is curr's iter_start_time field
                #       carried in HC2 to the next iter.
                # Consume: -
                # Produce: iter_start_time (curr field)
                if self.enable_iter_perf_stats:
                    iter_start_time = time.time()

                # Concern: schedule (+ disagg + iter_stats + spec_decode + resource)
                # Task: same fan-out as plain loop. See annotations on
                #       `_prepare_and_schedule_batch`.
                # Consume: -
                # Produce: scheduled_batch, iter_stats (curr fields);
                #          mutates self.active_requests, self.use_spec_decode
                scheduled_batch, iter_stats = self._prepare_and_schedule_batch()

                # Concern: control
                # Task: pause loop if a control request is pending.
                # Consume: -
                # Produce: -
                self._handle_control_request()

                # Concern: loop_control
                # Task: shutdown rendezvous.
                # Consume: scheduled_batch
                # Produce: -
                if scheduled_batch is None:
                    break

                # Concern: benchmark_disagg_gate
                # Task: gate curr's forward until benchmark fill complete.
                # Consume: scheduled_batch
                # Produce: can_forward (cross-iter loop state)
                can_forward, should_retry = self._check_benchmark_disagg_gate(
                    scheduled_batch, can_forward)
                if should_retry:
                    continue

                # Concern: schedule (paused-request terminate)
                # Task: V1-only — terminate paused requests at iter top.
                #       NOTE: the *pause* counterpart of this terminate is
                #       deferred to after prev's P_APPLY below (V1 KV
                #       handoff ordering). Same `schedule` concern, two
                #       split sites within curr's P_SCHEDULE.
                # Consume: scheduled_batch.paused_requests
                # Produce: terminated requests (out-of-band)
                if not self._scheduler_manages_kv_suspend:
                    self._terminate_requests(scheduled_batch.paused_requests)

                # Concern: schedule (collective queue gate)
                # Task: collective check that every TP rank has a non-empty
                #       batch. ADP exposes a per-rank flag too, used to
                #       drive the empty-rank branch of HC2 below.
                # Consume: scheduled_batch.batch_size
                # Produce: can_queue, can_queue_this_rank
                can_queue, can_queue_this_rank = self._can_queue(
                    scheduled_batch)

                if can_queue:
                    if self.kv_cache_transceiver:
                        # Concern: disagg
                        # Task: promote DISAGG_GENERATION_TRANS_COMPLETE
                        #       gen requests to GENERATION_IN_PROGRESS;
                        #       prepend first_gen logits/logprobs.
                        # Consume: scheduled_batch.generation_requests
                        # Produce: per-request state (out-of-band)
                        # For generation requests which have completed KV cache transfer
                        self._prepare_disagg_gen_transmission_complete(
                            scheduled_batch)

                    # Concern: spec_decode (overlap-mode draft gating)
                    # Task: decide whether the draft model should run this
                    #       iter. Reads self.previous_batch — that's an
                    #       allowed *coarse* read (presence/absence), not
                    #       a half-concern, since it does not forward any
                    #       per-batch data into curr's storage. The fine
                    #       per-batch data forwarding happens later, in
                    #       HC1.
                    # Consume: self.previous_batch (cross-iter loop state)
                    # Produce: has_draft_batch (local), self.use_spec_decode,
                    #          model_engine.enable_spec_decode,
                    #          per-request py_draft_tokens (when reset)
                    has_draft_batch = self.drafter is not None and self.previous_batch is not None and self.use_spec_decode and self.drafter.should_forward_draft_model(
                        scheduled_batch)
                    # Reset the draft tokens to avoid preparing resources for the draft model.
                    if self.drafter is not None and self.use_spec_decode and not has_draft_batch:
                        self.use_spec_decode = False
                        # We are not running the draft model. Remove the draft tokens and turn off spec
                        # decode so that the requests get handled correctly.
                        # One corner case: when we have at least one context request, we have to keep spec
                        # dec on. This ensures that we capture hidden states for requests that haven't done
                        # prefill yet.
                        self.use_spec_decode = False
                        self.model_engine.enable_spec_decode = scheduled_batch.num_context_requests > 0
                        if not self.model_engine.enable_spec_decode:
                            for request in scheduled_batch.all_requests():
                                request.py_draft_tokens = []

                    # Concern: spec_decode (dynamic draft length)
                    # Task: pad/truncate py_draft_tokens to a uniform
                    #       length so CUDA-graph capture works.
                    # Consume: scheduled_batch
                    # Produce: per-request py_draft_tokens (out-of-band),
                    #          model_engine.runtime_draft_len
                    self._handle_dynamic_draft_len(scheduled_batch)

                    # Concern: resource
                    # Task: allocate KV blocks + per-resource state.
                    # Consume: scheduled_batch
                    # Produce: KV blocks, resource state (out-of-band)
                    self.resource_manager.prepare_resources(scheduled_batch)

                if self.kv_connector_manager:
                    # Concern: kv_connector
                    # Task: refresh connector metadata.
                    # Consume: -
                    # Produce: connector metadata (out-of-band)
                    self.kv_connector_manager.handle_metadata()

                if can_queue:
                    # Concern: kv_connector
                    # Task: queue async KV-load (paired with
                    #       `_kv_connector_wait_for_save` inside
                    #       `_forward_step` at curr P_FORWARD).
                    # Consume: scheduled_batch
                    # Produce: connector load ops (out-of-band)
                    self._kv_connector_start_batch(scheduled_batch)

                # if using a kv connector, we need to call can_queue again since scheduled_batch might have changed
                if self.kv_connector_manager:
                    # Concern: schedule (collective queue re-check)
                    # Task: re-evaluate can_queue after kv_connector may
                    #       have evicted/added requests.
                    # Consume: scheduled_batch.batch_size
                    # Produce: can_queue, can_queue_this_rank
                    can_queue, can_queue_this_rank = self._can_queue(
                        scheduled_batch)

                if not can_queue:
                    # Concern: resource (revert)
                    # Task: undo V2 scheduler's per-gen KV growth.
                    # Consume: scheduled_batch
                    # Produce: KV cache state (out-of-band)
                    self._revert_gen_alloc(scheduled_batch)

                # Scheduler control flow (NOT a half concern — no
                # per-batch data forwarded). Decides whether prev's
                # step (2) and step (4) will be invoked this iter.
                # If the batch is not empty on this rank, but empty on other ranks,
                # we need to delay the update of the previous batch's sample state,
                # and let the later iteration to update it.
                should_process_previous_batch = can_queue or not can_queue_this_rank
                if can_queue:

                    # Concern: schedule (gen-request reorder)
                    # Task: stable-sort gen requests so those without
                    #       py_batch_idx come first — required by
                    #       model_engine._forward_step's batch-layout
                    #       assumption in disagg mode.
                    # Consume: scheduled_batch.generation_requests
                    # Produce: scheduled_batch.generation_requests reordered
                    # The generation requests that do not have batch_idx
                    # need to be in front of the batch due to the assumptions
                    # made in model_engine.py::_forward_step. This is only important
                    # for disaggregated serving. For non-disaggregated serving,
                    # the generation requests always have batch_idx.
                    scheduled_batch.generation_requests = sorted(  # stable sort
                        scheduled_batch.generation_requests,
                        key=lambda req: int(req.py_batch_idx is not None),
                    )

                    if self.kv_cache_transceiver:
                        # Concern: response (first-token)
                        # Task: emit first-token response for newly
                        #       promoted disagg-gen requests so the
                        #       client can start streaming.
                        # Consume: scheduled_batch.generation_requests
                        # Produce: enqueued first-token responses
                        # Return the first token to the client
                        self._handle_first_token_response(scheduled_batch)

                    # Concern: guided_decoder
                    # Task: register batch with constraint matcher;
                    #       init disagg-gen request matchers.
                    # Consume: scheduled_batch
                    # Produce: matcher per-batch state (out-of-band)
                    # init_disagg_gen_requests must be before engine forward, where the prev_seq_slot is updated.
                    if self.guided_decoder is not None and self.kv_cache_transceiver:
                        self.guided_decoder.add_batch(scheduled_batch)
                        self.guided_decoder.init_disagg_gen_requests()

                    # =====================================================
                    # ===== HC1 BRIDGE: prev.P_SAMPLE -> curr.P_FORWARD ===
                    # =====================================================
                    # The next ~30 lines are the HC1 cross-batch bridge.
                    # Inside a coroutine refactor this lives in the
                    # scheduler iter, between step (1)'s P_SCHEDULE-end
                    # rendezvous and curr's P_FORWARD body.
                    # =====================================================

                    # Half concern: HC1 (consume half on prev)
                    # Bridge: prev.P_SAMPLE.sample_state
                    #         -> curr.P_FORWARD.previous_tensors_device
                    # Task: snapshot prev's sample-state container.
                    # Consume: prev.sample_state (per-batch, prev side)
                    # Produce: previous_tensors (local; staged for HC1
                    #          produce half below)
                    previous_tensors = self.previous_batch and self.previous_batch.sample_state

                    # Half concern: HC3 (consume half this iter, cross-iter)
                    # Bridge: curr_at_prev_iter.P_FORWARD.has_previous_draft_tokens
                    #         -> this iter's HC1 selection.
                    # Task: read the side-channel flag set by the *previous*
                    #       iter's `_handle_speculative_decoding` (HC3
                    #       produce). Drives the HC1 produce-half branch
                    #       below.
                    # Consume: self.has_previous_draft_tokens (cross-iter
                    #          side channel)
                    # Produce: use_previous_draft_tokens (local)
                    # If there are previous draft tokens, we need to update the target requests to accept some draft tokens.
                    # When there's any accepted tokens, we can't directly use the previous batch's outputs in this iteration for the target model,
                    # so we'll set the target model's input to None and skip updating the target requests after target model forward.
                    use_previous_draft_tokens = self.has_previous_draft_tokens
                    num_accepted_tokens_device = None

                    target_inputs = None
                    num_accepted_tokens_device = None

                    if has_draft_batch:
                        # Half concern: HC1 (in-bridge compute)
                        # Task: run the draft model on prev's sample
                        #       tensors, computing accepted-token counts
                        #       and packaged target_inputs ready for
                        #       curr's forward. Internally also sets
                        #       `self.has_previous_draft_tokens` — the
                        #       HC3 produce half for next iter.
                        # Consume: prev.sample_state (via previous_tensors,
                        #          previous_tensors_device); scheduled_batch
                        # Produce: target_inputs, num_accepted_tokens_device
                        #          (locals; feed HC1 produce half below);
                        #          self.has_previous_draft_tokens (HC3
                        #          produce half, cross-iter)
                        self.execution_stream.wait_stream(
                            torch.cuda.current_stream())
                        with torch.cuda.stream(self.execution_stream):
                            target_inputs, num_accepted_tokens_device = self._handle_speculative_decoding(
                                scheduled_batch, previous_tensors,
                                previous_tensors_device)
                        torch.cuda.current_stream().wait_stream(
                            self.execution_stream)

                    # Half concern: HC1 (produce half on curr — selection)
                    # Task: pick the right source for curr.P_FORWARD's
                    #       `previous_tensors_device` slot — draft-model
                    #       outputs if the draft ran, else prev's raw
                    #       sample tensors.
                    # Consume: target_inputs, use_previous_draft_tokens
                    #          (locals from HC1/HC3 consume halves);
                    #          prev.sample_state.device
                    # Produce: previous_tensors_device (local; flows into
                    #          curr.P_FORWARD via _forward_step args)
                    # Use the draft_model's outputs if we've launched the draft model.
                    # Otherwise, use the previous batch's outputs.
                    if (target_inputs is not None
                            and target_inputs.next_draft_tokens
                            is not None) or use_previous_draft_tokens:
                        previous_tensors_device = target_inputs
                    else:
                        previous_tensors_device = self.previous_batch and self.previous_batch.sample_state and self.previous_batch.sample_state.device

                    # ===== curr PHASE: P_FORWARD =====

                    # Concern: perf_metric (curr)
                    # Task: allocate CUDA event handles for curr's
                    #       forward+sample timing window.
                    # Consume: -
                    # Produce: gpu_forward_start, gpu_forward_end,
                    #          gpu_sample_end (CUDA events; curr fields)
                    # GPU timing for perf metrics
                    gpu_forward_start, gpu_forward_end, gpu_sample_end = self.perf_manager.create_timing_events(
                    )

                    with self.perf_manager.record_perf_events(
                            gpu_forward_start, gpu_forward_end) as fwd_timing:
                        # Concern: forward (curr) (+ kv_connector wait_for_save)
                        # Task: queue model forward on execution_stream;
                        #       internally calls `_kv_connector_wait_for_save`
                        #       (deadline-wait paired with start_batch
                        #       above). The HC1 produce-half values
                        #       (`previous_tensors_device`,
                        #       `num_accepted_tokens_device`) are consumed
                        #       here as forward inputs.
                        # Consume: scheduled_batch, prepared resources,
                        #          previous_tensors_device,
                        #          num_accepted_tokens_device (HC1 produce)
                        # Produce: batch_outputs (logits + extras),
                        #          fwd_timing (CPU ts), forward CUDA events
                        batch_outputs = self._forward_step(
                            scheduled_batch, previous_tensors_device,
                            num_accepted_tokens_device)

                # =========================================================
                # ===== STEP (2) : prev P_APPLY ===========================
                # =========================================================
                # Drives prev through P_APPLY: sampler-event sync, token
                # apply, ctx-side KV send. Sequenced AFTER curr's
                # P_FORWARD queue (so the sync can overlap curr's GPU
                # forward) and BEFORE curr's P_SAMPLE (which mutates
                # the default stream).
                # =========================================================

                if self.previous_batch is not None and should_process_previous_batch:
                    # Concern: sample (prev — apply tokens)
                    # Task: BLOCK on prev's sampler_event, then apply
                    #       sampled tokens to prev's requests via
                    #       sampler.update_requests. This is the only
                    #       point this iter that synchronously waits on
                    #       prev's sampler event.
                    # Consume: prev.sample_state (sampler_event +
                    #          host tensors)
                    # Produce: per-request new tokens, py_decoding_iter
                    #          (out-of-band on prev's requests)
                    self._update_requests(self.previous_batch.sample_state)

                    # Concern: disagg (prev — KV send) (+ kv_connector)
                    # Task: start ctx->gen KV send for prev's finished
                    #       ctx-only requests; opportunistic completion
                    #       probe at the end. Multi-concern function (see
                    #       annotations on `_send_kv_async`).
                    # Consume: prev.scheduled_requests
                    # Produce: KV send handles, terminated ctx requests
                    #          (out-of-band on prev)
                    self._send_kv_async(
                        self.previous_batch.scheduled_requests.all_requests())

                # Flush outside the conditional so that all DP ranks
                # participate in the tp_gather collective even when
                # should_process_previous_batch differs between ranks.
                self._flush_pending_transfer_responses()

                if self.drafter is not None and self.use_spec_decode and should_process_previous_batch:
                    # Concern: spec_decode (prev — draft cleanup)
                    # Task: free draft KV / spec resources used by prev's
                    #       draft model run (paired with this iter's HC1
                    #       in-bridge `_handle_speculative_decoding`,
                    #       which is the *next* iter's draft prep that
                    #       used prev's slots).
                    # Consume: prev's spec_decode state
                    # Produce: draft resources released (out-of-band)
                    # Cleanup previous draft resources used in the draft model
                    self.drafter.cleanup_previous_draft_resources()

                # Concern: schedule (curr — paused-request pause; deferred)
                # Task: V1-only — pause curr's preempted requests. Same
                #       `schedule` concern as `_terminate_requests`
                #       above; placement deferred to after prev's
                #       P_APPLY because V1 needs prev's KV slots to be
                #       freed by `_update_requests` before curr's pauses
                #       can rewrite them.
                # Consume: scheduled_batch.paused_requests
                # Produce: paused requests (out-of-band)
                if not self._scheduler_manages_kv_suspend:
                    self._pause_requests(scheduled_batch.paused_requests)

                # =========================================================
                # ===== STEP (3) : curr P_SAMPLE -> P_STATE_UPD ===========
                # =========================================================

                if can_queue:
                    # ===== curr PHASE: P_SAMPLE =====

                    guided_decoder_failed_requests = None
                    with self.perf_manager.record_perf_events(
                            None, gpu_sample_end) as sample_timing:
                        if self.guided_decoder is not None:
                            # Concern: guided_decoder (curr)
                            # Task: re-add curr to matcher with refreshed
                            #       new_tokens, then mask logits via
                            #       execute. Must run before _sample_async
                            #       so masking affects sampling.
                            # Consume: scheduled_batch,
                            #          batch_outputs (logits)
                            # Produce: guided_decoder_failed_requests,
                            #          masked logits
                            # add_batch must be called again to have updated new tokens.
                            self.guided_decoder.add_batch(scheduled_batch)
                            guided_decoder_failed_requests = self.guided_decoder.execute(
                                batch_outputs['logits'])

                        # Concern: sample (curr)
                        # Task: queue sampling kernel + D2H copy of
                        #       sample-state tensors; record sampler_event
                        #       used by *next iter*'s step (2) to sync.
                        # Consume: scheduled_batch, batch_outputs (logits)
                        # Produce: sample_state (sampler_event + .device
                        #          tensors), sample_timing
                        sample_state = self._sample_async(
                            scheduled_batch, batch_outputs)

                    assert sample_state is not None, "Sampling failed"

                    # ===== curr PHASE: P_STATE_UPD =====

                    # Concern: guided_decoder (curr — error mark)
                    # Task: mark grammar-failed curr requests as errored.
                    #       Strict ordering: AFTER _sample_async (else
                    #       GENERATION_COMPLETE breaks sample's
                    #       context_chunk_size access).
                    # Consume: scheduled_batch,
                    #          guided_decoder_failed_requests
                    # Produce: failed-request error responses (out-of-band)
                    # Handle guided decoder errors after _sample_async to avoid state conflicts.
                    # If called before, failed requests would be marked as GENERATION_COMPLETE,
                    # causing _sample_async to fail when accessing context_chunk_size property.
                    self._handle_guided_decoder_errors(
                        scheduled_batch, guided_decoder_failed_requests)

                    # Concern: sample (curr — state advance)
                    # Task: advance context_chunk_position; transition
                    #       finished-context requests to GENERATION_*;
                    #       drop ADP dummy. Independent of sample_state
                    #       value (which is still un-synced — that
                    #       sync happens at next iter's step (2)).
                    # Consume: scheduled_batch
                    # Produce: per-request state (out-of-band)
                    self._update_request_states(scheduled_batch)

                # =========================================================
                # ===== STEP (4) : prev P_RESPOND =========================
                # =========================================================

                if self.previous_batch is not None and should_process_previous_batch:
                    # Concern: response (prev) + resource (prev) +
                    #          kv_cache_events (prev) + iter_stats (prev)
                    # Task: prev's full P_RESPOND fan-out — see
                    #       `_process_previous_batch`. Internally calls
                    #       `_handle_canceled_requests`,
                    #       `_handle_responses`, `update_resources`,
                    #       `_add_kv_cache_events`, and
                    #       `_process_iter_stats`. (`_process_iter_stats`
                    #       lives here, not at end-of-iter like the plain
                    #       loop, because iter_stats is a prev field
                    #       populated across two iters.)
                    # Consume: prev.scheduled_requests, prev.iter_stats,
                    #          prev.iter_start_time
                    # Produce: prev's responses, freed KV blocks, kv-cache
                    #          events, iter_stats record (all out-of-band)
                    self._process_previous_batch()

                    # Concern: perf_metric (prev)
                    # Task: read prev's forward/sample CUDA events into
                    #       per-request metric records. Deferred from
                    #       prev's P_STATE_UPD (= last iter's tail) to
                    #       this iter to avoid the next iter's
                    #       create_timing_events overwriting them.
                    # Consume: prev.scheduled_requests, prev's gpu events
                    # Produce: per-request gpu timings (out-of-band on prev)
                    self.perf_manager.compute_batch_gpu_times(
                        self.previous_batch.scheduled_requests.all_requests())
                else:
                    # Concern: response (collective fallback)
                    # Task: when there's no prev to retire (first iter or
                    #       this rank skipped), still call into the
                    #       response collective so other ranks don't hang
                    #       waiting for the gather/allgather inside
                    #       `_enqueue_responses`.
                    # Consume: -
                    # Produce: -
                    self._enqueue_responses([])

                # Concern: spec_decode (curr — late state mark)
                # Task: mark curr's gen requests that will finish next iter
                #       (sets state=GENERATION_TO_COMPLETE +
                #       exclude_last_generation_logits=False).
                #       Cross-batch ordering: MUST run after prev's
                #       P_RESPOND because prev's `_handle_responses`
                #       reads exclude_last_generation_logits while
                #       building responses for shared-by-id requests.
                #       Belongs to curr's P_STATE_UPD conceptually but
                #       the placement is dictated by prev's response.
                # Consume: scheduled_batch.generation_requests
                # Produce: per-request state, exclude_last_generation_logits
                #          (out-of-band on curr)
                # Call set_exclude_last_generation_logits after _process_previous_batch.
                # If set before, the response of a request may be incorrect, as it will
                # use the wrong indices for generation logits when streaming is enabled.
                if can_queue:
                    self._update_generation_requests_that_will_complete_next_iteration(
                        scheduled_batch.generation_requests)

                if can_queue:
                    # Concern: perf_metric (curr — tail snapshot)
                    # Task: snapshot CUDA event handles + CPU timing onto
                    #       curr's per-request perf records. Times are
                    #       finalized at next iter's `compute_batch_gpu_times`.
                    #       This block could in principle run inside
                    #       step (3) (curr's P_STATE_UPD) — its data
                    #       deps (`fwd_timing`, `sample_timing`) are
                    #       resolved by then. NOTE: a possible reorder
                    #       candidate; left here for now to keep this
                    #       diff comments-only.
                    # Consume: scheduled_batch, gpu_*_start/_end (events),
                    #          fwd_timing, sample_timing
                    # Produce: per-request perf records (out-of-band)
                    self.perf_manager.save_timing_to_requests(
                        scheduled_batch.all_requests(), gpu_forward_start,
                        gpu_forward_end, gpu_sample_end, fwd_timing.start_time,
                        fwd_timing.end_time, sample_timing.start_time,
                        sample_timing.end_time)
                    if self.enable_iter_perf_stats:
                        # Concern: iter_stats (curr — num_ctx_tokens key)
                        # Task: snapshot model_engine.iter_states.num_ctx_tokens
                        #       onto curr's iter_stats. Read at next iter's
                        #       step (4) inside `_process_iter_stats`.
                        # Consume: model_engine.iter_states (engine state)
                        # Produce: iter_stats.inflight_batching_stats.num_ctx_tokens
                        #          (curr field)
                        iter_stats.inflight_batching_stats.num_ctx_tokens = self.model_engine.iter_states[
                            'num_ctx_tokens']

                    # =====================================================
                    # ===== HC2 PRODUCE: curr -> next-iter prev slot ======
                    # =====================================================
                    # Half concern: HC2 (produce half this iter, cross-iter)
                    # Bridge: curr.P_STATE_UPD.{scheduled_requests,
                    #         sample_state, iter_stats, iter_start_time}
                    #         -> next iter's `self.previous_batch` slot.
                    # Pair: next iter's HC1 consume + step (2) + step (4),
                    #       which read self.previous_batch as prev.
                    # Task: package curr into a BatchState record so next
                    #       iter sees it as prev. This is the
                    #       `handle_prev = handle_t` of the docstring
                    #       pseudo-code.
                    # Consume: scheduled_batch, sample_state,
                    #          iter_start_time, iter_stats (curr fields)
                    # Produce: self.previous_batch (cross-iter scheduler
                    #          slot)
                    self.previous_batch = BatchState(
                        scheduled_requests=scheduled_batch,
                        sample_state=sample_state,
                        iter_start_time=iter_start_time,
                        iter_stats=iter_stats)
                elif not can_queue_this_rank:
                    # Half concern: HC2 (produce half — empty-rank cleanup)
                    # Task: cleanup variant of HC2's produce half — when
                    #       this rank is empty, clear the prev slot so
                    #       next iter sees no prev. The "consume half"
                    #       on the next iter still pairs with this — it
                    #       just sees `self.previous_batch is None` and
                    #       skips its step (2)/(4).
                    # Consume: -
                    # Produce: self.previous_batch (cleared)
                    # If the batch is empty on this rank, we need to clear the previous batch.
                    self.previous_batch = None

                # =========================================================
                # ===== global / curr P_RESPOND tail ======================
                # =========================================================

                if self.kv_cache_transceiver and self.async_transfer_manager.has_any_inflight_requests(
                ):
                    # Concern: disagg
                    # Task: end-of-iter timeout sweep over inflight async
                    #       KV transfers (cross-batch — flags any
                    #       inflight transfer regardless of which batch
                    #       owns the request).
                    # Consume: -
                    # Produce: per-request py_kv_transfer_timed_out
                    self._check_kv_transfer_timeout()

                # Concern: kv_connector
                # Task: terminate connector-flagged finished requests
                #       (paired with `_kv_connector_start_batch` at curr
                #       P_SCHEDULE).
                # Consume: -
                # Produce: terminated requests (out-of-band)
                self._kv_connector_terminate_requests()

                # Concern: loop_control
                # Task: advance global iteration counter.
                # Consume: -
                # Produce: iter_counter (loop state)
                self.iter_counter += 1

    # REFACTOR: 2a (concern: spec_decode) -- overlap-mode HC1 in-bridge
    # helper; computes accepted-token count from prev's sample tensors.
    @nvtx_range("_accept_draft_tokens")
    def _accept_draft_tokens(
        self, scheduled_batch: ScheduledRequests,
        target_outputs: SampleStateTensors,
        target_inputs: Optional[SampleStateTensors]
    ) -> Tuple[SampleStateTensorsSpec, Optional[torch.Tensor]]:
        """
        Prepare target device inputs after computing draft token acceptance.

        This function:
        1. If draft tokens exist: compares sampled tokens with draft tokens to compute acceptance
        2. If no draft tokens: directly uses the first sampled token
        3. Creates new_tokens by extracting accepted tokens per request

        Args:
            scheduled_batch: The scheduled requests
            target_outputs: Contains new_tokens [max_draft_len + 1, batch_size, beam_width]
                                or [1, batch_size, beam_width] if no draft tokens
            target_inputs: Contains next_draft_tokens [batch_size, max_draft_len]
        Returns:
            Tuple of:
            - SampleStateTensorsSpec with new_tokens set to accepted tokens,
              new_tokens_lens and next_draft_tokens set to None
            - num_accepted_tokens: [batch_size] tensor with acceptance counts per request,
              or None if no draft tokens
        """
        has_draft_tokens = target_inputs is not None and isinstance(
            target_inputs, SampleStateTensorsSpec
        ) and target_inputs.next_draft_tokens is not None
        target_tokens = target_outputs.new_tokens  # [max_draft_len + 1, batch_size, beam_width] or [1, batch_size, beam_width]
        new_tokens = torch.zeros_like(target_tokens)

        # Squeeze the beam dimension (beam_width=1 for greedy or single beam)
        target_tokens = target_tokens.squeeze(
            -1)  # [max_draft_len + 1, batch_size] or [1, batch_size]

        batch_size = target_tokens.shape[1]
        device = target_tokens.device
        # Compute number of accepted tokens per request
        num_accepted_tokens = torch.zeros(batch_size,
                                          dtype=torch.int32,
                                          device=device)

        if has_draft_tokens:
            # Draft tokens exist, compute acceptance
            draft_tokens = target_inputs.next_draft_tokens  # [batch_size, max_draft_len]
            max_draft_len = draft_tokens.shape[1]

            # Compute number of accepted tokens per request
            # Generation requests: compare with draft tokens to find acceptance
            num_contexts = scheduled_batch.num_context_requests
            if batch_size > num_contexts:
                # Use .T to transpose: [max_draft_len + 1, num_gens] -> [num_gens, max_draft_len + 1]
                gen_target_tokens = target_tokens[:,
                                                  num_contexts:].T  # [num_gens, max_draft_len + 1]

                # Compare draft tokens with target tokens to find acceptance
                # Use cumprod to find the first rejection point
                draft_tokens_gen = draft_tokens[
                    num_contexts:, :].int()  # [num_gens, max_draft_len]
                num_accepted_tokens[num_contexts:] += torch.cumprod(
                    (draft_tokens_gen == gen_target_tokens[:, :max_draft_len]
                     ).int(),
                    dim=-1).sum(dim=1)

            # Vectorized extraction using advanced indexing (no GPU-CPU sync)
            # Use num_accepted_tokens as indices to gather the right tokens
            batch_indices = torch.arange(batch_size, device=device)
            new_tokens[0, :, 0] = target_tokens[num_accepted_tokens,
                                                batch_indices]
        else:
            # No draft tokens to accept, just use the first (and only) sampled token
            batch_indices = torch.arange(batch_size, device=device)
            new_tokens[0, :, 0] = target_tokens[0, batch_indices]

        # Create the updated SampleStateTensorsSpec
        # new_tokens_lens and next_draft_tokens are left as None
        result_tensors = SampleStateTensorsSpec(
            new_tokens=new_tokens,
            log_probs=target_outputs.log_probs,
            new_tokens_lens=None,
            next_draft_tokens=None)

        # Copy logits if available
        if hasattr(target_outputs, 'logits'):
            result_tensors.logits = target_outputs.logits

        return result_tensors, num_accepted_tokens

    # REFACTOR: 1b (multi-concern, inlined+split) -- annotated in-line as
    # overlap's prev-batch RESPOND_8 fan-out across response + resource
    # + kv_cache_events + iter_stats. Body splits into the respective
    # concerns' RESPOND_8 / FINALIZE_9 blocks.
    def _process_previous_batch(self):
        self._handle_canceled_requests()
        finished_requests = self._handle_responses()
        scheduled_requests = self.previous_batch.scheduled_requests
        attn_metadata = getattr(self.model_engine, 'attn_metadata', None)
        kv_cache_dtype_byte_size = getattr(self.model_engine,
                                           'kv_cache_dtype_byte_size', None)
        self.resource_manager.update_resources(scheduled_requests,
                                               attn_metadata,
                                               kv_cache_dtype_byte_size)
        if self.enable_kv_cache_events:
            self._add_kv_cache_events()

        if self.enable_iter_perf_stats:
            self._process_iter_stats(finished_requests, self.active_requests,
                                     self.previous_batch)

    # REFACTOR: 2a (concern: forward) -- PP non-last-rank variant of
    # forward; collapses forward + sample placeholder + state advance
    # into one call so ``BatchStorage`` has a unified shape across ranks.
    def _forward_step_inter_pp(self, scheduled_batch) -> SampleState:
        self._forward_step(scheduled_batch)
        sampler_event = torch.cuda.Event()
        sampler_event.record()
        self._update_request_states(scheduled_batch)
        sampling_requests = scheduled_batch.context_requests_last_chunk + scheduled_batch.generation_requests
        return self.sampler.SampleState(
            requests=sampling_requests,
            sampler_event=SamplerEvent(cuda_event=sampler_event),
            runtime_draft_len=self.model_engine.runtime_draft_len,
        )

    # REFACTOR: 2a (concern: schedule) -- request validation in fetch path.
    def _validate_token_id_range(self, request: LlmRequest) -> None:
        if isinstance(self.model_engine.model, DecoderModelForCausalLM):
            # Only skip token‐range checks for Llama4 when the request has multimodal data
            if isinstance(self.model_engine.model,
                          Llama4ForConditionalGeneration):
                has_mm = bool(request.py_multimodal_data)
                if has_mm:
                    logger.debug(
                        f"Skipping token-range validation for {type(self.model_engine.model).__name__} "
                        "(multimodal request)")
                    return

            # FIXME: This check is necessary because of how Qwen2ForProcessRewardModel
            #        subclasses DecoderModelForCausalLM. Perhaps the functionality
            #        of DecoderModelForCausalLM reused by Qwen2ForProcessRewardModel
            #        should be factored out into a separate class instead.
            if not hasattr(self.model_engine.model, "lm_head"):
                return

            if not request.check_token_id_range(
                    self.model_engine.model.lm_head.num_embeddings):
                raise ValueError("Token ID out of range")

    # REFACTOR: 2a (concern: schedule) -- request validation in fetch path.
    def _validate_request(self, request: LlmRequest):
        # Validate beam width
        sampling_config = request.sampling_config
        if sampling_config is not None:
            if sampling_config.beam_width != self.max_beam_width:
                raise ValueError(
                    f"Request beam width {sampling_config.beam_width} "
                    f"is not equal to max_beam_width {self.max_beam_width}. This is not supported!"
                )

        # Check token ID ranges
        self._validate_token_id_range(request)

        # Perform sampler-specific validation
        self.sampler.validate_request(request)

    # REFACTOR: 2a (concern: schedule) -- helper for the fetch path.
    def _fetch_and_enqueue_requests(self, waiting_queue: WaitingQueue,
                                    total_num_active_requests: int) -> None:
        """Fetch requests from request_queue and enqueue to waiting_queue."""
        # Block new requests while control requests are pending
        if len(self.control_requests) != 0:
            return

        # Calculate timeout
        idle = (total_num_active_requests == 0) and len(waiting_queue) == 0
        if idle:
            # In Ray path (TLLM_DISABLE_MPI=1), use a periodic heartbeat timeout so rank 0
            # reaches the broadcast path regularly to prevent trtllm-serve timeout when idle.
            timeout = datetime.timedelta(
                seconds=1200) if self._disable_mpi else None
        else:
            timeout = datetime.timedelta(0)

        # Fetch requests from rank 0
        new_requests = []
        if self.dist.rank == 0:
            # Process accumulated requests that were queued during control request handling.
            if len(self.request_accumulated) != 0:
                new_requests.extend(self.request_accumulated)
                self.request_accumulated.clear()
                # Reset timeout to 0 to avoid hanging when no new requests are available
                timeout = datetime.timedelta(0)
            with self.hang_detector.pause():
                new_requests.extend(
                    self.executor_request_queue.get_from_request_queue(timeout))

        # Broadcast requests and handle Python objects
        new_requests, py_request_objects = self.request_broadcaster.broadcast(
            new_requests)

        # Validate and filter requests
        new_requests = self._handle_special_queue_items(new_requests)

        # Attach Python objects to requests
        if py_request_objects and (self.dist.tp_size > 1 or self.dist.has_pp
                                   or self.dist.cp_size
                                   > 1) and self.dist.rank > 0:
            attach_py_objects_to_requests(new_requests, py_request_objects)

        waiting_queue.add_requests(new_requests)

    # REFACTOR: 2a (concern: schedule) -- helper for the fetch path.
    def _pop_from_waiting_queue(
        self,
        waiting_queue: WaitingQueue,
        total_num_active_requests: int,
        all_ranks_num_active_requests: Optional[List[int]] = None
    ) -> List[RequestQueueItem]:
        """Pop requests from waiting_queue based on available capacity."""
        if self.enable_attention_dp:
            total_max = self.dist.tp_size * self.max_num_active_requests
        else:
            total_max = self.max_num_active_requests

        max_new_requests = total_max - total_num_active_requests

        return get_from_waiting_queue(
            waiting_queue,
            max_new_requests,
            enable_attention_dp=self.enable_attention_dp,
            max_num_active_requests=self.max_num_active_requests,
            all_ranks_num_active_requests=all_ranks_num_active_requests)

    # REFACTOR: 2a (concern: schedule) -- main entry of the fetch step.
    @nvtx_range("_fetch_new_requests")
    def _fetch_new_requests(
            self, waiting_queue: WaitingQueue,
            active_requests: List[LlmRequest]) -> List[LlmRequest]:
        """Fetch new requests and return LlmRequests ready for execution."""
        # 1. Gather rank states and calculate total_num_active_requests
        if self.enable_attention_dp:
            # NOTE: gather_all_rank_states is called here (before step 3)
            # because _pop_from_waiting_queue needs all_ranks_num_active_requests
            # from the allgather result. Moving it to step 5 would require an
            # extra allgather. When introducing new router implementations
            # (e.g. KV-cache-aware) that need new_requests to gather additional
            # info, the allgather position may need to be revisited.

            all_rank_states = self.adp_router.gather_all_rank_states(
                active_requests)
            all_ranks_num_active_requests = [
                s.num_active_requests for s in all_rank_states
            ]
            total_num_active_requests = sum(all_ranks_num_active_requests)
        else:
            total_num_active_requests = len(active_requests)
            all_ranks_num_active_requests = None
            all_rank_states = None

        # 2. Fetch and enqueue to waiting queue
        self._fetch_and_enqueue_requests(waiting_queue,
                                         total_num_active_requests)

        # 3. Pop requests from waiting queue
        new_requests = self._pop_from_waiting_queue(
            waiting_queue, total_num_active_requests,
            all_ranks_num_active_requests)

        # 4. Update performance metrics (before DP scheduling to clear all start_times)
        if self.enable_iter_perf_stats and self.dist.rank == 0:
            self._update_new_active_requests_queue_latency(new_requests)

        # 5. Update total fetch counter (used by benchmark disagg gating)
        self.num_fetch_requests += len(new_requests)

        # 6. Schedule requests across ranks (DP only)
        if self.enable_attention_dp:
            if self.adp_router.needs_prefix_matches:
                self.adp_router.gather_prefix_matches(new_requests)

            all_ranks_new_requests, self.expected_num_active_requests = \
                self.adp_router.route_requests(
                    all_rank_states, new_requests,
                    self.max_num_active_requests)
            new_requests_cur_rank = all_ranks_new_requests[self.dist.tp_rank]

            # Update per-rank counter for DP
            self.num_fetch_requests_cur_rank += len(new_requests_cur_rank)

            new_requests = new_requests_cur_rank

        # 7. Merge requests
        return merge_requests(new_requests,
                              cp_config=self.dist.cp_config,
                              cp_rank=self.dist.cp_rank,
                              cp_size=self.dist.cp_size,
                              exclude_last_generation_logits=self.
                              _should_exclude_last_generation_logits())

    # REFACTOR: 1b (multi-concern, inlined+split) -- per-item dispatcher
    # at fetch time: shutdown signals -> ``MessagePort.is_shutdown``;
    # cancellation items -> ``response`` cancellation set; control items
    # -> ``control`` queue. Splits into the respective concerns'
    # SCHEDULE_0 input handling.
    def _handle_special_queue_items(
            self,
            new_requests: List[RequestQueueItem]) -> List[RequestQueueItem]:
        """Handle special signals."""
        accepted_new_requests = []
        for idx, req_item in enumerate(new_requests):
            if req_item.is_shutdown_request:
                self.is_shutdown = True
                break
            elif req_item.is_canceled_request:
                self.canceled_req_ids.append(req_item.id)
            elif req_item.is_control_request:
                self.control_requests.append(req_item)
                if self.dist.rank == 0:
                    self.request_accumulated.extend(new_requests[idx + 1:])
                break
            else:
                accepted_new_requests.append(req_item)

        return accepted_new_requests

    # REFACTOR: 2a (concern: iter_stats) -- queue-latency metric writer
    # invoked by the fetch path at SCHEDULE_0.
    def _update_new_active_requests_queue_latency(
            self, new_requests: List[RequestQueueItem]):
        """Update queue latency metrics for new requests."""
        now = time.time()
        latency = self.executor_request_queue.calculate_queue_latency(
            new_requests, now)
        self.new_active_requests_queue_latency_ms += latency

    # REFACTOR: 2a (concern: iter_stats) -- queue-latency metric reader.
    def _get_new_active_requests_queue_latency(self) -> float:
        return self.new_active_requests_queue_latency_ms

    # REFACTOR: 2a (concern: disagg) -- disagg's first-token logit
    # snapshot uses this flag (also accessed directly as the property
    # ``self.should_exclude_last_generation_logits``).
    def _should_exclude_last_generation_logits(self) -> bool:
        return self.should_exclude_last_generation_logits

    # REFACTOR: 2a (concern: schedule) -- fetch step entry point used by
    # ``_prepare_and_schedule_batch``; invokes validation + activation.
    # In the new design: the schedule concern's fetch step at
    # SCHEDULE_0; admits via ``ctx.svc.pool.add_active(reqs)`` instead
    # of ``self.active_requests.extend(...)``.
    def _fetch_and_activate_new_requests(self) -> List[LlmRequest]:

        def _respond_if_invalid(request: LlmRequest) -> bool:
            """Immediately fail invalid request.

            Return True if invalid request was encountered and
            handled.
            """
            try:
                self._validate_request(request)
                return False
            except Exception as e:
                self._handle_errors(str(e),
                                    requests=[request],
                                    charge_budget=False)
                return True

        new_requests_cur_rank = self._fetch_new_requests(
            self.waiting_queue, self.active_requests)

        validated_requests = [
            request for request in new_requests_cur_rank
            if not _respond_if_invalid(request)
        ]

        self.active_requests.extend(validated_requests)
        return validated_requests

    # REFACTOR: 2a (concern: kv_cache_events) -- RESPOND_8 step.
    def _add_kv_cache_events(self):
        kv_cache_manager = self.resource_manager.resource_managers.get(
            ResourceManagerType.KV_CACHE_MANAGER)
        if not kv_cache_manager:
            return
        # Flush iteration events at each iteration to ensure that events have enough time
        # to be transferred to main thread when user needs them.
        kv_cache_manager.flush_iteration_events()

    # REFACTOR: 2a (concern: schedule) -- ADP request balancer used by
    # ``_schedule``.
    def _balance_adp_requests(self, context_requests: list[LlmRequest],
                              generation_requests: list[LlmRequest]):
        balanced_context_requests = context_requests
        num_scheduled_context_requests = len(context_requests)
        num_scheduled_generation_requests = len(generation_requests)
        num_scheduled_tokens = sum(
            [len(req.get_tokens(0))
             for req in context_requests]) + num_scheduled_generation_requests
        # Note: We use tp_allgather instead of tp_cp_allgather because we want to
        # balance the requests across DP ranks; not CP ranks within those DP ranks.
        responses_list = self.dist.tp_allgather([
            num_scheduled_context_requests, num_scheduled_generation_requests,
            num_scheduled_tokens
        ])
        all_ranks_num_scheduled_context_requests = [
            response[0] for response in responses_list
        ]
        all_ranks_num_scheduled_generation_requests = [
            response[1] for response in responses_list
        ]
        all_ranks_have_free_ctx_slots = all([
            num_gen < self.max_batch_size
            for num_gen in all_ranks_num_scheduled_generation_requests
        ])
        all_ranks_have_ctx_requests = all([
            num_ctx > 0 for num_ctx in all_ranks_num_scheduled_context_requests
        ])
        all_ranks_have_gen_requests = all([
            num_gen > 0
            for num_gen in all_ranks_num_scheduled_generation_requests
        ])

        if self.attention_dp_enable_balance:
            # wait for all ranks have context requests
            if all_ranks_have_free_ctx_slots and all_ranks_have_ctx_requests:
                self.adp_ctx_waiting_iters_count = 0
                # balance number of context requests across ranks
                if all_ranks_have_gen_requests:
                    if self.adp_ctx_batching_wait_iters_count < self.attention_dp_batching_wait_iters:
                        self.adp_ctx_batching_wait_iters_count += 1
                        balanced_context_requests = []
                    else:
                        self.adp_ctx_batching_wait_iters_count = 0
            else:
                self.adp_ctx_waiting_iters_count += 1
                balanced_context_requests = []
                timeout_reached = self.adp_ctx_waiting_iters_count >= self.attention_dp_time_out_iters
                if timeout_reached or not all_ranks_have_gen_requests:
                    self.adp_ctx_waiting_iters_count = 0
                    balanced_context_requests = context_requests
        return balanced_context_requests

    # REFACTOR: 2a (concern: schedule) -- batch-wait helper.
    @staticmethod
    def _compute_scheduled_tokens(context_requests, generation_requests):
        """Compute the total number of scheduled tokens for batch waiting decisions.

        For context requests, we estimate the actual compute tokens for this
        iteration (excluding tokens served from KV cache).

        For generation requests, each contributes 1 + num_draft_tokens.

        Note on reusable token handling:
        estimated_reusable_tokens is an absolute count from position 0.
        Depending on the scheduler, context_current_position may or may not
        have been advanced past the reusable prefix by the time this method
        is called:
        - V1 scheduler: prepare_context runs after scheduling, so
          context_current_position is still 0.
        - V2 scheduler: prepare_context runs during scheduling, so
          context_current_position is already advanced to the reused offset.
        To handle both correctly, the reusable credit applied to the current
        chunk is max(0, reusable - context_current_position), i.e. only the
        portion of the reusable range that falls within this chunk's span.
        """
        num_scheduled_ctx_tokens = 0
        for ctx_req in context_requests:
            reusable = (ctx_req.estimated_reusable_tokens
                        if ctx_req.is_first_context_chunk else 0)
            # Credit only the reusable tokens that overlap with the current
            # chunk: if context_current_position has already been advanced past
            # the reusable prefix (V2), the credit is 0; if not (V1), the full
            # reusable count is subtracted.
            reusable_in_chunk = max(0,
                                    reusable - ctx_req.context_current_position)
            remaining = ctx_req.context_remaining_length
            if reusable_in_chunk <= 0:
                compute = ctx_req.context_chunk_size
            elif reusable_in_chunk + ctx_req.context_chunk_size < remaining:
                compute = ctx_req.context_chunk_size
            else:
                compute = max(1, remaining - reusable_in_chunk)
            num_scheduled_ctx_tokens += compute
        num_scheduled_gen_tokens = sum(1 + gen_req.num_draft_tokens
                                       for gen_req in generation_requests)
        return num_scheduled_ctx_tokens + num_scheduled_gen_tokens

    # REFACTOR: 2a (concern: schedule) -- batch-wait policy applied by
    # ``_schedule``.
    def _waiting_requests(self, context_requests: list[LlmRequest],
                          generation_requests: list[LlmRequest]):
        """
        Return an empty list if scheduled requests fulfill the waiting conditions, otherwise return the original context requests.
        Waiting conditions:
        - The number of scheduled tokens (both context and generation) is smaller than `self.batch_wait_max_tokens_ratio * self.max_num_tokens`
        - The number of waiting iterations is smaller than `self.batch_wait_timeout_iters`.
        """

        num_scheduled_tokens = self._compute_scheduled_tokens(
            context_requests, generation_requests)

        should_waiting = self.batch_wait_iters_count < self.batch_wait_timeout_iters and num_scheduled_tokens < self.batch_wait_max_tokens_ratio * self.max_num_tokens
        if should_waiting:
            self.batch_wait_iters_count += 1
            return []

        self.batch_wait_iters_count = 0
        return context_requests

    # REFACTOR: 2a (concern: schedule) -- main scheduling pass.
    @nvtx_range("_schedule")
    def _schedule(self):
        scheduler_output = self.scheduler.schedule_request(
            self.active_requests, self.inflight_req_ids)

        scheduled_context_requests = scheduler_output.context_requests
        if self.enable_attention_dp and self.attention_dp_enable_balance:
            scheduled_context_requests = self._balance_adp_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests)

        # If no generation requests, no need to wait, to avoid dead waiting
        should_check_waiting = not self.enable_attention_dp and self.enable_batch_waiting and len(
            scheduler_output.context_requests) > 0 and len(
                scheduler_output.generation_requests) > 0
        if should_check_waiting:
            scheduled_context_requests = self._waiting_requests(
                scheduler_output.context_requests,
                scheduler_output.generation_requests)

        scheduled_requests = ScheduledRequests()
        scheduled_requests.reset_context_requests(scheduled_context_requests)
        scheduled_requests.generation_requests = scheduler_output.generation_requests
        scheduled_requests.paused_requests = scheduler_output.paused_requests

        return scheduled_requests, scheduler_output.fitting_disagg_gen_init_requests, scheduler_output.num_fitting_requests

    # REFACTOR: 2a (concern: disagg) -- SCHEDULE_0 probe.
    @nvtx_range("_check_disagg_gen_transfer_status")
    def _check_disagg_gen_transfer_status(self):

        need_check = any([
            req.is_disagg_generation_transmission_in_progress
            for req in self.active_requests
        ])
        non_gen_first_reqs = [
            req for req in self.active_requests
            if req.py_disaggregated_params and req.py_disaggregated_params.
            schedule_style != DisaggScheduleStyle.GENERATION_FIRST
        ]
        need_check_one = bool(non_gen_first_reqs) and all(
            req.is_disagg_generation_transmission_in_progress
            for req in non_gen_first_reqs)

        if need_check:
            at_least_num = 1 if need_check_one else 0
            self._check_disagg_gen_cache_transfer_status(at_least_num)

        return

    # REFACTOR: 2a (concern: disagg) -- SCHEDULE_0 / RESPOND_8 sweep.
    @nvtx_range("_check_kv_transfer_timeout")
    def _check_kv_transfer_timeout(self):
        if not self.kv_cache_transceiver:
            return
        timeout_ms = self.kv_cache_transceiver.kv_transfer_timeout_ms
        if timeout_ms is None:
            return

        def flag_if_kv_transfer_timed_out(req: LlmRequest, type: str) -> None:
            current_time = time.time()
            if req.py_kv_transfer_start_time is None:
                return
            elapsed_time = (current_time - req.py_kv_transfer_start_time) * 1000
            if elapsed_time > timeout_ms and not req.py_kv_transfer_timed_out:
                logger.warning(
                    f"Terminating {type} request {req.py_request_id} due to KV cache transfer timeout"
                )
                req.py_kv_transfer_timed_out = True

        for req in self.async_transfer_manager.requests_in_transfer().values():
            flag_if_kv_transfer_timed_out(req, "context")

        for req in self.active_requests:
            if req.is_disagg_generation_transmission_in_progress:
                flag_if_kv_transfer_timed_out(req, "generation")

        return

    # REFACTOR: 2a (concern: disagg) -- SCHEDULE_0 probe.
    @nvtx_range("_check_disagg_ctx_schedulable_status")
    def _check_disagg_ctx_schedulable_status(self,
                                             new_requests: List[LlmRequest]):
        """
        In context-first mode, context requests are schedulable immediately,
        otherwise, we need to check if context requests are ready to be scheduled by querying kv cache transceiver
        """
        if not self.kv_cache_transceiver:
            return
        gen_first_ctx_requests = [
            req for req in new_requests
            if req.is_context_only_request and req.py_disaggregated_params.
            schedule_style == DisaggScheduleStyle.GENERATION_FIRST
        ]
        # Always call prepare_context_requests when there are new requests
        # or previously-waiting requests, so the tp_allgather consensus
        # can promote requests whose peer info has arrived on all ranks.
        self.kv_cache_transceiver.prepare_context_requests(
            gen_first_ctx_requests)

    # REFACTOR: 2a (concern: schedule) -- ADP padding helper.
    # Body becomes ``ctx.svc.pool.count_schedulable()`` -- it filters
    # the pool by the "not awaiting KV transfer" predicate. Lives on
    # the pool service, not the schedule concern, because other
    # callers may want it too.
    def _count_schedulable_active_requests(self) -> int:
        """Count active requests that are ready for scheduling.

        In non-disaggregated mode, all active requests are schedulable.
        In disaggregated mode, requests still waiting for KV cache
        transfer (in INIT or transmission-in-progress state) are
        excluded because they cannot participate in the forward pass
        until transfer completes.

        Returns:
            The number of active requests eligible for scheduling.
        """
        if self.kv_cache_transceiver is None:
            return len(self.active_requests)

        def _is_awaiting_kv_transfer(req) -> bool:
            return (req.is_disagg_generation_init_state
                    or req.is_disagg_generation_transmission_in_progress)

        return sum(1 for req in self.active_requests
                   if not _is_awaiting_kv_transfer(req))

    # REFACTOR: 2a (concern: schedule) -- ADP padding helper used during
    # benchmark-disagg fill phase.
    def _should_skip_dummy_for_benchmark_disagg(
            self, num_schedulable_requests: int) -> bool:
        """Decide whether to skip ADP dummy insertion during benchmark disagg fill.

        During the fill phase (``_benchmark_fill_phase_active`` is True),
        the ``can_forward`` gate prevents forward-pass collectives, so
        temporarily-empty ranks don't need dummies.  Dummies added during
        the fill phase would never be cleaned up (termination only runs
        after a forward pass), permanently wasting KV cache slots.

        Once the fill phase completes and the gate opens, the flag is
        cleared and this method stops skipping — the normal dummy
        add-forward-terminate lifecycle handles taper-down correctly
        (e.g., when ranks empty out at different rates due to varied
        speculative decoding acceptance rates).

        Args:
            num_schedulable_requests: Number of active requests that have
                completed KV transfer and are ready for the forward pass.

        Returns:
            True if dummy insertion should be skipped for this iteration.
        """
        if not self._benchmark_fill_phase_active or self.is_warmup:
            return False

        logger.info(f"Skipped adding dummy requests: "
                    f"num_fetch_requests={self.num_fetch_requests}, "
                    f"num_schedulable_requests={num_schedulable_requests}")
        return True

    # REFACTOR: 2a (concern: schedule) -- ADP dummy padding at SCHEDULE_0.
    # Admit the dummy via ``ctx.svc.pool.add_active(dummy)``; the later
    # eviction (in sample's state-advance) calls
    # ``ctx.svc.pool.remove_active([dummy])``.
    @nvtx_range("_pad_attention_dp_dummy_request")
    def _pad_attention_dp_dummy_request(self):
        """
        Pad with a generation dummy request, if required, to ensure every attention_dp rank has at least one active request.
        """
        if not self.enable_attention_dp:
            return

        assert self.expected_num_active_requests >= len(self.active_requests)
        num_active_request = self._count_schedulable_active_requests()

        if self._should_skip_dummy_for_benchmark_disagg(num_active_request):
            return

        # Other ranks have work but this rank is idle — insert a dummy so
        # it can participate in collective operations during the forward pass.
        if num_active_request == 0 and self.expected_num_active_requests > 0:
            llm_request = self.kv_cache_manager.add_dummy_requests(
                request_ids=[0],
                is_gen=True,
                prepare_resource=True,
                max_num_draft_tokens=self.max_total_draft_tokens,
            )[0]
            llm_request.is_attention_dp_dummy = True
            spec_resource_manager = self.resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER)
            if spec_resource_manager is not None:
                spec_resource_manager.add_dummy_requests([0])
            self.active_requests.append(llm_request)

    # REFACTOR: 1b (multi-concern, inlined+split) -- body sequence
    # ``disagg -> resource -> disagg``: build a ``ScheduledRequests``
    # holder (disagg side; could equally be built by schedule), prep
    # KV resources for it (resource side), submit the async KV recv
    # (disagg side).
    #
    # Split across SCHEDULE_0 -> RESOURCE_PREP_1 (the latter phase
    # was added precisely to make this split possible -- the
    # happens-before rule on storage views means writes at phase P
    # are visible only at phases > P, so two concerns at the SAME
    # phase cannot exchange storage data):
    #
    #   * SCHEDULE_0: ``schedule`` writes
    #     ``fitting_disagg_gen_init_requests``;
    #     ``disagg`` packages it into ``disagg_gen_init_to_prepare``
    #     (both writes go to SCHEDULE_0's write view -- legal because
    #     the writers are different concerns).
    #   * RESOURCE_PREP_1: ``resource`` reads
    #     ``disagg_gen_init_to_prepare`` and runs ``prepare_resources``
    #     for each resource manager type;
    #     ``disagg`` reads ``fitting_disagg_gen_init_requests`` and
    #     calls ``_recv_disagg_gen_cache`` to submit the async KV recv.
    #
    # See ``BatchPhase.RESOURCE_PREP_1`` and the
    # ``disagg_gen_init_to_prepare`` field in
    # ``batch_storage.py`` for the storage / phase contract.
    @nvtx_range("_prepare_disagg_gen_init")
    def _prepare_disagg_gen_init(self, fitting_disagg_gen_init_requests):
        if fitting_disagg_gen_init_requests:
            disagg_gen_init_to_prepare = ScheduledRequests()
            disagg_gen_init_to_prepare.context_requests_last_chunk = fitting_disagg_gen_init_requests

            for resource_mgr_type in (
                    ResourceManagerType.KV_CACHE_MANAGER,
                    ResourceManagerType.SPEC_RESOURCE_MANAGER,
                    ResourceManagerType.DRAFT_KV_CACHE_MANAGER):
                if (resource_mgr_type in self.resource_manager.resource_managers
                        and self.resource_manager.
                        resource_managers[resource_mgr_type] is not None):
                    self.resource_manager.resource_managers[
                        resource_mgr_type].prepare_resources(
                            disagg_gen_init_to_prepare)

            # Trigger KV cache exchange for new disagg_gen_init_requests
            self._recv_disagg_gen_cache(fitting_disagg_gen_init_requests)

    # REFACTOR: 2a (concern: disagg) -- FORWARD_2 step that promotes
    # transmission-complete gen requests + sets up sampler step.
    @nvtx_range("_prepare_disagg_gen_transmission_complete")
    def _prepare_disagg_gen_transmission_complete(self, scheduled_batch):
        cache_trans_complete_requests = []
        for req in scheduled_batch.generation_requests:
            if req.is_disagg_generation_transmission_complete:
                cache_trans_complete_requests.append(req)
        if len(cache_trans_complete_requests) > 0:
            requests = ScheduledRequests()
            requests.context_requests_last_chunk = cache_trans_complete_requests
            self.resource_manager.resource_managers[
                ResourceManagerType.SEQ_SLOT_MANAGER].prepare_resources(
                    requests)
            self._setup_sampler_step(requests)

        for req in scheduled_batch.generation_requests:
            if req.is_disagg_generation_transmission_complete:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.context_current_position = req.prompt_len
                req.decoding_iter = 1
                req.py_decoding_iter = 1
                req.py_kv_transfer_start_time = None
                req.py_kv_transfer_timed_out = False
                first_gen_tokens = req.context_phase_params.first_gen_tokens
                ctx_draft_tokens = req.context_phase_params.draft_tokens
                req.py_draft_tokens = [] if ctx_draft_tokens is None else ctx_draft_tokens
                beam_width = req.sampling_config.beam_width
                for beam in range(0, beam_width):
                    req.add_new_token(first_gen_tokens[beam], beam)

                self._maybe_prepend_logprobs_and_logits(req, beam_width)

    # REFACTOR: 2a (concern: disagg) -- helper of
    # ``_prepare_disagg_gen_transmission_complete``.
    def _maybe_prepend_logprobs_and_logits(self, req, beam_width):
        """Prepend logprobs and generation logits for first_gen_tokens
        if transferred from prefill."""
        disagg_params = getattr(req, 'py_disaggregated_params', None)
        if disagg_params is None:
            return

        if getattr(disagg_params, 'first_gen_log_probs', None) is not None:
            if beam_width != 1:
                logger.warning(
                    "Skipping first_gen_log_probs prepend for "
                    "request %s: beam_width=%s is not supported.",
                    req.py_request_id, beam_width)
            else:
                req.py_result.append_log_probs(
                    [disagg_params.first_gen_log_probs])

        if getattr(disagg_params, 'first_gen_logits', None) is not None:
            if beam_width != 1:
                logger.warning(
                    "Skipping first_gen_logits prepend for "
                    "request %s: beam_width=%s is not supported.",
                    req.py_request_id, beam_width)
            else:
                device = torch.device('cuda', self.device_id)
                for logits_tensor in disagg_params.first_gen_logits:
                    req.py_result.append_generation_logits(
                        logits_tensor.to(device))

    # REFACTOR: 2a (concern: response) -- used by ``_handle_first_token_response``.
    def _has_prepended_logits(self, req) -> bool:
        """Check whether the request has first-gen logits prepended from
        prefill that need a snapshot before response creation."""
        if not self.should_exclude_last_generation_logits:
            return False
        disagg_params = getattr(req, 'py_disaggregated_params', None)
        if disagg_params is None:
            return False
        return getattr(disagg_params, 'first_gen_logits', None) is not None

    # REFACTOR: 2a (concern: disagg) -- helper of ``_prepare_disagg_gen_init``
    # (KV recv submission to ctx worker).
    @nvtx_range("_recv_disagg_gen_cache")
    def _recv_disagg_gen_cache(self, new_gen_reqs):

        # For gen-only benchmarking, mark new gen request as transmission complete right away
        if os.getenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY") == "1":
            for req in new_gen_reqs:
                req.state = LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE
            return

        if os.getenv("TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP") == "1":
            for req in new_gen_reqs:
                self.kv_cache_transceiver.request_and_receive_sync(req)
        else:
            for req in new_gen_reqs:
                self.kv_cache_transceiver.request_and_receive_async(req)

        if self.kv_cache_transceiver.kv_transfer_timeout_ms is not None:
            for req in new_gen_reqs:
                if req.state == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS:
                    req.py_kv_transfer_start_time = time.time()

        non_gen_first_active = [
            req for req in self.active_requests
            if req.py_disaggregated_params and req.py_disaggregated_params.
            schedule_style != DisaggScheduleStyle.GENERATION_FIRST
        ]
        block_transfer = bool(non_gen_first_active) and all(
            req.is_disagg_generation_transmission_in_progress
            for req in non_gen_first_active)
        self._check_disagg_gen_cache_transfer_status(1 if block_transfer else 0)

        return

    # REFACTOR: 1b (multi-concern, inlined+split) -- annotated in-line as
    # RESPOND_8 fan-out across disagg (start ctx KV send + opportunistic
    # probe) + kv_connector (flag finished requests for async save).
    # Body splits into the respective concerns' RESPOND_8 blocks.
    @nvtx_range("_send_kv_async")
    def _send_kv_async(self, scheduled_requests: List[LlmRequest]):
        # ===================================================================
        # Multi-concern. All work here is at P_RESPOND.
        #   * `disagg` block: starts ctx-side KV send for finished context
        #     requests (paired with the ctx-side recv promotion at
        #     P_SCHEDULE in `_check_disagg_ctx_schedulable_status`).
        #   * `kv_connector` block: kicks off async save for finished
        #     requests via the connector manager (paired with
        #     `_kv_connector_start_batch` and `wait_for_save` at
        #     P_SCHEDULE/P_FORWARD, and `_kv_connector_terminate_requests`
        #     later at P_RESPOND).
        #   * trailing `disagg` probe: opportunistic completion sweep for
        #     ctx KV transfers (same concern as the call later in
        #     `_executor_loop`, kept here so freshly-started transfers can
        #     be reaped if they completed synchronously).
        # ===================================================================

        def kv_connector_request_finished(req: LlmRequest):
            try:
                cache_block_ids = self.kv_cache_manager.get_cache_indices(req)
            except Exception as e:
                logger.warning(
                    f"Unable to get cache blocks for request {req.py_request_id}. Skipping asynchronous saving: {e}"
                )
            else:
                if self.kv_connector_manager.request_finished(
                        req, cache_block_ids):
                    self.async_transfer_manager.start_transfer(req)

        if self.kv_cache_transceiver:
            # Concern: disagg
            # Task: start async ctx->gen KV send for finished ctx-only
            #       requests; respond_and_send_async also produces the
            #       ctx-side response packet.
            # Consume: scheduled_requests
            # Produce: ctx-side responses + send handles (out-of-band)
            for req in scheduled_requests:
                if req.is_context_only_request and (
                        req.is_context_finished or req.is_finished_due_to_length
                ) and not req.is_finished_due_to_cancellation:
                    # Order is important here: we need to start the transfer before responding
                    # to make sure the blocks are stored for reuse before they are sent.
                    self.async_transfer_manager.start_transfer(req)
                    self.kv_cache_transceiver.respond_and_send_async(req)

                    if self.kv_cache_transceiver.kv_transfer_timeout_ms is not None:
                        req.py_kv_transfer_start_time = time.time()

        if self.kv_connector_manager:
            # Concern: kv_connector
            # Task: ask the connector to flag finished requests for async
            #       save and start their transfer. In overlap mode the
            #       finished requests come from the *previous* batch; in
            #       non-overlap mode they come from the current
            #       scheduled_requests.
            # Consume: scheduled_requests / self.previous_batch
            # Produce: connector save handles (out-of-band)
            if not self.disable_overlap_scheduler:
                requests = self.previous_batch.scheduled_requests.all_requests(
                ) if self.previous_batch is not None else []
            else:
                requests = scheduled_requests
            for req in requests:
                if req.is_finished:
                    kv_connector_request_finished(req)

        if self.kv_cache_transceiver:
            # Concern: disagg (opportunistic completion probe)
            # Task: non-blocking sweep for ctx transfers that already
            #       completed since iter top — terminates those requests
            #       so their KV blocks can be reused next iter.
            # Consume: -
            # Produce: terminated requests (out-of-band)
            self._check_disagg_ctx_cache_transfer_status(0)

    # REFACTOR: 2a (concern: disagg) -- error-state lookup helper.
    def _get_disagg_reqs_in_error_state(self):
        return [
            req for req in self.active_requests
            if req.state == LlmRequestState.DISAGG_TRANS_ERROR
        ]

    # REFACTOR: 2a (concern: disagg) -- common error sweep helper for
    # ctx / gen cache transfer status checks.
    def _check_cache_transfer_errors(self, error_msg_prefix: str):
        """Common helper to check for and handle cache transfer errors."""
        error_requests = self._get_disagg_reqs_in_error_state()
        if error_requests:
            self._handle_errors(
                f"Error in kv cache transfer for {error_msg_prefix}",
                requests=error_requests,
                charge_budget=False)

    # REFACTOR: 2a (concern: disagg) -- ctx cache transfer status sweep.
    @nvtx_range("_check_disagg_ctx_cache_transfer_status")
    def _check_disagg_ctx_cache_transfer_status(self, atLeastNum: int = 0):
        finished_requests, error_requests = self.kv_cache_transceiver.check_context_transfer_status(
            atLeastNum)

        completed_req_ids = set(finished_requests + error_requests)

        requests_in_transfer = self.async_transfer_manager.requests_in_transfer(
        )

        for request_id in completed_req_ids:

            if request_id not in requests_in_transfer:
                logger.warning(
                    f"Request {request_id} not found in transfer manager")
                continue

            request = requests_in_transfer[request_id]

            self._end_transfer_and_maybe_terminate(request)

        # The set of requests in transfer may have changed since we terminated some requests.
        requests_in_transfer = self.async_transfer_manager.requests_in_transfer(
        )

        for request_id in list(requests_in_transfer.keys()):
            request = requests_in_transfer[request_id]
            if request.py_kv_transfer_timed_out and request_id not in completed_req_ids:
                is_cancelled = self.kv_cache_transceiver.cancel_request(request)
                # If cancel is successful, mark as complete so it can be cleaned up
                # Otherwise, try at next iteration
                if is_cancelled:
                    request.py_kv_transfer_start_time = None
                    request.state = LlmRequestState.DISAGG_CONTEXT_COMPLETE

                    self._end_transfer_and_maybe_terminate(request)

        self._check_cache_transfer_errors("context requests")

    # REFACTOR: 2a (concern: disagg) -- gen cache transfer status sweep.
    @nvtx_range("_check_disagg_gen_cache_transfer_status")
    def _check_disagg_gen_cache_transfer_status(self, atLeastNum: int = 0):
        result = self.kv_cache_transceiver.check_gen_transfer_status(atLeastNum)
        if isinstance(result, tuple):
            _, _, cancelled_reqs = result
            user_canceled_set = set(self.canceled_req_ids)
            for req in cancelled_reqs:
                req_id = req.py_request_id if not req.is_child else req.parent_request_id
                if req_id not in user_canceled_set:
                    req.state = LlmRequestState.DISAGG_TRANS_ERROR
        self._check_cache_transfer_errors("generation requests")

    # REFACTOR: 2a (concern: forward) -- FORWARD_2 main step. Internally
    # invokes ``_kv_connector_wait_for_save`` (kv_connector deadline
    # wait) -- in the new design that becomes a kv_connector method
    # called from forward's body, not folded in here.
    def _forward_step(
            self,
            scheduled_requests: ScheduledRequests,
            new_tensors_device: Optional[SampleStateTensors] = None,
            num_accepted_tokens_device: Optional[torch.Tensor] = None):
        ExpertStatistic.set_iter(self.iter_counter)

        @nvtx_range(
            f"[Executor] _forward_step {self.iter_counter}: {scheduled_requests.num_context_requests} ctx reqs, {scheduled_requests.num_generation_requests} gen reqs"
        )
        def forward(scheduled_requests, resource_manager, new_tensors_device,
                    gather_context_logits, cache_indirection_buffer,
                    num_accepted_tokens_device):
            return self.model_engine.forward(
                scheduled_requests,
                resource_manager,
                new_tensors_device,
                gather_context_logits=gather_context_logits,
                cache_indirection_buffer=cache_indirection_buffer,
                num_accepted_tokens_device=num_accepted_tokens_device)

        try:
            # Concern: forward
            # Task: collect per-request flags (gather_context_logits,
            #       cache indirection); queue the model forward on
            #       execution_stream so it overlaps with main-stream
            #       KVCacheTransferManager onboard/offload work.
            # Consume: scheduled_requests, prepared resources,
            #          new_tensors_device, num_accepted_tokens_device
            # Produce: outputs (logits + extras)
            gather_context_logits = any(
                a.py_return_context_logits
                for a in scheduled_requests.context_requests)
            cache_indirection_buffer = self.sampler.get_cache_indirection()

            # Run model forward on the execution stream for proper synchronization
            # with KVCacheTransferManager's onboard/offload operations.
            self.execution_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.execution_stream):
                outputs = forward(scheduled_requests, self.resource_manager,
                                  new_tensors_device, gather_context_logits,
                                  cache_indirection_buffer,
                                  num_accepted_tokens_device)

            # Ensure the default stream waits for execution_stream to complete
            # before downstream operations use the outputs.
            torch.cuda.current_stream().wait_stream(self.execution_stream)

            # Concern: kv_connector
            # Task: deadline-wait counterpart of `_kv_connector_start_batch`
            #       at P_SCHEDULE — must complete before the next iter's
            #       prepare_resources reuses block layouts.
            # Consume: -
            # Produce: -
            self._kv_connector_wait_for_save()

            return outputs
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(
                f"Encountered an error in forward function: {error_msg}")
            self._handle_errors(error_msg)
            return None

    # REFACTOR: 2a (concern: spec_decode) -- late state mark used in
    # overlap / PP variants.
    def _update_generation_requests_that_will_complete_next_iteration(
            self, generation_requests: list[LlmRequest]):
        """ Update the generation requests that will complete next iteration.

        If overlap scheduling is enabled, we need update the state of generation requests that will complete next iteration
        and adjust the exclude_last_generation_logits flag accordingly.
        """
        for request in generation_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE and request.will_complete_next_iteration(
            ):
                request.set_exclude_last_generation_logits(False)
                request.state = LlmRequestState.GENERATION_TO_COMPLETE

    # REFACTOR: 2a (concern: sample) -- TP variant of state advance.
    def _update_request_states_tp(self, scheduled_requests: ScheduledRequests):
        # handle potential attention dp dummy request
        if self.active_requests and self.active_requests[
                -1].is_attention_dp_dummy:
            request = self.active_requests[-1]
            request.state = LlmRequestState.GENERATION_COMPLETE
            self.inflight_req_ids.erase(request.py_request_id)
            self._terminate_request(request)
            self.active_requests.remove(request)

        for request in scheduled_requests.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:  # skip failed requests
                request.py_last_context_chunk = (
                    request.context_current_position,
                    request.context_current_position +
                    request.context_chunk_size)
                request.move_to_next_context_chunk()
            if request.context_remaining_length == 0:
                # Prefill is done for this request; drop pinned encoder outputs
                # (multimodal_embedding) and raw pre-encoder tensors that multimodal models stashed
                # on `py_multimodal_data`. Without this, encoder inputs and outputs for multi-modal
                # requests stay pinned on GPU through the full decode lifetime and can lead to OOMs
                # at high concurrency.
                _strip_py_multimodal_data_post_prefill(request)
                if not self.disable_overlap_scheduler and request.will_complete_next_iteration(
                ):
                    request.set_exclude_last_generation_logits(False)
                    request.state = LlmRequestState.GENERATION_TO_COMPLETE
                else:
                    request.state = LlmRequestState.GENERATION_IN_PROGRESS

    # REFACTOR: 2a (concern: sample) -- star-attention variant of state advance.
    def _update_request_states_star_attention(
            self, scheduled_requests: ScheduledRequests):
        for request in scheduled_requests.context_requests:
            if request.ctx_iters >= len(request.ctx_blocks) - 2:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS
            request.ctx_iters += 1

        for request in scheduled_requests.generation_requests:
            request.gen_iters += 1

    # REFACTOR: 2a (concern: sample) -- STATE_UPD_4 dispatch (TP/CP/HELIX).
    @nvtx_range("_update_request_states")
    def _update_request_states(self, scheduled_requests: ScheduledRequests):
        cp_config = self.dist.cp_config
        if 'cp_type' in cp_config:
            cp_type = cp_config['cp_type']
            if cp_type == CpType.STAR:
                self._update_request_states_star_attention(scheduled_requests)
            elif cp_type == CpType.HELIX:
                # Take the usual route with _update_request_states_tp().
                pass
            else:
                raise NotImplementedError(
                    f'Unsupported cp type {cp_type.name}.')
        self._update_request_states_tp(scheduled_requests)

    # REFACTOR: 2a (concern: sample) -- SAMPLE_3 main step.
    @nvtx_range("_sample_async")
    def _sample_async(self, scheduled_batch,
                      batch_outputs) -> SampleState | None:
        try:
            if batch_outputs is not None:
                num_context_logits_prefix_sum = [0]
                prefix_sum = 0
                num_context_tokens = 0
                for request in scheduled_batch.context_requests:
                    context_chunk_size = request.context_chunk_size
                    prefix_sum += context_chunk_size if request.py_return_context_logits else 1
                    num_context_logits_prefix_sum.append(prefix_sum)
                    num_context_tokens += context_chunk_size

                beam_width = self.sampler.beam_width(
                    scheduled_batch.all_requests())

                HandleLogits()(scheduled_batch.context_requests,
                               scheduled_batch.generation_requests,
                               batch_outputs["logits"], beam_width,
                               num_context_logits_prefix_sum,
                               self.sampler.is_generation_model())

                HandleAdditionalOutputs()(scheduled_batch.context_requests,
                                          scheduled_batch.generation_requests,
                                          batch_outputs, beam_width,
                                          num_context_tokens)

                return self.sampler.sample_async(scheduled_batch, batch_outputs,
                                                 num_context_logits_prefix_sum)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

    # REFACTOR: 2a (concern: disagg) -- thin wrapper around
    # ``self.sampler.setup_sampler_step``; only caller is
    # ``_prepare_disagg_gen_transmission_complete`` (disagg). After
    # refactor disagg's coroutine calls ``sampler.setup_sampler_step``
    # directly and this wrapper goes away.
    @nvtx_range("_setup_sampler_step")
    def _setup_sampler_step(self, requests: ScheduledRequests):
        try:
            return self.sampler.setup_sampler_step(requests)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

    # REFACTOR: 2a (concern: sample) -- APPLY_7 main step (blocks on
    # sampler_event, applies sampled tokens to requests).
    @nvtx_range("_update_requests")
    def _update_requests(self,
                         sample_state: SampleState,
                         resource_manager: Optional[ResourceManager] = None):
        try:
            self.sampler.update_requests(sample_state, resource_manager)
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in sampling: {error_msg}")
            self._handle_errors(error_msg)

    # REFACTOR: 1b (legacy helper conflating TWO failure modes; splits
    # in the new design):
    #
    # Mode A -- CATASTROPHIC (callers pass NO ``requests=``; default
    # is "fail every active request + clear active_requests"). The
    # concern can't recover and the rest of the batch / loop can't
    # safely proceed. Today's call sites: hang detector, exceptions
    # caught inside ``_forward_step`` / ``_sample_async`` /
    # ``_setup_sampler_step`` / ``_update_requests`` /
    # ``_prepare_draft_requests``.
    #
    # In the new design the concern's coroutine just RAISES (no
    # special helper, no try/except wrapper around the body). The
    # exception propagates: concern -> BATCH -> SCHEDULER. The
    # SCHEDULER catches at the top of its loop, calls
    # ``fail_requests(active_requests, msg)`` (Mode B's utility) on
    # the active set, and sets the shutdown latch. The intermediate
    # ``GeneratorExit`` thrown into still-suspended concerns by the
    # Driver runs their ``finally:`` for concern-LOCAL cleanup
    # (CUDA event handles, etc.) -- but NOT for the failing of
    # requests, which is the SCHEDULER's job.
    #
    # Mode B -- PER-REQUEST FAIL-FAST (callers pass an explicit
    # ``requests=`` arg). A specific subset of requests is bad; the
    # rest of the batch and the loop continue. Today's call sites:
    # ``_validate_request`` failure (one bad request), guided
    # decoder errors (grammar-failed subset), KV transfer timeout
    # (timed-out subset), cache transfer error (DISAGG_TRANS_ERROR
    # subset), benchmark gen-only KV exhaustion (the active set,
    # explicitly named).
    #
    # In the new design this becomes a 2b free-function utility:
    # ``fail_requests(ctx, reqs, msg)`` in ``concerns/_shared.py``.
    # Body composes three services::
    #
    #     def fail_requests(ctx, reqs, msg):
    #         error_responses = []
    #         for req in reqs:
    #             req.state = LlmRequestState.GENERATION_COMPLETE
    #             error_responses.append((req.py_request_id, LlmResponse(
    #                 request_id=req.py_request_id,
    #                 error_msg=msg,
    #                 client_id=req.py_client_id,
    #             )))
    #         ctx.svc.pool.remove_active(reqs)
    #         ctx.svc.client.enqueue(error_responses)
    #         for req in reqs:
    #             ctx.svc.termination.terminate(req)
    #
    # Touches multiple service-owned bits of state but each call is
    # itself a single-purpose service method -- the utility is just
    # the composition. Same trade-off as ``_terminate_request``.
    def _handle_errors(self,
                       error_msg: Optional[str] = None,
                       *,
                       requests: Optional[List[LlmRequest]] = None,
                       charge_budget: bool = True) -> None:
        """Fail requests and optionally initiate shutdown on fatal errors.

        When ``charge_budget`` is True (the default), classifies the error
        via the error budget.  If deemed fatal (immediate-fatal pattern or
        budget exhausted), **all** active requests are failed and a shutdown
        is enqueued.  Otherwise only the requests in *requests* are failed.

        When ``charge_budget`` is False, the error is treated as a
        per-request failure: only the specified requests are failed, the
        error budget is not consumed, and shutdown is never triggered.
        Use this for request-scoped errors (validation, KV-transfer
        timeout, guided-decoder) that should not affect server health.

        .. note::
            The ``charge_budget=False`` path reuses the full
            ``_handle_errors`` machinery (queue drain, response
            enqueue, terminate) even though it only needs to fail a
            single request.  A future improvement would be to extract
            a lightweight ``_fail_request(request, error_msg)`` helper
            for request-scoped failures, keeping ``_handle_errors``
            focused on system-level errors that may crash the engine.

        Args:
            error_msg: Human-readable error description.  Defaults to
                ``"error"`` when ``None``.
            requests: Subset of active requests to fail.  When ``None``
                (or when the error is fatal), all ``active_requests`` are
                failed.
            charge_budget: Whether to consume the error budget.  Set to
                False for request-scoped errors that should not affect
                server health.
        """
        error_responses: Dict[int, LlmResponse] = {}
        error_msg = error_msg or "error"

        is_fatal = (self._error_budget.consume(error_msg)
                    if charge_budget else False)
        if is_fatal and self._error_budget.budget < 1e-9:
            logger.error(f"Error budget exhausted "
                         f"(budget={self._error_budget.budget:.3f}), "
                         "treating as fatal")

        if is_fatal:
            self._fatal_error = RuntimeError(f"Fatal error: {error_msg}")
            self.is_shutdown = True
            logger.error(
                f"Fatal error detected, initiating shutdown: {error_msg}")
            requests = None

            # Drain waiting_queue so that queued-but-not-yet-activated
            # requests don't get picked up on the next iteration.
            # These are RequestQueueItems (not yet LlmRequests), so we
            # fail them via error responses.  Buffer all responses and
            # call _enqueue_responses once after the loop so every rank
            # enters the same number of collectives (attention-DP /
            # gather-all modes use collective gathers internally).
            waiting_responses: List[Tuple[int, LlmResponse]] = []
            while self.waiting_queue:
                item = self.waiting_queue.pop_request()
                if (self.gather_all_responses
                        or self.dist.rank == 0) and item.request is not None:
                    waiting_responses.append(
                        (item.id,
                         LlmResponse(request_id=item.id,
                                     error_msg=error_msg,
                                     client_id=getattr(item.request,
                                                       'client_id', None))))
            # Also drain executor_request_queue so items already queued
            # but not yet fetched by the main loop are not scheduled
            # after the CUDA context is corrupted.  Safe to use empty()
            # here because is_shutdown is True and the queue's active
            # flag is about to be set False, so no new items arrive.
            raw_queue = self.executor_request_queue.get_request_queue()
            while not raw_queue.empty():
                item = raw_queue.get_nowait()
                if item.is_shutdown_request:
                    continue
                if ((self.gather_all_responses or self.dist.rank == 0)
                        and item.request is not None):
                    waiting_responses.append(
                        (item.id,
                         LlmResponse(request_id=item.id,
                                     error_msg=error_msg,
                                     client_id=getattr(item.request,
                                                       'client_id', None))))

            if waiting_responses:
                self._enqueue_responses(waiting_responses)
                logger.info(f"Drained {len(waiting_responses)} queued requests "
                            "on fatal error")

        failed_requests = (list(self.active_requests)
                           if requests is None else requests)
        for request in failed_requests:
            req_id = request.py_request_id
            request.state = LlmRequestState.GENERATION_COMPLETE
            error_responses[req_id] = LlmResponse(
                request_id=req_id,
                error_msg=error_msg,
                client_id=request.py_client_id)
        if requests is None:
            self.active_requests.clear()
        else:
            self.active_requests = [
                request for request in self.active_requests
                if request not in requests
            ]
        self._enqueue_responses(list(error_responses.items()))
        for request in failed_requests:
            self._terminate_request(request)

        if self._fatal_error is not None:
            self.executor_request_queue.enqueue_shutdown_request()

    # REFACTOR: 2a-on-service -- body becomes
    # ``ctx.svc.termination.terminate(req)``. The
    # ``DisaggPPTerminationHandler`` reference lives as instance
    # state on ``TerminationService``; the dispatch (PP handler vs
    # direct path) is the body of ``terminate``. The legacy
    # ``_do_terminate_request`` collapses into the same method.
    #
    # Callers in the new design come from THREE sites:
    # * Normal path: response concern's RESPOND_8 (handle_responses
    #   + cancellation), schedule concern's paused-request
    #   termination, the disagg / kv_connector
    #   ``_end_transfer_and_maybe_terminate`` replacement.
    # * Error Mode A (catastrophic): SCHEDULER's top-of-loop
    #   exception handler iterates the active set and calls
    #   ``ctx.svc.termination.terminate(req)`` on each.
    # * Error Mode B (per-request fail-fast): the
    #   ``fail_requests(ctx, reqs, msg)`` utility iterates the
    #   named subset and calls it on each.
    def _terminate_request(self, request: LlmRequest):
        # Dummy requests don't participate in disagg KV cache transfers,
        # so they must bypass the PP termination handler to avoid stale
        # sequences in the KV cache manager (the handler delays removal,
        # but the dummy ID is reused every iteration).
        if (self._disagg_pp_termination_handler is not None
                and not request.is_dummy_request):
            self._disagg_pp_termination_handler.terminate(request)
        else:
            self._do_terminate_request(request)

    # REFACTOR: 2a-on-service -- body folds INTO
    # ``ctx.svc.termination.terminate(req)`` (collapsed with
    # ``_terminate_request``'s dispatch). Was the "direct path"
    # branch + the ``DisaggPPTerminationHandler`` callback; in the
    # new design the service exposes one ``terminate`` method that
    # picks the right path internally.
    def _do_terminate_request(self, request: LlmRequest):
        self.resource_manager.free_resources(request)

        if self.gather_all_responses or self.dist.rank == 0:
            self.result_wait_queues.pop(request.py_request_id, None)

    # REFACTOR: 2a (concern: response) -- helper of ``_try_cancel_request``.
    def _is_request_in_transmission(self, request) -> bool:
        """Check if a request is currently in transmission state."""
        return (request.state
                == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS
                or request.state
                == LlmRequestState.DISAGG_GENERATION_TRANS_IN_PROGRESS)

    # REFACTOR: 2a (concern: response) -- cancellation attempt helper.
    def _try_cancel_request(self, request) -> bool:
        """Check if a request can be canceled and attempt cancellation if needed.

        Returns:
            bool: True if the request can be canceled (either successfully cancelled or doesn't need cancellation).
        """
        if self.kv_cache_transceiver is None:
            return True

        if not self._is_request_in_transmission(request):
            return True

        return self.kv_cache_transceiver.cancel_request(request)

    # REFACTOR: 2a (concern: response) -- RESPOND_8 cancellation handler.
    # Reads ``ctx.port.canceled_req_ids`` (cross-thread incoming),
    # builds error responses for cancelled requests via
    # ``ctx.svc.client.enqueue(...)``, terminates via
    # ``ctx.svc.termination.terminate(req)``.
    @nvtx_range("_handle_canceled_requests")
    def _handle_canceled_requests(self):
        if len(self.canceled_req_ids) == 0:
            return

        # Create set from list of canceled request ids to speed up canceled test
        canceled_req_ids_set = set(self.canceled_req_ids)

        # Remove canceled requests from the waiting queue
        self.waiting_queue.remove_by_ids(canceled_req_ids_set)

        still_pending_canceled_ids = []
        for request in self.active_requests:
            req_id = request.py_request_id if not request.is_child else request.parent_request_id
            if req_id not in canceled_req_ids_set:
                continue

            is_cancelled = self._try_cancel_request(request)
            if is_cancelled:
                # Mark requests as finished, then, we reuse all existing code
                # to clean up the KV cache resources.
                request.finish_by_reason(FinishReason.CANCELLED)
                request.decoding_iter = request.py_decoding_iter
            else:
                still_pending_canceled_ids.append(req_id)

        # Clear list of requests marked for cancellation and add back those that failed to cancel.
        self.canceled_req_ids.clear()
        self.canceled_req_ids.extend(still_pending_canceled_ids)

    # REFACTOR: 2a-on-service -- body moves to
    # ``ctx.svc.client.enqueue(items)`` (the ``ClientChannel``
    # service). Encapsulates TP gather + cross-thread put on
    # ``MessagePort.responses`` + ``response_cv.notify_all`` +
    # per-request fan-out to ``MessagePort.result_wait_queues``.
    # All concerns / utilities that emit responses (response,
    # ``fail_requests``) call this via the service.
    @nvtx_range("_enqueue_responses")
    def _enqueue_responses(self, responses: Iterable[Tuple[int, LlmResponse]]):
        if 0 not in self.dist.mapping.tp_group and not self.gather_all_responses:
            return

        if self.enable_attention_dp and self.dist.world_size != 1:
            if not self.gather_all_responses:
                responses_list = self.dist.tp_gather(responses)
            else:
                responses_list = self.dist.allgather(responses)
            if self.dist.rank == 0 or self.gather_all_responses:
                gather_responses = []
                if responses_list is not None:
                    for resp in responses_list:
                        if resp is not None:
                            gather_responses.extend(resp)
                    responses = gather_responses
        logger.debug(
            f'after gather, rank = {self.dist.rank}, responses = {responses}')

        if self.dist.rank == 0 or self.gather_all_responses:
            with self.response_cv:
                for req_id, resp in responses:
                    if req_id in self.responses.keys():
                        self.responses[req_id].append(resp)
                    else:
                        self.responses.update({req_id: [resp]})
                    # (TODO: joyang) There are other types of responses, we need to sort out.
                    if type(
                            resp
                    ) == LlmResponse and req_id in self.result_wait_queues and self.result_wait_queues[
                            req_id] is not None:
                        self.result_wait_queues[req_id].put_response.remote(
                            resp.client_id, resp)
                self.response_cv.notify_all()

    # REFACTOR: 2a (concern: response) -- SCHEDULE_0 first-token emission
    # for newly-promoted disagg-gen requests. Body builds first-token
    # responses and calls ``ctx.svc.client.enqueue(items)``.
    @nvtx_range("_handle_first_token_response")
    def _handle_first_token_response(self, scheduled_batch):
        new_responses = []
        for req in scheduled_batch.generation_requests:
            if req.py_decoding_iter == 1:
                logger.debug(
                    f'Send first token response for request {req.py_request_id}'
                )
                # Snapshot prepended first_gen_logits for the response:
                #
                # 1. generation_logits is not stored on LlmResult; every
                #    access goes through __getattr__ -> PyResult property
                #    -> LogitsStorage, re-reading shared mutable state.
                # 2. All streaming responses reference the same PyResult,
                #    so later tokens mutate what earlier responses would
                #    read.
                # 3. With the overlap scheduler, exclude_last_generation_logits
                #    is True, which would hide the prepended logits since
                #    they are the only entry at this point.
                #
                # WAR: read the logits now (bypassing exclusion), then set
                # the tensor directly on LlmResult as an instance attribute.
                # This shadows __getattr__ so the consumer always gets the
                # correct, frozen logits.
                has_prepended_logits = self._has_prepended_logits(req)
                logits_snapshot = (req.py_result.get_latest_logits_unexcluded()
                                   if has_prepended_logits else None)
                response = req.create_response(False, self.dist.rank)
                if logits_snapshot is not None and response is not None:
                    response.result.generation_logits = logits_snapshot
                new_responses.append((req.py_request_id, response))

        self._enqueue_responses(new_responses)

    # REFACTOR: 1b (multi-concern, inlined+split) -- annotated in-line as
    # RESPOND_8 fan-out across response (build / enqueue / terminate) +
    # perf_metric (per-step metrics) + spec_decode (rolling acceptance
    # gate -- cross-iter feedback) + disagg (timeout cleanup +
    # ctx-complete terminate-policy fork). Body splits into the
    # respective concerns' RESPOND_8 blocks.
    #
    # Response concern's body (the bulk of this method) becomes:
    # iterates ``ctx.svc.pool``, builds responses, calls
    # ``ctx.svc.client.enqueue(items)`` for output, calls
    # ``ctx.svc.pool.remove_active(finished)`` for the post-iter
    # rewrite, calls ``ctx.svc.termination.terminate(req)`` for the
    # retire-now branch.
    @nvtx_range("_handle_responses")
    def _handle_responses(self):
        # ===================================================================
        # Multi-concern. Primary concern is `response`; this body also
        # interleaves three other concerns that share the per-request
        # iteration:
        #   * `perf_metric`: append_step_metrics / update_perf_metrics.
        #   * `disagg`: KV-transfer timeout cleanup, disagg ctx-state
        #     terminate-policy fork.
        #   * `spec_decode`: speculation_gate rolling acceptance update
        #     (which can flip self.speculation_permanently_disabled and
        #     thus disable spec_decode at the *next* iter's P_SCHEDULE).
        # All work here is at P_RESPOND.
        # ===================================================================
        new_responses = []
        requests_to_terminate = []
        # Requests terminated by _check_disagg_ctx_cache_transfer_status (DISAGG_CONTEXT_COMPLETE);
        # included in the return value for stats but not re-terminated here.
        requests_finished_by_transfer = []
        new_active_requests = []
        logger.debug(
            f'------before _handle_responses, rank = {self.dist.rank}, output = {self.active_requests}'
        )

        batch_token_time = self.perf_manager.get_timestamp()

        for request in self.active_requests:
            req_id = request.py_request_id
            # Concern: response (dummy)
            # Task: ADP dummy never produces a response; just terminate.
            # no responses for dummy request, and finish it
            if request.is_attention_dp_dummy:
                requests_to_terminate.append(request)
                continue

            # Concern: disagg (timeout cleanup)
            # Task: requests whose KV transfer timed out (flagged by
            #       `_check_kv_transfer_timeout` at P_SCHEDULE / P_RESPOND)
            #       are cancelled and erred. Multi-concern leak: this
            #       branch belongs to the `disagg` coroutine in the
            #       refactored design, not `response`.
            # Check if generation request needs cleanup due to KV cache transfer timeout
            if request.py_kv_transfer_timed_out:
                is_cancelled = self.kv_cache_transceiver.cancel_request(request)
                if is_cancelled:
                    self._handle_errors(
                        error_msg=f"Request {request.py_request_id} timed out",
                        requests=[request],
                        charge_budget=False)
                continue

            # Concern: response (skip-emit) + perf_metric (step)
            # Task: gen request still in transmission OR overlap-mode's
            #       first-token already emitted at P_SCHEDULE — skip the
            #       response but still record per-step metrics.
            if request.is_generation_only_request() and not request.is_finished:
                # If request is in transmission, so we don't need to emit a response
                # Also, for the first iteration with overlap, we should skip since first
                # token has already been emitted previously
                if request.is_disagg_generation_transmission_in_progress or (
                        not self.disable_overlap_scheduler
                        and request.py_decoding_iter <= 1):
                    self.perf_manager.append_step_metrics(
                        request,
                        self.iter_counter,
                        batch_token_time=batch_token_time)
                    new_active_requests.append(request)
                    continue

            # Concern: response (per-request snapshot)
            # Task: copy py_draft_tokens / py_decoding_iter onto the C++
            #       response-visible fields so the response packet sees a
            #       consistent view of the request.
            request.draft_tokens = request.py_draft_tokens if get_draft_token_length(
                request) > 0 else []
            request.decoding_iter = request.py_decoding_iter

            # Concern: perf_metric
            # Task: append the per-step metric record for this request
            #       (token time, decode latency, etc).
            self.perf_manager.append_step_metrics(
                request, self.iter_counter, batch_token_time=batch_token_time)

            # Ensure C++ perf metrics (lastTokenTime, etc.) are always updated
            # independently of whether append_step_metrics early-returned.
            # This is critical for E2E latency computation in tracing/Prometheus.
            if request.return_perf_metrics and request.py_decoding_iter >= 1:
                request.update_perf_metrics(self.iter_counter)

            # Concern: response (build / enqueue)
            # Task: build the response packet on iter==1, on finish, or
            #       on the streaming interval; stash for batch enqueue.
            request_done = False
            if request.py_decoding_iter == 1 or request.is_finished or \
                    request.py_decoding_iter % self.stream_interval == 0:
                response = request.create_response(False, self.dist.rank)
                if response:
                    request_done = request.is_finished
                    response.result.cached_tokens = request.cached_tokens
                    new_responses.append((req_id, response))

            if request_done:
                # Concern: spec_decode (rolling acceptance gate)
                # Task: feed this request's avg_decoded_tokens_per_iter
                #       into speculation_gate; flipping
                #       speculation_permanently_disabled here is what
                #       makes `_prepare_and_schedule_batch` set
                #       use_spec_decode=False on the *next* iter.
                #       Cross-iter feedback edge: P_RESPOND -> next
                #       iter's P_SCHEDULE.
                # Consume: per-request avg_decoded_tokens_per_iter
                # Produce: self.speculation_permanently_disabled (loop state)
                if (self.drafter is not None and getattr(
                        self.model_engine, 'enable_spec_decode', False)
                        and not self.speculation_permanently_disabled
                        and not request.is_dummy and not self.is_warmup):
                    if self.speculation_gate is not None:
                        # Response handling runs on multiple PP ranks. Only the last PP rank performs
                        # sampling; restrict rolling stat updates to it to avoid overcounting.
                        if (not getattr(self.dist, 'has_pp',
                                        False)) or self.dist.is_last_pp_rank:
                            avg_decoded = getattr(
                                request, 'avg_decoded_tokens_per_iter', None)
                            if avg_decoded is not None:
                                disabled_now, _ = self.speculation_gate.record_avg_decoded(
                                    avg_decoded,
                                    request_id=getattr(request, 'py_request_id',
                                                       None))
                                if disabled_now:
                                    # disable speculation permanently
                                    # starting from next iteration, _prepare_and_schedule_batch will set self.use_spec_decode to False
                                    self.speculation_permanently_disabled = True
                            else:
                                logger.debug(
                                    f"Request {request.py_request_id} has no avg_decoded_tokens_per_iter"
                                )

                # Concern: disagg / response (terminate-policy fork)
                # Task: pick the right termination bucket. Disagg context-
                #       complete requests were already terminated by
                #       `_check_disagg_ctx_cache_transfer_status`; only
                #       record them for stats. Otherwise terminate now,
                #       except for ctx requests still in transmission
                #       (their KV is still being sent).
                # If partial reuse is enabled, and the KV cache manager is not VSWA, and the PP size is 1,
                # then we need to terminate the request. TODO: Remove this once disagg support from KVCache reuse
                # path is fixed.
                force_terminate_for_partial_reuse = (
                    self.enable_partial_reuse_for_disagg
                    and not self.kv_cache_manager.is_vswa
                    and self.dist.pp_size == 1)
                if request.is_disagg_context_complete_state:
                    # Already terminated by _check_disagg_ctx_cache_transfer_status;
                    # track for stats only to avoid double-free (nvbug/5961736).
                    requests_finished_by_transfer.append(request)
                elif force_terminate_for_partial_reuse:
                    requests_to_terminate.append(request)
                elif not request.is_disagg_context_transmission_state:
                    requests_to_terminate.append(request)
            else:
                new_active_requests.append(request)

        self.active_requests.clear()
        self.active_requests.extend(new_active_requests)
        # Request should be terminated after enqueueing response to ensure we can enqueue response successfully.
        self._enqueue_responses(new_responses)
        for request in requests_to_terminate:
            self._terminate_request(request)
        return requests_to_terminate + requests_finished_by_transfer

    # REFACTOR: 1a (public API) -- main-thread blocking wait used by
    # ``await_responses``.
    def _await_any_response(self,
                            timeout: Optional[float] = None
                            ) -> List[LlmResponse]:

        def any_responses_ready():
            return len(self.responses) > 0 or self.is_shutdown

        responses = []
        with self.response_cv:
            self.response_cv.wait_for(any_responses_ready, timeout=timeout)
            for req_id, response in self.responses.items():
                responses += response
            self.responses = {}

        return responses

    # REFACTOR: 1a (public API) -- main-thread blocking wait used by
    # ``await_responses``.
    def _await_single_response(
            self,
            id: int,
            timeout: Optional[float] = None) -> List[LlmResponse]:
        with self.response_cv:

            def key_has_response():
                return id in self.responses.keys()

            self.response_cv.wait_for(key_has_response, timeout=timeout)
            response = self.responses[id]
            self.responses.pop(id)
            return response

    # REFACTOR: 2a (concern: schedule) -- terminate paused requests
    # batch helper at SCHEDULE_0; only caller is the schedule concern's
    # paused-request lifecycle block.
    def _terminate_requests(self, requests_to_terminate):
        # todo: support work with self.inflight_req_ids.
        #       Currently, self.inflight_req_ids is not updated.
        for req in requests_to_terminate:
            self._terminate_request(req)

    # REFACTOR: 2a (concern: schedule) -- paused requests batch helper
    # at SCHEDULE_0 (paired with ``_terminate_requests``).
    def _pause_requests(self, requests_to_pause):
        for req in requests_to_pause:
            req.pause(self.max_input_len)

    # REFACTOR: 2a (concern: schedule) -- PP inflight-id tracking add at
    # SCHEDULE_0 (paired with ``_remove_inflight_ids`` at P_RETIRE).
    # Body becomes ``ctx.svc.pool.mark_inflight(reqs)``.
    def _add_inflight_ids(self, scheduled_requests: ScheduledRequests):
        """Add request IDs of current sampling requests to self.inflight_req_ids.

        Non-final context chunks should not be added to the inflight set, so the scheduler can keep scheduling
        further context chunks while earlier ones are in the PP pipeline.
        Only requests that sample new tokens should be added to the inflight set since their next iteration depends
        on these new tokens, so they should be skipped in the scheduler until the new tokens are generated.
        This includes context requests that finish context phase and generation requests.
        """
        for req in scheduled_requests.context_requests_last_chunk:
            logger.debug(
                f"Context request with ID {req.request_id} added to DECODER model inflight set"
            )
            self.inflight_req_ids.insert(req.request_id)
        for req in scheduled_requests.generation_requests:
            logger.debug(
                f"Generation request with ID {req.request_id} added to DECODER model inflight set"
            )
            self.inflight_req_ids.insert(req.request_id)

    # REFACTOR: 2a (concern: schedule) -- PP inflight-id tracking remove
    # at P_RETIRE inside ``_handle_executed_batch``.
    # Body becomes ``ctx.svc.pool.unmark_inflight(reqs)``.
    def _remove_inflight_ids(self, scheduled_requests: ScheduledRequests):
        """Remove request IDs of current sampling requests from self.inflight_req_ids."""
        for req in scheduled_requests.context_requests_last_chunk:
            logger.debug(
                f"Context request with ID {req.request_id} removed from DECODER model inflight set"
            )
            self.inflight_req_ids.erase(req.request_id)
        for req in scheduled_requests.generation_requests:
            logger.debug(
                f"Generation request with ID {req.request_id} removed from DECODER model inflight set"
            )
            self.inflight_req_ids.erase(req.request_id)

    # REFACTOR: 2a (concern: spec_decode) -- overlap-mode HC1 in-bridge
    # body that runs the draft model on prev's sample tensors.
    def _handle_speculative_decoding(
        self, scheduled_batch, previous_tensors, target_inputs
    ) -> Tuple[Optional[SampleStateTensorsSpec], Optional[torch.Tensor]]:
        with request_context(is_draft=self.draft_model_engine is not None,
                             scheduled_requests=scheduled_batch):
            target_outputs = self.previous_batch.sample_state and self.previous_batch.sample_state.device
            assert target_outputs is not None, "target_outputs should not be None"
            new_target_inputs, num_accepted_tokens_device = self._accept_draft_tokens(
                scheduled_batch=scheduled_batch,
                target_inputs=target_inputs,
                target_outputs=target_outputs)

            self.drafter.generate_draft_tokens_with_overlap(
                scheduled_batch, self.resource_manager,
                previous_tensors.device if previous_tensors else None,
                new_target_inputs, num_accepted_tokens_device)

            # Pad draft tokens to the max draft length for CUDA graph compatibility
            self.has_previous_draft_tokens = new_target_inputs is not None and new_target_inputs.next_draft_tokens is not None

        return new_target_inputs, num_accepted_tokens_device

    # REFACTOR: 1a (public API).
    def reset_prefix_cache(self):
        self.kv_cache_manager.reset_reuse_state()

    # REFACTOR: 2a (concern: guided_decoder) -- APPLY_7 step (mark
    # grammar-failed requests as errored).
    def _handle_guided_decoder_errors(
            self, scheduled_batch: ScheduledRequests,
            failed_requests: Optional[List[Tuple[int, str]]]):
        """Handle errors that occurred during guided decoding.

        Args:
            scheduled_batch: The current batch of scheduled requests
            failed_requests: List of (request_id, error_message) tuples for failed requests,
                           or None if no failures occurred
        """
        if not failed_requests:
            return

        failed_req_id_to_err = {req_id: err for req_id, err in failed_requests}

        for request in scheduled_batch.all_requests():
            if request.py_request_id not in failed_req_id_to_err:
                continue
            error_msg = failed_req_id_to_err[request.py_request_id]
            self._handle_errors(error_msg,
                                requests=[request],
                                charge_budget=False)


class DisaggPPTerminationHandler:
    """Handles termination synchronization across pipeline parallel ranks under disaggregated serving.

    We require synchronization when terminating requests in disaggregated PP when
    KV cache reuse is enabled. All PP ranks need to reach consensus before freeing
    resources to avoid a NCCL hang.
    """

    def __init__(self, dist, terminator_func: Callable[[LlmRequest], None]):
        self._dist = dist
        self._terminator_func = terminator_func
        self._pending_termination = {}
        self._terminating_iteration = 0
        self._send_handle = None
        self._comm_tag = PPCommTag.TERMINATION

    def terminate(self, request: LlmRequest):
        self._pending_termination[request.py_request_id] = request

    @nvtx_range("_disagg_pp_termination_handler_sync")
    def terminate_pending_requests(self):
        """
        Ring-style communicating to decide which requests to be terminated and avoid bubbles.
        This ensures that one request is terminated from rank_0 to rank_(pp_size-1) in order.
        """
        terminate_req_ids = []
        term_state = None
        if self._send_handle:
            self._send_handle.wait()

        if not (self._dist.is_first_pp_rank
                and self._terminating_iteration == 0):
            term_state = self._dist.recv_object(src=self._dist.prev_pp_rank,
                                                tag=self._comm_tag)

        ready_req_map = term_state["ready"] if term_state else {
        }  # {req_id: num_ranks} ranks vote in the ready dict
        terminate_req_ids = term_state["term"] if term_state else [
        ]  # request ids to be terminated in the current iteration

        reqs_to_terminate = {
            req_id: self._pending_termination.pop(req_id, None)
            for req_id in terminate_req_ids
            if req_id in self._pending_termination
        }

        if self._dist.is_first_pp_rank:
            # rank0 proposes the requests to be terminated
            ready_req_map = {req_id: 1 for req_id in self._pending_termination}
        else:
            # if a rank agrees to terminate a request, increase the vote count for the request id
            for req_id in ready_req_map.keys():
                if req_id in self._pending_termination:
                    ready_req_map[req_id] += 1

        if self._dist.is_last_pp_rank:
            new_terminate_req_ids = [
                req_id for req_id, num_ranks in ready_req_map.items()
                if num_ranks == self._dist.pp_size
            ]
            # by determining the terminate ids in the last rank, we can save the overhead of sending the ready dict back to rank0
            new_term_state = {"ready": {}, "term": new_terminate_req_ids}
        else:
            # other pp ranks pass the updated ready dict and terminate request ids to the next rank, and the
            # terminate_req_ids will not change in a given iteration, so we can terminate the requests synchronously
            new_term_state = {"ready": ready_req_map, "term": terminate_req_ids}

        self._send_handle = self._dist.isend_object(
            new_term_state, dest=self._dist.next_pp_rank, tag=self._comm_tag)

        if reqs_to_terminate:
            logger.debug(
                f'rank {self._dist.pp_rank} terminates {list(reqs_to_terminate.keys())} in iter {self._terminating_iteration}'
            )
        for req_id, req in reqs_to_terminate.items():
            if req:
                self._terminator_func(req)
        self._terminating_iteration += 1
