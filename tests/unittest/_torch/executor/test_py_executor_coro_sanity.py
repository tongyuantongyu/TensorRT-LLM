"""Sanity smoke-test: drive a real model through ``PyExecutorCoro``.

Runs TinyLlama-1.1B-Chat through the new coroutine-based executor by
monkey-patching the legacy ``PyExecutor`` reference that the
``LLM`` API path uses (in ``_util.create_py_executor_instance``)
with ``PyExecutorCoro``.

Conservative KV-cache settings keep the WSL-host VRAM footprint
small enough not to crash the Windows host:

* ``free_gpu_memory_fraction=0.15`` (small slice of the visible
  GPU's free memory).
* ``max_batch_size=2``, ``max_num_tokens=512`` -- bounded prefill
  budget.
* Single short prompt, ``max_tokens=8`` -- the run completes in a
  handful of iters.

The test asserts:

* Loop spins up + shuts down without exceptions.
* The generated output has the expected ``max_tokens=8`` length and
  the right shape.

It does NOT assert exact token equality with the legacy executor --
the goal is "the new loop runs end-to-end on a real model", not
correctness vs reference. A tighter comparison test against the
legacy executor can be added once both code paths support the same
features.
"""

from __future__ import annotations

import pytest

# Skip cleanly if there's no GPU available -- the rest of the test
# would just crash on torch.cuda.set_device.
try:
    import torch
    _HAS_CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover - import-time failure
    _HAS_CUDA = False

pytestmark = pytest.mark.skipif(
    not _HAS_CUDA, reason="needs CUDA to drive a real model")


def _patch_in_pyexecutor_coro(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap ``PyExecutor`` with ``PyExecutorCoro`` in the LLM API path.

    The LLM API instantiates the executor inside
    ``tensorrt_llm._torch.pyexecutor._util.create_py_executor_instance``
    via ``PyExecutor(...)``. Replacing that module-level binding
    re-routes the call without touching any other code.
    """
    from tensorrt_llm._torch.pyexecutor import _util
    from tensorrt_llm._torch.pyexecutor.py_executor_coro import PyExecutorCoro
    monkeypatch.setattr(_util, "PyExecutor", PyExecutorCoro)


@pytest.mark.parametrize(
    "disable_overlap_scheduler,pipeline_parallel_size",
    [
        (True, 1),
        (False, 1),
        (True, 2),
    ],
    ids=["plain", "overlap", "pp2"],
)
def test_py_executor_coro_runs_tinyllama_end_to_end(
    monkeypatch,
    disable_overlap_scheduler,
    pipeline_parallel_size,
):
    """One prompt in, one response out, clean shutdown.

    Parametrized over scheduler-iter variants:

    * ``plain`` -- ``scheduler_iter_plain``: one batch per iter
      driven through every phase. Single rank.
    * ``overlap`` -- ``scheduler_iter_overlap``: two batches alive,
      ``current`` runs through STATE_UPD_4 while ``previous`` runs
      APPLY_7 -> FINALIZE_9 (the legacy overlap pattern -- HC1
      batch-to-batch bridge of ``previous_tensors_device``). Single
      rank.
    * ``pp2`` -- ``scheduler_iter_pp`` with ``pp_size=2``. Requires
      2 GPUs; the LLM API auto-launches the second rank. Skipped
      automatically if fewer than 2 GPUs are visible.
    """
    if pipeline_parallel_size > 1:
        if not _HAS_CUDA or torch.cuda.device_count() < pipeline_parallel_size:
            pytest.skip(
                f"PP={pipeline_parallel_size} needs at least "
                f"{pipeline_parallel_size} GPUs; have "
                f"{torch.cuda.device_count() if _HAS_CUDA else 0}.")

    if monkeypatch is not None:
        _patch_in_pyexecutor_coro(monkeypatch)

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    # Conservative VRAM footprint -- 15% of free GPU memory (~2-3 GiB
    # on a 16 GiB consumer card) for KV. block_reuse off so the
    # bring-up doesn't depend on the reuse machinery.
    kv_cache_config = KvCacheConfig(
        free_gpu_memory_fraction=0.15,
        enable_block_reuse=False,
    )

    sampling_params = SamplingParams(max_tokens=8)

    llm_kwargs = dict(
        model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        kv_cache_config=kv_cache_config,
        max_batch_size=2,
        max_num_tokens=8,
        disable_overlap_scheduler=disable_overlap_scheduler,
        # No CUDA graph -- the resource concern doesn't yet wire the
        # graph-capture-friendly attn_metadata path.
        cuda_graph_config=None,
    )
    if pipeline_parallel_size > 1:
        llm_kwargs["pipeline_parallel_size"] = pipeline_parallel_size

    with LLM(**llm_kwargs) as llm:
        outputs = llm.generate(["A B C"], sampling_params=sampling_params)

    assert len(outputs) == 1, f"expected one RequestOutput, got {len(outputs)}"
    completion = outputs[0].outputs[0]
    print(
        f"[result] prompt_token_ids={outputs[0].prompt_token_ids} "
        f"token_ids={completion.token_ids} "
        f"text={completion.text!r} "
        f"finish_reason={completion.finish_reason}",
        flush=True,
    )
    assert completion.text, "completion text should be non-empty"
    # ``max_tokens=8`` so the completion's token_ids should be at
    # most 8 (could be fewer if a stop token was hit).
    assert 1 <= len(completion.token_ids) <= 8, (
        f"unexpected token count: {len(completion.token_ids)}")


if __name__ == '__main__':
    import sys

    # Default: overlap (the new code path). ``legacy-*`` modes run
    # the original PyExecutor with the matching variant, no monkey-
    # patching -- useful as reference runs.
    mode = sys.argv[1] if len(sys.argv) > 1 else "overlap"

    use_coro = mode not in ("legacy-plain", "legacy-overlap", "legacy-pp2")
    if use_coro:
        from tensorrt_llm._torch.pyexecutor import _util
        from tensorrt_llm._torch.pyexecutor.py_executor_coro import PyExecutorCoro
        setattr(_util, "PyExecutor", PyExecutorCoro)

    if mode in ("plain", "both"):
        print("=== plain (PyExecutorCoro) ===", flush=True)
        test_py_executor_coro_runs_tinyllama_end_to_end(
            None,
            disable_overlap_scheduler=True,
            pipeline_parallel_size=1)
    if mode in ("overlap", "both"):
        print("=== overlap (PyExecutorCoro) ===", flush=True)
        test_py_executor_coro_runs_tinyllama_end_to_end(
            None,
            disable_overlap_scheduler=False,
            pipeline_parallel_size=1)
    if mode == "pp2":
        print("=== pp2 (PyExecutorCoro) ===", flush=True)
        test_py_executor_coro_runs_tinyllama_end_to_end(
            None,
            disable_overlap_scheduler=True,
            pipeline_parallel_size=2)
    if mode == "legacy-plain":
        print("=== plain (legacy PyExecutor) ===", flush=True)
        test_py_executor_coro_runs_tinyllama_end_to_end(
            None,
            disable_overlap_scheduler=True,
            pipeline_parallel_size=1)
    if mode == "legacy-overlap":
        print("=== overlap (legacy PyExecutor) ===", flush=True)
        test_py_executor_coro_runs_tinyllama_end_to_end(
            None,
            disable_overlap_scheduler=False,
            pipeline_parallel_size=1)
    if mode == "legacy-pp2":
        print("=== pp2 (legacy PyExecutor) ===", flush=True)
        test_py_executor_coro_runs_tinyllama_end_to_end(
            None,
            disable_overlap_scheduler=True,
            pipeline_parallel_size=2)
    print("=== done ===", flush=True)
