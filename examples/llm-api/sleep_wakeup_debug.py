import os.path

import torch.cuda

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.virtual_memory import (materialize_with_tag,
                                                release_with_tag)
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
from tensorrt_llm._torch.pyexecutor.py_executor_creator import ExecutorMemoryType


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(max_tokens=32)
    kv_cache_config = KvCacheConfig(max_tokens=32768 * 32,
                                    enable_block_reuse=False,
                                    enable_partial_reuse=False)
    cuda_graph_config = CudaGraphConfig()

    llama3 = os.path.expandvars(
        "$LLM_MODELS_ROOT/llama-3.1-model/Llama-3.1-8B-Instruct")

    llm = LLM(model=llama3,
              enable_sleep=True,
              cuda_graph_config=None,  # CUDA Graph unsupported
              kv_cache_config=kv_cache_config)
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")

    marks = [
        ExecutorMemoryType.SAMPLER,
        ExecutorMemoryType.DRAFTER,
        ExecutorMemoryType.GUIDED_DECODER,
        ExecutorMemoryType.SPEC_RESOURCES,
        ExecutorMemoryType.MODEL_EXTRA,
        ExecutorMemoryType.EXTRA_RESOURCES,
        ExecutorMemoryType.KV_CACHE,
        ExecutorMemoryType.MODEL_ENGINE_MAIN,
        ExecutorMemoryType.MODEL_ENGINE_DRAFT,
    ]

    input('Wake')

    torch.cuda.synchronize()
    print("Start sleep")
    # Need TLLM_WORKER_USE_SINGLE_PROCESS=1
    # TODO: Add LLM methods to call this in worker
    release_with_tag(*marks)
    print("Finish sleep")
    torch.cuda.synchronize()

    torch.cuda.empty_cache()

    input('Sleeping...')

    torch.cuda.synchronize()
    materialize_with_tag(*marks)
    torch.cuda.synchronize()

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
