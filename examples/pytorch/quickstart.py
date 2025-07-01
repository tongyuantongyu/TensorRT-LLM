import torch.cuda

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.virtual_memory import (materialize_with_mark,
                                                release_with_mark)
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig


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

    llama3 = '/home/scratch.trt_llm_data/llm-models/llama-3.1-model/Llama-3.1-8B-Instruct'

    llm = LLM(model=llama3,
              enable_sleep=True,
              cuda_graph_config=cuda_graph_config,
              kv_cache_config=kv_cache_config)
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")

    marks = [
        'model',
        'draft_model',
        'kv_cache',
        'spec',
        'drafter',
        'extra',
    ]

    input('Wake')

    torch.cuda.synchronize()
    release_with_mark(*marks)
    torch.cuda.synchronize()

    torch.cuda.empty_cache()

    input('Sleeping...')

    torch.cuda.synchronize()
    materialize_with_mark(*marks)
    torch.cuda.synchronize()

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == '__main__':
    main()
