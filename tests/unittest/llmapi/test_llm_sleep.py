from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

from .test_llm import llama_model_path


def test_llm_sleep():
    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    max_tokens=4096)

    llm = LLM(model=llama_model_path,
              enable_sleep=True,
              cuda_graph_config=None,  # CUDA Graph unsupported yet
              kv_cache_config=kv_cache_config)

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0)

    with llm:
        outputs = llm.generate(prompts, sampling_params)
        generated_before_sleep = [output.outputs[0].text for output in outputs]

        llm.sleep(-1)
        llm.wakeup(-1)

        outputs = llm.generate(prompts, sampling_params)
        generated_after_sleep = [output.outputs[0].text for output in outputs]

    for before, after in zip(generated_before_sleep, generated_after_sleep):
        assert before == after, "Generated result mismatch before and after sleep"
