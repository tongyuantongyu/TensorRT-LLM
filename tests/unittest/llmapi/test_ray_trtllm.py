# SPDX-License-Identifier: Apache-2.0
"""
a simple demonstration of RLHF with vLLM, inspired by
the OpenRLHF framework https://github.com/OpenRLHF/OpenRLHF .
It follows the design that, training processes and inference processes
are different, and they live on different GPUs.
Training processes send prompts to inference processes to generate data,
and also synchronize the weights of the model by broadcasting the weights
from the training process to the inference process.
Note that this is a simple demonstration of one training instance and one
inference instance. In practice, there could be multiple training instances
and multiple inference instances. For the full implementation, please refer
to the OpenRLHF framework.
"""
import os

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoModelForCausalLM

from tensorrt_llm import LLM


class RayLLM:
    def __init__(self, *args, **kwargs):
        import torch
        dev_count = torch.cuda.device_count()
        print("dev_count: ", dev_count)
        self.mpi_session = None
        for i in range(dev_count):
            device = torch.cuda.get_device_properties(i)
            print(f"pid: {os.getpid()}, device {i}: {device.name}, UUID: {device.uuid}, Memory: {device.total_memory / 1024**2:.0f}MB")
        self.llm = LLM(*args, **kwargs)

    def generate(self, prompts):
        ret = []

        llm_ret = self.llm.generate(prompts)
        for r in llm_ret:
            ret.append(r.outputs[0].text)

        return ret

"""
Start the training process, here we use huggingface transformers
as an example to hold a model on GPU 0.
"""


"""
Start the inference process, here we use vLLM to hold a model on GPU 1 and
GPU 2. For the details on how to use ray, please refer to the ray
documentation https://docs.ray.io/en/latest/ .
"""
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
ray.init(include_dashboard=False)

pg_inference = placement_group([{"GPU": 2, "CPU": 0}])
ray.get(pg_inference.ready())
scheduling_inference = PlacementGroupSchedulingStrategy(
    placement_group=pg_inference,
    placement_group_capture_child_tasks=True,
    placement_group_bundle_index=0,
)
"""
launch the vLLM inference engine.
here we use `enforce_eager` to reduce the start time.
"""
llm = ray.remote(
    num_cpus=0,
    num_gpus=2,
    scheduling_strategy=scheduling_inference,
)(RayLLM).remote(
     model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
     tensor_parallel_size=2,
    # enforce_eager=True,
    # worker_extension_cls="rlhf_utils.WorkerExtension",
    # tensor_parallel_size=2,
    # distributed_executor_backend="ray",
)

# Generate texts from the prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

#sampling_params = SamplingParams(temperature=0)

outputs = ray.get(llm.generate.remote(prompts))

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt!r}, "
          f"Generated text: {output!r}")

# set up the communication between the training process
# and the inference engine.
# master_address = get_ip()
# master_port = get_open_port()

# handle = llm.collective_rpc.remote("init_weight_update_group",
#                                    args=('master_address', 5, 0, 3))

# # model_update_group = stateless_init_process_group(master_address, master_port,
# #                                                   0, 3, torch.device("cuda:0"))
# # ray.get(handle)

# # simulate training, modify the weights of the model.
# # for name, p in train_model.named_parameters():
# #     p.data.zero_()

# # # sync weight from the training process to the inference engine.
# # for name, p in train_model.named_parameters():
# #     handle = llm.collective_rpc.remote("update_weight",
# #                                        args=(name, p.dtype, p.shape))
# #     model_update_group.broadcast(p, src=0, stream=torch.cuda.current_stream())
# #     ray.get(handle)

# # check if the weights are updated.
# # assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))

# # use the updated model to generate texts, they will be nonsense
# # because the weights are all zeros.
# outputs_updated = ray.get(llm.generate.remote(prompts, sampling_params))
# for output in outputs_updated:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, "
#           f"Generated text: {generated_text!r}")
