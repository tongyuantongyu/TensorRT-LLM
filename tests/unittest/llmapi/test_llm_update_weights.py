import argparse
import torch
import torch.distributed as dist
import atexit
import os
from typing import Any, Optional
from tensorrt_llm import SamplingParams
from tensorrt_llm import LLM
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    MixedPrecision,
    ShardedStateDictConfig,
    FullStateDictConfig
)
#from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.tensor import DTensor

def init_distributed():
    """Initialize distributed training"""
    if "LOCAL_RANK" not in os.environ:
        return 1, 0, torch.device("cuda:0")

    # Set default environment variables if not already set
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return world_size, rank, torch.device(f"cuda:{rank}")

def exit_distributed():
    """Exit distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def report_device_id() -> str:
    """Report the UUID of the current CUDA device using NVML.
    Returns:
        str: UUID of the device in the format "GPU-xxxxx"
    """
    from tensorrt_llm._torch.utils import get_device_uuid
    # Get current device index from torch
    device_idx = torch.cuda.current_device()
    # Get device UUID using NVML
    return get_device_uuid(device_idx)

class fsdp_interface:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = torch.device(f"cuda:{self.rank}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = self.load_fsdp_model(model_dir)

    def load_fsdp_model(self, model_dir):
        """Load and initialize FSDP model"""
        # Initialize distributed setup
        print(f"World size: {self.world_size}, Rank: {self.rank}, Device: {self.device}")

        # Setup mixed precision policy for FSDP
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32
        )

        if self.rank == 0:
            print(f"Loading FSDP model from {model_dir}")

        # Initialize FSDP model
        fsdp_model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )

        # Print model info
        if self.rank == 0:
            total_params = sum(p.numel() for p in fsdp_model.parameters())
            trainable_params = sum(p.numel() for p in fsdp_model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Model device: {next(fsdp_model.parameters()).device}")

        # Wrap model with FSDP
        fsdp_model = FSDP(
            fsdp_model,
            mixed_precision=mixed_precision_policy,
            device_id=torch.cuda.current_device(),
            use_orig_params=True
        )

        if self.rank == 0:
            print("FSDP model initialized successfully")

        self._held_streamed_param_reference = None
        self._held_sharded_state_dict_reference = None

        return fsdp_model



    def per_tensor_generator(self):
        # If the model is not FSDP, then we need to manually move it to the GPU
        # For an FSDP model, model.state_dict() will move the params to the GPU
        if not isinstance(self.model, FSDP):
            self.model = self.manual_load_to_gpu(self.model)
            self._held_sharded_state_dict_reference = self.model.state_dict()
        else:
            # Get sharded state dict instead of full state dict for FSDP1
            with FSDP.state_dict_type(
                self.model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig()
            ):
                self._held_sharded_state_dict_reference = self.model.state_dict()
        for name, param in self._held_sharded_state_dict_reference.items():
            yield name, param

    @torch.no_grad()
    def prepare_weights_for_ipc(self) -> tuple[list[tuple[str, int]], float]:
        # If the model is not FSDP, then we need to manually move it to the GPU
        # For an FSDP model, model.state_dict() will move the params to the GPU
        if not isinstance(self.model, FSDP):
            self.model = self.manual_load_to_gpu(self.model)
            self._held_sharded_state_dict_reference = self.model.state_dict()
        else:
            # Get sharded state dict instead of full state dict for FSDP1
            with FSDP.state_dict_type(
                self.model,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig()
            ):
                self._held_sharded_state_dict_reference = self.model.state_dict()

        # Collect info for streaming multiple tensors
        ### state_dict_info = []
        ### for name, tensor in self._held_sharded_state_dict_reference.items():
        ###     # dtensor's numel will return complete tensor instead of only local tensor
        ###     size_in_bytes = tensor.element_size() * tensor.numel()
        ###     state_dict_info.append((name, size_in_bytes))
        self.refit_param_info = []
        for name, tensor in self._held_sharded_state_dict_reference.items():
            # dtensor's numel will return complete tensor instead of only local tensor
            size_in_bytes = tensor.element_size() * tensor.numel()
            self.refit_param_info.append((name, size_in_bytes))

        from tensorrt_llm._torch.utils import get_free_memory_bytes
        #print(f"State dict info: {state_dict_info}")
        # Collect current available memory for refit
        ## Get current device index from torch
        device_idx = torch.cuda.current_device()
        ## Get device free memory using NVML
        total_available_bytes = get_free_memory_bytes(device_idx)
        ## Use 80% of the free memory for safety
        memory_ratio = os.getenv("NRL_REFIT_BUFFER_MEMORY_RATIO", "0.8")
        total_available_bytes *= float(memory_ratio)

        return self.refit_param_info, total_available_bytes

    @torch.no_grad()
    def get_weights_ipc_handles(self, keys: list[str]) -> dict[str, Any]:
        from torch.distributed.tensor import DTensor
        from torch.multiprocessing.reductions import reduce_tensor

        assert self._held_sharded_state_dict_reference is not None, (
            "prepare_weights_for_ipc must be called before get_weights_ipc_handles"
        )

        # Clean up the held tensors to reduce peak memory
        if self._held_streamed_param_reference is not None:
            del self._held_streamed_param_reference
            self._held_streamed_param_reference = None

        converted_params = {}
        for key in keys:
            # Get full_tensor for dtensor (GPU > 1)
            if not key.startswith("model."):
                continue
            tensor = self._held_sharded_state_dict_reference[key]
            if isinstance(tensor, DTensor):
                full_tensor = tensor.full_tensor()
            else:
                full_tensor = tensor
            # Convert parameters to the configured dtype
            #print(f"FSDP rank {self.rank} name: {key}, shape: {full_tensor.shape}, {full_tensor[0]}")
            converted_params[key] = full_tensor

        # Temporary record the full tensor for cleanup
        # It is needed for cleanup the last full_tensor in the refit process
        self._held_streamed_param_reference = converted_params

        # Get device UUID for IPC
        device_uuid = report_device_id()
        # Create handles for the tensors
        all_handles = []
        for key, p in converted_params.items():
            handle = reduce_tensor(p.detach())
            all_handles.append((key, handle))

        #print(f"device_uuid: {device_uuid}, All handles keys: {[key for key, _ in all_handles]}")
        print(f"device_uuid: {device_uuid}")
        return {device_uuid: all_handles}

    @torch.no_grad()
    def prepare_weights_for_ipc_refit(
        self, _refit_buffer_size_gb: Optional[int] = None
    ) -> list[list[str]]:
        """Prepare the weights for IPC.

        Returns:
            list: A list containing the keys of the parameters, which is grouped by size.
        """
        # Get the state_dict_info and available memory from all workers
        state_dict_info = self.refit_param_info

        if _refit_buffer_size_gb is not None:
            total_available_bytes = _refit_buffer_size_gb * (1024**3)
        else:
            # Get the minimum available memory from all workers
            total_available_bytes = min(result[1] for result in state_dict_info)

        # Group tensors by size
        cur_available_bytes = total_available_bytes
        grouped_param_keys: list[list[str]] = []
        keys: list[str] = []

        for key, size_in_bytes in state_dict_info:
            if size_in_bytes > cur_available_bytes:
                if keys:
                    grouped_param_keys.append(keys)
                    keys = []
                cur_available_bytes = total_available_bytes

            keys.append(key)
            cur_available_bytes -= size_in_bytes

        if keys:
            grouped_param_keys.append(keys)

        return grouped_param_keys

class NamedParam:
    def __init__(self, name, size, param):
        self.name = name
        self.size = size
        self.param = param

class GateAndUp:
    def __init__(self):
        self.gate = None
        self.up = None
    def set_gate(self, gate):
        self.gate = gate
    def set_up(self, up):
        self.up = up
    def get_size(self):
        return self.gate.size + self.up.size
    def is_complete(self):
        return self.gate is not None and self.up is not None

class trtllm_interface:
    def __init__(self, model_dir, tensor_parallel_size):
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.device = torch.device(f"cuda:{self.rank}")
        self.model_dir = model_dir
        self.tensor_parallel_size = tensor_parallel_size
        self.llm = self.load_trtllm_model(model_dir, tensor_parallel_size)

    def load_trtllm_model(self, model_dir, tensor_parallel_size):
        if self.rank == 0:
            print("Loading TensorRT-LLM model")
            return LLM(
                model=model_dir,
                tensor_parallel_size=tensor_parallel_size,
                #disable_overlap_scheduler=True,
                #load_format='auto'
                load_format='dummy',
                kv_cache_config=KvCacheConfig(
                    free_gpu_memory_fraction=0.85,
                    enable_block_reuse=False
                )
            )
        else:
            return None

    def update_weights_from_ipc_handles(self, rank, device_handles):
        if rank == 0:
            gathered_handles = [None for _ in range(dist.get_world_size())]
        else:
            gathered_handles = None
        dist.gather_object(
            obj=device_handles,
            object_gather_list=gathered_handles,
            dst=0
        )
        if rank == 0:
            all_handles = {k: v for d in gathered_handles for k, v in d.items()}
            self.llm.update_weights_from_ipc_handles(all_handles)

    def update_weights_from_tensor_generator(self, tensor_generator):
        device_uuid = report_device_id()
        rank = dist.get_rank()
        from torch.multiprocessing.reductions import reduce_tensor
        total_available_bytes = 0.7 * (1024**3)
        cur_available_bytes = total_available_bytes
        converted_params = {}
        cur_handles = []
        gate_up = {}
        for name, param in tensor_generator:
            size_in_bytes = param.element_size() * param.numel()
            if isinstance(param, DTensor):
                param = param.full_tensor()
            gate_up_name = None
            gate_up_pair = None
            if "gate_proj" in name:
                gate_up_name = name.replace("gate_proj", "")
                if (gate_up_name not in gate_up):
                    gate_up[gate_up_name] = GateAndUp()
                assert gate_up[gate_up_name].gate is None
                gate_up[gate_up_name].set_gate(NamedParam(name, size_in_bytes, param))
            elif "up_proj" in name:
                gate_up_name = name.replace("up_proj", "")
                if (gate_up_name not in gate_up):
                    gate_up[gate_up_name] = GateAndUp()
                assert gate_up[gate_up_name].up is None
                gate_up[gate_up_name].set_up(NamedParam(name, size_in_bytes, param))
            if (gate_up_name is not None):
                if gate_up[gate_up_name].is_complete():
                    gate_up_pair = gate_up.pop(gate_up_name)
                    size_in_bytes = gate_up_pair.get_size()
                else:
                    continue

            if size_in_bytes > cur_available_bytes:
                device_handles = {device_uuid: cur_handles}
                self.update_weights_from_ipc_handles(rank, device_handles)
                cur_available_bytes = total_available_bytes
                del converted_params
                converted_params = {}
                cur_handles = []

            assert cur_available_bytes >= size_in_bytes
            cur_available_bytes -= size_in_bytes
            if (gate_up_pair is not None):
                converted_params[gate_up_pair.gate.name] = gate_up_pair.gate.param
                converted_params[gate_up_pair.up.name] = gate_up_pair.up.param
                handle = reduce_tensor(gate_up_pair.gate.param.detach())
                cur_handles.append((gate_up_pair.gate.name, handle))
                handle = reduce_tensor(gate_up_pair.up.param.detach())
                cur_handles.append((gate_up_pair.up.name, handle))
                gate_up_pair = None
            else:
                converted_params[name] = param
                handle = reduce_tensor(param.detach())
                cur_handles.append((name, handle))

        assert len(gate_up) == 0

        if cur_handles:
            device_handles = {device_uuid: cur_handles}
            self.update_weights_from_ipc_handles(rank, device_handles)
            cur_available_bytes = total_available_bytes
            del converted_params
            converted_params = {}
            cur_handles = []

def get_total_available_bytes(pg: dist.ProcessGroup, message: str = "") -> int:
    mem_allocated = torch.cuda.memory_allocated()
    mem_reserved = torch.cuda.memory_reserved()
    mem_free, mem_total = torch.cuda.mem_get_info()
    print(f"{message} mem_free: {mem_free:,}, mem_total: {mem_total:,}, mem_allocated: {mem_allocated:,}, mem_reserved: {mem_reserved:,}")
    mem_free = torch.tensor(mem_free)
    dist.all_reduce(mem_free, op=dist.ReduceOp.MIN, group=pg)
    mem_free = mem_free.item()
    print(f"{message} gathered_mem_free: {mem_free:,}")
    return mem_free * 0.2

def cleanup():
    """Cleanup function to destroy process group"""
    if dist.is_initialized():
        print(f"Cleaning up process group on rank {dist.get_rank()}")
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(
        description="LLM models with the PyTorch workflow.")

    parser.add_argument('--model_dir',
                        type=str,
                        required=True,
                        default='/model/Qwen2.5-0.5B-Instruct',
                        help="Model checkpoint directory.")

    parser.add_argument('--tensor_parallel_size',
                        type=int,
                        default=2,
                        help="Tensor parallel size (number of GPUs to use)")

    parser.add_argument('--use_fsdp',
                        action='store_true',
                        help="Use FSDP model loading instead of direct TensorRT-LLM loading")

    args = parser.parse_args()

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    world_size, rank, device = init_distributed()

    sampling_params = SamplingParams(max_tokens=32)

    # Load FSDP model
    fsdp = fsdp_interface(args.model_dir)
    trtllm = trtllm_interface(args.model_dir, args.tensor_parallel_size)

    if rank == 0:
        print(f"Collected handles from all {world_size} ranks:")

    # For FSDP mode, we would need additional logic to integrate withTensorRT-LLM
    # This is a placeholder for now
    if rank == 0:
        outputs = trtllm.llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")

        ## load the model from fsdp
        ## then generate the output again
        result = trtllm.llm.sleep(1)
        print(f"sleep result: {result}")

        result = trtllm.llm.wakeup()
        print(f"wakeup result: {result}")

    trtllm.update_weights_from_tensor_generator(fsdp.per_tensor_generator())

    # generate the output again
    if rank == 0:
        outputs = trtllm.llm.generate(prompts, sampling_params)
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"[{i}] Prompt: {prompt!r}, Generated text: {generated_text!r}")

    exit_distributed()
if __name__ == '__main__':
    main()

# torchrun --nproc_per_node=2 tests/unittest/llmapi/test_llm_update_weights.py --model_dir /model/Qwen2.5-0.5B-Instruct --tensor_parallel_size 2
# torchrun --nproc_per_node=2 tests/unittest/llmapi/test_llm_update_weights.py --model_dir /model/Qwen2.5-3B-Instruct/ --tensor_parallel_size 2