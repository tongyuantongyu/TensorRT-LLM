# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import enum
import itertools
import os
import pathlib
import re
import threading
from typing import Optional, ParamSpec, Tuple

import pynvml


class GPUAffinityError(Exception):
    pass


class StrEnum(enum.Enum):
    def __str__(self):
        return self.value


class Mode(StrEnum):
    UNIQUE_INTERLEAVED = "unique_interleaved"
    UNIQUE_CONTIGUOUS = "unique_contiguous"


class Scope(StrEnum):
    NODE = "node"
    SOCKET = "socket"


_T = ParamSpec("_T")

CoreList = list[int]

SiblingGroup = tuple[int, ...]
SiblingGroups = tuple[SiblingGroup, ...]
NodeGroups = tuple[SiblingGroups, ...]

AffinityList = None


class Device:
    # assume nvml returns list of 64 bit ints
    _nvml_bit_affinity: int = 64

    _nvml_affinity_elements: int = (os.cpu_count() + _nvml_bit_affinity - 1) // _nvml_bit_affinity

    def __init__(self, device_idx: int) -> None:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            if device_idx >= len(visible_devices):
                msg = (
                    f"Requested device_idx={device_idx} is out of bounds for "
                    f"the specified CUDA_VISIBLE_DEVICES={visible_devices}"
                )
                raise GPUAffinityError(msg)
            try:
                # interpret elements of CUDA_VISIBLE_DEVICES as integer indices
                remapped_device_ids = [int(id) for id in visible_devices]
                device_idx = remapped_device_ids[device_idx]
            except ValueError:
                # interpret elements of CUDA_VISIBLE_DEVICES as UUID strings
                num_devices = pynvml.nvmlDeviceGetCount()
                devices_uuids = [
                    pynvml.nvmlDeviceGetUUID(handle)
                    for handle in [
                        pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in range(num_devices)
                    ]
                ]
                indices_with_matching_prefixes = [
                    i
                    for i in range(num_devices)
                    if devices_uuids[i].startswith(visible_devices[device_idx])
                ]
                if len(indices_with_matching_prefixes) == 0:
                    msg = (
                        f"None of the devices matched the specified UUID "
                        f"prefix, prefix={visible_devices[device_idx]}, "
                        f"devices={devices_uuids}"
                    )
                    raise GPUAffinityError(msg)
                elif len(indices_with_matching_prefixes) > 1:
                    msg = (
                        f"Multiple devices matched the specified UUID "
                        f"prefix, prefix={visible_devices[device_idx]}, "
                        f"devices={devices_uuids}"
                    )
                    raise GPUAffinityError(msg)
                device_idx = indices_with_matching_prefixes[0]

        try:
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        except Exception as ex:
            msg = f"Unable to get NVML handle for device {device_idx}"
            raise GPUAffinityError(msg) from ex

    def get_name(self) -> str:
        return pynvml.nvmlDeviceGetName(self.handle)

    def get_uuid(self) -> str:
        return pynvml.nvmlDeviceGetUUID(self.handle)

    def get_cpu_affinity(self, scope: Scope) -> CoreList:
        if scope == Scope.SOCKET:
            nvml_scope = pynvml.NVML_AFFINITY_SCOPE_SOCKET
        elif scope == Scope.NODE:
            nvml_scope = pynvml.NVML_AFFINITY_SCOPE_NODE
        else:
            raise GPUAffinityError("Unknown affinity scope")

        affinity_string = ""
        for j in pynvml.nvmlDeviceGetCpuAffinityWithinScope(
            self.handle, Device._nvml_affinity_elements, nvml_scope
        ):
            # assume nvml returns list of 64 bit ints
            affinity_string = "{:064b}".format(j) + affinity_string

        affinity_list = [int(x) for x in affinity_string]
        affinity_list.reverse()  # so core 0 is in 0th element of list

        ret = [i for i, e in enumerate(affinity_list) if e != 0]
        return ret


# ============================================================================
# CPU Topology Helpers
# ============================================================================


def get_thread_siblings_list() -> list[SiblingGroup]:
    path = "/sys/devices/system/cpu/cpu*/topology/thread_siblings_list"
    thread_siblings_list: list[SiblingGroup] = []
    pattern = re.compile(r"(\d+)\D(\d+)")
    for fname in pathlib.Path(path[0]).glob(path[1:]):
        with open(fname) as f:
            content = f.read().strip()
            res = pattern.findall(content)
            if res:
                pair = tuple(sorted(map(int, res[0])))
                thread_siblings_list.append(pair)
    thread_siblings_list = sorted(set(thread_siblings_list))
    return thread_siblings_list


def build_thread_siblings_dict(siblings_list: list[SiblingGroup]) -> dict[int, SiblingGroup]:
    siblings_dict: dict[int, SiblingGroup] = {}
    for siblings_tuple in siblings_list:
        for core in siblings_tuple:
            siblings_dict[core] = siblings_tuple

    return siblings_dict


def group_list_by_key(the_list: list[int], key) -> list[Tuple[int, ...]]:
    sorted_list = sorted(the_list, key=key)
    grouped = [tuple(group) for key, group in itertools.groupby(sorted_list, key=key)]
    return grouped


def group_by_siblings(affinities: list[CoreList]) -> list[SiblingGroups]:
    siblings_list: list[SiblingGroup] = get_thread_siblings_list()
    siblings_dict: dict[int, SiblingGroup] = build_thread_siblings_dict(siblings_list)
    affinities_grouped = [
        tuple(group_list_by_key(affinity, key=lambda x: siblings_dict.get(x, (x,))))
        for affinity in affinities
    ]
    return affinities_grouped


def group_by_node(
    socket_affinities: list[SiblingGroups], node_affinities: list[SiblingGroups]
) -> list[NodeGroups]:
    socket_node_assigned_cores: dict[SiblingGroups, list[SiblingGroup]] = collections.defaultdict(
        list
    )
    for socket, node_cores in zip(socket_affinities, node_affinities):
        socket_node_assigned_cores[socket].extend(node_cores)

    socket_node_assigned_cores_tuples: dict[SiblingGroups, Tuple[SiblingGroup, ...]] = {
        key: tuple(sorted(set(value))) for key, value in socket_node_assigned_cores.items()
    }

    node_grouping: dict[SiblingGroup, list[SiblingGroup]] = collections.defaultdict(list)

    for socket_cores, assigned_cores in socket_node_assigned_cores_tuples.items():
        unassigned_cores = sorted(list(set(socket_cores) - set(assigned_cores)))

        for assigned_core in assigned_cores:
            node_grouping[assigned_core].append(assigned_core)

        for assigned, unassigned in zip(itertools.cycle(assigned_cores), unassigned_cores):
            node_grouping[assigned].append(unassigned)

    node_grouping_tuples: dict[SiblingGroup, Tuple[SiblingGroup, ...]] = {
        key: tuple(value) for key, value in node_grouping.items()
    }

    grouped_affinities = [
        tuple(node_grouping_tuples[item] for item in node_affinity)
        for node_affinity in node_affinities
    ]
    return grouped_affinities


def ungroup_by_nodes(
    affinities: list[list[SiblingGroups]], scope: Scope
) -> list[list[SiblingGroup]]:
    result: list[list[SiblingGroup]] = []
    if scope == Scope.SOCKET:
        result = [list(itertools.chain(*zip(*affinity))) for affinity in affinities]
    elif scope == Scope.NODE:
        result = [[group[0] for group in affinity] for affinity in affinities]
    return result


def check_core_count(
    affinities: list[list[SiblingGroup]],
    min_physical_cores: int = 1,
    max_physical_cores: Optional[int] = None,
) -> list[list[SiblingGroup]]:
    for gpu_id, affinity in enumerate(affinities):
        if len(affinity) < min_physical_cores:
            raise GPUAffinityError(
                f"Number of available physical cores for GPU {gpu_id} is less "
                f"the predefinied minimum, "
                f"min_physical_cores={min_physical_cores}, "
                f"available physical cores: {affinity} (count={len(affinity)})"
            )

    if max_physical_cores is not None:
        affinities = [affinity[:max_physical_cores] for affinity in affinities]

    return affinities


def ungroup_by_nodes_and_check_count(
    affinities: list[list[SiblingGroups]],
    scope: Scope,
    min_physical_cores: int = 1,
    max_physical_cores: Optional[int] = None,
) -> list[list[SiblingGroup]]:
    """Ungroup by nodes and check core count, keeping sibling groups intact."""
    affinities_ungrouped = ungroup_by_nodes(affinities, scope)
    affinities_checked = check_core_count(
        affinities_ungrouped, min_physical_cores, max_physical_cores
    )
    return affinities_checked


def check_affinities(affinities: list[CoreList]) -> None:
    if not len(affinities):
        raise GPUAffinityError(f"List of all affinities is empty: {affinities}")

    for idx, affinity in enumerate(affinities):
        if not len(affinities):
            raise GPUAffinityError(f"Affinity {idx} is empty: {affinity}")

    # sets of cores should be either identical or disjoint
    for i, j in itertools.product(affinities, affinities):
        if not set(i) == set(j) and not set(i).isdisjoint(set(j)):
            raise GPUAffinityError(
                f"Sets of cores should be either identical or disjoint, but got {i} and {j}."
            )


def get_affinities(
    nproc_per_node: int, scope: Scope, exclude_unavailable_cores: bool = True
) -> list[CoreList]:
    devices = [Device(i) for i in range(nproc_per_node)]
    affinities = [dev.get_cpu_affinity(scope) for dev in devices]

    if exclude_unavailable_cores:
        available_cores = os.sched_getaffinity(0)
        affinities = [sorted(list(set(affinity) & available_cores)) for affinity in affinities]

    check_affinities(affinities)

    return affinities


def get_grouped_affinities(
    nproc_per_node: int, exclude_unavailable_cores: bool = True
) -> list[NodeGroups]:
    socket_affinities = get_affinities(nproc_per_node, Scope.SOCKET, exclude_unavailable_cores)
    node_affinities = get_affinities(nproc_per_node, Scope.NODE, exclude_unavailable_cores)

    sibling_socket_affinities = group_by_siblings(socket_affinities)
    sibling_node_affinities = group_by_siblings(node_affinities)

    grouped_affinities = group_by_node(sibling_socket_affinities, sibling_node_affinities)

    return grouped_affinities


def get_unique(
    nproc_per_node: int,
    scope: Scope,
    mode: Mode,
    min_physical_cores: int,
    max_physical_cores: Optional[int],
    balanced: bool = True,
) -> list[list[SiblingGroup]]:
    grouped_affinities: list[NodeGroups] = get_grouped_affinities(nproc_per_node)

    grouped_affinities_to_device_ids: dict[NodeGroups, list[int]] = collections.defaultdict(list)

    for idx, grouped_affinity in enumerate(grouped_affinities):
        grouped_affinities_to_device_ids[grouped_affinity].append(idx)

    # compute minimal number of physical cores per GPU across all GPUs and
    # sockets, code assigns this number of cores per GPU if balanced == True
    min_physical_cores_per_gpu = min(
        [len(cores) // len(gpus) for cores, gpus in grouped_affinities_to_device_ids.items()]
    )

    grouped_unique_affinities: list[Optional[list[SiblingGroups]]] = [None] * nproc_per_node

    for (
        grouped_affinity,
        device_ids,
    ) in grouped_affinities_to_device_ids.items():
        devices_per_group = len(device_ids)
        if balanced:
            cores_per_device = min_physical_cores_per_gpu
            grouped_affinity = grouped_affinity[: devices_per_group * min_physical_cores_per_gpu]
        else:
            cores_per_device = len(grouped_affinity) // devices_per_group

        for subgroup_id, device_id in enumerate(device_ids):
            if mode == Mode.UNIQUE_INTERLEAVED:
                unique_grouped_affinity = list(grouped_affinity[subgroup_id::devices_per_group])
            elif mode == Mode.UNIQUE_CONTIGUOUS:
                unique_grouped_affinity = list(
                    grouped_affinity[
                        subgroup_id * cores_per_device : (subgroup_id + 1) * cores_per_device
                    ]
                )
            else:
                raise GPUAffinityError("Unknown set_unique mode")

            grouped_unique_affinities[device_id] = unique_grouped_affinity

    # Ungroup by nodes and check count, but keep sibling groups intact
    ungrouped_affinities = ungroup_by_nodes_and_check_count(
        grouped_unique_affinities,
        scope,
        min_physical_cores,
        max_physical_cores,
    )

    # Return as list of sibling groups (each group is a tuple of logical cores)
    return ungrouped_affinities


def list_native_threads() -> list[int]:
    pid = os.getpid()
    task_dir = f"/proc/{pid}/task"
    tids: list[int] = []
    for entry in os.scandir(task_dir):
        if entry.is_dir():
            tids.append(int(entry.name))
    return tids


def configure_thread_affinity(
    gpu_id: int,
    nproc_per_node: int,
    *,
    extra_target_threads: Optional[list[int]] = None,
) -> dict[int, set[int]]:
    r"""Configures CPU affinity for the current process to match CPU-GPU topology.

    This function pins the target thread to all logical cores of a dedicated
    physical core that matches the CPU-GPU hardware topology, and pins other
    threads to the remaining logical cores from other physical cores. This
    improves and stabilizes performance of deep learning workloads.

    The function uses UNIQUE_CONTIGUOUS mode with NODE scope, which is the
    recommended configuration for NVIDIA DGX servers.

    Args:
        gpu_id (int): GPU index, value from 0 to `nproc_per_node` - 1
        nproc_per_node (int): number of processes per node
        extra_target_threads (int, optional): native thread ID to pin to dedicated core.
            If None, uses the current thread. Default: None

    Returns:
        ThreadAffinityResult: Object with target_thread_affinity and
            other_threads_affinity - sets of CPU cores assigned to the target
            thread and other threads respectively.

    Raises:
        GPUAffinityError: If no suitable CPU cores can be found for the given
            GPU, or if affinity cannot be set properly.

    Example:
        import gpu_affinity
        import torch

        gpu_id = 0
        nproc_per_node = torch.cuda.device_count()

        try:
            affinity = gpu_affinity.configure_thread_affinity(gpu_id, nproc_per_node)
            print(f'GPU {gpu_id}: target thread affinity: {affinity.target_thread_affinity}')
            print(f'GPU {gpu_id}: other threads affinity: {affinity.other_threads_affinity}')
        except gpu_affinity.GPUAffinityError as e:
            print(f'Failed to configure affinity: {e}')

    WARNING: Intel's OpenMP implementation resets affinity on the first call to
    an OpenMP function after a fork. Set KMP_AFFINITY=disabled environment
    variable to preserve the affinity after a fork (e.g., in PyTorch DataLoader
    workers).
    """
    if gpu_id >= nproc_per_node:
        msg = f"gpu_id={gpu_id} should be smaller than nproc_per_node={nproc_per_node}"
        raise GPUAffinityError(msg)

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as ex:
        msg = "Error while initializing pynvml"
        raise GPUAffinityError(msg) from ex

    # Get CPU affinity for all GPUs using recommended settings
    # Returns cores already grouped by siblings
    extra_target_threads = extra_target_threads or []

    try:
        affinity = get_unique(
            nproc_per_node=nproc_per_node,
            scope=Scope.NODE,
            mode=Mode.UNIQUE_CONTIGUOUS,
            min_physical_cores=2,
            max_physical_cores=None,
            balanced=True,
        )
    except GPUAffinityError as ex:
        msg = (
            f"Failed to get CPU affinity for GPU {gpu_id}. "
            f"This may happen if the container has incorrect CPU assignment."
        )
        raise GPUAffinityError(msg) from ex

    # Get core groups (sibling groups) assigned to this GPU
    core_groups: list[SiblingGroup] = affinity[gpu_id]

    try:
        target_threads = [threading.current_thread().native_id] + extra_target_threads

        # Get all other threads
        all_threads = list_native_threads()
        other_threads = list(set(all_threads).difference(target_threads))

        # Assign a dedicated core for each target thread
        target_affinity = core_groups.pop(1)
        for tid in target_threads:
            os.sched_setaffinity(tid, target_affinity)

        # Set affinity for other threads
        other_affinity = list(itertools.chain.from_iterable(core_groups))
        for tid in other_threads:
            os.sched_setaffinity(tid, other_affinity)

        # Verify affinity was set correctly
        affinities = {tid: os.sched_getaffinity(tid) for tid in target_threads}
        affinities[0] = os.sched_getaffinity(other_threads[0]) if other_threads else None

        return affinities

    except OSError as ex:
        msg = f"Error while setting thread affinity: {ex}"
        raise GPUAffinityError(msg) from ex
    except AttributeError as ex:
        msg = "OS affinity functions are not available on this platform"
        raise GPUAffinityError(msg) from ex
