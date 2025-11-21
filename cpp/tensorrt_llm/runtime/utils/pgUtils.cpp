/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

#include <linux/futex.h>
#include <memory>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"

#if defined(__x86_64__)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <intrin.h>
#endif

namespace tensorrt_llm::pg_utils
{

c10::intrusive_ptr<c10d::ProcessGroup> pg_world;
c10::intrusive_ptr<c10d::ProcessGroup> pg_local;

c10::intrusive_ptr<c10d::ProcessGroup> get_world_pg()
{
    return pg_world;
}

c10::intrusive_ptr<c10d::ProcessGroup> get_local_pg()
{
    return pg_local;
}

void init_pg(c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_world,
    c10::intrusive_ptr<c10d::ProcessGroup> const& process_group_local)
{
    TLLM_LOG_DEBUG(process_group_world->getRank(), "Init process group on rank %d", process_group_world->getRank());
    pg_world = process_group_world;
    pg_local = process_group_local;
}

/**
 * Shared data structure for the barrier.
 */
struct LocalNodeBarrier::BarrierData
{
    struct AtomicPair
    {
        // Avoid waking up everyone when new participant joins.
        alignas(std::hardware_destructive_interference_size) std::atomic<uint32_t> count
            = 0; // Number of processes that have arrived at the barrier
        alignas(std::hardware_destructive_interference_size) std::atomic<uint32_t> generation
            = 0; // Generation counter for reusable barrier
    };

    AtomicPair futex;
    AtomicPair spin;
};

namespace
{
constexpr mode_t kSharedMemoryPermissions = 0666;
constexpr int32_t kWakeupAllWaiters = std::numeric_limits<int32_t>::max();
constexpr timespec kFutexTimeout = {1, 0};
constexpr std::chrono::steady_clock::duration kSyncTimeout = std::chrono::seconds(30);

// Wrapper for futex syscall
int64_t futex(uint32_t* uaddr, int futex_op, uint32_t val, timespec const* timeout = nullptr, int32_t* uaddr2 = nullptr,
    int32_t val3 = 0)
{
    return syscall(SYS_futex, uaddr, futex_op, val, timeout, uaddr2, val3);
}

} // namespace

LocalNodeBarrier::LocalNodeBarrier(std::string memPath)
    : mMemPath(std::move(memPath))
{
}

void LocalNodeBarrier::init()
{
    int const fd = shm_open(mMemPath.c_str(), O_CREAT | O_RDWR, kSharedMemoryPermissions);
    // Open or create shared memory
    if (fd == -1)
    {
        mMemPath.clear();
        throw std::runtime_error("Failed to open shared memory (" + std::to_string(errno) + "): " + mMemPath);
    }

    // Get current size
    struct stat file_stat
    {
    };

    if (fstat(fd, &file_stat) == -1)
    {
        close(fd);
        throw std::runtime_error("Failed to stat shared memory (" + std::to_string(errno) + ")");
    }

    bool const need_init = file_stat.st_size == 0;

    // Set the size if needed
    if (need_init)
    {
        if (ftruncate(fd, sizeof(BarrierData)) == -1)
        {
            close(fd);
            throw std::runtime_error("Failed to set shared memory size (" + std::to_string(errno) + ")");
        }
    }

    // Map the shared memory
    auto* const addr = mmap(nullptr, sizeof(BarrierData), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    // After a call to mmap(2) the file descriptor may be closed without affecting the memory mapping.
    close(fd);

    if (addr == MAP_FAILED)
    {
        throw std::runtime_error("Failed to map shared memory (" + std::to_string(errno) + ")");
    }

    mData = static_cast<BarrierData*>(addr);

    // Initialize if this is the first process to create the shared memory
    if (need_init)
    {
        new (mData) BarrierData();
    }
}

LocalNodeBarrier::~LocalNodeBarrier()
{
    if (mData != nullptr)
    {
        munmap(mData, sizeof(BarrierData));
    }
    if (!mMemPath.empty())
    {
        shm_unlink(mMemPath.c_str());
    }
}

void LocalNodeBarrier::spin_sync(uint32_t participants) const
{
    if (participants < 2)
    {
        return;
    }

    auto& data = mData->spin;

    // Load the current generation
    auto const gen = data.generation.load(std::memory_order_acquire);

    // Increment the arrival count atomically
    auto const count = data.count.fetch_add(1, std::memory_order_relaxed) + 1;

    if (count < participants)
    {
        // Not the last process to arrive, wait for the generation to change
        while (data.generation.load(std::memory_order_acquire) == gen)
        {
#if defined(__x86_64__)
            _mm_pause();
#elif defined(__aarch64__)
            __yield();
#endif
        }
    }
    else
    {
        // Last process to arrive at the barrier
        // Reset the count for the next round
        data.count.store(0, std::memory_order_relaxed);

        // Increment the generation to signal completion
        data.generation.fetch_add(1, std::memory_order_release);
    }
}

void LocalNodeBarrier::futex_sync(uint32_t participants, bool spin) const
{
    if (participants < 2)
    {
        return;
    }

    auto& data = mData->futex;

    // Load the current generation
    auto const gen = data.generation.load(std::memory_order_acquire);

    // Increment the arrival count atomically
    auto const count = data.count.fetch_add(1, std::memory_order_relaxed) + 1;

    if (count < participants)
    {
        auto const end = std::chrono::steady_clock::now() + kSyncTimeout;
        // Not the last process to arrive, wait for the generation to change
        while (data.generation.load(std::memory_order_acquire) == gen)
        {
            if (std::chrono::steady_clock::now() > end)
            {
                throw std::runtime_error("FutexBarrier sync timeout");
            }

            // FUTEX_WAIT: If *uaddr == gen, sleep until woken by FUTEX_WAKE
            // If the value has changed, futex returns immediately with EAGAIN
            futex(reinterpret_cast<uint32_t*>(&data.generation), FUTEX_WAIT, gen, &kFutexTimeout);
        }
    }
    else
    {
        // Last process to arrive at the barrier
        // Reset the count for the next round
        data.count.store(0, std::memory_order_relaxed);

        // Increment the generation to signal completion
        data.generation.fetch_add(1, std::memory_order_release);

        // Wake up all waiting processes
        futex(reinterpret_cast<uint32_t*>(&data.generation), FUTEX_WAKE, kWakeupAllWaiters);
    }

    if (spin)
    {
        spin_sync(participants);
    }
}

} // namespace tensorrt_llm::pg_utils
