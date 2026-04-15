# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tensorrt_llm.tools.cuda_graph_optimizer import (CudaGraphOptimizer,
                                                      load_histogram)


class TestCudaGraphOptimizer:

    def test_empty_histogram(self):
        opt = CudaGraphOptimizer({})
        sizes, cost = opt.best(8)
        assert sizes == []
        assert cost == 0

    def test_single_batch_size(self):
        hist = {64: 100}
        opt = CudaGraphOptimizer(hist)
        sizes, cost = opt.best(1)
        assert len(sizes) == 1
        assert sizes[0] >= 64
        assert cost == sizes[0] * 100

    def test_single_graph_covers_all(self):
        hist = {32: 50, 64: 50}
        opt = CudaGraphOptimizer(hist)
        sizes, cost = opt.best(1)
        assert len(sizes) == 1
        # The single graph must cover all batch sizes
        assert sizes[0] >= 64

    def test_two_graphs_better_than_one(self):
        hist = {32: 100, 256: 100}
        opt = CudaGraphOptimizer(hist)
        _, cost1 = opt.best(1)
        _, cost2 = opt.best(2)
        assert cost2 <= cost1

    def test_more_graphs_never_worse(self):
        hist = {16: 50, 64: 100, 128: 200, 256: 150, 512: 50}
        opt = CudaGraphOptimizer(hist)
        prev_cost = float('inf')
        for n in range(1, 6):
            _, cost = opt.best(n)
            assert cost <= prev_cost
            prev_cost = cost

    def test_candidate_alignment(self):
        hist = {500: 100}
        opt = CudaGraphOptimizer(hist)
        # All candidates < 32 should be powers of 2
        small = [c for c in opt.size_candidates if c < 32]
        for s in small:
            assert s & (s - 1) == 0, f"{s} is not a power of 2"
        # All candidates >= 32 should be multiples of 32
        large = [c for c in opt.size_candidates if c >= 32]
        for s in large:
            assert s % 32 == 0, f"{s} is not a multiple of 32"

    def test_perfect_cost_achievable(self):
        # If histogram only has values that are valid candidates,
        # enough graphs should achieve perfect cost
        hist = {32: 100, 64: 100, 128: 100}
        opt = CudaGraphOptimizer(hist)
        sizes, cost = opt.best(16)
        perfect_cost = sum(b * f for b, f in hist.items())
        assert cost == perfect_cost

    def test_cap_n_at_candidates(self):
        hist = {4: 10}
        opt = CudaGraphOptimizer(hist)
        n_candidates = len(opt.size_candidates)
        sizes, cost = opt.best(100)
        assert len(sizes) <= n_candidates

    def test_zero_graphs(self):
        hist = {64: 100}
        opt = CudaGraphOptimizer(hist)
        sizes, cost = opt.best(0)
        assert sizes == []
        assert cost == 0

    def test_known_result(self):
        # Known small case: batch sizes 1 and 2, both with equal frequency
        hist = {1: 50, 2: 50}
        opt = CudaGraphOptimizer(hist)
        sizes, cost = opt.best(2)
        # With 2 graphs, should be able to exactly cover both
        # Candidates include 1, 2, so cost = 1*50 + 2*50 = 150
        assert cost == 150
        assert 1 in sizes
        assert 2 in sizes


class TestLoadHistogram:

    def test_load_from_jsonl(self, tmp_path):
        jsonl_file = tmp_path / "rank0.jsonl"
        events = [
            {"step": None, "event": "warm_up", "phase": "start"},
            {"step": None, "event": "warm_up", "phase": "end"},
            {"step": 0, "event": "forward", "gen_batch_size": 128},
            {"step": 1, "event": "forward", "gen_batch_size": 256},
            {"step": 2, "event": "forward", "gen_batch_size": 128},
            {"step": 3, "event": "ctx_schedule", "trigger": "token_ratio"},
            {"step": 4, "event": "forward", "gen_batch_size": 512},
        ]
        with open(jsonl_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + '\n')

        hist = load_histogram([str(jsonl_file)])
        assert hist == {128: 2, 256: 1, 512: 1}

    def test_load_skips_warmup(self, tmp_path):
        jsonl_file = tmp_path / "rank0.jsonl"
        events = [
            {"step": None, "event": "warm_up", "phase": "start"},
            {"step": None, "event": "forward", "gen_batch_size": 64},
            {"step": None, "event": "forward", "gen_batch_size": 64},
            {"step": None, "event": "warm_up", "phase": "end"},
            {"step": 0, "event": "forward", "gen_batch_size": 128},
            {"step": 1, "event": "forward", "gen_batch_size": 256},
        ]
        with open(jsonl_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + '\n')

        hist = load_histogram([str(jsonl_file)])
        # warmup forward events (batch size 64) should be skipped
        assert hist == {128: 1, 256: 1}

    def test_load_skips_multiple_warmups(self, tmp_path):
        """Multiple warmup phases — all should be skipped."""
        jsonl_file = tmp_path / "rank0.jsonl"
        events = [
            {"step": None, "event": "warm_up", "phase": "start"},
            {"step": None, "event": "forward", "gen_batch_size": 2048},
            {"step": None, "event": "warm_up", "phase": "end"},
            {"step": 0, "event": "forward", "gen_batch_size": 128},
            {"step": 0, "event": "warm_up", "phase": "start"},
            {"step": 0, "event": "forward", "gen_batch_size": 2048},
            {"step": 0, "event": "warm_up", "phase": "end"},
            {"step": 0, "event": "forward", "gen_batch_size": 256},
            {"step": 1, "event": "forward", "gen_batch_size": 512},
        ]
        with open(jsonl_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + '\n')

        hist = load_histogram([str(jsonl_file)])
        assert hist == {128: 1, 256: 1, 512: 1}

    def test_load_multiple_files(self, tmp_path):
        file1 = tmp_path / "instance1_rank0.jsonl"
        file2 = tmp_path / "instance2_rank0.jsonl"
        events1 = [
            {"step": None, "event": "warm_up", "phase": "end"},
            {"step": 0, "event": "forward", "gen_batch_size": 128},
            {"step": 1, "event": "forward", "gen_batch_size": 256},
        ]
        events2 = [
            {"step": None, "event": "warm_up", "phase": "end"},
            {"step": 0, "event": "forward", "gen_batch_size": 128},
            {"step": 1, "event": "forward", "gen_batch_size": 512},
        ]
        for f_path, evts in [(file1, events1), (file2, events2)]:
            with open(f_path, "w") as f:
                for e in evts:
                    f.write(json.dumps(e) + '\n')

        hist = load_histogram([str(file1), str(file2)])
        assert hist == {128: 2, 256: 1, 512: 1}

    def test_load_filters_by_event(self, tmp_path):
        jsonl_file = tmp_path / "rank0.jsonl"
        events = [
            {"step": None, "event": "warm_up", "phase": "end"},
            {"step": 0, "event": "forward", "gen_batch_size": 128},
            {"step": 0, "event": "ctx_schedule", "trigger": "token_ratio"},
        ]
        with open(jsonl_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + '\n')

        hist = load_histogram([str(jsonl_file)])
        assert hist == {128: 1}

    def test_load_custom_event_and_field(self, tmp_path):
        jsonl_file = tmp_path / "rank0.jsonl"
        events = [
            {"step": None, "event": "warm_up", "phase": "end"},
            {
                "step": 0,
                "event": "ctx_schedule",
                "wait_iters": 10
            },
            {
                "step": 1,
                "event": "ctx_schedule",
                "wait_iters": 64
            },
            {
                "step": 2,
                "event": "ctx_schedule",
                "wait_iters": 10
            },
        ]
        with open(jsonl_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + '\n')

        hist = load_histogram([str(jsonl_file)],
                              event_name="ctx_schedule",
                              field="wait_iters")
        assert hist == {10: 2, 64: 1}

    def test_load_empty_file(self, tmp_path):
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")
        hist = load_histogram([str(jsonl_file)])
        assert hist == {}

    def test_load_no_warmup_end(self, tmp_path):
        """If no warm_up end event, all forward events should be skipped."""
        jsonl_file = tmp_path / "rank0.jsonl"
        events = [
            {"step": 0, "event": "forward", "gen_batch_size": 128},
            {"step": 1, "event": "forward", "gen_batch_size": 256},
        ]
        with open(jsonl_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + '\n')

        hist = load_histogram([str(jsonl_file)])
        assert hist == {}


class TestEndToEnd:

    def test_profiler_to_optimizer(self, tmp_path):
        """Full pipeline: create JSONL, load histogram, optimize."""
        jsonl_file = tmp_path / "rank0.jsonl"
        events = [
            {"step": None, "event": "warm_up", "phase": "start"},
            {"step": None, "event": "forward", "gen_batch_size": 64},
            {"step": None, "event": "warm_up", "phase": "end"},
        ]
        # Simulate workload: mostly batch size 256, some 128 and 512
        for step in range(100):
            if step < 20:
                bs = 128
            elif step < 80:
                bs = 256
            else:
                bs = 512
            events.append({
                "step": step,
                "event": "forward",
                "gen_batch_size": bs
            })

        with open(jsonl_file, "w") as f:
            for e in events:
                f.write(json.dumps(e) + '\n')

        hist = load_histogram([str(jsonl_file)])
        # warmup forward event (batch size 64) should be excluded
        assert sum(hist.values()) == 100

        opt = CudaGraphOptimizer(hist)
        sizes, cost = opt.best(4)
        assert len(sizes) <= 4
        assert cost > 0

        # Perfect cost for reference
        perfect_cost = sum(b * f for b, f in hist.items())
        assert cost >= perfect_cost
