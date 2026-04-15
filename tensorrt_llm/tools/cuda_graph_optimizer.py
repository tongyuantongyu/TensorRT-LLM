# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Find optimal CUDA graph batch sizes from StepProfiler data.

Reads StepProfiler JSONL files, builds a histogram of generation batch sizes
(skipping warmup iterations), and uses dynamic programming to find the set of
CUDA graph batch sizes that minimizes total padding cost.

Usage:
    python -m tensorrt_llm.tools.cuda_graph_optimizer step_profiler/*.jsonl -n 16

Constraints:
- Graph sizes >= 32 must be aligned to 32; sizes < 32 must be power of 2.
- To serve a step with batch size n, use smallest captured graph with size >= n.
"""

from __future__ import annotations

import argparse
import functools
import itertools
import json
import math
from collections import defaultdict


def load_histogram(paths: list[str],
                   event_name: str = "forward",
                   field: str = "gen_batch_size") -> dict[int, int]:
    """Load a histogram from one or more StepProfiler JSONL files.

    Filters events by name, skips warmup iterations (steps before the first
    warm_up phase="end" event), extracts the given field, and counts
    occurrences. Results from multiple files are merged.

    Args:
        paths: Paths to .jsonl files from StepProfiler.
        event_name: Event type to filter (default: "forward").
        field: Field name to histogram (default: "gen_batch_size").

    Returns:
        Mapping from field value to occurrence count.
    """
    hist: dict[int, int] = defaultdict(int)
    for path in paths:
        in_warmup = True
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                # Track warmup state transitions
                if record.get("event") == "warm_up":
                    if record.get("phase") == "start":
                        in_warmup = True
                    elif record.get("phase") == "end":
                        in_warmup = False
                    continue
                if in_warmup:
                    continue
                if record.get("event") != event_name:
                    continue
                value = record.get(field)
                if value is not None:
                    hist[int(value)] += 1
    return dict(hist)


def compute_cost(hist: dict[int, int], graph_sizes: list[int]) -> int:
    """Compute total padding cost for a given set of graph sizes.

    For each batch size in the histogram, the cost is the smallest graph size
    that can serve it (>= batch size), multiplied by the frequency. Batch sizes
    larger than all graph sizes use the batch size itself (no graph).
    """
    sorted_sizes = sorted(graph_sizes)
    total = 0
    for batch, freq in hist.items():
        # Find smallest graph size >= batch
        padded = batch
        for s in sorted_sizes:
            if s >= batch:
                padded = s
                break
        total += padded * freq
    return total


def default_graph_sizes(max_batch_size: int) -> list[int]:
    """Generate default CUDA graph batch sizes matching TRT-LLM's convention.

    Reproduces CudaGraphConfig._generate_cuda_graph_batch_sizes with
    enable_padding=True.
    """
    batch_sizes = [1, 2, 4] + [i * 8 for i in range(1, 17)]
    batch_sizes += [
        2**i for i in range(8, math.ceil(math.log(max_batch_size, 2)))
    ]
    batch_sizes = sorted(s for s in batch_sizes if s <= max_batch_size)
    if max_batch_size != batch_sizes[-1]:
        batch_sizes.append(max_batch_size)
    return batch_sizes


class CudaGraphOptimizer:
    """DP optimizer for CUDA graph batch size selection.

    Given a histogram of actual generation batch sizes and a budget of N
    CUDA graph slots, finds the set of batch sizes that minimizes total
    padding cost.

    Candidate sizes: powers of 2 below alignment, then multiples of alignment.
    """

    def __init__(self, hist: dict[int, int], alignment: int = 32) -> None:
        self.hist = hist
        if not hist:
            return

        max_batch = max(hist.keys())

        small = [2**i for i in range(int(math.log2(alignment)))]
        possible_candidates = itertools.chain(
            small, itertools.count(alignment, alignment))

        self.size_candidates: list[int] = list(
            itertools.takewhile(
                lambda n: n / max_batch < 2
                if n < alignment else n < max_batch + alignment,
                possible_candidates))

        # Precompute prefix sums for O(1) cost lookups.
        max_size = self.size_candidates[-1]
        freq_prefix = [0] * (max_size + 1)
        for b, freq in hist.items():
            if b <= max_size:
                freq_prefix[b] = freq

        for i in range(1, max_size + 1):
            freq_prefix[i] += freq_prefix[i - 1]

        self.freq_prefix = freq_prefix

    def get_cost(self, from_idx: int, to_idx: int) -> int:
        """Cost of padding requests between candidates[from_idx] and candidates[to_idx]."""
        low = 0 if from_idx == -1 else self.size_candidates[from_idx]
        high = self.size_candidates[to_idx]
        request_count = self.freq_prefix[high] - self.freq_prefix[low]
        return request_count * high

    @functools.cache
    def _best_subrange(
            self, graphs_left: int,
            end_idx: int) -> tuple[list[int], float]:
        """Find optimal sizes to cover workload up to end_idx with graphs_left graphs."""
        if graphs_left == 1:
            return [self.size_candidates[end_idx]], self.get_cost(-1, end_idx)

        if end_idx + 1 < graphs_left:
            return [], float('inf')

        best_cost = float('inf')
        best_choices: list[int] = []

        for u in range(graphs_left - 2, end_idx):
            prev_choices, prev_cost = self._best_subrange(
                graphs_left - 1, u)
            curr_cost = prev_cost + self.get_cost(u, end_idx)

            if curr_cost < best_cost:
                best_cost = curr_cost
                best_choices = prev_choices + [self.size_candidates[end_idx]]

        return best_choices, best_cost

    def best(self, n: int) -> tuple[list[int], int]:
        """Find the optimal n batch sizes minimizing total padding cost.

        Args:
            n: Maximum number of CUDA graph batch sizes.

        Returns:
            Tuple of (optimal_sizes, total_cost).
        """
        if not self.hist or n <= 0:
            return [], 0

        n = min(n, len(self.size_candidates))
        best_choices, best_cost = self._best_subrange(
            n, len(self.size_candidates) - 1)
        return best_choices, int(best_cost)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=
        "Find optimal CUDA graph batch sizes from StepProfiler data.")
    parser.add_argument(
        "input_files",
        nargs="+",
        help="StepProfiler .jsonl files (supports glob patterns)")
    parser.add_argument(
        "-n",
        "--num-graphs",
        type=int,
        default=16,
        help="Max number of CUDA graphs to select (default: 16)")
    parser.add_argument(
        "--alignment",
        type=int,
        default=32,
        help=
        "Candidate alignment. Powers of 2 below this, multiples of this above "
        "(default: 32). TRT-LLM default graph sizes use alignment=8.")
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help=
        "Reference for overhead comparison. Either a max_batch_size (integer) "
        "to compute default graph sizes, or a comma-separated list of sizes "
        "(e.g. '4,16,32,64,128,256,512'). Default: perfect cost (no padding)."
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write results to JSON file")
    args = parser.parse_args()

    hist = load_histogram(args.input_files)

    if not hist:
        print("No data found.")
        return

    optimizer = CudaGraphOptimizer(hist, alignment=args.alignment)
    total_steps = sum(hist.values())
    perfect_cost = sum(batch * freq for batch, freq in hist.items())

    # Determine reference cost for overhead comparison.
    ref_sizes = None
    if args.reference is not None:
        if ',' in args.reference:
            ref_sizes = [int(s) for s in args.reference.split(',')]
        else:
            ref_sizes = default_graph_sizes(int(args.reference))
        ref_cost = compute_cost(hist, ref_sizes)
    else:
        ref_cost = perfect_cost

    print(
        f"Histogram: {len(hist)} unique batch sizes, {total_steps} steps "
        f"(from {len(args.input_files)} file(s))")
    if ref_sizes is not None:
        print(
            f"Reference: {len(ref_sizes)} graphs, cost {ref_cost} "
            f"(+{(ref_cost - perfect_cost) / perfect_cost * 100:.2f}% vs perfect)"
        )
    print()

    results = []
    last_extra_cost = None
    for n in range(1, args.num_graphs + 1):
        graph_sizes, min_cost = optimizer.best(n)
        extra_cost = (min_cost - ref_cost) / perfect_cost
        reduction = 0.0 if last_extra_cost is None else last_extra_cost - extra_cost
        last_extra_cost = extra_cost
        print(f"{n:>2} graphs: {min_cost} "
              f"({extra_cost * 100:+.2f}%,\t -{reduction * 100:.2f}%)"
              f", \t{graph_sizes}")
        results.append({
            "num_graphs": n,
            "batch_sizes": graph_sizes,
            "cost": min_cost,
            "overhead_pct": round(extra_cost * 100, 2),
        })

    print()
    a = args.alignment
    print(
        f"Alignment: sizes >= {a} are multiples of {a}; sizes < {a} are powers of 2."
    )
    print(
        "Cost model: time ~ batch_size + const per step; minimizing total batch_size*steps."
    )

    if args.output_json:
        output = {
            "histogram": hist,
            "total_steps": total_steps,
            "perfect_cost": perfect_cost,
            "ref_cost": ref_cost,
            "ref_sizes": ref_sizes,
            "results": results,
        }
        with open(args.output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
