"""
Dimension-split baseline for Harmony-style evaluation.

This script:
1. Loads preprocessed base and query vectors
2. Splits vectors across dimension blocks
3. Computes partial similarities block by block
4. Supports early stopping using a running threshold
5. Measures latency, throughput, pruning, and block load
"""

from pathlib import Path
import time
import argparse
import heapq
import numpy as np


DATA_DIR = Path("data/processed")


def load_data(query_set: str):
    base = np.load(DATA_DIR / "base_vectors.npy")

    if query_set == "uniform":
        queries = np.load(DATA_DIR / "queries_uniform.npy")
    elif query_set == "skewed":
        queries = np.load(DATA_DIR / "queries_skewed.npy")
    else:
        raise ValueError("query_set must be 'uniform' or 'skewed'")

    return base.astype(np.float32), queries.astype(np.float32)


def split_dimension_ranges(dim: int, num_blocks: int):
    block_edges = np.linspace(0, dim, num_blocks + 1, dtype=int)
    ranges = []
    for i in range(num_blocks):
        ranges.append((block_edges[i], block_edges[i + 1]))
    return ranges


def full_search(query: np.ndarray, base: np.ndarray, top_k: int):
    scores = base @ query
    k = min(top_k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    results = [(float(scores[i]), int(i)) for i in idx]
    return heapq.nlargest(top_k, results, key=lambda x: x[0])


def dimension_search(query: np.ndarray, base: np.ndarray, top_k: int, num_blocks: int):
    dim_ranges = split_dimension_ranges(base.shape[1], num_blocks)
    partial_scores = np.zeros(len(base), dtype=np.float32)
    active = np.ones(len(base), dtype=bool)
    block_loads = np.zeros(num_blocks, dtype=np.int64)
    pruned_counts = np.zeros(num_blocks, dtype=np.int64)

    warmup_scores = base[:, dim_ranges[0][0]:dim_ranges[0][1]] @ query[dim_ranges[0][0]:dim_ranges[0][1]]
    partial_scores += warmup_scores
    block_loads[0] += len(base)

    k = min(top_k, len(base))
    top_idx = np.argpartition(-partial_scores, k - 1)[:k]
    threshold = float(partial_scores[top_idx].min())

    suffix_max = []
    for start, end in dim_ranges:
        block = base[:, start:end]
        q_block = query[start:end]
        block_max = np.sum(np.abs(block) * np.abs(q_block), axis=1)
        suffix_max.append(block_max)

    suffix_remaining = [None] * num_blocks
    running = np.zeros(len(base), dtype=np.float32)
    for i in range(num_blocks - 1, -1, -1):
        running = running + suffix_max[i]
        suffix_remaining[i] = running.copy()

    for block_id in range(1, num_blocks):
        start, end = dim_ranges[block_id]

        optimistic_bound = partial_scores + suffix_remaining[block_id]
        keep_mask = optimistic_bound >= threshold
        pruned_now = active & (~keep_mask)
        pruned_counts[block_id] += pruned_now.sum()
        active = active & keep_mask

        if not np.any(active):
            break

        active_idx = np.where(active)[0]
        q_block = query[start:end]
        scores = base[active_idx, start:end] @ q_block
        partial_scores[active_idx] += scores

        block_loads[block_id] += len(active_idx)

        active_scores = partial_scores[active_idx]
        if len(active_scores) >= k:
            threshold = float(np.partition(active_scores, -k)[-k])
        else:
            threshold = float(active_scores.min())

    final_idx = np.where(active)[0]
    final_scores = partial_scores[final_idx]
    results = [(float(final_scores[i]), int(final_idx[i])) for i in range(len(final_idx))]
    results = heapq.nlargest(top_k, results, key=lambda x: x[0])

    return {
        "results": results,
        "block_loads": block_loads,
        "pruned_counts": pruned_counts,
    }


def run_dimension_baseline(base, queries, top_k: int, num_blocks: int):
    latencies = []
    all_results = []
    total_block_loads = np.zeros(num_blocks, dtype=np.int64)
    total_pruned = np.zeros(num_blocks, dtype=np.int64)

    for query in queries:
        start = time.perf_counter()
        out = dimension_search(query, base, top_k, num_blocks)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        all_results.append(out["results"])
        total_block_loads += out["block_loads"]
        total_pruned += out["pruned_counts"]

    return {
        "latencies": np.array(latencies),
        "results": all_results,
        "block_loads": total_block_loads,
        "pruned_counts": total_pruned,
    }


def run_full_baseline(base, queries, top_k: int):
    latencies = []
    all_results = []

    for query in queries:
        start = time.perf_counter()
        results = full_search(query, base, top_k)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        all_results.append(results)

    return {
        "latencies": np.array(latencies),
        "results": all_results,
    }


def summarize_full(name: str, run_output: dict):
    lat = run_output["latencies"]
    total_time = float(lat.sum())
    qps = len(lat) / total_time if total_time > 0 else 0.0

    print(f"\n{name}")
    print(f"Queries: {len(lat)}")
    print(f"Mean latency (s): {lat.mean():.6f}")
    print(f"P95 latency (s): {np.percentile(lat, 95):.6f}")
    print(f"P99 latency (s): {np.percentile(lat, 99):.6f}")
    print(f"Throughput (QPS): {qps:.2f}")


def summarize_dimension(name: str, run_output: dict):
    lat = run_output["latencies"]
    total_time = float(lat.sum())
    qps = len(lat) / total_time if total_time > 0 else 0.0

    loads = run_output["block_loads"].astype(np.float64)
    imbalance_ratio = loads.max() / loads.mean() if loads.mean() > 0 else 0.0

    print(f"\n{name}")
    print(f"Queries: {len(lat)}")
    print(f"Mean latency (s): {lat.mean():.6f}")
    print(f"P95 latency (s): {np.percentile(lat, 95):.6f}")
    print(f"P99 latency (s): {np.percentile(lat, 99):.6f}")
    print(f"Throughput (QPS): {qps:.2f}")
    print(f"Load imbalance ratio: {imbalance_ratio:.4f}")
    print(f"Block loads: {loads.astype(int).tolist()}")
    print(f"Pruned candidates by block: {run_output['pruned_counts'].astype(int).tolist()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-set", choices=["uniform", "skewed"], default="uniform")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-blocks", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    base, queries = load_data(args.query_set)

    full_out = run_full_baseline(
        base=base,
        queries=queries,
        top_k=args.top_k,
    )

    dim_out = run_dimension_baseline(
        base=base,
        queries=queries,
        top_k=args.top_k,
        num_blocks=args.num_blocks,
    )

    summarize_full("Full-vector baseline", full_out)
    summarize_dimension("Dimension-split baseline", dim_out)


if __name__ == "__main__":
    main()