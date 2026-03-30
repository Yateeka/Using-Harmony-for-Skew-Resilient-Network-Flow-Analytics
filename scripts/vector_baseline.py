"""
Vector-only baseline for Harmony-style evaluation.

This script:
1. Loads preprocessed base and query vectors
2. Partitions base vectors across shards
3. Searches each shard independently
4. Merges per-shard top-k results
5. Measures latency, throughput, and shard load
"""

from pathlib import Path
import time
import argparse
import heapq
import numpy as np
from sklearn.cluster import KMeans


DATA_DIR = Path("data/processed")
RANDOM_STATE = 42


def load_data(query_set: str):
    base = np.load(DATA_DIR / "base_vectors.npy")

    if query_set == "uniform":
        queries = np.load(DATA_DIR / "queries_uniform.npy")
    elif query_set == "skewed":
        queries = np.load(DATA_DIR / "queries_skewed.npy")
    else:
        raise ValueError("query_set must be 'uniform' or 'skewed'")

    return base.astype(np.float32), queries.astype(np.float32)


def build_random_shards(base: np.ndarray, num_shards: int):
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(len(base))
    rng.shuffle(indices)
    split_indices = np.array_split(indices, num_shards)

    shards = []
    for shard_id, idx in enumerate(split_indices):
        shards.append(
            {
                "shard_id": shard_id,
                "indices": idx,
                "vectors": base[idx],
            }
        )
    return shards


def build_cluster_shards(base: np.ndarray, num_shards: int):
    kmeans = KMeans(n_clusters=num_shards, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(base)

    shards = []
    for shard_id in range(num_shards):
        idx = np.where(labels == shard_id)[0]
        shards.append(
            {
                "shard_id": shard_id,
                "indices": idx,
                "vectors": base[idx],
            }
        )
    return shards, kmeans


def shard_search(query: np.ndarray, shard: dict, top_k: int):
    if len(shard["vectors"]) == 0:
        return []

    scores = shard["vectors"] @ query
    k = min(top_k, len(scores))
    top_local_idx = np.argpartition(-scores, k - 1)[:k]
    results = []

    for local_idx in top_local_idx:
        global_idx = shard["indices"][local_idx]
        score = float(scores[local_idx])
        results.append((score, int(global_idx), shard["shard_id"]))

    return results


def merge_results(candidates, top_k: int):
    return heapq.nlargest(top_k, candidates, key=lambda x: x[0])


def run_random_partition(base, queries, num_shards: int, top_k: int):
    shards = build_random_shards(base, num_shards)
    shard_loads = np.zeros(num_shards, dtype=np.int64)

    latencies = []
    all_results = []

    for query in queries:
        start = time.perf_counter()
        candidates = []

        for shard in shards:
            local_results = shard_search(query, shard, top_k)
            shard_loads[shard["shard_id"]] += len(shard["vectors"])
            candidates.extend(local_results)

        merged = merge_results(candidates, top_k)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        all_results.append(merged)

    return {
        "latencies": np.array(latencies),
        "results": all_results,
        "shard_loads": shard_loads,
    }


def run_cluster_partition(base, queries, num_shards: int, top_k: int, nprobe: int):
    shards, kmeans = build_cluster_shards(base, num_shards)
    shard_loads = np.zeros(num_shards, dtype=np.int64)

    latencies = []
    all_results = []

    centroid_scores = kmeans.cluster_centers_ @ queries.T

    for i, query in enumerate(queries):
        start = time.perf_counter()
        query_centroid_scores = centroid_scores[:, i]
        probe_ids = np.argpartition(-query_centroid_scores, nprobe - 1)[:nprobe]

        candidates = []
        for shard_id in probe_ids:
            shard = shards[int(shard_id)]
            local_results = shard_search(query, shard, top_k)
            shard_loads[int(shard_id)] += len(shard["vectors"])
            candidates.extend(local_results)

        merged = merge_results(candidates, top_k)
        elapsed = time.perf_counter() - start

        latencies.append(elapsed)
        all_results.append(merged)

    return {
        "latencies": np.array(latencies),
        "results": all_results,
        "shard_loads": shard_loads,
    }


def summarize(name: str, run_output: dict):
    lat = run_output["latencies"]
    total_time = float(lat.sum())
    qps = len(lat) / total_time if total_time > 0 else 0.0

    loads = run_output["shard_loads"].astype(np.float64)
    imbalance_ratio = loads.max() / loads.mean() if loads.mean() > 0 else 0.0

    print(f"\n{name}")
    print(f"Queries: {len(lat)}")
    print(f"Mean latency (s): {lat.mean():.6f}")
    print(f"P95 latency (s): {np.percentile(lat, 95):.6f}")
    print(f"P99 latency (s): {np.percentile(lat, 99):.6f}")
    print(f"Throughput (QPS): {qps:.2f}")
    print(f"Load imbalance ratio: {imbalance_ratio:.4f}")
    print(f"Shard loads: {loads.astype(int).tolist()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-set", choices=["uniform", "skewed"], default="uniform")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--nprobe", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    base, queries = load_data(args.query_set)

    random_out = run_random_partition(
        base=base,
        queries=queries,
        num_shards=args.num_shards,
        top_k=args.top_k,
    )

    cluster_out = run_cluster_partition(
        base=base,
        queries=queries,
        num_shards=args.num_shards,
        top_k=args.top_k,
        nprobe=args.nprobe,
    )

    summarize("Random shard baseline", random_out)
    summarize("Cluster shard baseline", cluster_out)


if __name__ == "__main__":
    main()