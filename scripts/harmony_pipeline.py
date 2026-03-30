"""
Harmony-style pipeline prototype aligned with Algorithm 1 in the paper.

This script:
1. Loads preprocessed base vectors and query workloads
2. Builds vector partitions with k-means
3. Uses a prewarm stage to initialize a global top-k threshold
4. Routes queries to vector partitions
5. Executes a vector pipeline over routed partitions
6. Executes a dimension pipeline within each vector partition
7. Applies threshold-based early pruning across dimension blocks
8. Reports latency, throughput, routing load, compute load, and pruning statistics
"""

from pathlib import Path
import argparse
import heapq
import time
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


def split_dimension_ranges(dim: int, num_blocks: int):
    edges = np.linspace(0, dim, num_blocks + 1, dtype=int)
    ranges = []
    for i in range(num_blocks):
        ranges.append((edges[i], edges[i + 1]))
    return ranges


def build_vector_partitions(base: np.ndarray, num_partitions: int):
    kmeans = KMeans(
        n_clusters=num_partitions,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = kmeans.fit_predict(base)

    partitions = []
    for partition_id in range(num_partitions):
        idx = np.where(labels == partition_id)[0]
        partitions.append(
            {
                "partition_id": partition_id,
                "indices": idx,
                "vectors": base[idx],
            }
        )

    return partitions, kmeans


def route_query_to_partitions(query: np.ndarray, kmeans: KMeans, nprobe: int):
    centroid_scores = kmeans.cluster_centers_ @ query
    nprobe = min(nprobe, len(centroid_scores))
    top_idx = np.argpartition(-centroid_scores, nprobe - 1)[:nprobe]
    top_idx = top_idx[np.argsort(-centroid_scores[top_idx])]
    return top_idx


def merge_topk(results, top_k: int):
    if not results:
        return []
    return heapq.nlargest(top_k, results, key=lambda x: x[0])


def compute_exact_scores(query: np.ndarray, vectors: np.ndarray):
    return vectors @ query


def prewarm_heap(
    queries: np.ndarray,
    base: np.ndarray,
    top_k: int,
    warmup_queries: int,
    warmup_vectors: int,
):
    rng = np.random.default_rng(RANDOM_STATE)

    q_count = min(warmup_queries, len(queries))
    v_count = min(warmup_vectors, len(base))

    q_idx = rng.choice(len(queries), size=q_count, replace=False)
    v_idx = rng.choice(len(base), size=v_count, replace=False)

    sampled_queries = queries[q_idx]
    sampled_vectors = base[v_idx]

    heap = []

    for query in sampled_queries:
        scores = compute_exact_scores(query, sampled_vectors)
        local_k = min(top_k, len(scores))
        top_local = np.argpartition(-scores, local_k - 1)[:local_k]
        for idx in top_local:
            score = float(scores[idx])
            heapq.heappush(heap, score)
            if len(heap) > top_k:
                heapq.heappop(heap)

    if len(heap) < top_k:
        return -np.inf

    return float(heap[0])


def make_suffix_upper_bounds(vectors: np.ndarray, query: np.ndarray, dim_ranges):
    num_blocks = len(dim_ranges)
    block_bounds = []

    for start, end in dim_ranges:
        v_block = vectors[:, start:end]
        q_block = query[start:end]
        bound = np.sum(np.abs(v_block) * np.abs(q_block), axis=1)
        block_bounds.append(bound.astype(np.float32))

    suffix_bounds = [None] * num_blocks
    running = np.zeros(len(vectors), dtype=np.float32)

    for block_id in range(num_blocks - 1, -1, -1):
        running = running + block_bounds[block_id]
        suffix_bounds[block_id] = running.copy()

    return suffix_bounds


def update_threshold_from_scores(scores: np.ndarray, current_threshold: float, top_k: int):
    if len(scores) == 0:
        return current_threshold

    if len(scores) >= top_k:
        kth = float(np.partition(scores, -top_k)[-top_k])
        return max(current_threshold, kth)

    return max(current_threshold, float(scores.min()))


def dimension_pipeline(
    query: np.ndarray,
    partition: dict,
    dim_ranges,
    top_k: int,
    current_threshold: float,
):
    vectors = partition["vectors"]
    indices = partition["indices"]

    num_blocks = len(dim_ranges)
    block_loads = np.zeros(num_blocks, dtype=np.int64)
    pruned_counts = np.zeros(num_blocks, dtype=np.int64)

    if len(vectors) == 0:
        return {
            "results": [],
            "block_loads": block_loads,
            "pruned_counts": pruned_counts,
            "final_threshold": current_threshold,
        }

    partial_scores = np.zeros(len(vectors), dtype=np.float32)
    active = np.ones(len(vectors), dtype=bool)
    suffix_bounds = make_suffix_upper_bounds(vectors, query, dim_ranges)

    threshold = current_threshold

    for block_id, (start, end) in enumerate(dim_ranges):
        optimistic = partial_scores + suffix_bounds[block_id]
        keep_mask = optimistic >= threshold
        pruned_now = active & (~keep_mask)

        pruned_counts[block_id] += int(pruned_now.sum())
        active = active & keep_mask

        if not np.any(active):
            break

        active_idx = np.where(active)[0]
        q_block = query[start:end]
        scores = vectors[active_idx, start:end] @ q_block
        partial_scores[active_idx] += scores
        block_loads[block_id] += len(active_idx)

        threshold = update_threshold_from_scores(
            partial_scores[active_idx],
            threshold,
            top_k,
        )

    active_idx = np.where(active)[0]
    active_scores = partial_scores[active_idx]

    if len(active_scores) == 0:
        results = []
    else:
        local_k = min(top_k, len(active_scores))
        best_idx = np.argpartition(-active_scores, local_k - 1)[:local_k]
        results = []
        for pos in best_idx:
            local_pos = int(active_idx[pos])
            global_idx = int(indices[local_pos])
            score = float(active_scores[pos])
            results.append((score, global_idx, partition["partition_id"]))
        results = merge_topk(results, top_k)
        threshold = max(threshold, results[-1][0]) if len(results) >= top_k else threshold

    return {
        "results": results,
        "block_loads": block_loads,
        "pruned_counts": pruned_counts,
        "final_threshold": threshold,
    }


def vector_pipeline(
    queries_in_partition,
    partition: dict,
    dim_ranges,
    top_k: int,
    global_threshold: float,
):
    partition_results = {}
    total_block_loads = np.zeros(len(dim_ranges), dtype=np.int64)
    total_pruned_counts = np.zeros(len(dim_ranges), dtype=np.int64)
    running_threshold = global_threshold

    for query_id, query in queries_in_partition:
        out = dimension_pipeline(
            query=query,
            partition=partition,
            dim_ranges=dim_ranges,
            top_k=top_k,
            current_threshold=running_threshold,
        )

        partition_results[query_id] = out["results"]
        total_block_loads += out["block_loads"]
        total_pruned_counts += out["pruned_counts"]
        running_threshold = max(running_threshold, out["final_threshold"])

    return {
        "partition_results": partition_results,
        "block_loads": total_block_loads,
        "pruned_counts": total_pruned_counts,
        "final_threshold": running_threshold,
    }


def query_pipeline(
    queries: np.ndarray,
    base: np.ndarray,
    num_partitions: int,
    nprobe: int,
    num_blocks: int,
    top_k: int,
    warmup_queries: int,
    warmup_vectors: int,
):
    partitions, kmeans = build_vector_partitions(base, num_partitions)
    dim_ranges = split_dimension_ranges(base.shape[1], num_blocks)

    global_threshold = prewarm_heap(
        queries=queries,
        base=base,
        top_k=top_k,
        warmup_queries=warmup_queries,
        warmup_vectors=warmup_vectors,
    )

    latencies = []
    all_results = []

    routing_loads = np.zeros(num_partitions, dtype=np.int64)
    partition_compute_loads = np.zeros(num_partitions, dtype=np.int64)
    block_compute_loads = np.zeros(num_blocks, dtype=np.int64)
    block_pruned_counts = np.zeros(num_blocks, dtype=np.int64)

    for query_id, query in enumerate(queries):
        start_time = time.perf_counter()

        routed_partitions = route_query_to_partitions(query, kmeans, nprobe)
        candidates = []
        running_threshold = global_threshold

        grouped_queries = {}
        for partition_id in routed_partitions:
            routing_loads[int(partition_id)] += 1
            grouped_queries[int(partition_id)] = [(query_id, query)]

        for partition_id in routed_partitions:
            partition = partitions[int(partition_id)]
            batch = grouped_queries[int(partition_id)]

            out = vector_pipeline(
                queries_in_partition=batch,
                partition=partition,
                dim_ranges=dim_ranges,
                top_k=top_k,
                global_threshold=running_threshold,
            )

            local_results = out["partition_results"].get(query_id, [])
            candidates.extend(local_results)

            partition_compute_loads[int(partition_id)] += int(out["block_loads"].sum())
            block_compute_loads += out["block_loads"]
            block_pruned_counts += out["pruned_counts"]

            merged_so_far = merge_topk(candidates, top_k)
            if len(merged_so_far) >= top_k:
                running_threshold = max(running_threshold, float(merged_so_far[-1][0]))
            running_threshold = max(running_threshold, out["final_threshold"])

        final_results = merge_topk(candidates, top_k)
        elapsed = time.perf_counter() - start_time

        latencies.append(elapsed)
        all_results.append(final_results)
        global_threshold = max(global_threshold, running_threshold)

    return {
        "latencies": np.array(latencies),
        "results": all_results,
        "routing_loads": routing_loads,
        "partition_compute_loads": partition_compute_loads,
        "block_compute_loads": block_compute_loads,
        "block_pruned_counts": block_pruned_counts,
        "initial_threshold": global_threshold,
    }


def summarize(results: dict):
    lat = results["latencies"]
    total_time = float(lat.sum())
    qps = len(lat) / total_time if total_time > 0 else 0.0

    routing = results["routing_loads"].astype(np.float64)
    part_compute = results["partition_compute_loads"].astype(np.float64)
    block_compute = results["block_compute_loads"].astype(np.float64)
    pruned = results["block_pruned_counts"].astype(np.int64)

    routing_imbalance = routing.max() / routing.mean() if routing.mean() > 0 else 0.0
    partition_imbalance = part_compute.max() / part_compute.mean() if part_compute.mean() > 0 else 0.0
    block_imbalance = block_compute.max() / block_compute.mean() if block_compute.mean() > 0 else 0.0

    print("\nHarmony Pipeline Results")
    print(f"Queries: {len(lat)}")
    print(f"Mean latency (s): {lat.mean():.6f}")
    print(f"P95 latency (s): {np.percentile(lat, 95):.6f}")
    print(f"P99 latency (s): {np.percentile(lat, 99):.6f}")
    print(f"Throughput (QPS): {qps:.2f}")
    print(f"Routing imbalance ratio: {routing_imbalance:.4f}")
    print(f"Partition compute imbalance ratio: {partition_imbalance:.4f}")
    print(f"Block compute imbalance ratio: {block_imbalance:.4f}")
    print(f"Routing loads: {routing.astype(int).tolist()}")
    print(f"Partition compute loads: {part_compute.astype(int).tolist()}")
    print(f"Block compute loads: {block_compute.astype(int).tolist()}")
    print(f"Pruned candidates by block: {pruned.tolist()}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-set", choices=["uniform", "skewed"], default="uniform")
    parser.add_argument("--num-partitions", type=int, default=4)
    parser.add_argument("--nprobe", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--warmup-queries", type=int, default=32)
    parser.add_argument("--warmup-vectors", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()
    base, queries = load_data(args.query_set)

    results = query_pipeline(
        queries=queries,
        base=base,
        num_partitions=args.num_partitions,
        nprobe=args.nprobe,
        num_blocks=args.num_blocks,
        top_k=args.top_k,
        warmup_queries=args.warmup_queries,
        warmup_vectors=args.warmup_vectors,
    )

    summarize(results)


if __name__ == "__main__":
    main()