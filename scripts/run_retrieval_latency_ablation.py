import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from hybrid_search import reciprocal_rank_fusion
from run_chunking_retrieval_ablation import (
    ChunkingAblationRetriever,
    openai_client,
)

CONFIG = "500_50"
DEPTHS = [10, 20, 40]
REPEATS = 3

MANIFEST_PATH = Path(
    "outputs/chunking_ablation/gold_remap_reviewed.json"
)

OUTPUT_PATH = Path(
    "outputs/reranker_ablation/retrieval_latency_ablation.json"
)


def ms(seconds):
    return seconds * 1000.0


def percentile(values, p):
    values = sorted(values)

    if not values:
        return 0.0

    if len(values) == 1:
        return values[0]

    pos = (len(values) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    fraction = pos - lower

    return (
        values[lower] * (1 - fraction)
        + values[upper] * fraction
    )


def stats(values):
    return {
        "n": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def semantic_from_embedding(
    retriever,
    embedding,
    *,
    n_results,
    where,
):
    raw = retriever.collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        where=where,
    )

    results = []

    for i, chunk_id in enumerate(raw["ids"][0]):
        results.append({
            "rank": i + 1,
            "chunk_id": chunk_id,
            "text": raw["documents"][0][i],
            "metadata": raw["metadatas"][0][i],
            "distance": raw["distances"][0][i],
        })

    return results


def main():
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )

    queries = manifest["results"]

    retriever = ChunkingAblationRetriever(
        CONFIG,
        candidate_k=max(DEPTHS),
    )

    print("\nCaching query embeddings and measuring API latency...")

    cached_embeddings = {}
    embedding_latencies = []

    for i, record in enumerate(queries, 1):
        query = record["query"]

        start = time.perf_counter()

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )

        elapsed = ms(time.perf_counter() - start)

        cached_embeddings[record["query_id"]] = (
            response.data[0].embedding
        )

        embedding_latencies.append(elapsed)

        print(
            f"[embed {i:02d}/{len(queries)}] "
            f"{record['query_id']}: "
            f"{elapsed:.1f} ms"
        )

    # Warm up local systems/model.
    warm = queries[0]
    warm_where = {
        "company_id": warm["company_id"]
    }
    warm_embedding = cached_embeddings[
        warm["query_id"]
    ]

    retriever.bm25_search(
        warm["query"],
        n_results=5,
        where=warm_where,
    )

    semantic_from_embedding(
        retriever,
        warm_embedding,
        n_results=5,
        where=warm_where,
    )

    warm_sem = semantic_from_embedding(
        retriever,
        warm_embedding,
        n_results=10,
        where=warm_where,
    )

    warm_bm = retriever.bm25_search(
        warm["query"],
        n_results=10,
        where=warm_where,
    )

    warm_fused = reciprocal_rank_fusion(
        warm_sem,
        warm_bm,
    )

    retriever.reranker.rerank(
        warm["query"],
        warm_fused,
        top_k=5,
    )

    standalone = {
        "bm25": [],
        "semantic_local": [],
    }

    depth_timings = {
        str(depth): {
            "hybrid_local": [],
            "reranker_incremental": [],
            "hybrid_rerank_local": [],
            "candidate_pool_sizes": [],
        }
        for depth in DEPTHS
    }

    print("\nRunning local latency benchmark...")

    for repeat in range(REPEATS):
        print(f"\nRepeat {repeat + 1}/{REPEATS}")

        for i, record in enumerate(queries, 1):
            query = record["query"]
            where = {
                "company_id":
                record["company_id"]
            }
            embedding = cached_embeddings[
                record["query_id"]
            ]

            # Standalone BM25 Top-5.
            start = time.perf_counter()
            retriever.bm25_search(
                query,
                n_results=5,
                where=where,
            )
            standalone["bm25"].append(
                ms(time.perf_counter() - start)
            )

            # Standalone semantic Top-5, excluding API embedding.
            start = time.perf_counter()
            semantic_from_embedding(
                retriever,
                embedding,
                n_results=5,
                where=where,
            )
            standalone["semantic_local"].append(
                ms(time.perf_counter() - start)
            )

            for depth in DEPTHS:
                # Full local hybrid:
                # semantic vector lookup + BM25 + RRF.
                start = time.perf_counter()

                semantic = semantic_from_embedding(
                    retriever,
                    embedding,
                    n_results=depth,
                    where=where,
                )

                bm25 = retriever.bm25_search(
                    query,
                    n_results=depth,
                    where=where,
                )

                fused = reciprocal_rank_fusion(
                    semantic,
                    bm25,
                )

                hybrid_ms = ms(
                    time.perf_counter() - start
                )

                # Incremental reranker latency only.
                start = time.perf_counter()

                retriever.reranker.rerank(
                    query,
                    fused,
                    top_k=5,
                )

                rerank_ms = ms(
                    time.perf_counter() - start
                )

                block = depth_timings[str(depth)]

                block["hybrid_local"].append(
                    hybrid_ms
                )
                block["reranker_incremental"].append(
                    rerank_ms
                )
                block["hybrid_rerank_local"].append(
                    hybrid_ms + rerank_ms
                )
                block["candidate_pool_sizes"].append(
                    len(fused)
                )

            print(
                f"  [{i:02d}/{len(queries)}] "
                f"{record['query_id']}"
            )

    embedding_stats = stats(
        embedding_latencies
    )

    summary = {
        "embedding_api": embedding_stats,
        "standalone": {
            name: stats(values)
            for name, values in standalone.items()
        },
        "depths": {},
    }

    embedding_median = (
        embedding_stats["median_ms"]
    )

    for depth in DEPTHS:
        raw = depth_timings[str(depth)]

        hybrid_stats = stats(
            raw["hybrid_local"]
        )
        rerank_stats = stats(
            raw["reranker_incremental"]
        )
        final_stats = stats(
            raw["hybrid_rerank_local"]
        )

        summary["depths"][str(depth)] = {
            "hybrid_local": hybrid_stats,
            "reranker_incremental": rerank_stats,
            "hybrid_rerank_local": final_stats,
            "candidate_pool_size": stats(
                raw["candidate_pool_sizes"]
            ),
            "estimated_median_end_to_end_ms": {
                "hybrid": (
                    embedding_median
                    + hybrid_stats["median_ms"]
                ),
                "hybrid_rerank": (
                    embedding_median
                    + final_stats["median_ms"]
                ),
            },
        }

    output = {
        "config": CONFIG,
        "candidate_depths": DEPTHS,
        "repeats": REPEATS,
        "query_count": len(queries),
        "methodology": {
            "embedding": (
                "OpenAI query embedding measured once "
                "per benchmark query."
            ),
            "local": (
                "Local retrieval and reranking measured "
                "with cached query embeddings."
            ),
            "end_to_end_estimate": (
                "Median API embedding latency plus "
                "median local pipeline latency."
            ),
            "warmup": (
                "One unrecorded warm-up pass before "
                "local timing."
            ),
        },
        "summary": summary,
        "raw": {
            "embedding_api_ms":
                embedding_latencies,
            "standalone":
                standalone,
            "depths":
                depth_timings,
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 95)
    print("RETRIEVAL LATENCY SUMMARY")
    print("=" * 95)

    e = summary["embedding_api"]

    print(
        "\nOpenAI query embedding"
        f"\n  median={e['median_ms']:.1f} ms"
        f" | mean={e['mean_ms']:.1f} ms"
        f" | p95={e['p95_ms']:.1f} ms"
    )

    for name, s in summary[
        "standalone"
    ].items():
        print(
            f"\n{name}"
            f"\n  median={s['median_ms']:.1f} ms"
            f" | mean={s['mean_ms']:.1f} ms"
            f" | p95={s['p95_ms']:.1f} ms"
        )

    for depth in DEPTHS:
        block = summary["depths"][str(depth)]

        print(f"\ncandidate_k={depth}")

        for name in [
            "hybrid_local",
            "reranker_incremental",
            "hybrid_rerank_local",
        ]:
            s = block[name]

            print(
                f"  {name:24s}"
                f" median={s['median_ms']:.1f} ms"
                f" | mean={s['mean_ms']:.1f} ms"
                f" | p95={s['p95_ms']:.1f} ms"
            )

        pool = block["candidate_pool_size"]

        print(
            "  candidate_pool_size      "
            f"median={pool['median_ms']:.1f}"
            f" | mean={pool['mean_ms']:.1f}"
        )

        est = block[
            "estimated_median_end_to_end_ms"
        ]

        print(
            "  estimated E2E hybrid      "
            f"{est['hybrid']:.1f} ms"
        )

        print(
            "  estimated E2E + rerank    "
            f"{est['hybrid_rerank']:.1f} ms"
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
