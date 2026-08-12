import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from hybrid_search import reciprocal_rank_fusion
from run_chunking_retrieval_ablation import (
    ChunkingAblationRetriever,
    evidence_rank,
)

CONFIG = "500_50"
DEPTHS = [10, 20, 40]
TOP_K = 5

MANIFEST_PATH = Path(
    "outputs/chunking_ablation/gold_remap_reviewed.json"
)

OUTPUT_PATH = Path(
    "outputs/reranker_ablation/"
    "candidate_depth_ablation.json"
)


def evaluate_result(
    query_id,
    mode,
    candidate_k,
    retrieved,
    gold,
):
    retrieved_ids = [
        row["chunk_id"]
        for row in retrieved
    ]

    completion_rank = evidence_rank(
        retrieved_ids,
        gold,
    )

    return {
        "query_id": query_id,
        "candidate_k": candidate_k,
        "mode": mode,
        "retrieved_chunk_ids": retrieved_ids,
        "completion_rank": completion_rank,
        "hit_at_5": completion_rank is not None,
        "reciprocal_rank": (
            1.0 / completion_rank
            if completion_rank is not None
            else 0.0
        ),
    }


def summarize(rows):
    summary = {}

    for depth in DEPTHS:
        summary[str(depth)] = {}

        for mode in ["hybrid", "hybrid_rerank"]:
            subset = [
                row
                for row in rows
                if row["candidate_k"] == depth
                and row["mode"] == mode
            ]

            hits = sum(
                row["hit_at_5"]
                for row in subset
            )

            mrr = (
                sum(
                    row["reciprocal_rank"]
                    for row in subset
                )
                / len(subset)
            )

            summary[str(depth)][mode] = {
                "queries": len(subset),
                "hits": hits,
                "hit_rate": hits / len(subset),
                "mrr": mrr,
            }

        hybrid = {
            row["query_id"]: row
            for row in rows
            if row["candidate_k"] == depth
            and row["mode"] == "hybrid"
        }

        rerank = {
            row["query_id"]: row
            for row in rows
            if row["candidate_k"] == depth
            and row["mode"] == "hybrid_rerank"
        }

        recovered = []
        lost = []
        rank_improved = []
        rank_degraded = []

        for qid in hybrid:
            h = hybrid[qid]["completion_rank"]
            r = rerank[qid]["completion_rank"]

            if h is None and r is not None:
                recovered.append({
                    "query_id": qid,
                    "rerank_rank": r,
                })

            elif h is not None and r is None:
                lost.append({
                    "query_id": qid,
                    "hybrid_rank": h,
                })

            elif h is not None and r is not None:
                if r < h:
                    rank_improved.append({
                        "query_id": qid,
                        "hybrid_rank": h,
                        "rerank_rank": r,
                    })
                elif r > h:
                    rank_degraded.append({
                        "query_id": qid,
                        "hybrid_rank": h,
                        "rerank_rank": r,
                    })

        summary[str(depth)]["reranker_changes"] = {
            "recovered": recovered,
            "lost": lost,
            "rank_improved": rank_improved,
            "rank_degraded": rank_degraded,
        }

    return summary


def main():
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    max_depth = max(DEPTHS)

    # Load model/index once.
    retriever = ChunkingAblationRetriever(
        CONFIG,
        candidate_k=max_depth,
    )

    rows = []

    for i, record in enumerate(
        manifest["results"],
        1,
    ):
        print(
            f"[{i}/"
            f"{len(manifest['results'])}] "
            f"{record['query_id']}"
        )

        where = {
            "company_id":
            record["company_id"]
        }

        # Retrieve each base ranking only once
        # at the maximum depth.
        semantic_full = retriever.semantic_search(
            record["query"],
            n_results=max_depth,
            where=where,
        )

        bm25_full = retriever.bm25_search(
            record["query"],
            n_results=max_depth,
            where=where,
        )

        gold = record["configs"][CONFIG]

        for depth in DEPTHS:
            semantic = semantic_full[:depth]
            bm25 = bm25_full[:depth]

            fused = reciprocal_rank_fusion(
                semantic,
                bm25,
            )

            hybrid_top5 = fused[:TOP_K]

            reranked_top5 = (
                retriever.reranker.rerank(
                    record["query"],
                    fused,
                    top_k=TOP_K,
                )
            )

            rows.append(
                evaluate_result(
                    record["query_id"],
                    "hybrid",
                    depth,
                    hybrid_top5,
                    gold,
                )
            )

            rows.append(
                evaluate_result(
                    record["query_id"],
                    "hybrid_rerank",
                    depth,
                    reranked_top5,
                    gold,
                )
            )

    summary = summarize(rows)

    output = {
        "config": CONFIG,
        "depths": DEPTHS,
        "top_k": TOP_K,
        "query_count": len(
            manifest["results"]
        ),
        "summary": summary,
        "results": rows,
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

    print("\n" + "=" * 90)
    print("RERANKER CANDIDATE-DEPTH ABLATION")
    print("=" * 90)

    for depth in DEPTHS:
        block = summary[str(depth)]

        print(f"\ncandidate_k={depth}")

        for mode in [
            "hybrid",
            "hybrid_rerank",
        ]:
            m = block[mode]

            print(
                f"  {mode:15s} "
                f"Hit@5="
                f"{m['hit_rate']:.3f} "
                f"({m['hits']}/"
                f"{m['queries']}) "
                f"MRR={m['mrr']:.3f}"
            )

        changes = block[
            "reranker_changes"
        ]

        print(
            "  recovered:",
            len(changes["recovered"]),
            [
                x["query_id"]
                for x in changes["recovered"]
            ],
        )

        print(
            "  lost     :",
            len(changes["lost"]),
            [
                x["query_id"]
                for x in changes["lost"]
            ],
        )

        print(
            "  rank ↑   :",
            len(changes["rank_improved"]),
        )

        print(
            "  rank ↓   :",
            len(changes["rank_degraded"]),
        )

    # Regression guard against yesterday's
    # independently run candidate_k=20 result.
    k20 = summary["20"]

    assert (
        k20["hybrid"]["hits"] == 25
    ), (
        "candidate_k=20 hybrid no longer "
        "matches prior result (25/27)."
    )

    assert (
        k20["hybrid_rerank"]["hits"] == 26
    ), (
        "candidate_k=20 rerank no longer "
        "matches prior result (26/27)."
    )

    print(
        "\nPASS: candidate_k=20 reproduces "
        "the previous 25/27 -> 26/27 result."
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
