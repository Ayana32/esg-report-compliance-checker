"""Evaluate retrieval modes against gold evidence chunks."""

import argparse
import json
from pathlib import Path
from typing import Any

from retrieval_modes import AblationRetriever


DEFAULT_GOLD_PATH = Path(
    "data/evaluation/retrieval_gold.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("outputs/retrieval_ablation")


def load_gold_queries(path: Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records."""

    records = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line.strip():
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON on line {line_number} of {path}"
            ) from exc

    return records


def evaluate_query(
    retriever: AblationRetriever,
    record: dict[str, Any],
    *,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    """Evaluate one gold query for one retrieval mode."""

    results = retriever.search(
        record["query"],
        mode=mode,
        n_results=top_k,
        where={"company_id": record["company_id"]},
    )

    retrieved_chunk_ids = [
        result["chunk_id"] for result in results
    ]
    gold_chunk_ids = set(record["gold_chunk_ids"])

    relevant_ranks = [
        rank
        for rank, chunk_id in enumerate(
            retrieved_chunk_ids,
            1,
        )
        if chunk_id in gold_chunk_ids
    ]

    first_relevant_rank = (
        min(relevant_ranks)
        if relevant_ranks
        else None
    )

    return {
        "query_id": record["query_id"],
        "company_id": record["company_id"],
        "requirement_id": record["requirement_id"],
        "slot_id": record["slot_id"],
        "mode": mode,
        "top_k": top_k,
        "gold_chunk_ids": sorted(gold_chunk_ids),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "relevant_ranks": relevant_ranks,
        "first_relevant_rank": first_relevant_rank,
        "hit_at_k": first_relevant_rank is not None,
        "reciprocal_rank": (
            1 / first_relevant_rank
            if first_relevant_rank
            else 0.0
        ),
    }


def summarize(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate hit rate and MRR by retrieval mode."""

    summary = {}

    modes = sorted(
        {result["mode"] for result in results}
    )

    for mode in modes:
        mode_results = [
            result
            for result in results
            if result["mode"] == mode
        ]

        total = len(mode_results)
        hits = sum(
            1 for result in mode_results
            if result["hit_at_k"]
        )
        reciprocal_rank_sum = sum(
            result["reciprocal_rank"]
            for result in mode_results
        )

        summary[mode] = {
            "total_queries": total,
            "hits": hits,
            "hit_rate": hits / total if total else 0.0,
            "mrr": (
                reciprocal_rank_sum / total
                if total
                else 0.0
            ),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["bm25", "semantic", "hybrid"],
        choices=["bm25", "semantic", "hybrid"],
    )

    args = parser.parse_args()

    gold_records = load_gold_queries(args.gold)

    if not gold_records:
        raise ValueError(
            f"No gold queries found in {args.gold}"
        )

    retriever = AblationRetriever("reports")
    evaluation_results = []

    for record in gold_records:
        for mode in args.modes:
            print(
                f"Evaluating {record['query_id']} "
                f"with mode={mode}"
            )

            result = evaluate_query(
                retriever,
                record,
                mode=mode,
                top_k=args.top_k,
            )
            evaluation_results.append(result)

    output = {
        "top_k": args.top_k,
        "query_count": len(gold_records),
        "modes": args.modes,
        "summary": summarize(evaluation_results),
        "results": evaluation_results,
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output_dir
        / f"retrieval_ablation_top{args.top_k}.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSummary")

    for mode, metrics in output["summary"].items():
        print(
            f"{mode}: "
            f"Hit@{args.top_k}="
            f"{metrics['hit_rate']:.3f}, "
            f"MRR={metrics['mrr']:.3f}"
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
