import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_chunking_retrieval_ablation import (
    ChunkingAblationRetriever,
)

CONFIG = "500_50"
CANDIDATE_K = 10
TOP_K = 5

MANIFEST_PATH = Path(
    "outputs/chunking_ablation/"
    "gold_remap_reviewed.json"
)

OUTPUT_PATH = Path(
    "outputs/verifier_benchmark/"
    "candidate_packets_unreviewed.json"
)


SLOT_TO_ELEMENT = {
    "slot_a": (
        "Total Scope 1 emissions in tCO2e"
    ),
    "slot_b": (
        "Gases included in calculation "
        "(CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)"
    ),
    "slot_c": (
        "Biogenic CO2 emissions "
        "(reported separately)"
    ),
    "slot_d": (
        "Base year for calculation"
    ),
    "slot_e": (
        "Emission factors and GWP source "
        "(e.g. IPCC AR5/AR6)"
    ),
    "slot_f": (
        "Consolidation approach "
        "(equity share, financial control, "
        "operational control)"
    ),
    "slot_g": (
        "Standards and methodologies used "
        "(e.g. GHG Protocol, "
        "third-party verification)"
    ),
}


def main():
    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    retriever = ChunkingAblationRetriever(
        CONFIG,
        candidate_k=CANDIDATE_K,
    )

    packets = []

    for i, record in enumerate(
        manifest["results"],
        1,
    ):
        query_id = record["query_id"]
        company_id = record["company_id"]
        slot_id = record["slot_id"]

        print(
            f"[{i:02d}/"
            f"{len(manifest['results'])}] "
            f"{query_id}"
        )

        evidence = retriever.search(
            record["query"],
            mode="hybrid_rerank",
            n_results=TOP_K,
            where={
                "company_id": company_id
            },
        )

        packet = {
            "case_id": query_id,
            "company_id": company_id,
            "requirement_id": "305-1",
            "requirement_name": (
                "Direct (Scope 1) GHG emissions"
            ),
            "slot_id": slot_id,
            "element": SLOT_TO_ELEMENT[
                slot_id
            ],
            "retrieval_query": record["query"],
            "retrieval_config": {
                "chunking": CONFIG,
                "candidate_k":
                    CANDIDATE_K,
                "mode": "hybrid_rerank",
                "top_k": TOP_K,
            },

            # Annotation aid only.
            # Do NOT use this automatically
            # as verifier gold status.
            "reviewed_retrieval_gold": {
                "gold_chunk_ids":
                    record["configs"][
                        CONFIG
                    ]["gold_chunk_ids"],
                "complementary_groups":
                    record["configs"][
                        CONFIG
                    ][
                        "complementary_groups"
                    ],
            },

            "evidence_chunks": [
                {
                    "rank": rank,
                    "chunk_id":
                        row["chunk_id"],
                    "text":
                        row["text"],
                    "metadata":
                        row.get(
                            "metadata",
                            {},
                        ),
                    "reranker_score":
                        row.get(
                            "reranker_score"
                        ),
                }
                for rank, row in enumerate(
                    evidence,
                    1,
                )
            ],

            # Human annotation fields.
            # Must be filled BEFORE running
            # verifier evaluation.
            "gold_status": None,
            "gold_supporting_chunk_ids": [],
            "review_rationale": "",
            "review_status": "UNREVIEWED",
        }

        packets.append(packet)

    output = {
        "benchmark": (
            "GRI 305-1 verifier-only "
            "candidate packets"
        ),
        "query_count": len(packets),
        "source_query_set": (
            "pre-existing 27-query "
            "retrieval benchmark"
        ),
        "annotation_policy": {
            "gold_status": (
                "Assigned manually from the "
                "frozen Top-5 evidence packet "
                "using the current "
                "llm_verifier.SLOT_RULES."
            ),
            "retrieval_gold": (
                "May be used only as an "
                "annotation aid for locating "
                "reviewed evidence. It does "
                "not determine verifier "
                "gold status."
            ),
            "prediction_blind": (
                "Verifier predictions must "
                "not be generated or "
                "inspected until gold "
                "annotation is frozen."
            ),
        },
        "cases": packets,
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

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    main()
