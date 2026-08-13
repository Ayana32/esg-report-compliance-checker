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

OUTPUT_PATH = Path(
    "outputs/verifier_benchmark/"
    "challenge_packets_unreviewed.json"
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

# Selection is frozen BEFORE verifier prediction.
# Existing document-level labels are used only
# to select likely challenge cases.
CHALLENGE_CASES = [
    # Pre-existing PARTIAL candidates
    ("IBK", "slot_b", "partial"),
    ("Kepco", "slot_d", "partial"),
    ("Heathrow", "slot_e", "partial"),
    ("Siemens", "slot_f", "partial"),
    ("Samsung", "slot_f", "partial"),
    ("Kepco", "slot_g", "partial"),

    # Pre-existing MISSING candidates
    ("Incheon", "slot_b", "missing"),
    ("IBK", "slot_c", "missing"),
    ("Kepco", "slot_e", "missing"),
    ("Kepco", "slot_f", "missing"),
    ("Incheon", "slot_d", "missing"),
    ("Incheon", "slot_e", "missing"),
]


def main():
    retriever = ChunkingAblationRetriever(
        CONFIG,
        candidate_k=CANDIDATE_K,
    )

    packets = []

    for i, (
        company_id,
        slot_id,
        selection_label,
    ) in enumerate(CHALLENGE_CASES, 1):

        element = SLOT_TO_ELEMENT[slot_id]

        # Element-focused query.
        # The historical document-level label
        # does NOT determine packet-level gold.
        query = (
            f"{element} "
            f"Scope 1 greenhouse gas emissions"
        )

        case_id = (
            f"{company_id}_305-1_"
            f"{slot_id}_challenge"
        )

        print(
            f"[{i:02d}/{len(CHALLENGE_CASES)}] "
            f"{case_id}"
        )

        evidence = retriever.search(
            query,
            mode="hybrid_rerank",
            n_results=TOP_K,
            where={
                "company_id": company_id
            },
        )

        packets.append({
            "case_id": case_id,
            "company_id": company_id,
            "requirement_id": "305-1",
            "requirement_name": (
                "Direct (Scope 1) GHG emissions"
            ),
            "slot_id": slot_id,
            "element": element,
            "retrieval_query": query,
            "retrieval_config": {
                "chunking": CONFIG,
                "candidate_k": CANDIDATE_K,
                "mode": "hybrid_rerank",
                "top_k": TOP_K,
            },

            # Selection provenance only.
            # MUST NOT be copied into gold_status.
            "selection_source": {
                "preexisting_document_label":
                    selection_label,
                "used_for":
                    "challenge-case selection only",
            },

            "evidence_chunks": [
                {
                    "rank": rank,
                    "chunk_id": row["chunk_id"],
                    "text": row["text"],
                    "metadata": row.get(
                        "metadata",
                        {},
                    ),
                    "reranker_score": row.get(
                        "reranker_score"
                    ),
                }
                for rank, row in enumerate(
                    evidence,
                    1,
                )
            ],

            "gold_status": None,
            "gold_supporting_chunk_ids": [],
            "review_rationale": "",
            "review_status": "UNREVIEWED",
        })

    output = {
        "benchmark": (
            "GRI 305-1 verifier challenge packets"
        ),
        "case_count": len(packets),
        "selection_policy": (
            "Cases were selected before verifier "
            "prediction using only pre-existing "
            "human document-level annotations. "
            "Packet-level gold must be assigned "
            "independently from frozen Top-5 evidence."
        ),
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

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
