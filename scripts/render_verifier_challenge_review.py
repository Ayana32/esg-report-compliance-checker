import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_verifier import SLOT_RULES

INPUT = Path(
    "outputs/verifier_benchmark/"
    "challenge_packets_unreviewed.json"
)

OUTPUT = Path(
    "outputs/verifier_benchmark/"
    "challenge_review.txt"
)


def main():
    data = json.loads(
        INPUT.read_text(encoding="utf-8")
    )

    lines = []

    lines += [
        "=" * 110,
        "VERIFIER CHALLENGE GOLD REVIEW",
        "=" * 110,
        "",
        "IMPORTANT:",
        "- Use frozen evidence only.",
        "- Apply current SLOT_RULES.",
        "- Ignore the pre-existing document label when assigning packet gold.",
        "- Do not inspect verifier predictions.",
        "",
    ]

    for i, case in enumerate(
        data["cases"],
        1,
    ):
        element = case["element"]

        lines += [
            "",
            "#" * 110,
            f"CASE {i:02d} / {len(data['cases'])}",
            "#" * 110,
            f"case_id: {case['case_id']}",
            f"company: {case['company_id']}",
            f"slot: {case['slot_id']}",
            f"element: {element}",
            (
                "selection label "
                "(DO NOT COPY AS GOLD): "
                f"{case['selection_source']['preexisting_document_label']}"
            ),
            "",
            "CURRENT SLOT RULE",
            "-" * 110,
            SLOT_RULES[element].strip(),
            "",
            "FROZEN TOP-5 EVIDENCE",
            "-" * 110,
        ]

        for chunk in case[
            "evidence_chunks"
        ]:
            page = (
                chunk.get(
                    "metadata",
                    {}
                ).get(
                    "page_num",
                    "?"
                )
            )

            lines += [
                "",
                (
                    f"[Chunk {chunk['rank']}] "
                    f"page={page}"
                ),
                (
                    "chunk_id: "
                    f"{chunk['chunk_id']}"
                ),
                chunk["text"].strip(),
            ]

        lines += [
            "",
            "HUMAN REVIEW",
            "-" * 110,
            (
                "gold_status: "
                "[covered / partial / missing]"
            ),
            (
                "gold_supporting_chunk_ids: "
                "[...]"
            ),
            "review_rationale:",
            "",
        ]

    OUTPUT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
