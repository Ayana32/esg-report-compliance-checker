import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_verifier import SLOT_RULES

INPUT_PATH = Path(
    "outputs/verifier_benchmark/"
    "candidate_packets_unreviewed.json"
)

OUTPUT_DIR = Path(
    "outputs/verifier_benchmark/review_batches"
)

BATCH_SIZE = 7


def main():
    data = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    cases = data["cases"]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for batch_start in range(
        0,
        len(cases),
        BATCH_SIZE,
    ):
        batch = cases[
            batch_start:
            batch_start + BATCH_SIZE
        ]

        batch_no = (
            batch_start // BATCH_SIZE
        ) + 1

        lines = []

        lines.append(
            "=" * 110
        )
        lines.append(
            f"VERIFIER GOLD REVIEW — BATCH {batch_no}"
        )
        lines.append(
            "=" * 110
        )
        lines.append("")
        lines.append(
            "IMPORTANT:"
        )
        lines.append(
            "- Assign gold_status ONLY from the frozen evidence packet below."
        )
        lines.append(
            "- Use the current SLOT_RULES exactly."
        )
        lines.append(
            "- Do not use prior verifier predictions."
        )
        lines.append(
            "- reviewed retrieval gold is annotation aid only."
        )
        lines.append(
            "- Supporting chunks must directly justify the selected status."
        )
        lines.append("")

        for local_idx, case in enumerate(
            batch,
            1,
        ):
            global_idx = (
                batch_start + local_idx
            )

            element = case["element"]

            lines.append("")
            lines.append(
                "#" * 110
            )
            lines.append(
                f"CASE {global_idx:02d} / "
                f"{len(cases)}"
            )
            lines.append(
                "#" * 110
            )

            lines.append(
                f"case_id: {case['case_id']}"
            )
            lines.append(
                f"company: {case['company_id']}"
            )
            lines.append(
                f"slot: {case['slot_id']}"
            )
            lines.append(
                f"element: {element}"
            )
            lines.append("")

            lines.append(
                "CURRENT SLOT RULE"
            )
            lines.append(
                "-" * 110
            )
            lines.append(
                SLOT_RULES[element].strip()
            )
            lines.append("")

            reviewed = case[
                "reviewed_retrieval_gold"
            ]

            gold_ids = set(
                reviewed["gold_chunk_ids"]
            )

            complementary_ids = {
                cid
                for group in reviewed[
                    "complementary_groups"
                ]
                for cid in group
            }

            lines.append(
                "FROZEN TOP-5 EVIDENCE"
            )
            lines.append(
                "-" * 110
            )

            for chunk in case[
                "evidence_chunks"
            ]:
                cid = chunk["chunk_id"]

                markers = []

                if cid in gold_ids:
                    markers.append(
                        "REVIEWED_RETRIEVAL_GOLD"
                    )

                if cid in complementary_ids:
                    markers.append(
                        "REVIEWED_COMPLEMENTARY"
                    )

                marker_text = (
                    " | " + ", ".join(markers)
                    if markers
                    else ""
                )

                page = (
                    chunk.get(
                        "metadata",
                        {}
                    ).get(
                        "page_num",
                        "?"
                    )
                )

                lines.append("")
                lines.append(
                    f"[Chunk {chunk['rank']}] "
                    f"page={page}"
                    f"{marker_text}"
                )
                lines.append(
                    f"chunk_id: {cid}"
                )
                lines.append(
                    chunk["text"].strip()
                )

            lines.append("")
            lines.append(
                "HUMAN REVIEW"
            )
            lines.append(
                "-" * 110
            )
            lines.append(
                "gold_status: "
                "[covered / partial / missing]"
            )
            lines.append(
                "gold_supporting_chunk_ids: "
                "[...]"
            )
            lines.append(
                "review_rationale: "
            )
            lines.append("")

        output_path = (
            OUTPUT_DIR
            / f"batch_{batch_no}.txt"
        )

        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        print(
            f"Saved {output_path} "
            f"({len(batch)} cases)"
        )


if __name__ == "__main__":
    main()
