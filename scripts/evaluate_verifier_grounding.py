import json
from pathlib import Path

PATH = Path(
    "outputs/verifier_benchmark/"
    "verifier_predictions.json"
)

data = json.loads(
    PATH.read_text(encoding="utf-8")
)

rows = data["predictions"]

assert len(rows) == 39

non_missing_pred = 0
non_missing_with_evidence = 0
missing_pred = 0
missing_with_empty_evidence = 0

groundable_correct = 0
grounding_overlap = 0

invalid_selected_ids = []
non_missing_no_evidence = []
missing_with_evidence = []
correct_no_overlap = []

for row in rows:
    pred_status = row["prediction"]["status"]
    gold_status = row["gold"]["status"]

    selected_ids = set(
        row["prediction"].get(
            "selected_chunk_ids",
            []
        )
    )

    gold_ids = set(
        row["gold"].get(
            "supporting_chunk_ids",
            []
        )
    )

    # Reconstruct frozen packet IDs from
    # the corresponding gold source later
    # using selected IDs already normalized
    # by LLMVerifier. Here we additionally
    # verify no blank IDs survive.
    if "" in selected_ids:
        invalid_selected_ids.append(
            row["case_id"]
        )

    if pred_status == "missing":
        missing_pred += 1

        if not selected_ids:
            missing_with_empty_evidence += 1
        else:
            missing_with_evidence.append(
                row["case_id"]
            )

    else:
        non_missing_pred += 1

        if selected_ids:
            non_missing_with_evidence += 1
        else:
            non_missing_no_evidence.append(
                row["case_id"]
            )

    # Grounding correctness is most
    # meaningful when classification is
    # also correct and gold has support.
    if (
        pred_status == gold_status
        and gold_status in {
            "covered",
            "partial",
        }
    ):
        groundable_correct += 1

        if selected_ids & gold_ids:
            grounding_overlap += 1
        else:
            correct_no_overlap.append({
                "case_id": row["case_id"],
                "status": gold_status,
                "selected": sorted(
                    selected_ids
                ),
                "gold_support": sorted(
                    gold_ids
                ),
            })

print("=" * 80)
print("VERIFIER GROUNDING METRICS")
print("=" * 80)

print(
    "Predicted covered/partial:",
    non_missing_pred,
)

print(
    "Non-missing predictions with evidence:",
    f"{non_missing_with_evidence}/"
    f"{non_missing_pred}",
)

if non_missing_pred:
    print(
        "Evidence-present rate:",
        f"{non_missing_with_evidence / non_missing_pred:.3f}",
    )

print()

print(
    "Predicted missing:",
    missing_pred,
)

print(
    "Missing predictions with empty evidence:",
    f"{missing_with_empty_evidence}/"
    f"{missing_pred}",
)

if missing_pred:
    print(
        "Missing-empty rate:",
        f"{missing_with_empty_evidence / missing_pred:.3f}",
    )

print()

print(
    "Correct covered/partial cases:",
    groundable_correct,
)

print(
    "Correct cases with >=1 gold-support overlap:",
    f"{grounding_overlap}/"
    f"{groundable_correct}",
)

if groundable_correct:
    print(
        "Grounding overlap rate:",
        f"{grounding_overlap / groundable_correct:.3f}",
    )

print()
print("Diagnostics")

print(
    "non-missing without evidence:",
    len(non_missing_no_evidence),
)

print(
    "missing with evidence:",
    len(missing_with_evidence),
)

print(
    "blank selected IDs:",
    len(invalid_selected_ids),
)

print(
    "correct classification but no gold overlap:",
    len(correct_no_overlap),
)

if non_missing_no_evidence:
    print()
    print("NON-MISSING WITHOUT EVIDENCE")
    for case_id in non_missing_no_evidence:
        print(" -", case_id)

if missing_with_evidence:
    print()
    print("MISSING WITH EVIDENCE")
    for case_id in missing_with_evidence:
        print(" -", case_id)

if correct_no_overlap:
    print()
    print("CORRECT STATUS BUT NO GOLD SUPPORT OVERLAP")
    for item in correct_no_overlap:
        print(
            " -",
            item["case_id"],
            "| selected=",
            item["selected"],
            "| gold=",
            item["gold_support"],
        )
