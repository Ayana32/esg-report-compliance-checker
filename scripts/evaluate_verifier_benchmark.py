import json
from collections import Counter, defaultdict
from pathlib import Path

PATH = Path(
    "outputs/verifier_benchmark/"
    "verifier_predictions.json"
)

data = json.loads(
    PATH.read_text(encoding="utf-8")
)

rows = data["predictions"]

labels = [
    "covered",
    "partial",
    "missing",
]

assert len(rows) == 39

remaining_errors = [
    r["case_id"]
    for r in rows
    if r["diagnostics"].get("call_error")
]

assert not remaining_errors, (
    f"Remaining call errors: {remaining_errors}"
)

gold = [
    r["gold"]["status"]
    for r in rows
]

pred = [
    r["prediction"]["status"]
    for r in rows
]

correct = sum(
    g == p
    for g, p in zip(gold, pred)
)

accuracy = correct / len(rows)

matrix = {
    g: Counter()
    for g in labels
}

for g, p in zip(gold, pred):
    matrix[g][p] += 1

metrics = {}

for label in labels:
    tp = sum(
        1
        for g, p in zip(gold, pred)
        if g == label and p == label
    )

    fp = sum(
        1
        for g, p in zip(gold, pred)
        if g != label and p == label
    )

    fn = sum(
        1
        for g, p in zip(gold, pred)
        if g == label and p != label
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0.0
    )

    metrics[label] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": sum(
            g == label
            for g in gold
        ),
    }

macro_f1 = sum(
    metrics[label]["f1"]
    for label in labels
) / len(labels)

false_covered = [
    r
    for r in rows
    if r["prediction"]["status"] == "covered"
    and r["gold"]["status"] != "covered"
]

severe_missing_to_covered = [
    r
    for r in rows
    if r["gold"]["status"] == "missing"
    and r["prediction"]["status"] == "covered"
]

covered_to_partial = [
    r
    for r in rows
    if r["gold"]["status"] == "covered"
    and r["prediction"]["status"] == "partial"
]

covered_to_missing = [
    r
    for r in rows
    if r["gold"]["status"] == "covered"
    and r["prediction"]["status"] == "missing"
]

partial_to_missing = [
    r
    for r in rows
    if r["gold"]["status"] == "partial"
    and r["prediction"]["status"] == "missing"
]

print("=" * 80)
print("VERIFIER CLASSIFICATION METRICS")
print("=" * 80)
print(f"Cases:     {len(rows)}")
print(f"Correct:   {correct}")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Macro-F1:  {macro_f1:.3f}")
print()

print("Per-class:")
for label in labels:
    m = metrics[label]
    print(
        f"{label:8s} "
        f"P={m['precision']:.3f} "
        f"R={m['recall']:.3f} "
        f"F1={m['f1']:.3f} "
        f"n={m['support']}"
    )

print()
print("Confusion matrix")
print(
    "gold\\pred     covered partial missing"
)

for g in labels:
    print(
        f"{g:12s}"
        f"{matrix[g]['covered']:8d}"
        f"{matrix[g]['partial']:8d}"
        f"{matrix[g]['missing']:8d}"
    )

print()
print("Error slices:")
print(
    "false-covered:",
    len(false_covered),
)
print(
    "missing->covered:",
    len(severe_missing_to_covered),
)
print(
    "covered->partial:",
    len(covered_to_partial),
)
print(
    "covered->missing:",
    len(covered_to_missing),
)
print(
    "partial->missing:",
    len(partial_to_missing),
)

for title, items in [
    (
        "FALSE COVERED",
        false_covered,
    ),
    (
        "COVERED -> PARTIAL",
        covered_to_partial,
    ),
    (
        "COVERED -> MISSING",
        covered_to_missing,
    ),
    (
        "PARTIAL -> MISSING",
        partial_to_missing,
    ),
]:
    if not items:
        continue

    print()
    print(title)

    for row in items:
        print(
            " -",
            row["case_id"],
            "| gold=",
            row["gold"]["status"],
            "| pred=",
            row["prediction"]["status"],
        )
