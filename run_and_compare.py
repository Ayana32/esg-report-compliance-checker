"""
run_and_compare.py
Run compliance checker on all 3 companies → compare with ground truth → save error cases
"""

import csv
from pathlib import Path
from compliance_checker_v2_hybrid import ComplianceCheckerV2
from ground_truth import GROUND_TRUTH, SLOTS

# YAML element text → slot key 매핑
ELEMENT_TO_SLOT = {
    "Total Scope 1 emissions in tCO2e"                                          : "slot_a",
    "Gases included in calculation (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)"      : "slot_b",
    "Biogenic CO2 emissions (reported separately)"                               : "slot_c",
    "Base year for calculation"                                                  : "slot_d",
    "Emission factors and GWP source (e.g. IPCC AR5/AR6)"                       : "slot_e",
    "Consolidation approach (equity share, financial control, operational control)": "slot_f",
    "Standards and methodologies used (e.g. GHG Protocol, third-party verification)": "slot_g",
}

OUTPUT = "day3_baseline_output.csv"
COMPANIES = ["IBK", "Shinhan", "Kepco", "Heathrow", "Schneider",
             "Siemens", "Samsung", "Hyundai", "Incheon", "SC", "HSBC"]
STANDARD  = "gri_305"
REQ_ID    = "305-1"


def extract_error_cases(company: str, result: dict, gt: dict) -> list[dict]:
    """
    Compare system prediction vs ground truth at slot level.
    Returns one row per mismatched slot + one row for overall mismatch.
    """
    errors = []
    evidence      = result.get("evidence", [])
    elem_coverage = result.get("element_coverage", [])
    chunk_count   = len(evidence)

    GRI_KEYWORDS = [
        "scope 1", "ghg", "greenhouse gas", "co2", "emissions",
        "metric tonnes", "tonne", "carbon", "direct emission", "biogenic"
    ]
    retrieved_text = " ".join(e.get("text", "").lower() for e in evidence)
    keyword_hit    = any(kw in retrieved_text for kw in GRI_KEYWORDS)

    # ── Overall mismatch ──────────────────────────────────────────────────────
    predicted_overall = result.get("overall_status")
    gt_overall        = gt.get("overall")

    if predicted_overall != gt_overall:
        errors.append({
            "company_id"    : company,
            "level"         : "overall",
            "slot"          : "overall",
            "predicted"     : predicted_overall,
            "ground_truth"  : gt_overall,
            "chunk_count"   : chunk_count,
            "keyword_hit"   : keyword_hit,
            "method_used"   : result.get("verification_mode", "?"),
            "error_type"    : classify_error_type(chunk_count, keyword_hit, predicted_overall, gt_overall),
            "reasoning"     : "",
            "notes"         : gt.get("review_notes", ""),
        })

    # ── Slot-level mismatch ───────────────────────────────────────────────────
    # Map element_coverage list back to slot keys by element name
    for elem in elem_coverage:
        elem_name  = elem.get("element", "")
        slot_key   = ELEMENT_TO_SLOT.get(elem_name)
        if slot_key is None:
            continue   # 매핑 없는 요소는 스킵
        predicted  = elem.get("status")
        gt_slot    = gt.get(slot_key)
        method     = elem.get("verification_method", "?")
        reasoning  = elem.get("reasoning", "")

        if predicted != gt_slot:
            errors.append({
                "company_id"   : company,
                "level"        : "slot",
                "slot"         : slot_key,
                "predicted"    : predicted,
                "ground_truth" : gt_slot,
                "chunk_count"  : chunk_count,
                "keyword_hit"  : keyword_hit,
                "method_used"  : method,
                "error_type"   : classify_error_type(chunk_count, keyword_hit, predicted, gt_slot),
                "reasoning"    : reasoning[:200],
                "notes"        : SLOTS[slot_key],
            })

    return errors


def classify_error_type(chunk_count: int, keyword_hit: bool, predicted: str, gt: str) -> str:
    """
    Heuristic: retrieval_error if chunks look bad, verifier_error if chunks look fine.
    """
    STATUS_RANK = {"covered": 2, "partial": 1, "missing": 0}
    pred_rank = STATUS_RANK.get(predicted, -1)
    gt_rank   = STATUS_RANK.get(gt, -1)

    # False negative: predicted lower than ground truth
    direction = "false_negative" if pred_rank < gt_rank else "false_covered"

    if chunk_count <= 1 or not keyword_hit:
        return f"retrieval_error ({direction})"
    else:
        return f"verifier_error ({direction})"


def main():
    Path("outputs").mkdir(exist_ok=True)
    checker    = ComplianceCheckerV2()
    all_errors = []
    all_results = []

    for company in COMPANIES:
        print(f"\n{'='*60}")
        print(f"Running: {company}")
        print(f"{'='*60}")

        result = checker.check_requirement(
            company_id=company,
            standard=STANDARD,
            req_id=REQ_ID,
            verification_mode="hybrid"
        )

        gt         = GROUND_TRUTH[company]
        predicted  = result["overall_status"]
        gt_overall = gt["overall"]
        match      = predicted == gt_overall

        print(f"  Predicted : {predicted.upper()}")
        print(f"  Ground truth: {gt_overall.upper()}")
        print(f"  Match     : {'OK' if match else 'MISMATCH'}")

        errors = extract_error_cases(company, result, gt)
        all_errors.extend(errors)
        all_results.append({
            "company"    : company,
            "predicted"  : predicted,
            "gt"         : gt_overall,
            "match"      : match,
            "error_count": len(errors),
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        symbol = "OK" if r["match"] else "MISMATCH"
        print(f"  {r['company']:12} predicted={r['predicted']:8} gt={r['gt']:8} -> {symbol} ({r['error_count']} errors)")

    total_errors     = len(all_errors)
    retrieval_errors = sum(1 for e in all_errors if "retrieval_error" in e["error_type"])
    verifier_errors  = sum(1 for e in all_errors if "verifier_error"  in e["error_type"])
    false_negatives  = sum(1 for e in all_errors if "false_negative"  in e["error_type"])
    false_covered    = sum(1 for e in all_errors if "false_covered"   in e["error_type"])

    print(f"\n  Total mismatches  : {total_errors}")
    print(f"  Retrieval errors  : {retrieval_errors}")
    print(f"  Verifier errors   : {verifier_errors}")
    print(f"  False negatives   : {false_negatives}")
    print(f"  False covered     : {false_covered}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    if all_errors:
        fields = list(all_errors[0].keys())
        with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_errors)
        print(f"\n  Saved -> {OUTPUT}")
    else:
        print(f"\n  No mismatches found!")

    print("\nNext: open day2_error_cases.csv -> Day 3 pattern analysis")


if __name__ == "__main__":
    main()