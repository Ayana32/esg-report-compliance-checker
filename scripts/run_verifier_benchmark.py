import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_verifier import LLMVerifier, SLOT_RULES


POSITIVE_PATH = Path(
    "outputs/verifier_benchmark/"
    "positive_gold_frozen.json"
)

CHALLENGE_PATH = Path(
    "outputs/verifier_benchmark/"
    "challenge_gold_frozen.json"
)

OUTPUT_PATH = Path(
    "outputs/verifier_benchmark/"
    "verifier_predictions.json"
)

EXPECTED_GOLD_COMMIT = "66a6977"


def git_output(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=ROOT,
        text=True,
    ).strip()


def slot_rules_hash():
    payload = json.dumps(
        SLOT_RULES,
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def load_cases(path, split_name):
    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    cases = []

    for case in data["cases"]:
        # IMPORTANT:
        # Gold fields remain available only for later scoring.
        # They are NOT passed to LLMVerifier.
        cases.append({
            "split": split_name,
            "case_id": case["case_id"],
            "company_id": (
                case.get("company_id")
                or case.get("company")
            ),
            "requirement_id": (
                case.get("requirement_id", "305-1")
            ),
            "requirement_name": case.get(
                "requirement_name",
                "Direct (Scope 1) GHG emissions",
            ),
            "slot_id": (
                case.get("slot_id")
                or case.get("slot")
            ),
            "element": case["element"],
            "evidence_chunks": case["evidence_chunks"],

            # Kept only for post-hoc evaluation.
            "gold_status": case["gold_status"],
            "gold_supporting_chunk_ids": (
                case["gold_supporting_chunk_ids"]
            ),
        })

    return cases


def detect_parse_fallback(result):
    reasoning = str(
        result.get("reasoning", "")
    )

    return reasoning.startswith(
        "Failed to parse LLM response:"
    )


def main():
    current_commit = git_output(
        "rev-parse",
        "HEAD",
    )

    positive = load_cases(
        POSITIVE_PATH,
        "positive",
    )

    challenge = load_cases(
        CHALLENGE_PATH,
        "challenge",
    )

    cases = positive + challenge

    assert len(positive) == 27
    assert len(challenge) == 12
    assert len(cases) == 39

    gold_distribution = Counter(
        case["gold_status"]
        for case in cases
    )

    print("=" * 80)
    print("GRI 305-1 FIXED-EVIDENCE VERIFIER BENCHMARK")
    print("=" * 80)
    print(f"Gold freeze commit: {EXPECTED_GOLD_COMMIT}")
    print(f"Current commit:     {current_commit}")
    print(f"Cases:              {len(cases)}")
    print(
        "Gold distribution:",
        dict(gold_distribution),
    )
    print()

    verifier = LLMVerifier()

    predictions = []

    for i, case in enumerate(cases, 1):
        print(
            f"[{i:02d}/{len(cases)}] "
            f"{case['case_id']}",
            flush=True,
        )

        started = time.perf_counter()

        try:
            # CRITICAL:
            # Only the frozen evidence packet + slot context
            # are passed to the verifier.
            #
            # gold_status and gold_supporting_chunk_ids
            # are NEVER passed into verify_element().
            result = verifier.verify_element(
                element=case["element"],
                evidence_chunks=case[
                    "evidence_chunks"
                ],
                requirement_name=case[
                    "requirement_name"
                ],
                max_chunks=5,
            )

            call_error = None

        except Exception as exc:
            result = {
                "status": "missing",
                "covered": False,
                "partial": False,
                "confidence": 0.0,
                "reasoning": (
                    "Benchmark call failed: "
                    f"{type(exc).__name__}: {exc}"
                ),
                "evidence_chunk_indices": [],
                "selected_evidence": [],
                "page_numbers": [],
            }

            call_error = (
                f"{type(exc).__name__}: {exc}"
            )

        elapsed_ms = (
            time.perf_counter() - started
        ) * 1000

        selected_evidence = result.get(
            "selected_evidence",
            [],
        )

        selected_chunk_ids = [
            item.get("chunk_id", "")
            for item in selected_evidence
            if item.get("chunk_id")
        ]

        prediction = {
            "split": case["split"],
            "case_id": case["case_id"],
            "company_id": case["company_id"],
            "requirement_id": (
                case["requirement_id"]
            ),
            "slot_id": case["slot_id"],
            "element": case["element"],

            "prediction": {
                "status": result.get(
                    "status",
                    "missing",
                ),
                "confidence": result.get(
                    "confidence",
                    0.0,
                ),
                "reasoning": result.get(
                    "reasoning",
                    "",
                ),
                "evidence_chunk_indices": (
                    result.get(
                        "evidence_chunk_indices",
                        [],
                    )
                ),
                "selected_chunk_ids": (
                    selected_chunk_ids
                ),
                "page_numbers": result.get(
                    "page_numbers",
                    [],
                ),
            },

            "diagnostics": {
                "latency_ms": elapsed_ms,
                "parse_fallback": (
                    detect_parse_fallback(result)
                ),
                "call_error": call_error,
            },

            # Gold is copied into the result file ONLY
            # for scoring after inference is complete.
            # It was not passed to the verifier.
            "gold": {
                "status": case["gold_status"],
                "supporting_chunk_ids": (
                    case[
                        "gold_supporting_chunk_ids"
                    ]
                ),
            },
        }

        predictions.append(prediction)

        print(
            "     "
            f"pred={prediction['prediction']['status']} "
            f"conf={prediction['prediction']['confidence']} "
            f"evidence="
            f"{prediction['prediction']['evidence_chunk_indices']} "
            f"{elapsed_ms:.0f} ms",
            flush=True,
        )

        # Persist after every case so an interrupted run
        # does not destroy completed API results.
        partial_output = {
            "metadata": {
                "benchmark": (
                    "GRI 305-1 fixed-evidence "
                    "verifier benchmark"
                ),
                "gold_freeze_commit": (
                    EXPECTED_GOLD_COMMIT
                ),
                "runner_commit": current_commit,
                "slot_rules_sha256": (
                    slot_rules_hash()
                ),
                "inference_timestamp_utc": (
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
                "inference_policy": (
                    "Frozen Top-5 evidence packets "
                    "passed directly to "
                    "LLMVerifier.verify_element; "
                    "no retrieval performed during "
                    "verifier evaluation; gold "
                    "labels not passed to verifier."
                ),
                "max_chunks": 5,
                "temperature": 0.0,
                "completed_cases": len(
                    predictions
                ),
                "total_cases": len(cases),
            },
            "predictions": predictions,
        }

        OUTPUT_PATH.write_text(
            json.dumps(
                partial_output,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    pred_distribution = Counter(
        row["prediction"]["status"]
        for row in predictions
    )

    parse_fallbacks = sum(
        row["diagnostics"]["parse_fallback"]
        for row in predictions
    )

    call_errors = sum(
        row["diagnostics"]["call_error"]
        is not None
        for row in predictions
    )

    final_output = {
        "metadata": {
            "benchmark": (
                "GRI 305-1 fixed-evidence "
                "verifier benchmark"
            ),
            "gold_freeze_commit": (
                EXPECTED_GOLD_COMMIT
            ),
            "runner_commit": current_commit,
            "slot_rules_sha256": (
                slot_rules_hash()
            ),
            "inference_timestamp_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "inference_policy": (
                "Frozen Top-5 evidence packets "
                "passed directly to "
                "LLMVerifier.verify_element; "
                "no retrieval performed during "
                "verifier evaluation; gold labels "
                "not passed to verifier."
            ),
            "max_chunks": 5,
            "temperature": 0.0,
            "completed_cases": len(
                predictions
            ),
            "total_cases": len(cases),
            "gold_distribution": dict(
                gold_distribution
            ),
            "prediction_distribution": dict(
                pred_distribution
            ),
            "parse_fallback_count": (
                parse_fallbacks
            ),
            "call_error_count": call_errors,
        },
        "predictions": predictions,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            final_output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("INFERENCE COMPLETE")
    print("=" * 80)
    print(
        "Prediction distribution:",
        dict(pred_distribution),
    )
    print(
        "Parse fallbacks:",
        parse_fallbacks,
    )
    print(
        "Call errors:",
        call_errors,
    )
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
