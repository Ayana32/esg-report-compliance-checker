import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_verifier import LLMVerifier


PRED_PATH = Path(
    "outputs/verifier_benchmark/"
    "verifier_predictions.json"
)

POSITIVE_PATH = Path(
    "outputs/verifier_benchmark/"
    "positive_gold_frozen.json"
)

CHALLENGE_PATH = Path(
    "outputs/verifier_benchmark/"
    "challenge_gold_frozen.json"
)


def load_case_map():
    case_map = {}

    for path in [
        POSITIVE_PATH,
        CHALLENGE_PATH,
    ]:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        for case in data["cases"]:
            case_map[case["case_id"]] = case

    return case_map


def main():
    data = json.loads(
        PRED_PATH.read_text(encoding="utf-8")
    )

    case_map = load_case_map()

    failed_rows = [
        row
        for row in data["predictions"]
        if row["diagnostics"].get(
            "call_error"
        )
    ]

    print(
        f"Retrying {len(failed_rows)} "
        "failed cases"
    )

    verifier = LLMVerifier()

    recovered = 0
    still_failed = 0

    for i, row in enumerate(
        failed_rows,
        1,
    ):
        case_id = row["case_id"]
        case = case_map[case_id]

        print(
            f"[{i:02d}/{len(failed_rows)}] "
            f"{case_id}",
            flush=True,
        )

        last_error = None

        for attempt in range(1, 4):
            try:
                started = time.perf_counter()

                result = verifier.verify_element(
                    element=case["element"],
                    evidence_chunks=case[
                        "evidence_chunks"
                    ],
                    requirement_name=case.get(
                        "requirement_name",
                        (
                            "Direct (Scope 1) "
                            "GHG emissions"
                        ),
                    ),
                    max_chunks=5,
                )

                elapsed_ms = (
                    time.perf_counter()
                    - started
                ) * 1000

                selected = result.get(
                    "selected_evidence",
                    [],
                )

                row["prediction"] = {
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
                    "evidence_chunk_indices":
                        result.get(
                            "evidence_chunk_indices",
                            [],
                        ),
                    "selected_chunk_ids": [
                        item.get(
                            "chunk_id",
                            "",
                        )
                        for item in selected
                        if item.get("chunk_id")
                    ],
                    "page_numbers": result.get(
                        "page_numbers",
                        [],
                    ),
                }

                row["diagnostics"] = {
                    "latency_ms": elapsed_ms,
                    "parse_fallback": str(
                        result.get(
                            "reasoning",
                            "",
                        )
                    ).startswith(
                        "Failed to parse "
                        "LLM response:"
                    ),
                    "call_error": None,
                    "retry_attempt": attempt,
                }

                print(
                    "     "
                    f"RECOVERED "
                    f"pred={row['prediction']['status']} "
                    f"conf={row['prediction']['confidence']} "
                    f"attempt={attempt}",
                    flush=True,
                )

                recovered += 1
                break

            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                print(
                    "     "
                    f"attempt {attempt} failed: "
                    f"{last_error}",
                    flush=True,
                )

                if attempt < 3:
                    time.sleep(
                        5 * attempt
                    )

        else:
            row["diagnostics"][
                "call_error"
            ] = last_error

            still_failed += 1

        # Save after every case
        PRED_PATH.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Small spacing between API calls
        time.sleep(2)

    remaining = [
        row["case_id"]
        for row in data["predictions"]
        if row["diagnostics"].get(
            "call_error"
        )
    ]

    data["metadata"][
        "call_error_count"
    ] = len(remaining)

    data["metadata"][
        "completed_cases"
    ] = (
        len(data["predictions"])
        - len(remaining)
    )

    PRED_PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("RETRY COMPLETE")
    print("=" * 80)
    print("Recovered:", recovered)
    print(
        "Still failed:",
        still_failed,
    )

    if remaining:
        print("Remaining cases:")
        for case_id in remaining:
            print(" -", case_id)


if __name__ == "__main__":
    main()
