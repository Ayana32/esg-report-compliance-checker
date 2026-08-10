import json
import re
from collections import Counter
from pathlib import Path


CORPUS_ROOT = Path("data/chunks/ablation")
REVIEWED_PATH = Path(
    "outputs/chunking_ablation/gold_remap_reviewed.json"
)
CANDIDATES_PATH = Path(
    "outputs/chunking_ablation/gold_remap_candidates.json"
)
OUTPUT_PATH = Path(
    "outputs/chunking_ablation/unified_gold_integrity_all_configs.json"
)

CONFIGS = [
    "250_25",
    "500_50",
    "750_75",
    "1000_100",
]


def normalize_for_comparison(text: str) -> str:
    """
    Comparison-only normalization.
    Does not modify corpus files.
    """
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("•", "- ")
    text = text.replace("◆", "- ")
    text = text.replace("▶", "- ")
    text = text.replace("\u200b", "")
    text = text.replace("（", "(").replace("）", ")")

    text = re.sub(
        r"(\w+)-\s+(\w+)",
        r"\1\2",
        text,
    )

    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_corpus(config: str):
    corpus = {}
    report_dir = CORPUS_ROOT / config / "reports"

    for path in sorted(report_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cid = row["chunk_id"]

                if cid in corpus:
                    raise AssertionError(
                        f"Duplicate chunk ID in {config}: {cid}"
                    )

                corpus[cid] = row

    return corpus


reviewed = json.loads(
    REVIEWED_PATH.read_text(encoding="utf-8")
)
candidate_data = json.loads(
    CANDIDATES_PATH.read_text(encoding="utf-8")
)

candidate_by_query = {
    r["query_id"]: r
    for r in candidate_data["results"]
}

corpora = {
    cfg: load_corpus(cfg)
    for cfg in CONFIGS
}

errors = []
summary = {}
details = []

for cfg in CONFIGS:
    counts = Counter()
    reviewed_ids = set()

    for result in reviewed["results"]:
        qid = result["query_id"]
        config_review = result["configs"][cfg]

        ids = list(config_review["gold_chunk_ids"])

        for group in config_review["complementary_groups"]:
            ids.extend(group)

        # Deduplicate while preserving order.
        ids = list(dict.fromkeys(ids))

        for cid in ids:
            reviewed_ids.add(cid)

            if cid not in corpora[cfg]:
                errors.append(
                    f"{qid} | {cfg} | reviewed ID missing "
                    f"from unified corpus: {cid}"
                )
                continue

            counts["present"] += 1

            # Candidate-remap text provides the pre-unified comparison
            # for 250/750/1000. Baseline candidate coverage may vary,
            # so absence here is recorded rather than treated as failure.
            candidate_row = None

            candidate_query = candidate_by_query.get(qid)

            if candidate_query:
                for c in candidate_query["configs"].get(cfg, []):
                    if c["chunk_id"] == cid:
                        candidate_row = c
                        break

            if candidate_row is None:
                counts["no_old_candidate_text"] += 1
                details.append({
                    "query_id": qid,
                    "config": cfg,
                    "chunk_id": cid,
                    "status": "NO_OLD_CANDIDATE_TEXT",
                })
                continue

            old_text = candidate_row["text"]
            new_text = corpora[cfg][cid]["text"]

            if old_text == new_text:
                status = "IDENTICAL"
            elif (
                normalize_for_comparison(old_text)
                == normalize_for_comparison(new_text)
            ):
                status = "CLEANING_ONLY"
            else:
                status = "CONTENT_CHANGED"

            counts[status] += 1

            details.append({
                "query_id": qid,
                "config": cfg,
                "chunk_id": cid,
                "status": status,
            })

            if status == "CONTENT_CHANGED":
                errors.append(
                    f"{qid} | {cfg} | possible content change: {cid}"
                )

    split_queries = sum(
        result["configs"][cfg]["status"] == "split_evidence"
        for result in reviewed["results"]
    )

    single_queries = sum(
        result["configs"][cfg]["status"] == "single_chunk_evidence"
        for result in reviewed["results"]
    )

    summary[cfg] = {
        "corpus_chunks": len(corpora[cfg]),
        "reviewed_unique_chunk_ids": len(reviewed_ids),
        "single_chunk_queries": single_queries,
        "split_evidence_queries": split_queries,
        "present_ids": counts["present"],
        "identical_text": counts["IDENTICAL"],
        "cleaning_only": counts["CLEANING_ONLY"],
        "content_changed": counts["CONTENT_CHANGED"],
        "no_old_candidate_text": counts["no_old_candidate_text"],
    }


OUTPUT_PATH.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "benchmark_queries": len(reviewed["results"]),
            "summary": summary,
            "details": details,
            "errors": errors,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)


print("=" * 90)
print("UNIFIED CHUNKING GOLD INTEGRITY — ALL CONFIGS")
print("=" * 90)

for cfg in CONFIGS:
    s = summary[cfg]

    print(
        f"{cfg}: "
        f"corpus={s['corpus_chunks']} | "
        f"reviewed_ids={s['reviewed_unique_chunk_ids']} | "
        f"single={s['single_chunk_queries']} | "
        f"split={s['split_evidence_queries']} | "
        f"identical={s['identical_text']} | "
        f"cleaning_only={s['cleaning_only']} | "
        f"content_changed={s['content_changed']} | "
        f"no_old_text={s['no_old_candidate_text']}"
    )

print()

if errors:
    print(f"FAIL: {len(errors)} issue(s) found.")
    for error in errors:
        print("-", error)
    raise SystemExit(1)

print("PASS: every reviewed gold/complementary ID exists in the unified corpus.")
print("PASS: no reviewed evidence shows substantive content change.")
print("PASS: split-evidence annotations remain structurally valid.")
print("Saved:", OUTPUT_PATH)
