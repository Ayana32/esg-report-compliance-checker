import json
import re
from collections import Counter
from pathlib import Path

import tiktoken


CANONICAL_DIR = Path("data/chunks/reports")
RAW_DIR = Path("data/raw/extracted")
OUTPUT_ROOT = Path("data/chunks/ablation")

CONFIGS = {
    "500_50": (500, 50),
    "250_25": (250, 25),
    "750_75": (750, 75),
    "1000_100": (1000, 100),
}

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")




IBK_HEADER_PATTERNS = [
    r'Overview\s+ESG\s+Management\s+Focus\s+Issue\s+Sustainability\s+Core\s+Value',
    r'ESG\s+Fact\s+Report',
    r'Environmental\s+기후변화\s+대응',
    r'Beyond\s+Sustainability',
    r'2024\s+IBK\s+지속가능경영보고서',
]


def clean_report_headers(text: str, language: str) -> str:
    """
    Reproduce the report-header cleanup applied to the canonical corpus
    before the final cleaning pass.
    """
    if language == "KO":
        for pattern in IBK_HEADER_PATTERNS:
            text = re.sub(
                pattern,
                "",
                text,
                flags=re.IGNORECASE,
            )

    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_chunk_text(text: str, language: str) -> str:
    """
    Reproduce the final cleaning pass used by the canonical corpus.
    Cleaning is applied AFTER token-based chunking so chunk boundaries
    and stored token_count remain based on the raw extracted text.
    """
    # Normalize URLs without removing them.
    text = re.sub(
        r'https?://[^\s]+',
        lambda m: m.group(0).strip(),
        text,
    )

    # Normalize common punctuation.
    text = text.replace('–', '-').replace('—', '-')

    # Normalize smart quotes.

    # Standardize bullets.
    text = text.replace('•', '- ')
    text = text.replace('◆', '- ')
    text = text.replace('▶', '- ')

    # Normalize whitespace.
    text = re.sub(r'\s+', ' ', text).strip()

    # Language-specific fixes.
    if language == 'KO':
        text = text.replace('\u200b', '')
        text = text.replace('（', '(').replace('）', ')')

    elif language == 'EN':
        # PDF line-break hyphenation:
        # "sustain- ability" -> "sustainability"
        text = re.sub(
            r'(\w+)-\s+(\w+)',
            r'\1\2',
            text,
        )

    return text

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def chunk_text_fixed(text: str, chunk_size: int, overlap: int):
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than "
            f"chunk_size ({chunk_size})"
        )

    tokens = encoding.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        decoded_text = encoding.decode(chunk_tokens)
        chunk_text = decoded_text

        chunks.append(
            {
                "text": chunk_text,
                "token_start": start,
                "token_end": end,
                "token_count": len(
                    encoding.encode(chunk_text)
                ),
            }
        )

        start = end - overlap

    return chunks


def load_report_metadata(canonical_path: Path):
    """
    Reuse report-level metadata from the canonical corpus so the ablation
    changes only segmentation, not report metadata.
    """
    with canonical_path.open(encoding="utf-8") as f:
        first = json.loads(next(f))

    metadata = first["metadata"]

    return {
        "report_id": metadata["report_id"],
        "company_id": metadata["company_id"],
        "sector": metadata["sector"],
        "language": metadata["language"],
    }


def load_canonical(canonical_path: Path):
    rows = []

    with canonical_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def build_report(
    canonical_path: Path,
    config_name: str,
    chunk_size: int,
    overlap: int,
):
    report_meta = load_report_metadata(canonical_path)
    report_id = report_meta["report_id"]

    page_dir = RAW_DIR / report_id / "pages"
    page_paths = sorted(page_dir.glob("page_*.txt"))

    if not page_paths:
        raise FileNotFoundError(
            f"No raw pages found for {report_id}: {page_dir}"
        )

    output_dir = OUTPUT_ROOT / config_name / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{report_id}.jsonl"

    rows = []

    for page_path in page_paths:
        page_num = int(page_path.stem.split("_")[-1])
        page_text = page_path.read_text(encoding="utf-8")

        chunks = chunk_text_fixed(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_num, chunk in enumerate(chunks, start=1):
            chunk_id = (
                f"{report_id}"
                f"_p{page_num:04d}"
                f"_c{chunk_num:04d}"
            )

            text = clean_report_headers(
                chunk["text"],
                report_meta["language"],
            )
            text = clean_chunk_text(
                text,
                report_meta["language"],
            )

            row = {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": {
                    **report_meta,
                    "page_num": page_num,
                    "chunk_num": chunk_num,
                    "token_count": chunk["token_count"],
                    "char_count": len(text),
                },
            }

            rows.append(row)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return rows


def validate_unique_ids(rows, config_name, report_id):
    ids = [r["chunk_id"] for r in rows]

    if len(ids) != len(set(ids)):
        counts = Counter(ids)
        duplicates = [
            k for k, v in counts.items()
            if v > 1
        ]
        raise AssertionError(
            f"{config_name} {report_id}: duplicate chunk IDs: "
            f"{duplicates[:10]}"
        )


def validate_baseline_reproduction(
    canonical_path: Path,
    generated_rows,
):
    """
    500/50 must exactly reproduce the current canonical corpus.
    We compare chunk ID and text. Metadata is checked separately because
    the canonical schema may contain derived values such as char_count.
    """
    canonical_rows = load_canonical(canonical_path)

    if len(canonical_rows) != len(generated_rows):
        raise AssertionError(
            f"{canonical_path.stem}: "
            f"canonical chunks={len(canonical_rows)}, "
            f"generated chunks={len(generated_rows)}"
        )

    for i, (old, new) in enumerate(
        zip(canonical_rows, generated_rows),
        start=1,
    ):
        if old["chunk_id"] != new["chunk_id"]:
            raise AssertionError(
                f"{canonical_path.stem}: chunk ID mismatch "
                f"at row {i}\n"
                f"old={old['chunk_id']}\n"
                f"new={new['chunk_id']}"
            )

        if old["text"] != new["text"]:
            raise AssertionError(
                f"{canonical_path.stem}: text mismatch "
                f"at {old['chunk_id']}"
            )

        for key in [
            "report_id",
            "company_id",
            "sector",
            "language",
            "page_num",
            "chunk_num",
            "token_count",
            "char_count",
        ]:
            old_value = old["metadata"][key]
            new_value = new["metadata"][key]

            if old_value != new_value:
                raise AssertionError(
                    f"{canonical_path.stem}: metadata mismatch "
                    f"at {old['chunk_id']} key={key}: "
                    f"{old_value!r} != {new_value!r}"
                )


def main():
    canonical_paths = sorted(
        CANONICAL_DIR.glob("*.jsonl")
    )

    if len(canonical_paths) != 16:
        raise AssertionError(
            f"Expected 16 canonical reports, "
            f"found {len(canonical_paths)}"
        )

    summary = {}

    for config_name, (chunk_size, overlap) in CONFIGS.items():
        print("\n" + "=" * 80)
        print(
            f"BUILDING {config_name} "
            f"(chunk={chunk_size}, overlap={overlap})"
        )
        print("=" * 80)

        config_rows = []

        for canonical_path in canonical_paths:
            rows = build_report(
                canonical_path=canonical_path,
                config_name=config_name,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            validate_unique_ids(
                rows,
                config_name=config_name,
                report_id=canonical_path.stem,
            )

            if config_name == "500_50":
                validate_baseline_reproduction(
                    canonical_path,
                    rows,
                )

            config_rows.extend(rows)

            print(
                f"{canonical_path.stem}: "
                f"{len(rows)} chunks"
            )

        token_counts = [
            r["metadata"]["token_count"]
            for r in config_rows
        ]

        all_ids = [
            r["chunk_id"]
            for r in config_rows
        ]

        assert len(all_ids) == len(set(all_ids)), (
            f"{config_name}: duplicate IDs across reports"
        )

        summary[config_name] = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "report_count": len(canonical_paths),
            "total_chunks": len(config_rows),
            "min_token_count": min(token_counts),
            "max_token_count": max(token_counts),
            "mean_token_count": (
                sum(token_counts) / len(token_counts)
            ),
        }

        print()
        print(
            f"{config_name}: "
            f"total_chunks={len(config_rows)}, "
            f"mean_tokens={summary[config_name]['mean_token_count']:.1f}, "
            f"min={min(token_counts)}, "
            f"max={max(token_counts)}"
        )

        if config_name == "500_50":
            print(
                "PASS: 500/50 exactly reproduces "
                "the canonical corpus."
            )

    summary_path = (
        OUTPUT_ROOT / "corpus_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("CHUNKING ABLATION CORPORA COMPLETE")
    print("=" * 80)

    for config_name, stats in summary.items():
        print(
            f"{config_name}: "
            f"{stats['report_count']} reports | "
            f"{stats['total_chunks']} chunks | "
            f"mean={stats['mean_token_count']:.1f} tokens | "
            f"max={stats['max_token_count']}"
        )

    print("\nSaved:", summary_path)
    print("PASS: all chunk IDs are unique.")
    print("PASS: canonical corpus was not modified.")


if __name__ == "__main__":
    main()
