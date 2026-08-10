import json
import re
from pathlib import Path
from statistics import mean

import tiktoken


CANONICAL_DIR = Path("data/chunks/reports")
RAW_DIR = Path("data/raw/extracted")
OUTPUT_ROOT = Path("data/chunks/ablation")

CONFIGS = {
    "250_25": (250, 25),
    "500_50": (500, 50),
    "750_75": (750, 75),
    "1000_100": (1000, 100),
}

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def clean_chunk_text(text: str, language: str) -> str:
    """
    Deterministic final text-cleaning policy used for every report and
    every chunking configuration.
    """

    # Keep URLs but remove surrounding whitespace artefacts.
    text = re.sub(
        r"https?://[^\s]+",
        lambda m: m.group(0).strip(),
        text,
    )

    # Normalize dash variants.
    text = text.replace("–", "-").replace("—", "-")

    # Standardize common bullet symbols.
    text = text.replace("•", "- ")
    text = text.replace("◆", "- ")
    text = text.replace("▶", "- ")

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    if language == "KO":
        text = text.replace("\u200b", "")
        text = text.replace("（", "(").replace("）", ")")

    elif language == "EN":
        # Repair PDF line-break hyphenation:
        # "sustain- ability" -> "sustainability"
        text = re.sub(
            r"(\w+)-\s+(\w+)",
            r"\1\2",
            text,
        )

    return text


def load_report_metadata(canonical_path: Path):
    """
    Reuse stable report-level metadata only.
    Chunk text and segmentation are always regenerated from raw pages.
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


def chunk_page(
    raw_text: str,
    chunk_size: int,
    overlap: int,
):
    """
    Reproduce the original page-level stride semantics.

    Note:
    start advances by chunk_size - overlap even when the final slice is
    shorter than chunk_size. This intentionally retains small tail chunks.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap={overlap} must be smaller than "
            f"chunk_size={chunk_size}"
        )

    tokens = encoding.encode(raw_text)
    chunks = []

    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        if not chunk_tokens:
            break

        decoded_text = encoding.decode(chunk_tokens)

        # Historical metadata counted the decoded text by re-encoding it.
        token_count = len(
            encoding.encode(decoded_text)
        )

        chunks.append(
            {
                "decoded_text": decoded_text,
                "token_count": token_count,
            }
        )

        start = end - overlap

    return chunks


def build_report(
    canonical_path: Path,
    config_name: str,
    chunk_size: int,
    overlap: int,
):
    report_meta = load_report_metadata(canonical_path)
    report_id = report_meta["report_id"]

    pages_dir = RAW_DIR / report_id / "pages"
    page_paths = sorted(
        pages_dir.glob("page_*.txt")
    )

    if not page_paths:
        raise FileNotFoundError(
            f"No raw page files for {report_id}"
        )

    out_dir = (
        OUTPUT_ROOT
        / config_name
        / "reports"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path = out_dir / f"{report_id}.jsonl"

    rows = []

    for page_path in page_paths:
        page_num = int(
            page_path.stem.split("_")[-1]
        )

        raw_text = page_path.read_text(
            encoding="utf-8"
        )

        page_chunks = chunk_page(
            raw_text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_num, chunk in enumerate(
            page_chunks,
            start=1,
        ):
            text = clean_chunk_text(
                chunk["decoded_text"],
                report_meta["language"],
            )

            chunk_id = (
                f"{report_id}"
                f"_p{page_num:04d}"
                f"_c{chunk_num:04d}"
            )

            # Empty chunks can arise when a tiny page-end tail or a
            # known header/footer artefact is fully removed by cleaning.
            # Exclude them from retrieval while preserving the original
            # segmentation IDs of all remaining chunks.
            if not text.strip():
                continue

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

    ids = [
        row["chunk_id"]
        for row in rows
    ]

    if len(ids) != len(set(ids)):
        raise AssertionError(
            f"Duplicate chunk IDs in {report_id}"
        )

    with out_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    return rows


def main():
    canonical_paths = sorted(
        CANONICAL_DIR.glob("*.jsonl")
    )

    if len(canonical_paths) != 16:
        raise AssertionError(
            f"Expected 16 reports, "
            f"found {len(canonical_paths)}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {}

    for config_name, (
        chunk_size,
        overlap,
    ) in CONFIGS.items():

        print("\n" + "=" * 80)
        print(
            f"BUILDING {config_name} "
            f"(chunk={chunk_size}, overlap={overlap})"
        )
        print("=" * 80)

        all_rows = []

        for canonical_path in canonical_paths:
            rows = build_report(
                canonical_path,
                config_name,
                chunk_size,
                overlap,
            )

            all_rows.extend(rows)

            print(
                f"{canonical_path.stem}: "
                f"{len(rows)} chunks"
            )

        ids = [
            row["chunk_id"]
            for row in all_rows
        ]

        if len(ids) != len(set(ids)):
            raise AssertionError(
                f"{config_name}: duplicate IDs "
                f"across reports"
            )

        token_counts = [
            row["metadata"]["token_count"]
            for row in all_rows
        ]

        char_counts = [
            row["metadata"]["char_count"]
            for row in all_rows
        ]

        empty_chunks = sum(
            row["text"] == ""
            for row in all_rows
        )

        summary[config_name] = {
            "chunk_size": chunk_size,
            "overlap": overlap,
            "stride": chunk_size - overlap,
            "report_count": 16,
            "total_chunks": len(all_rows),
            "min_token_count": min(token_counts),
            "max_token_count": max(token_counts),
            "mean_token_count": mean(token_counts),
            "mean_char_count": mean(char_counts),
            "empty_text_chunks": empty_chunks,
        }

        print()
        print(
            f"{config_name}: "
            f"chunks={len(all_rows)}, "
            f"mean_tokens={mean(token_counts):.1f}, "
            f"min={min(token_counts)}, "
            f"max={max(token_counts)}, "
            f"empty={empty_chunks}"
        )

    manifest = {
        "schema_version": 1,
        "purpose": (
            "Deterministic chunk-size ablation corpus. "
            "All configurations are regenerated from identical "
            "raw page extractions using the same preprocessing policy."
        ),
        "tokenizer": {
            "library": "tiktoken",
            "model": "gpt-3.5-turbo",
        },
        "preprocessing_order": [
            "raw_page_text",
            "tokenize",
            "fixed_token_slice",
            "decode",
            "decoded_text_token_count",
            "generic_final_text_cleaning",
            "metadata_and_jsonl_serialization",
        ],
        "configs": {
            name: {
                "chunk_size": size,
                "overlap": overlap,
                "stride": size - overlap,
            }
            for name, (
                size,
                overlap,
            ) in CONFIGS.items()
        },
        "summary": summary,
    }

    summary_path = (
        OUTPUT_ROOT
        / "corpus_summary.json"
    )

    manifest_path = (
        OUTPUT_ROOT
        / "preprocessing_manifest.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("UNIFIED ABLATION CORPUS COMPLETE")
    print("=" * 80)

    for config, stats in summary.items():
        print(
            f"{config}: "
            f"{stats['total_chunks']} chunks | "
            f"mean={stats['mean_token_count']:.1f} | "
            f"empty={stats['empty_text_chunks']}"
        )

    print("\nPASS: 16 reports generated per configuration.")
    print("PASS: chunk IDs unique within each configuration.")
    print("PASS: identical preprocessing policy used across configs.")
    print("Saved:", summary_path)
    print("Saved:", manifest_path)


if __name__ == "__main__":
    main()
