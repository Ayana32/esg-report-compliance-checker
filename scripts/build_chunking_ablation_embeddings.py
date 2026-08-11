import argparse
import json
import os
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


CONFIGS = {
    "250_25",
    "500_50",
    "750_75",
    "1000_100",
}

MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100

load_dotenv(dotenv_path=".env")

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def load_chunks(config: str):
    chunk_dir = (
        Path("data/chunks/ablation")
        / config
        / "reports"
    )

    chunks = []

    for path in sorted(chunk_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

    if not chunks:
        raise RuntimeError(
            f"No chunks found for {config}"
        )

    return chunks


def embed_batch(texts):
    response = client.embeddings.create(
        model=MODEL,
        input=texts,
    )

    embeddings = sorted(
        response.data,
        key=lambda item: item.index,
    )

    return [
        item.embedding
        for item in embeddings
    ]


def build_config(
    config: str,
    batch_size: int,
):
    chunks = load_chunks(config)

    db_path = (
        Path("data/chromadb_ablation")
        / config
    )

    db_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    chroma = chromadb.PersistentClient(
        path=str(db_path)
    )

    # Make rebuilds deterministic.
    try:
        chroma.delete_collection("reports")
    except Exception:
        pass

    collection = chroma.create_collection(
        name="reports"
    )

    print("=" * 80)
    print(f"CONFIG: {config}")
    print(f"Chunks: {len(chunks)}")
    print(f"DB: {db_path}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {batch_size}")
    print("=" * 80)

    stored = 0

    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc=config,
    ):
        batch = chunks[
            start:start + batch_size
        ]

        texts = [
            row["text"]
            for row in batch
        ]

        embeddings = embed_batch(texts)

        if len(embeddings) != len(batch):
            raise RuntimeError(
                "Embedding count mismatch: "
                f"{len(embeddings)} != {len(batch)}"
            )

        collection.add(
            ids=[
                row["chunk_id"]
                for row in batch
            ],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                row["metadata"]
                for row in batch
            ],
        )

        stored += len(batch)

        # Mild pacing between requests.
        time.sleep(0.05)

    actual = collection.count()

    if actual != len(chunks):
        raise AssertionError(
            f"{config}: expected {len(chunks)} "
            f"embeddings, found {actual}"
        )

    print()
    print(
        f"PASS: {actual} embeddings stored "
        f"for {config}"
    )

    return {
        "config": config,
        "chunks": len(chunks),
        "stored": actual,
        "db_path": str(db_path),
        "model": MODEL,
        "batch_size": batch_size,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        choices=sorted(CONFIGS),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError(
            "batch-size must be >= 1"
        )

    result = build_config(
        args.config,
        args.batch_size,
    )

    out_dir = Path(
        "outputs/chunking_ablation"
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = (
        out_dir
        / f"embedding_build_{args.config}.json"
    )

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Saved:", out)


if __name__ == "__main__":
    main()
