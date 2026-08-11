import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from hybrid_search import reciprocal_rank_fusion
from reranker import CrossEncoderReranker


CONFIGS = {
    "250_25",
    "500_50",
    "750_75",
    "1000_100",
}

DEFAULT_MANIFEST = Path(
    "outputs/chunking_ablation/gold_remap_reviewed.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "outputs/chunking_ablation/retrieval"
)

MODES = [
    "bm25",
    "semantic",
    "hybrid",
    "hybrid_rerank",
]

load_dotenv(dotenv_path=".env")
openai_client = OpenAI()


class ChunkingAblationRetriever:
    def __init__(
        self,
        config: str,
        *,
        candidate_k: int = 20,
    ):
        if config not in CONFIGS:
            raise ValueError(f"Unknown config: {config}")

        self.config = config
        self.candidate_k = candidate_k

        self.chunk_dir = (
            Path("data/chunks/ablation")
            / config
            / "reports"
        )

        self.chroma_dir = (
            Path("data/chromadb_ablation")
            / config
        )

        self.chunks = self._load_chunks()

        self.tokenized_corpus = [
            self._tokenize(row["text"])
            for row in self.chunks
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        self.chroma_client = (
            chromadb.PersistentClient(
                path=str(self.chroma_dir)
            )
        )

        self.collection = (
            self.chroma_client
            .get_collection("reports")
        )

        self.reranker = CrossEncoderReranker()

        print(
            f"{config}: BM25={len(self.chunks)} | "
            f"Chroma={self.collection.count()}"
        )

        if self.collection.count() != len(self.chunks):
            raise AssertionError(
                f"{config}: BM25/Chroma count mismatch"
            )

    def _load_chunks(self):
        chunks = []

        for path in sorted(
            self.chunk_dir.glob("*.jsonl")
        ):
            with path.open(encoding="utf-8") as f:
                for line in f:
                    chunks.append(
                        json.loads(line)
                    )

        if not chunks:
            raise RuntimeError(
                f"No chunks found for {self.config}"
            )

        return chunks

    @staticmethod
    def _tokenize(text: str):
        text = text.lower()
        tokens = word_tokenize(text)

        stop_words = set(
            stopwords.words("english")
        )

        return [
            token
            for token in tokens
            if (
                token.isalnum()
                and token not in stop_words
                and len(token) > 2
            )
        ]

    def bm25_search(
        self,
        query: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
    ):
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(
            query_tokens
        )

        candidates = []

        for idx, row in enumerate(self.chunks):
            if where and not all(
                row["metadata"].get(k) == v
                for k, v in where.items()
            ):
                continue

            candidates.append(
                (idx, float(scores[idx]))
            )

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for rank, (idx, score) in enumerate(
            candidates[:n_results],
            1,
        ):
            row = self.chunks[idx]

            results.append({
                "rank": rank,
                "chunk_id": row["chunk_id"],
                "text": row["text"],
                "metadata": row["metadata"],
                "score": score,
            })

        return results

    @staticmethod
    def _embed_query(query: str):
        response = (
            openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
            )
        )

        return response.data[0].embedding

    def semantic_search(
        self,
        query: str,
        *,
        n_results: int,
        where: dict[str, Any] | None,
    ):
        embedding = self._embed_query(query)

        raw = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
        )

        results = []

        for i, chunk_id in enumerate(
            raw["ids"][0]
        ):
            results.append({
                "rank": i + 1,
                "chunk_id": chunk_id,
                "text": raw["documents"][0][i],
                "metadata": raw["metadatas"][0][i],
                "distance": raw["distances"][0][i],
            })

        return results

    def hybrid_candidates(
        self,
        query: str,
        *,
        where: dict[str, Any] | None,
    ):
        semantic = self.semantic_search(
            query,
            n_results=self.candidate_k,
            where=where,
        )

        keyword = self.bm25_search(
            query,
            n_results=self.candidate_k,
            where=where,
        )

        return reciprocal_rank_fusion(
            semantic,
            keyword,
        )

    def search(
        self,
        query: str,
        *,
        mode: str,
        n_results: int,
        where: dict[str, Any] | None,
    ):
        if mode == "bm25":
            return self.bm25_search(
                query,
                n_results=n_results,
                where=where,
            )

        if mode == "semantic":
            return self.semantic_search(
                query,
                n_results=n_results,
                where=where,
            )

        if mode == "hybrid":
            return self.hybrid_candidates(
                query,
                where=where,
            )[:n_results]

        if mode == "hybrid_rerank":
            candidates = (
                self.hybrid_candidates(
                    query,
                    where=where,
                )
            )

            return self.reranker.rerank(
                query,
                candidates,
                top_k=n_results,
            )

        raise ValueError(
            f"Unsupported mode: {mode}"
        )


def evidence_rank(
    retrieved_ids: list[str],
    config_gold: dict[str, Any],
):
    rank_lookup = {
        chunk_id: rank
        for rank, chunk_id in enumerate(
            retrieved_ids,
            1,
        )
    }

    # Single-chunk evidence:
    # any reviewed independently sufficient chunk.
    single_ids = config_gold[
        "gold_chunk_ids"
    ]

    single_ranks = [
        rank_lookup[cid]
        for cid in single_ids
        if cid in rank_lookup
    ]

    candidate_completion_ranks = (
        list(single_ranks)
    )

    # Split evidence:
    # every member of a complementary group
    # must be retrieved before the evidence is complete.
    for group in config_gold[
        "complementary_groups"
    ]:
        if all(
            cid in rank_lookup
            for cid in group
        ):
            completion_rank = max(
                rank_lookup[cid]
                for cid in group
            )

            candidate_completion_ranks.append(
                completion_rank
            )

    if not candidate_completion_ranks:
        return None

    return min(
        candidate_completion_ranks
    )


def evaluate_config(
    config: str,
    manifest: dict[str, Any],
    *,
    top_k: int,
    candidate_k: int,
):
    retriever = ChunkingAblationRetriever(
        config,
        candidate_k=candidate_k,
    )

    results = []

    for idx, record in enumerate(
        manifest["results"],
        1,
    ):
        print(
            f"[{config}] "
            f"{idx}/{len(manifest['results'])} "
            f"{record['query_id']}"
        )

        for mode in MODES:
            retrieved = retriever.search(
                record["query"],
                mode=mode,
                n_results=top_k,
                where={
                    "company_id":
                    record["company_id"]
                },
            )

            retrieved_ids = [
                row["chunk_id"]
                for row in retrieved
            ]

            completion_rank = evidence_rank(
                retrieved_ids,
                record["configs"][config],
            )

            results.append({
                "config": config,
                "query_id": record["query_id"],
                "company_id": record["company_id"],
                "slot_id": record["slot_id"],
                "mode": mode,
                "top_k": top_k,
                "evidence_status": (
                    record["configs"][config][
                        "status"
                    ]
                ),
                "gold_chunk_ids": (
                    record["configs"][config][
                        "gold_chunk_ids"
                    ]
                ),
                "complementary_groups": (
                    record["configs"][config][
                        "complementary_groups"
                    ]
                ),
                "retrieved_chunk_ids":
                    retrieved_ids,
                "completion_rank":
                    completion_rank,
                "hit_at_k":
                    completion_rank is not None,
                "reciprocal_rank": (
                    1 / completion_rank
                    if completion_rank
                    else 0.0
                ),
            })

    return results


def summarize(results):
    summary = {}

    configs = sorted(
        {row["config"] for row in results}
    )

    for config in configs:
        summary[config] = {}

        for mode in MODES:
            rows = [
                row
                for row in results
                if (
                    row["config"] == config
                    and row["mode"] == mode
                )
            ]

            hits = sum(
                row["hit_at_k"]
                for row in rows
            )

            mrr = sum(
                row["reciprocal_rank"]
                for row in rows
            ) / len(rows)

            summary[config][mode] = {
                "queries": len(rows),
                "hits": hits,
                "hit_rate": hits / len(rows),
                "mrr": mrr,
            }

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        choices=[
            "all",
            *sorted(CONFIGS),
        ],
        default="all",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    manifest = json.loads(
        args.manifest.read_text(
            encoding="utf-8"
        )
    )

    configs = (
        sorted(CONFIGS)
        if args.config == "all"
        else [args.config]
    )

    all_results = []

    for config in configs:
        all_results.extend(
            evaluate_config(
                config,
                manifest,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
            )
        )

    summary = summarize(
        all_results
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "query_count":
            manifest["query_count"],
        "configs": configs,
        "modes": MODES,
        "summary": summary,
        "results": all_results,
    }

    output_path = (
        args.output_dir
        / "chunking_retrieval_ablation.json"
    )

    output_path.write_text(
        json.dumps(
            output,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 90)
    print("CHUNKING RETRIEVAL SUMMARY")
    print("=" * 90)

    for config in configs:
        print(f"\n{config}")

        for mode in MODES:
            metrics = summary[config][mode]

            print(
                f"  {mode:15s} "
                f"Hit@{args.top_k}="
                f"{metrics['hit_rate']:.3f} "
                f"({metrics['hits']}/"
                f"{metrics['queries']}) "
                f"MRR={metrics['mrr']:.3f}"
            )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
