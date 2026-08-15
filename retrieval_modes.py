"""Common interface for retrieval ablation modes."""

from typing import Any, Literal

from hybrid_search import reciprocal_rank_fusion
from keyword_search import KeywordRetriever
from reranker import CrossEncoderReranker
from semantic_search import SemanticRetriever


RetrievalMode = Literal[
    "bm25",
    "semantic",
    "hybrid",
    "hybrid_rerank",
]


class AblationRetriever:
    """Run multiple retrieval strategies through one interface."""

    def __init__(
        self,
        collection_name: str = "reports",
        *,
        candidate_k: int = 20,
        persist_directory: str = "data/chromadb",
        chunk_directory: str | None = None,
    ) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")

        self.collection_name = collection_name
        self.candidate_k = candidate_k

        self.keyword = KeywordRetriever(
            collection_name,
            chunk_directory=chunk_directory,
        )
        self.semantic = SemanticRetriever(
            collection_name,
            persist_directory=persist_directory,
        )
        self.reranker = CrossEncoderReranker()

    def _hybrid_candidates(
        self,
        query: str,
        *,
        where: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Retrieve and fuse semantic and BM25 candidate pools."""

        semantic_raw = self.semantic.search(
            query=query,
            n_results=self.candidate_k,
            where=where,
        )
        semantic_results = self.semantic.format_results(
            semantic_raw
        )

        keyword_results = self.keyword.search(
            query=query,
            n_results=self.candidate_k,
            where=where,
        )

        return reciprocal_rank_fusion(
            semantic_results,
            keyword_results,
        )

    def search(
        self,
        query: str,
        *,
        mode: RetrievalMode,
        n_results: int = 25,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve ranked chunks using the selected ablation mode."""

        if n_results < 1:
            raise ValueError("n_results must be at least 1.")

        if mode == "bm25":
            return self.keyword.search(
                query=query,
                n_results=n_results,
                where=where,
            )

        if mode == "semantic":
            raw_results = self.semantic.search(
                query=query,
                n_results=n_results,
                where=where,
            )
            return self.semantic.format_results(raw_results)

        if mode == "hybrid":
            candidates = self._hybrid_candidates(
                query,
                where=where,
            )
            return candidates[:n_results]

        if mode == "hybrid_rerank":
            candidates = self._hybrid_candidates(
                query,
                where=where,
            )
            return self.reranker.rerank(
                query,
                candidates,
                top_k=n_results,
            )

        raise ValueError(
            f"Unsupported retrieval mode: {mode}"
        )
