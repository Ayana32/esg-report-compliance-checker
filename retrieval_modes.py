"""Common interface for retrieval ablation modes."""

from typing import Any, Literal

from hybrid_search import HybridRetriever
from keyword_search import KeywordRetriever
from semantic_search import SemanticRetriever


RetrievalMode = Literal[
    "bm25",
    "semantic",
    "hybrid",
]


class AblationRetriever:
    """Run multiple retrieval strategies through one interface."""

    def __init__(self, collection_name: str = "reports") -> None:
        self.collection_name = collection_name

        self.keyword = KeywordRetriever(collection_name)
        self.semantic = SemanticRetriever(collection_name)
        self.hybrid = HybridRetriever(collection_name)

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
            return self.hybrid.search(
                query=query,
                n_results=n_results,
                where=where,
            )

        raise ValueError(f"Unsupported retrieval mode: {mode}")
