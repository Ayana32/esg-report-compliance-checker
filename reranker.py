"""Cross-encoder reranking for retrieval ablation."""

from typing import Any

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    ) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Rerank candidate results and return the top-k items."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if not results:
            return []

        pairs = [
            [query, result.get("text", "")]
            for result in results
        ]

        scores = self.model.predict(pairs)

        reranked = []

        for result, score in zip(results, scores):
            item = result.copy()
            item["reranker_score"] = float(score)
            reranked.append(item)

        reranked.sort(
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

        for rank, item in enumerate(reranked, 1):
            item["rank"] = rank

        return reranked[:top_k]
