"""Read-only MCP access to ESG report evidence."""

import argparse
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field, StringConstraints

from app.schemas import EvidenceItem
from keyword_search import KeywordRetriever


class EvidenceSearchResult(BaseModel):
    company_id: str
    evidence: list[EvidenceItem]


def create_server(retriever: KeywordRetriever, *, port: int = 8001) -> FastMCP:
    """Expose the existing company-filtered BM25 retriever as one MCP tool."""
    server = FastMCP(
        "ESG Report Evidence",
        host="127.0.0.1",
        port=port,
        stateless_http=True,
        json_response=True,
    )

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def search_esg_evidence(
        company_id: Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)
        ],
        disclosure_item: Annotated[
            str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)
        ],
        limit: Annotated[int, Field(ge=1, le=5)] = 5,
    ) -> EvidenceSearchResult:
        """Search indexed reports by exact company ID and disclosure text.

        Returns candidate passages and source pages. An empty list means no
        positive-scoring lexical match. This tool does not assess compliance.
        """
        rows = retriever.search(
            disclosure_item,
            n_results=limit,
            where={"company_id": company_id},
        )
        return EvidenceSearchResult(
            company_id=company_id,
            evidence=[
                EvidenceItem(
                    chunk_id=row["chunk_id"],
                    text=row["text"],
                    page_number=row["metadata"].get("page_num"),
                )
                for row in rows
                if row["score"] > 0
            ],
        )

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-directory", type=Path, default=Path("data/chunks/reports"))
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if not any(args.chunk_directory.glob("*.jsonl")):
        parser.error(f"No report chunks found in {args.chunk_directory}.")
    retriever = KeywordRetriever("reports", chunk_directory=str(args.chunk_directory))
    create_server(retriever, port=args.port).run(transport="streamable-http")


if __name__ == "__main__":
    main()
