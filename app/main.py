"""FastAPI entry point for the ESG Compliance Checker API."""
import logging

from fastapi import FastAPI
from app.api.routes import router as compliance_router
from app.schemas import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

app = FastAPI(
    title="ESG Compliance Checker API",
    description=(
        "A production-oriented REST API for evaluating corporate ESG "
        "disclosures against GRI requirements. The service combines "
        "ChromaDB semantic retrieval, BM25 keyword retrieval, reciprocal "
        "rank fusion, rule-based checks, and LLM verification to return "
        "structured element-level compliance decisions with supporting "
        "evidence and page references."
    ),
    version="1.0.0",
    contact={
        "name": "Misun Kim",
        "url": "https://github.com/Ayana32",
    },
)

app.include_router(compliance_router)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check API health",
    description="Returns the current health status of the API.",
)
def health_check() -> HealthResponse:
    """Return the current health status of the API."""

    return HealthResponse(status="healthy")