"""Mapping utilities between the RAG pipeline and API response models."""

from typing import Any

from app.schemas import (
    ComplianceResponse,
    ElementCoverage,
    EvidenceItem,
)


def map_compliance_result(
    result: dict[str, Any],
    *,
    n_results: int,
) -> ComplianceResponse:
    """Convert a raw compliance-check result into an API response model."""

    elements: list[ElementCoverage] = []

    for raw_element in result.get("element_coverage", []):
        evidence_items = [
            EvidenceItem(
                chunk_id=item.get("chunk_id", ""),
                text=item.get("text", ""),
                page_number=item.get("page_number"),
            )
            for item in raw_element.get("evidence", [])
        ]

        elements.append(
            ElementCoverage(
                element=raw_element["element"],
                status=raw_element["status"],
                confidence=raw_element["confidence"],
                reasoning=raw_element.get("reasoning", ""),
                verification_method=raw_element.get(
                    "verification_method",
                    result.get("verification_mode", "hybrid"),
                ),
                evidence=evidence_items,
            )
        )

    return ComplianceResponse(
        company_id=result["company_id"],
        requirement_id=result["requirement_id"],
        requirement_name=result["requirement_name"],
        overall_status=result["overall_status"],
        verification_mode=result["verification_mode"],
        element_coverage=elements,
        metadata={
            "n_results": n_results,
            "element_count": len(elements),
        },
    )