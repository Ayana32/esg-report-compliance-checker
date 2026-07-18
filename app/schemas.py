"""Pydantic request and response models for the ESG Compliance Checker API."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


VerificationMode = Literal["hybrid", "llm", "keyword"]
ComplianceStatus = Literal["covered", "partial", "missing"]


class HealthResponse(BaseModel):
    """Response returned by the health-check endpoint."""

    status: Literal["healthy"] = "healthy"

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
            }
        }
    )


class ComplianceRequest(BaseModel):
    """Input required to run a compliance check."""

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Internal identifier for the company report collection.",
        examples=["IBK"],
    )

    standard: str = Field(
        default="gri_305",
        min_length=1,
        max_length=50,
        description="ESG reporting standard identifier.",
        examples=["gri_305"],
    )

    requirement_id: str = Field(
        default="305-1",
        min_length=1,
        max_length=50,
        description="Requirement identifier within the selected standard.",
        examples=["305-1"],
    )

    verification_mode: VerificationMode = Field(
        default="hybrid",
        description=(
            "Verification strategy: hybrid combines keyword and LLM checks; "
            "llm uses only LLM verification; keyword uses lexical matching."
        ),
        examples=["hybrid"],
    )

    n_results: int = Field(
        default=25,
        ge=1,
        le=50,
        description="Maximum number of retrieved evidence chunks per element.",
        examples=[25],
    )

    @field_validator("company_id", "standard", "requirement_id")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        """Reject values containing only whitespace."""

        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("must not be blank")

        return cleaned_value

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "company_id": "IBK",
                "standard": "gri_305",
                "requirement_id": "305-1",
                "verification_mode": "hybrid",
                "n_results": 25,
            }
        },
    )


class EvidenceItem(BaseModel):
    """Evidence selected to support an element-level compliance decision."""

    chunk_id: str = Field(
        description="Unique identifier of the retrieved evidence chunk."
    )

    text: str = Field(
        description="Evidence text extracted from the sustainability report."
    )

    page_number: int | None = Field(
        default=None,
        description="Source report page number, when available.",
    )


class ElementCoverage(BaseModel):
    """Compliance result for one required disclosure element."""

    element: str = Field(
        description="Name of the disclosure element being evaluated."
    )

    status: ComplianceStatus = Field(
        description="Compliance classification for this element."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score assigned to the decision.",
    )

    reasoning: str = Field(
        description="Explanation supporting the compliance decision."
    )

    verification_method: str = Field(
        description="Method used to produce the final element decision."
    )

    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence chunks directly supporting the decision.",
    )


class ComplianceResponse(BaseModel):
    """Structured result returned by the compliance-check endpoint."""

    company_id: str
    requirement_id: str
    requirement_name: str
    overall_status: ComplianceStatus
    verification_mode: VerificationMode

    element_coverage: list[ElementCoverage] = Field(
        description="Element-by-element compliance results."
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional execution metadata such as retrieval counts "
            "or processing duration."
        ),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "company_id": "IBK",
                "requirement_id": "305-1",
                "requirement_name": "Direct GHG emissions (Scope 1)",
                "overall_status": "partial",
                "verification_mode": "hybrid",
                "element_coverage": [
                    {
                        "element": "Total Scope 1 emissions in tCO2e",
                        "status": "covered",
                        "confidence": 0.9,
                        "reasoning": (
                            "The report explicitly provides a numeric "
                            "Scope 1 emissions value."
                        ),
                        "verification_method": "hybrid-agree",
                        "evidence": [
                            {
                                "chunk_id": "IBK-report-64-1",
                                "text": (
                                    "Direct GHG emissions (Scope 1) "
                                    "were reported in tCO2eq."
                                ),
                                "page_number": 64,
                            }
                        ],
                    }
                ],
                "metadata": {
                    "n_results": 25,
                },
            }
        }
    )

class ErrorResponse(BaseModel):
    """Standard error response returned by the API."""

    error: str = Field(
        description="Stable machine-readable error identifier."
    )

    detail: str = Field(
        description="Human-readable explanation of the error."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "company_not_found",
                "detail": "No ESG report was found for company 'UNKNOWN'.",
            }
        }
    )