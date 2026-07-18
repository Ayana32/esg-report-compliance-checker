"""API routes for the ESG Compliance Checker."""
import logging

from fastapi import APIRouter, HTTPException, status

from app.exceptions import (
    CompanyNotFoundError,
    ExternalServiceError,
    NoEvidenceFoundError,
    RequirementNotFoundError,
)

from app.mappers import map_compliance_result
from app.schemas import (
    ComplianceRequest,
    ComplianceResponse,
    ErrorResponse,
)

from app.services import run_compliance_check
from compliance_checker_v2_hybrid import ComplianceCheckerV2

logger = logging.getLogger(__name__)

router = APIRouter()


# Temporary process-level instance.
# This will be moved to FastAPI lifespan and dependency injection later.
checker = ComplianceCheckerV2()


@router.post(
    "/query",
    response_model=ComplianceResponse,
    tags=["Compliance"],
    summary="Run an ESG compliance check",
    responses={
        400: {
            "model": ErrorResponse,
            "description": "The request is valid JSON but cannot be processed.",
        },
        404: {
            "model": ErrorResponse,
            "description": "The requested company or requirement was not found.",
        },
        503: {
            "model": ErrorResponse,
            "description": "An external dependency is unavailable.",
        },
        500: {
            "model": ErrorResponse,
            "description": "An unexpected internal error occurred.",
        },
    },
)
def run_compliance_query(
    request: ComplianceRequest,
) -> ComplianceResponse:
    """Run the RAG and LLM compliance-checking pipeline."""
    
    try:
        raw_result = run_compliance_check(
            checker,
            company_id=request.company_id,
            standard=request.standard,
            requirement_id=request.requirement_id,
            n_results=request.n_results,
            verification_mode=request.verification_mode,
        )

        return map_compliance_result(
            raw_result,
            n_results=request.n_results,
        )

    except CompanyNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "company_not_found",
                "detail": str(exc),
            },
        ) from exc

    except RequirementNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "requirement_not_found",
                "detail": str(exc),
            },
        ) from exc

    except NoEvidenceFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "no_evidence_found",
                "detail": str(exc),
            },
        ) from exc

    except ExternalServiceError as exc:
        logger.warning(
            "External service failure during compliance check: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "external_service_unavailable",
                "detail": str(exc),
            },
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_request",
                "detail": str(exc),
            },
        ) from exc

    except Exception as exc:
        # Full traceback is logged on the server, but never returned
        # to the API caller.
        logger.exception(
            "Unexpected failure while processing company=%s, "
            "standard=%s, requirement=%s",
            request.company_id,
            request.standard,
            request.requirement_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "internal_server_error",
                "detail": (
                    "An unexpected error occurred while processing "
                    "the compliance check."
                ),
            },
        ) from exc