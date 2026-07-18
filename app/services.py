"""Application service layer for compliance checks."""

from typing import Any

from app.exceptions import (
    CompanyNotFoundError,
    ExternalServiceError,
    NoEvidenceFoundError,
    RequirementNotFoundError,
)
from compliance_checker_v2_hybrid import ComplianceCheckerV2


def run_compliance_check(
    checker: ComplianceCheckerV2,
    *,
    company_id: str,
    standard: str,
    requirement_id: str,
    n_results: int,
    verification_mode: str,
) -> dict[str, Any]:
    """Run the compliance pipeline and translate expected failures."""

    try:
        result = checker.check_requirement(
            company_id=company_id,
            standard=standard,
            req_id=requirement_id,
            n_results=n_results,
            verification_mode=verification_mode,
        )

    except KeyError as exc:
        # RequirementLoader commonly exposes missing dictionary keys
        # as KeyError. Translate this into a domain-level error.
        raise RequirementNotFoundError(
            f"Standard '{standard}' or requirement "
            f"'{requirement_id}' was not found."
        ) from exc

    except FileNotFoundError as exc:
        raise RequirementNotFoundError(
            "A required compliance configuration or report file "
            "could not be found."
        ) from exc

    except TimeoutError as exc:
        raise ExternalServiceError(
            "The compliance check timed out while contacting "
            "an external service."
        ) from exc

    except ConnectionError as exc:
        raise ExternalServiceError(
            "An external service required for verification "
            "is currently unavailable."
        ) from exc

    # A successful execution with no retrieved evidence should not
    # masquerade as a valid compliance result.
    global_evidence = result.get("evidence", [])
    elements = result.get("element_coverage", [])

    if not global_evidence:
        raise CompanyNotFoundError(
            f"No ESG report was found for company '{company_id}'."
        )

    if not elements:
        raise NoEvidenceFoundError(
            "No evidence was retrieved for the requested requirement."
        )

    return result