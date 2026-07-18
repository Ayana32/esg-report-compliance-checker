"""Unit tests for the compliance service layer."""

from unittest.mock import MagicMock

import pytest

from app.exceptions import (
    CompanyNotFoundError,
    ExternalServiceError,
    RequirementNotFoundError,
)
from app.services import run_compliance_check


VALID_RESULT = {
    "company_id": "IBK",
    "requirement_id": "305-1",
    "requirement_name": "Direct (Scope 1) GHG emissions",
    "overall_status": "partial",
    "verification_mode": "keyword",
    "element_coverage": [
        {
            "element": "Total Scope 1 emissions in tCO2e",
            "status": "covered",
            "confidence": 0.9,
            "reasoning": "Numeric emissions data was found.",
            "verification_method": "keyword",
            "evidence": [
                {
                    "chunk_id": "IBK-report-64-1",
                    "text": "Scope 1 emissions were 1,234 tCO2e.",
                    "page_number": 64,
                }
            ],
        }
    ],
    "evidence": [
        {
            "chunk_id": "IBK-report-64-1",
            "text": "Scope 1 emissions were 1,234 tCO2e.",
            "page_number": 64,
        }
    ],
    "metadata": {
        "n_results": 5,
        "element_count": 1,
    },
}


def test_run_compliance_check_returns_valid_result():
    checker = MagicMock()
    checker.check_requirement.return_value = VALID_RESULT

    result = run_compliance_check(
        checker,
        company_id="IBK",
        standard="gri_305",
        requirement_id="305-1",
        n_results=5,
        verification_mode="keyword",
    )

    assert result["company_id"] == "IBK"
    assert result["overall_status"] == "partial"
    assert len(result["evidence"]) == 1


def test_run_compliance_check_raises_company_not_found():
    checker = MagicMock()
    checker.check_requirement.return_value = {
        "element_coverage": [
            {
                "element": "Total Scope 1 emissions in tCO2e",
                "status": "missing",
                "confidence": 0.3,
                "reasoning": "",
                "verification_method": "keyword",
                "evidence": [],
            }
        ],
        "evidence": [],
    }

    with pytest.raises(CompanyNotFoundError):
        run_compliance_check(
            checker,
            company_id="UNKNOWN",
            standard="gri_305",
            requirement_id="305-1",
            n_results=5,
            verification_mode="keyword",
        )


def test_run_compliance_check_converts_key_error():
    checker = MagicMock()
    checker.check_requirement.side_effect = KeyError("Unknown requirement")

    with pytest.raises(RequirementNotFoundError):
        run_compliance_check(
            checker,
            company_id="IBK",
            standard="gri_305",
            requirement_id="999-9",
            n_results=5,
            verification_mode="keyword",
        )


def test_run_compliance_check_converts_timeout_error():
    checker = MagicMock()
    checker.check_requirement.side_effect = TimeoutError(
        "LLM request timed out"
    )

    with pytest.raises(ExternalServiceError):
        run_compliance_check(
            checker,
            company_id="IBK",
            standard="gri_305",
            requirement_id="305-1",
            n_results=5,
            verification_mode="hybrid",
        )


def test_run_compliance_check_converts_connection_error():
    checker = MagicMock()
    checker.check_requirement.side_effect = ConnectionError(
        "External service unavailable"
    )

    with pytest.raises(ExternalServiceError):
        run_compliance_check(
            checker,
            company_id="IBK",
            standard="gri_305",
            requirement_id="305-1",
            n_results=5,
            verification_mode="hybrid",
        )