"""API endpoint tests for the ESG Compliance Checker."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def api():
    # Route tests replace the checker and can run without an index or a model.
    with patch("compliance_checker_v2_hybrid.ComplianceCheckerV2"):
        from app.api import routes
        from app.main import app

    with TestClient(app) as client:
        yield client, routes


MOCK_COMPLIANCE_RESULT = {
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
            "reasoning": "A numeric Scope 1 value was found.",
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


def test_health_endpoint_returns_200(api):
    client, _ = api
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_query_returns_200_with_mocked_checker(api, monkeypatch):
    client, routes = api
    mock_checker = MagicMock()
    mock_checker.check_requirement.return_value = MOCK_COMPLIANCE_RESULT

    monkeypatch.setattr(routes, "checker", mock_checker)

    response = client.post(
        "/query",
        json={
            "company_id": "IBK",
            "standard": "gri_305",
            "requirement_id": "305-1",
            "verification_mode": "keyword",
            "n_results": 5,
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["company_id"] == "IBK"
    assert body["requirement_id"] == "305-1"
    assert body["overall_status"] == "partial"
    assert body["element_coverage"][0]["status"] == "covered"

    mock_checker.check_requirement.assert_called_once_with(
        company_id="IBK",
        standard="gri_305",
        req_id="305-1",
        n_results=5,
        verification_mode="keyword",
    )


def test_query_rejects_blank_company_id(api):
    client, _ = api
    response = client.post(
        "/query",
        json={
            "company_id": "",
            "standard": "gri_305",
            "requirement_id": "305-1",
            "verification_mode": "hybrid",
            "n_results": 25,
        },
    )

    assert response.status_code == 422


def test_query_rejects_invalid_verification_mode(api):
    client, _ = api
    response = client.post(
        "/query",
        json={
            "company_id": "IBK",
            "standard": "gri_305",
            "requirement_id": "305-1",
            "verification_mode": "random",
            "n_results": 25,
        },
    )

    assert response.status_code == 422


def test_query_rejects_n_results_above_limit(api):
    client, _ = api
    response = client.post(
        "/query",
        json={
            "company_id": "IBK",
            "standard": "gri_305",
            "requirement_id": "305-1",
            "verification_mode": "keyword",
            "n_results": 1000,
        },
    )

    assert response.status_code == 422


def test_query_returns_404_when_company_has_no_evidence(api, monkeypatch):
    client, routes = api
    mock_checker = MagicMock()
    mock_checker.check_requirement.return_value = {
        "company_id": "NOT_A_REAL_COMPANY",
        "requirement_id": "305-1",
        "requirement_name": "Direct (Scope 1) GHG emissions",
        "overall_status": "missing",
        "verification_mode": "keyword",
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
        "metadata": {
            "n_results": 5,
            "element_count": 1,
        },
    }

    monkeypatch.setattr(routes, "checker", mock_checker)

    response = client.post(
        "/query",
        json={
            "company_id": "NOT_A_REAL_COMPANY",
            "standard": "gri_305",
            "requirement_id": "305-1",
            "verification_mode": "keyword",
            "n_results": 5,
        },
    )

    assert response.status_code == 404

    body = response.json()

    assert body["detail"]["error"] == "company_not_found"
    assert "NOT_A_REAL_COMPANY" in body["detail"]["detail"]


def test_query_returns_404_for_unknown_requirement(api, monkeypatch):
    client, routes = api
    mock_checker = MagicMock()
    mock_checker.check_requirement.side_effect = KeyError("999-9")

    monkeypatch.setattr(routes, "checker", mock_checker)

    response = client.post(
        "/query",
        json={
            "company_id": "IBK",
            "standard": "gri_305",
            "requirement_id": "999-9",
            "verification_mode": "keyword",
            "n_results": 5,
        },
    )

    assert response.status_code == 404

    body = response.json()

    assert body["detail"]["error"] == "requirement_not_found"


def test_query_returns_503_when_external_service_times_out(api, monkeypatch):
    client, routes = api
    mock_checker = MagicMock()
    mock_checker.check_requirement.side_effect = TimeoutError(
        "External service timed out"
    )

    monkeypatch.setattr(routes, "checker", mock_checker)

    response = client.post(
        "/query",
        json={
            "company_id": "IBK",
            "standard": "gri_305",
            "requirement_id": "305-1",
            "verification_mode": "hybrid",
            "n_results": 5,
        },
    )

    assert response.status_code == 503

    body = response.json()

    assert body["detail"]["error"] == "external_service_unavailable"
