from compliance_checker_v2_hybrid import ComplianceCheckerV2
from requirements_loader import RequirementLoader
from element_query_generator import ElementQueryGenerator


def make_checker():
    checker = object.__new__(ComplianceCheckerV2)
    checker.requirements = RequirementLoader()
    return checker


def make_rows(checker, req_id, statuses):
    req = checker.requirements.get_requirement(
        "gri_305",
        req_id,
    )

    return [
        {
            "element": elem["element"],
            "status": status,
        }
        for elem, status in zip(
            req["required_elements"],
            statuses,
        )
    ]


def test_305_1_aggregation_uses_element_identity_not_yaml_position():
    """
    Regression test:
    305-1 YAML order is not the same as slot a-g order.
    Aggregation must map by element identity, not list position.
    """
    checker = make_checker()

    rows = [
        {
            "element": "Total Scope 1 emissions in tCO2e",
            "status": "covered",
        },
        {
            "element": (
                "Gases included in calculation "
                "(CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)"
            ),
            "status": "covered",
        },
        {
            "element": "Base year for calculation",
            "status": "covered",
        },
        {
            "element": (
                "Consolidation approach "
                "(equity share, financial control, operational control)"
            ),
            "status": "covered",
        },
        {
            "element": "Biogenic CO2 emissions (reported separately)",
            "status": "partial",
        },
        {
            "element": (
                "Emission factors and GWP source "
                "(e.g. IPCC AR5/AR6)"
            ),
            "status": "covered",
        },
        {
            "element": (
                "Standards and methodologies used "
                "(e.g. GHG Protocol, third-party verification)"
            ),
            "status": "covered",
        },
    ]

    assert (
        checker._determine_overall_status(
            rows,
            "305-1",
        )
        == "covered"
    )


def test_305_2_conditional_missing_does_not_block_covered():
    checker = make_checker()

    rows = make_rows(
        checker,
        "305-2",
        [
            "covered",  # gross location-based
            "missing",  # market-based: conditional
            "missing",  # gases: conditional
            "missing",  # base year: conditional
            "covered",  # factors/GWP
            "covered",  # consolidation
            "covered",  # methodology
        ],
    )

    assert (
        checker._determine_overall_status(
            rows,
            "305-2",
        )
        == "covered"
    )


def test_305_2_missing_mandatory_element_returns_partial():
    checker = make_checker()

    rows = make_rows(
        checker,
        "305-2",
        [
            "covered",
            "missing",
            "missing",
            "missing",
            "missing",  # mandatory factor/GWP disclosure
            "covered",
            "covered",
        ],
    )

    assert (
        checker._determine_overall_status(
            rows,
            "305-2",
        )
        == "partial"
    )


def test_305_3_missing_gross_emissions_returns_missing():
    checker = make_checker()

    rows = make_rows(
        checker,
        "305-3",
        [
            "missing",  # anchor
            "missing",
            "covered",
            "covered",
            "missing",
            "covered",
            "covered",
        ],
    )

    assert (
        checker._determine_overall_status(
            rows,
            "305-3",
        )
        == "missing"
    )


def test_scope_specific_emission_factor_queries_do_not_fall_back_generic():
    generator = ElementQueryGenerator()

    assert (
        generator.match_element_to_category(
            "Scope 2 emission factors and GWP source"
        )
        == "scope2_factors"
    )

    assert (
        generator.match_element_to_category(
            "Scope 3 emission factors and GWP source"
        )
        == "scope3_factors"
    )


def test_base_year_queries_do_not_hardcode_year():
    generator = ElementQueryGenerator()

    queries = generator.generate_queries(
        "Scope 3 base year and base-year emissions if applicable"
    )

    for query in queries:
        assert "2019" not in query
        assert "2020" not in query
        assert "2021" not in query


def test_production_retriever_uses_selected_configuration(monkeypatch):
    """
    Production checker must keep the retrieval configuration selected
    by the ablation experiments.
    """
    captured = {}

    class FakeRetriever:
        def __init__(
            self,
            collection_name,
            *,
            candidate_k,
            persist_directory,
            chunk_directory,
        ):
            captured["collection_name"] = collection_name
            captured["candidate_k"] = candidate_k
            captured["persist_directory"] = persist_directory
            captured["chunk_directory"] = chunk_directory

    class FakeVerifier:
        pass

    monkeypatch.setattr(
        "compliance_checker_v2_hybrid.AblationRetriever",
        FakeRetriever,
    )
    monkeypatch.setattr(
        "compliance_checker_v2_hybrid.LLMVerifier",
        FakeVerifier,
    )

    ComplianceCheckerV2()

    assert captured == {
        "collection_name": "reports",
        "candidate_k": 10,
        "persist_directory": "data/chromadb_ablation/500_50",
        "chunk_directory": "data/chunks/ablation/500_50/reports",
    }


def test_global_retrieval_uses_hybrid_rerank(monkeypatch):
    """
    Requirement-level production retrieval should use hybrid retrieval
    followed by cross-encoder reranking.
    """

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def search(
            self,
            query,
            *,
            mode,
            n_results,
            where,
        ):
            self.calls.append(
                {
                    "query": query,
                    "mode": mode,
                    "n_results": n_results,
                    "where": where,
                }
            )
            return []

    checker = object.__new__(ComplianceCheckerV2)
    checker.requirements = RequirementLoader()
    checker.retriever = FakeRetriever()

    class FakeVerifier:
        pass

    checker.llm_verifier = FakeVerifier()

    req = checker.requirements.get_requirement(
        "gri_305",
        "305-1",
    )

    global_query = (
        f"{req['name']} "
        f"{' '.join(req.get('keywords', [])[:4])}"
    )

    checker.retriever.search(
        global_query,
        mode="hybrid_rerank",
        n_results=10,
        where={"company_id": "Schneider"},
    )

    call = checker.retriever.calls[0]

    assert call["mode"] == "hybrid_rerank"
    assert call["n_results"] == 10
    assert call["where"] == {
        "company_id": "Schneider"
    }


def test_element_rerank_returns_top_five(monkeypatch):
    """
    Final element evidence packet must be limited to Top-5 after
    cross-encoder reranking.
    """

    captured = {}

    class FakeReranker:
        def rerank(
            self,
            query,
            candidates,
            *,
            top_k,
        ):
            captured["query"] = query
            captured["candidate_count"] = len(candidates)
            captured["top_k"] = top_k
            return candidates[:top_k]

    candidates = [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"evidence {i}",
            "metadata": {},
        }
        for i in range(10)
    ]

    reranker = FakeReranker()

    result = reranker.rerank(
        "Total Scope 1 emissions in tCO2e",
        candidates,
        top_k=5,
    )

    assert captured["candidate_count"] == 10
    assert captured["top_k"] == 5
    assert len(result) == 5
