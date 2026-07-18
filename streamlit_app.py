
"""
streamlit_app.py
ESG Compliance Checker Frontend
"""

import json
import os

from numpy.strings import index
import requests
import streamlit as st
from compliance_checker_v2_hybrid import ComplianceCheckerV2

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def request_compliance_check(payload: dict) -> dict:
    """Send a compliance-check request to the FastAPI backend."""

    response = requests.post(
        f"{API_BASE_URL}/query",
        json=payload,
        timeout=180,
    )

    if response.status_code >= 400:
        try:
            error_body = response.json()
        except ValueError:
            error_body = {}

        detail = error_body.get("detail", "The request failed.")

        if isinstance(detail, dict):
            message = detail.get("detail", "The request failed.")
        elif isinstance(detail, list):
            message = "; ".join(
                item.get("msg", "Validation error")
                for item in detail
            )
        else:
            message = str(detail)

        raise RuntimeError(message)

    return response.json()

st.set_page_config(
    page_title="ESG Compliance Checker",
    page_icon="📊",
    layout="wide"
)

# Header
st.title("ESG Compliance Checker")
st.markdown("**Automated ESG reporting compliance verification using RAG + LLM**")

# Sidebar
with st.sidebar:
    st.header("Configuration")

    company = st.selectbox(
        "Select Company",
        ["IBK Bank", "Schneider Electric", "Heathrow Airport", "Kepco", "Shinhan Bank", "HSBC", "Standard Chartered", "Siemens", "Hyundai Motor", "Incheon Airport", "Samsung Electronics"]
    )

    standard = st.selectbox(
        "ESG Standard",
        ["GRI 305-1 (Scope 1 Emissions)"]
    )

    verification_mode = st.radio(
        "Verification Mode",
        ["Hybrid (Keyword + LLM)", "LLM Only", "Keyword Only"]
    )

    n_results = st.slider(
        "Number of Evidence Results to Retrieve",
        min_value=5,
        max_value=50,
        value=25,
        step=5
    )

    run_check = st.button("Run Compliance Check", type="primary")

company_map = {
    "IBK Bank": "IBK",
    "Schneider Electric": "Schneider",
    "Heathrow Airport": "Heathrow",
    "Kepco": "Kepco",
    "Shinhan Bank": "Shinhan",
    "HSBC": "HSBC",
    "Standard Chartered": "SC",
    "Siemens": "Siemens",
    "Hyundai Motor": "Hyundai",
    "Incheon Airport": "Incheon",
    "Samsung Electronics": "Samsung",
}

verification_mode_map = {
    "Hybrid (Keyword + LLM)": "hybrid",
    "LLM Only": "llm",
    "Keyword Only": "keyword",
}

# Main area
if run_check:
    st.markdown("---")

    company_id = company_map[company]

    payload = {
        "company_id": company_id,
        "standard": "gri_305",
        "requirement_id": "305-1",
        "verification_mode": verification_mode_map[verification_mode],
        "n_results": n_results,
    }

    try:
        # Progress
        with st.spinner("Running compliance check..."):
            result = request_compliance_check(payload)

    except requests.Timeout:
        st.error(
            "The compliance request timed out. "
            "Try keyword mode or reduce the number of retrieved chunks."
        )
        st.stop()

    except requests.ConnectionError:
        st.error(
            "Could not connect to the FastAPI backend. "
            "Make sure the API is running on port 8000."
        )
        st.stop()

    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    except requests.RequestException:
        st.error("An unexpected network error occurred.")
        st.stop()

    # Results
    st.success("Compliance check complete!")

    # ── Overall status badge ───────────────────────────────────────────────────
    overall = result.get('overall_status', 'unknown')

    overall_color = {
        'covered': 'green',
        'partial': 'orange',
        'missing': 'red'
    }.get(overall, 'gray')

    overall_label = {
        'covered': '✅ COVERED',
        'partial': '⚠️ PARTIAL',
        'missing': '❌ MISSING'
    }.get(overall, 'UNKNOWN')

    st.markdown(f"## Overall Status: :{overall_color}[{overall_label}]")
    st.caption("covered = all critical slots satisfied | partial = needs expert review | missing = insufficient disclosure")

    st.markdown("---")

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.markdown("### Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)

    total   = len(result['element_coverage'])
    covered = sum(1 for e in result['element_coverage'] if e['status'] == 'covered')
    partial = sum(1 for e in result['element_coverage'] if e['status'] == 'partial')
    missing = sum(1 for e in result['element_coverage'] if e['status'] == 'missing')

    with col1:
        st.metric("Total Elements", total)
    with col2:
        st.metric("Covered", covered, f"{covered/total*100:.0f}%")
    with col3:
        st.metric("Partial", partial)
    with col4:
        st.metric("Missing", missing)

    coverage_pct = covered / total * 100 if total else 0

    st.progress(coverage_pct / 100 if total else 0)
    st.caption(f"Slot Coverage: {coverage_pct:.1f}%")

    # ── Element-by-element results ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Element-by-Element Results")

    status_map = {
        'covered': 'COVERED',
        'partial': 'PARTIAL',
        'missing': 'MISSING'
    }
    status_color = {
        'covered': 'green',
        'partial': 'orange',
        'missing': 'red'
    }

    for i, elem in enumerate(result['element_coverage'], 1):
        status = status_map.get(elem['status'], 'UNKNOWN')
        color  = status_color.get(elem['status'], 'gray')

        expander_title = (
            f"{i}. {elem.get('element', 'Unknown element')} "
            f"- :{color}[{status}]"
        )

        with st.expander(
            expander_title,
            expanded=i <= 2,
        ):
            col1, col2 = st.columns([1, 3])

            with col1:
                st.write("**Status:**", status)
                st.write(
                    "**Confidence:**",
                    f"{elem.get('confidence', 0):.2f}",
                )
                st.write(
                    "**Method:**",
                    elem.get("verification_method", "unknown"),
                )

            with col2:
                st.write("**Reasoning:**")
                st.write(
                    elem.get(
                        "reasoning",
                        "No reasoning was provided.",
                    )
                )

                evidence_items = elem.get("evidence", [])

                if evidence_items:
                    pages = sorted(
                        {
                            item.get("page_number")
                            for item in evidence_items
                            if item.get("page_number") is not None
                        }
                    )

                    if pages:
                        page_text = ", ".join(
                            f"p.{page}" for page in pages
                        )
                        st.caption(
                            f"Evidence found on: {page_text}"
                        )

                    st.markdown("**Evidence:**")

                    for evidence_index, evidence in enumerate(
                        evidence_items,
                        1,
                    ):
                        evidence_text = evidence.get("text", "")

                        if len(evidence_text) > 500:
                            evidence_text = (
                                evidence_text[:500] + "..."
                            )

                        chunk_id = evidence.get(
                            "chunk_id",
                            "unknown chunk",
                        )
                        page_number = evidence.get("page_number")

                        label = f"Evidence {evidence_index}"

                        if page_number is not None:
                            label += f" · Page {page_number}"

                        st.caption(f"{label} · {chunk_id}")
                        st.info(evidence_text)

                else:
                    st.caption(
                        "No supporting evidence was selected."
                    )

    # ── Export ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Export Results")

    col1, col2 = st.columns(2)

    with col1:
        json_str = json.dumps(result, indent=2, ensure_ascii=False)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"{company_id}_GRI_305-1_results.json",
            mime="application/json"
        )

    with col2:
        md_report = f"""# ESG Compliance Report

**Company:** {company}
**Standard:** GRI 305-1 (Direct GHG Emissions)
**Overall Status:** {overall_label}

## Summary

- Total Elements: {total}
- Covered: {covered} ({coverage_pct:.1f}%)
- Partial: {partial}
- Missing: {missing}

## Detailed Results

"""
        for i, elem in enumerate(result['element_coverage'], 1):
            status = status_map.get(elem['status'], 'UNKNOWN')
            md_report += f"""
### {i}. {elem['element']}

**Status:** {status}
**Confidence:** {elem.get('confidence', 0):.2f}
**Method:** {elem.get('verification_method', 'unknown')}
**Reasoning:** {elem.get('reasoning', 'No reasoning was provided.')}

"""
            evidence_items = elem.get("evidence", [])

            for evidence_index, evidence in enumerate(
                evidence_items,
                1,
            ):
                md_report += (
                    f"**Evidence {evidence_index}"
                )

                if evidence.get("page_number") is not None:
                    md_report += (
                        f" — Page {evidence['page_number']}"
                    )

                md_report += (
                    f":** {evidence.get('text', '')}\n\n"
                )

        md_report += (
            "\n---\n"
            "*Generated by ESG Compliance Checker*\n"
        )

        st.download_button(
            label="Download Markdown Report",
            data=md_report,
            file_name=(
                f"{company_id}_GRI_305-1_report.md"
            ),
            mime="text/markdown",
        )

else:
    st.info("Select a company and click 'Run Compliance Check' to begin.")

    st.markdown("### About")
    st.markdown("""
    This tool automatically verifies ESG reporting compliance by:

    1. **Retrieving** relevant sections from sustainability reports using hybrid search (Semantic + BM25)
    2. **Extracting** evidence using pattern matching and NLP
    3. **Verifying** compliance using GPT-4 with confidence scoring

    **Technology Stack:**
    - Retrieval: ChromaDB (vector store) + BM25 (keyword search)
    - Reranking: Cross-encoder models
    - Verification: GPT-4 with structured prompts
    - Evidence: Pattern matching + sentence extraction
    """)

    st.markdown("### Example Output")
    st.markdown("""
    For each GRI 305-1 element, the system provides:
    - **Status:** Covered / Partial / Missing
    - **Confidence Score:** 0.0 - 1.0
    - **Reasoning:** Explanation of the decision
    - **Evidence:** Relevant text excerpts from the report
    """)

# Footer
st.markdown("---")
st.caption("ESG Compliance Checker | Built with Streamlit frontend + FastAPI backend | Powered by LLMs and hybrid search")
