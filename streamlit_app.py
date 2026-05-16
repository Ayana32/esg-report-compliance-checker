
"""
streamlit_app.py
ESG Compliance Checker Demo (Clean version)
"""
 
import streamlit as st
from compliance_checker_v2_hybrid import ComplianceCheckerV2
import json
 
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
    
    run_check = st.button("Run Compliance Check", type="primary")
 
# Main area
if run_check:
    st.markdown("---")
    
    # Progress
    with st.spinner("Running compliance check..."):
        checker = ComplianceCheckerV2()
        
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
            "Samsung Electronics": "Samsung"
        }
        company_id = company_map[company]
        
        result = checker.check_requirement(
            company_id=company_id,
            standard="gri_305",       # ← 버그 수정
            req_id="305-1",
            verification_mode=verification_mode.split()[0].lower()
        )
    
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
    
    coverage_pct = covered / total * 100
    st.progress(coverage_pct / 100)
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
        
        expander_title = f"{i}. {elem['element']} - :{color}[{status}]"
        
        with st.expander(expander_title, expanded=(i <= 2)):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write("**Status:**", status)
                st.write("**Confidence:**", f"{elem['confidence']:.2f}")
            
            with col2:
                st.write("**Reasoning:**")
                st.write(elem['reasoning'])
 
                # Page numbers
                pages = elem.get('page_numbers', [])
                if pages:
                    unique_pages = sorted(set(p for p in pages if p))
                    page_str = ", ".join(f"p.{p}" for p in unique_pages)
                    st.caption(f"📄 Evidence found on: {page_str}")
                
                if elem.get('evidence_text'):
                    st.markdown("**Evidence:**")
                    evidence = elem['evidence_text']
                    if len(evidence) > 300:
                        evidence = evidence[:300] + "..."
                    st.info(evidence)
    
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
**Confidence:** {elem['confidence']:.2f}  
**Reasoning:** {elem['reasoning']}
 
"""
            if elem.get('evidence_text'):
                evidence = elem['evidence_text']
                if len(evidence) > 200:
                    evidence = evidence[:200] + "..."
                md_report += f"**Evidence:** {evidence}\n\n"
        
        md_report += "\n---\n*Generated by ESG Compliance Checker*\n"
        
        st.download_button(
            label="Download Markdown Report",
            data=md_report,
            file_name=f"{company_id}_GRI_305-1_report.md",
            mime="text/markdown"
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
st.caption("ESG Compliance Checker | Built with Streamlit + GPT-4 + ChromaDB")