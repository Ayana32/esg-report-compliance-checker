"""
compliance_checker_v2_hybrid.py (v3)
Enhanced compliance checker — partial signal from LLM verifier now properly handled.

Changes from v2:
- _check_element_with_llm() reads result['status'] directly instead of result['covered'] bool
- partial from LLM verifier is no longer collapsed into missing
"""

from hybrid_search import HybridRetriever
from requirements_loader import RequirementLoader
from llm_verifier import LLMVerifier
from typing import Dict, List


class ComplianceCheckerV2:
    """Enhanced compliance checker with improved hybrid logic"""

    def __init__(self):
        self.requirements = RequirementLoader()
        self.retriever    = HybridRetriever(collection_name="reports")
        self.llm_verifier = LLMVerifier()

    def check_requirement(
        self,
        company_id: str,
        standard: str,
        req_id: str,
        n_results: int = 25,
        verification_mode: str = "hybrid"
    ) -> Dict:
        """
        Check requirement with configurable verification mode.

        Args:
            company_id: Company ID
            standard: Standard name (e.g. "gri_305")
            req_id: Requirement ID (e.g. "305-1")
            n_results: Number of chunks to retrieve
            verification_mode: "keyword", "llm", or "hybrid"
        """

        req      = self.requirements.get_requirement(standard, req_id)
        keywords = req.get('keywords', [])

        # Global evidence for fallback
        global_query    = f"{req['name']} {' '.join(keywords[:4])}"
        global_evidence = self.retriever.search(
            global_query,
            n_results=n_results,
            where={"company_id": company_id}
        )

        from element_query_generator import ElementQueryGenerator
        query_gen = ElementQueryGenerator()

        element_coverage = []

        for elem_data in req['required_elements']:
            element = elem_data['element']

            # Per-element evidence retrieval
            elem_queries = query_gen.generate_queries(element)
            if elem_queries:
                elem_evidence = []
                seen_ids = set()
                for eq in elem_queries[:3]:
                    results = self.retriever.search(
                        eq, n_results=10, where={"company_id": company_id}
                    )
                    for r in results:
                        if r.get('chunk_id') not in seen_ids:
                            elem_evidence.append(r)
                            seen_ids.add(r.get('chunk_id'))
                for r in global_evidence:
                    if r.get('chunk_id') not in seen_ids:
                        elem_evidence.append(r)
                        seen_ids.add(r.get('chunk_id'))
                evidence = elem_evidence[:n_results]
            else:
                evidence = global_evidence

            if verification_mode == "keyword":
                covered = self._check_element_with_keywords(element, evidence, keywords)
            elif verification_mode == "llm":
                covered = self._check_element_with_llm(element, evidence, req['name'])
            elif verification_mode == "hybrid":
                covered = self._check_element_hybrid(element, evidence, req['name'], keywords)
            else:
                raise ValueError(f"Unknown mode: {verification_mode}")

            element_coverage.append({
                'element'            : element,
                'status'             : covered['status'],
                'evidence_chunks'    : covered.get('chunks', []),
                'page_numbers'       : covered.get('page_numbers', []),
                'confidence'         : covered['confidence'],
                'reasoning'          : covered.get('reasoning', ''),
                'verification_method': covered.get('method', verification_mode)
            })

        overall_status = self._determine_overall_status(element_coverage)

        return {
            'company_id'        : company_id,
            'requirement_id'    : req_id,
            'requirement_name'  : req['name'],
            'overall_status'    : overall_status,
            'element_coverage'  : element_coverage,
            'evidence'          : evidence[:3],
            'verification_mode' : verification_mode
        }

    # ── Hybrid verification ────────────────────────────────────────────────────

    def _check_element_hybrid(
        self,
        element: str,
        evidence: List[Dict],
        requirement_name: str,
        keywords: List[str]
    ) -> Dict:
        """Keyword for recall, LLM for precision."""

        keyword_result = self._check_element_with_keywords(element, evidence, keywords)

        keyword_chunks = [
            chunk for chunk in evidence
            if chunk.get('chunk_id') in keyword_result.get('chunks', [])
        ]

        if not keyword_chunks:
            llm_evidence = evidence[:10]
        else:
            llm_evidence = keyword_chunks[:5] + evidence[:5]
            seen = set()
            llm_evidence = [
                c for c in llm_evidence
                if not (c.get('chunk_id') in seen or seen.add(c.get('chunk_id')))
            ][:10]

        llm_result   = self._check_element_with_llm(element, llm_evidence, requirement_name)
        final_result = self._combine_results(keyword_result, llm_result, element)

        return final_result

    def _combine_results(
        self,
        keyword_result: Dict,
        llm_result: Dict,
        element_name: str
    ) -> Dict:
        """Combine keyword and LLM results with status-first logic."""

        kw_status  = keyword_result['status']
        llm_status = llm_result['status']

        print(f"\n  DEBUG - {element_name[:40]}...")
        print(f"    Keyword: {kw_status} (conf: {keyword_result['confidence']:.2f})")
        print(f"    LLM:     {llm_status} (conf: {llm_result['confidence']:.2f})")

        STATUS_RANK = {'covered': 2, 'partial': 1, 'missing': 0}

        if kw_status == llm_status:
            final_status = kw_status
            final_conf   = max(keyword_result['confidence'], llm_result['confidence'])
            print(f"    → AGREE: {final_status} (conf: {final_conf:.2f})")
            return {
                'status'      : final_status,
                'confidence'  : final_conf,
                'reasoning'   : llm_result.get('reasoning', 'Both methods agree'),
                'chunks'      : keyword_result.get('chunks', []),
                'page_numbers': llm_result.get('page_numbers', []),
                'method'      : 'hybrid-agree'
            }

        print(f"    → DISAGREE")

        if STATUS_RANK[kw_status] > STATUS_RANK[llm_status]:
            # keyword=partial + LLM=missing → trust LLM (keyword false positive)
            if kw_status == 'partial' and llm_status == 'missing':
                print(f"    → Trust LLM: missing (keyword partial likely false positive)")
                return {
                    'status'      : 'missing',
                    'confidence'  : llm_result['confidence'],
                    'reasoning'   : f"LLM found no evidence ({llm_status}); keyword match likely incidental.",
                    'chunks'      : [],
                    'page_numbers': [],
                    'method'      : 'hybrid-llm-override'
                }
            # keyword=covered + LLM=partial → downgrade to partial
            print(f"    → Downgrade to partial (keyword={kw_status} > llm={llm_status})")
            return {
                'status'      : 'partial',
                'confidence'  : 0.7,
                'reasoning'   : f"Keyword found evidence ({kw_status}), but LLM verification inconclusive ({llm_status}). Needs review.",
                'chunks'      : keyword_result.get('chunks', []),
                'page_numbers': llm_result.get('page_numbers', []),
                'method'      : 'hybrid-downgrade'
            }
        else:
            print(f"    → Upgrade to {llm_status} (llm={llm_status} > keyword={kw_status})")
            return {
                'status'      : llm_status,
                'confidence'  : llm_result['confidence'],
                'reasoning'   : llm_result.get('reasoning', 'LLM found evidence missed by keywords'),
                'chunks'      : llm_result.get('chunks', []),
                'page_numbers': llm_result.get('page_numbers', []),
                'method'      : 'hybrid-upgrade'
            }
    def _check_element_with_llm(
        self,
        element: str,
        evidence: List[Dict],
        requirement_name: str
    ) -> Dict:
        """LLM-based element verification — reads status field directly."""

        result = self.llm_verifier.verify_element(
            element=element,
            evidence_chunks=evidence,
            requirement_name=requirement_name,
            max_chunks=10
        )

        # Read status directly from v3 verifier (covered / partial / missing)
        status = result.get('status', 'missing')

        chunks = [e.get('chunk_id', '') for e in evidence[:10]]

        return {
            'status'      : status,
            'confidence'  : result['confidence'],
            'reasoning'   : result['reasoning'],
            'page_numbers': result.get('page_numbers', []),
            'chunks'      : chunks if status != 'missing' else []
        }

    def _check_element_with_keywords(
        self,
        element: str,
        evidence: List[Dict],
        keywords: List[str]
    ) -> Dict:
        """Keyword-based element verification (baseline)."""

        element_lower  = element.lower()
        covered_chunks = []
        element_words  = [w for w in element_lower.split() if len(w) > 3]

        for chunk in evidence:
            text    = chunk.get('text', '').lower()
            matches = sum(1 for word in element_words if word in text)
            if matches >= 2:
                covered_chunks.append(chunk.get('chunk_id', ''))

        if len(covered_chunks) >= 2:
            status     = "covered"
            confidence = 0.9
        elif len(covered_chunks) == 1:
            status     = "partial"
            confidence = 0.6
        else:
            status     = "missing"
            confidence = 0.3

        return {
            'status'    : status,
            'chunks'    : covered_chunks,
            'confidence': confidence
        }

    def _determine_overall_status(self, element_coverage: List[Dict]) -> str:
        """
        Determine overall requirement coverage using ground_truth.py rules:
        - covered : no missing + at least 4 covered + all critical slots covered
        - partial : some evidence but at least one critical slot partial/missing
        - missing : majority of slots missing
        """
        from ground_truth import CRITICAL_SLOTS, SLOTS

        slot_keys = list(SLOTS.keys())
        total     = len(element_coverage)

        # Build slot_key → status map by position
        slot_status = {}
        for i, elem in enumerate(element_coverage):
            if i < len(slot_keys):
                slot_status[slot_keys[i]] = elem['status']

        covered_count = sum(1 for s in slot_status.values() if s == 'covered')
        missing_count = sum(1 for s in slot_status.values() if s == 'missing')

        # missing: majority missing
        if missing_count > total / 2:
            return "missing"

        # covered: no missing + 4+ covered + all critical covered
        all_critical_covered = all(
            slot_status.get(s) == 'covered' for s in CRITICAL_SLOTS
        )
        if missing_count == 0 and covered_count >= 4 and all_critical_covered:
            return "covered"

        return "partial"


# ── Main test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 80)
    print("COMPLIANCE CHECKER V3 - DEBUG MODE")
    print("=" * 80)

    checker = ComplianceCheckerV2()

    for company in ["IBK", "Schneider", "Heathrow"]:
        print(f"\n{'='*80}")
        print(f"Running: {company}")
        print(f"{'='*80}")

        result = checker.check_requirement(
            company_id=company,
            standard="gri_305",
            req_id="305-1",
            verification_mode="hybrid"
        )

        covered = sum(1 for e in result['element_coverage'] if e['status'] == 'covered')
        partial = sum(1 for e in result['element_coverage'] if e['status'] == 'partial')
        missing = sum(1 for e in result['element_coverage'] if e['status'] == 'missing')

        print(f"\n  Overall : {result['overall_status'].upper()}")
        print(f"  Elements: covered={covered}, partial={partial}, missing={missing}")
