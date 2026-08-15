"""
compliance_checker_v2_hybrid.py (v3)
Enhanced compliance checker — partial signal from LLM verifier now properly handled.

Changes from v2:
- _check_element_with_llm() reads result['status'] directly instead of result['covered'] bool
- partial from LLM verifier is no longer collapsed into missing
"""

from retrieval_modes import AblationRetriever
from requirements_loader import RequirementLoader
from llm_verifier import LLMVerifier
from typing import Dict, List


class ComplianceCheckerV2:
    """Enhanced compliance checker with improved hybrid logic"""

    def __init__(self):
        self.requirements = RequirementLoader()
        self.retriever = AblationRetriever(
            collection_name="reports",
            candidate_k=10,
            persist_directory="data/chromadb_ablation/500_50",
            chunk_directory="data/chunks/ablation/500_50/reports",
        )
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
            mode="hybrid_rerank",
            n_results=n_results,
            where={"company_id": company_id},
        )

        from element_query_generator import ElementQueryGenerator
        query_gen = ElementQueryGenerator()

        element_coverage = []

        for elem_data in req['required_elements']:
            element = elem_data['element']

            # Per-element evidence retrieval
            elem_queries = query_gen.generate_queries(element)
            if elem_queries:
                # Use multilingual queries for candidate recall, then perform
                # one final cross-encoder rerank over the merged candidate pool.
                elem_candidates = []
                seen_ids = set()

                for eq in elem_queries[:3]:
                    results = self.retriever.search(
                        eq,
                        mode="hybrid",
                        n_results=10,
                        where={"company_id": company_id},
                    )

                    for r in results:
                        chunk_id = r.get('chunk_id')
                        if chunk_id and chunk_id not in seen_ids:
                            elem_candidates.append(r)
                            seen_ids.add(chunk_id)

                # Add requirement-level evidence as fallback candidates.
                for r in global_evidence:
                    chunk_id = r.get('chunk_id')
                    if chunk_id and chunk_id not in seen_ids:
                        elem_candidates.append(r)
                        seen_ids.add(chunk_id)

                evidence = self.retriever.reranker.rerank(
                    element,
                    elem_candidates,
                    top_k=5,
                )
            else:
                evidence = global_evidence[:5]

            if verification_mode == "keyword":
                covered = self._check_element_with_keywords(element, evidence, keywords)
            elif verification_mode == "llm":
                covered = self._check_element_with_llm(element, evidence, req['name'])
            elif verification_mode == "hybrid":
                covered = self._check_element_hybrid(element, evidence, req['name'], keywords)
            else:
                raise ValueError(f"Unknown mode: {verification_mode}")

            # Convert selected chunk IDs into readable evidence text
            selected_chunk_ids = set(covered.get('chunks', []))

            selected_evidence = [
                chunk for chunk in evidence
                if chunk.get('chunk_id') in selected_chunk_ids
            ]

            evidence_text = "\n\n".join(
                chunk.get('text', '').strip()
                for chunk in selected_evidence
                if chunk.get('text', '').strip()
            )

            structured_evidence = [
                {
                    'chunk_id': chunk.get('chunk_id', ''),
                    'text': chunk.get('text', ''),
                    'page_number': chunk.get('metadata', {}).get('page_num')
                }
                for chunk in selected_evidence
            ]

            element_coverage.append({
                'element'            : element,
                'status'             : covered['status'],
                'evidence_chunks'    : covered.get('chunks', []),
                'evidence_text'      : evidence_text,
                'page_numbers'       : covered.get('page_numbers', []),
                'confidence'         : covered['confidence'],
                'reasoning'          : covered.get('reasoning', ''),
                'verification_method': covered.get('method', verification_mode),
                'evidence'           : structured_evidence
            })

        overall_status = self._determine_overall_status(
            element_coverage,
            req_id,
        )

        return {
            'company_id'        : company_id,
            'requirement_id'    : req_id,
            'requirement_name'  : req['name'],
            'overall_status'    : overall_status,
            'element_coverage'  : element_coverage,
            'evidence'          : global_evidence[:3],
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

            selected_chunks = (
                llm_result.get('chunks', [])
                or keyword_result.get('chunks', [])
            )

            selected_pages = (
                llm_result.get('page_numbers', [])
                if llm_result.get('chunks')
                else []
            )

            return {
                'status'      : final_status,
                'confidence'  : final_conf,
                'reasoning'   : llm_result.get('reasoning', 'Both methods agree'),
                'chunks'      : selected_chunks,
                'page_numbers': selected_pages,
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
                'chunks'      : (llm_result.get('chunks', []) or keyword_result.get('chunks', [])),
                'page_numbers': (llm_result.get('page_numbers', []) if llm_result.get('chunks') else []),
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
            max_chunks=5
        )

        # Read status directly from v3 verifier (covered / partial / missing)
        status = result.get('status', 'missing')

        selected_evidence = result.get('selected_evidence', [])

        selected_chunk_ids = [
            item.get('chunk_id', '')
            for item in selected_evidence
            if item.get('chunk_id')
        ]

        return {
            'status': status,
            'confidence': result['confidence'],
            'reasoning': result['reasoning'],
            'page_numbers': result.get('page_numbers', []),
            'chunks': selected_chunk_ids if status != 'missing' else [],
            'selected_evidence': selected_evidence if status != 'missing' else []
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

    def _determine_overall_status(
        self,
        element_coverage: List[Dict],
        req_id: str,
    ) -> str:
        """
        Determine requirement-level evidence completeness.

        GRI defines the disclosure elements themselves.
        covered/partial/missing aggregation is a project-specific
        conservative policy and is not an official GRI scoring rule.
        """
        if req_id == "305-1":
            from ground_truth import determine_coverage

            element_to_slot = {
                "Total Scope 1 emissions in tCO2e": "slot_a",
                "Gases included in calculation (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)": "slot_b",
                "Biogenic CO2 emissions (reported separately)": "slot_c",
                "Base year for calculation": "slot_d",
                "Emission factors and GWP source (e.g. IPCC AR5/AR6)": "slot_e",
                "Consolidation approach (equity share, financial control, operational control)": "slot_f",
                "Standards and methodologies used (e.g. GHG Protocol, third-party verification)": "slot_g",
            }

            slot_status = {}

            for elem in element_coverage:
                slot = element_to_slot.get(
                    elem.get("element")
                )
                if slot:
                    slot_status[slot] = elem.get(
                        "status",
                        "missing",
                    )

            return determine_coverage(slot_status)

        if req_id in {"305-2", "305-3"}:
            req = self.requirements.get_requirement(
                "gri_305",
                req_id,
            )

            metadata = {
                elem["element"]: elem
                for elem in req["required_elements"]
            }

            status_by_element = {
                elem["element"]: elem.get(
                    "status",
                    "missing",
                )
                for elem in element_coverage
            }

            required_elements = req[
                "required_elements"
            ]

            if not required_elements:
                return "missing"

            # The first disclosure element is the gross emissions
            # anchor for both 305-2 and 305-3.
            anchor = required_elements[0]["element"]

            if (
                status_by_element.get(
                    anchor,
                    "missing",
                )
                == "missing"
            ):
                return "missing"

            mandatory = [
                elem["element"]
                for elem in required_elements
                if not elem.get(
                    "conditional",
                    False,
                )
            ]

            all_mandatory_covered = all(
                status_by_element.get(name)
                == "covered"
                for name in mandatory
            )

            # A conditional disclosure that is absent does not,
            # by itself, prevent full coverage because GRI marks
            # these as "if applicable" or "if available".
            #
            # However, if evidence exists but is incomplete
            # (partial), retain the conservative partial status.
            conditional_partial = any(
                status_by_element.get(
                    elem["element"]
                )
                == "partial"
                for elem in required_elements
                if elem.get(
                    "conditional",
                    False,
                )
            )

            if (
                all_mandatory_covered
                and not conditional_partial
            ):
                return "covered"

            return "partial"

        # Generic conservative fallback for requirements that
        # do not yet have an explicit aggregation policy.
        statuses = [
            elem.get(
                "status",
                "missing",
            )
            for elem in element_coverage
        ]

        if not statuses:
            return "missing"

        if all(
            status == "covered"
            for status in statuses
        ):
            return "covered"

        if all(
            status == "missing"
            for status in statuses
        ):
            return "missing"

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
