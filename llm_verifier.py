"""
llm_verifier.py
Slot-aware LLM verifier with conservative bias and Korean language support.

Changes from v2:
- Slot-specific checklist injected into prompt
- Conservative bias: when in doubt → partial, not covered
- Korean language support
- Explicit distinction rules (base year vs data table, intensity vs consolidation)
"""
from llm_setup import LLMProvider
from typing import Dict, List
import json

# ── Slot-specific verification rules ──────────────────────────────────────────
# Injected into LLM prompt per slot to prevent common false covered errors.
SLOT_RULES = {
    "Total Scope 1 emissions in tCO2e": """
COVERED if: A specific numeric value is reported in tCO2e or tCO2eq for Scope 1 direct emissions.
PARTIAL if: Scope 1 emissions are mentioned but the unit or number is unclear/truncated.
MISSING if: No numeric Scope 1 emissions value is present.
NOTE: A percentage or reduction target is NOT a total emissions figure.""",

    "Gases included in calculation (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)": """
COVERED if: At least 3 specific gas types are explicitly named (e.g. CO2, CH4, N2O, SF6, HFCs).
PARTIAL if ANY of the following:
  - Fuel types or combustion sources are listed (e.g. LNG, diesel, gasoline, petrol, 휘발유, 경유, LPG)
  - Korean combustion terms appear: 고정연소, 이동연소
  - Only CO2 is mentioned without other gases
  → IMPORTANT: If you see LNG, diesel, gasoline, 휘발유, 경유, LPG, 고정연소, or 이동연소 → always return "partial", never "missing"
MISSING if: No gas types AND no fuel types AND no combustion sources are mentioned at all.
Korean equivalents: 이산화탄소(CO2), 메탄(CH4), 아산화질소(N2O), 육불화황(SF6)
Korean fuel terms that indicate PARTIAL: 휘발유, 경유, LPG, LNG, 고정연소, 이동연소""",

    "Base year for calculation": """
COVERED if ANY of the following:
  - A specific year is explicitly declared as the BASE YEAR for GHG calculations
  - Korean patterns: 기준연도: YYYY, YYYY년을 기준연도로, 기준년도, 기준 년도
  - A year marked as (기준) or (Base) in a data table
  - Text says e.g. 2021 (기준연도) or 2021 (Base year)
PARTIAL if: A year is mentioned in passing but not declared as base year.
MISSING if: No base year declaration or pattern is present.
NOTE: A table showing multiple years of data alone is NOT a base year declaration.
      BUT if one year is labeled (기준) or (Base) in that table → COVERED.
Korean equivalents: 기준연도, 기준 연도, 기준년도""",

    "Consolidation approach (equity share, financial control, operational control)": """
COVERED if ANY of the following:
  - equity share, financial control, or operational control is explicitly stated
  - Ownership percentage (e.g. 20% 이상 지분) is used as the basis for including entities → equity share approach
  - Korean patterns: 운영통제, 재무적 통제, 지분율, 지분법, 지배력, 소유지분
  - Statement that emissions are calculated based on equity ownership percentage
PARTIAL if: Reporting boundary is described (which entities included) but approach name not stated.
MISSING if: No consolidation approach or boundary information present at all.
NOTE: Describing WHAT is included is not the same as stating the APPROACH.
      BUT ownership percentage basis → equity share → COVERED.
Korean equivalents: 운영통제, 재무적 통제, 지분율, 지분법, 보고경계, 연결범위""",

    "Biogenic CO2 emissions (reported separately)": """
COVERED if: Biogenic CO2 emissions are explicitly reported as a SEPARATE numeric figure (e.g. "X tCO2e biogenic").
PARTIAL if: A specific biogenic source is quantified but not clearly labeled as biogenic CO2
            OR biogenic CO2 is explicitly discussed as "outside of scope" with a value given.
MISSING if ANY of the following:
  - Only a general mention of biogenic sources (biofuels, biomass, SAF) without a numeric figure
  - Only methodology notes about biogenic emissions without a figure
  - SAF savings or avoided emissions mentioned (not the same as reporting biogenic CO2)
  - The word "biogenic" appears only in passing without a separate disclosure
  → IMPORTANT: A mention or methodology note alone is NOT enough for partial. A numeric value must be present.
Korean equivalents: 바이오제닉, 생물성 탄소, 바이오매스""",

    "Emission factors and GWP source (e.g. IPCC AR5/AR6)": """
COVERED if ANY of the following:
  - A specific GWP source is cited (e.g. IPCC AR5, IPCC AR6, AR6, AR5)
  - IPCC 가이드라인 or IPCC guidelines mentioned as calculation basis
  - ISO 14064 mentioned as calculation standard
  - 국가별 배출계수 or country-specific emission factors explicitly referenced
  - Footnote or methodology note cites IPCC, ISO 14064, or national emission factor database
PARTIAL if ANY of the following:
  - Emission factors are mentioned but the GWP version is not specified
  - A general reference to internationally recognised standards without naming IPCC version
  - Emission factors are used but only the source organisation is named (e.g. ADEME, IEA, DEFRA)
  → IMPORTANT: If any emission factor source is named at all → at minimum partial, never missing
MISSING if: No mention of emission factors, GWP values, or calculation methodology sources at all.
Korean equivalents: 지구온난화지수, GWP, 배출계수, IPCC""",

    "Standards and methodologies used (e.g. GHG Protocol, third-party verification)": """
COVERED if: A specific standard or methodology is explicitly named (e.g. GHG Protocol, ISO 14064, WRI).
PARTIAL if ANY of the following:
  - Third-party verification is mentioned (e.g. DNV, SGS, Bureau Veritas) without naming the standard
  - internationally recognised methodology or similar vague reference
  - Assurance or verification report is referenced without specifying the standard
  → IMPORTANT: Any mention of third-party verification or assurance → at minimum partial
MISSING if: No mention of any standard, methodology, or verification at all.
Korean equivalents: GHG 프로토콜, 온실가스 프로토콜, 검증, 인증, 제3자 검증, ISO 14064""",

    "Emission intensity ratio (optional but common)": """
COVERED if: An emissions intensity ratio is reported (e.g. tCO2e per unit of revenue, per employee, per sqm).
PARTIAL if: Intensity is referenced but no actual ratio or denominator is given.
MISSING if: No intensity ratio is present.
NOTE: Reporting absolute emissions is NOT an intensity ratio.
      Consolidation approach text is NOT an intensity ratio.""",
}

# Default rules for any slot not in the above dict
DEFAULT_SLOT_RULES = """
COVERED if: The specific information is clearly and explicitly present in the evidence.
PARTIAL if: The information is mentioned but incomplete, ambiguous, or requires inference.
MISSING if: The information is absent from the evidence."""


class LLMVerifier:
    """Slot-aware LLM verifier with conservative bias"""

    def __init__(self):
        self.llm = LLMProvider()

    def verify_element(
        self,
        element: str,
        evidence_chunks: List[Dict],
        requirement_name: str,
        max_chunks: int = 10
    ) -> Dict:
        """
        Verify if evidence chunks cover a specific GRI 305-1 slot.

        Args:
            element: Slot description (must match a key in SLOT_RULES for best results)
            evidence_chunks: List of retrieved chunks
            requirement_name: Requirement name for context
            max_chunks: Maximum chunks to analyze (default: 10)

        Returns:
            {
                'status': str,
                'covered': bool,
                'partial': bool,
                'confidence': float,
                'reasoning': str,
                'evidence_chunk_indices': List[int],
                'selected_evidence': List[Dict],
                'page_numbers': List[int]
            }
        """

        # Prepare evidence text (with page numbers)
        analyzed_chunks = evidence_chunks[:max_chunks]

        evidence_text = "\n\n---\n\n".join([
            (
                f"[Chunk {i + 1} | "
                f"Page {chunk.get('metadata', {}).get('page_num', '?')}]\n"
                f"{chunk.get('text', '')}"
            )
            for i, chunk in enumerate(analyzed_chunks)
        ])

        # Get slot-specific rules
        slot_rules = SLOT_RULES.get(element, DEFAULT_SLOT_RULES)

        # ── System prompt ──────────────────────────────────────────────────────
        system_prompt = """You are a strict ESG compliance auditor verifying GRI 305-1 disclosures.

IMPORTANT PRINCIPLES:
1. Be conservative. When in doubt, return "partial" — never "covered".
2. Evidence may be in Korean, English, or mixed. Extract information regardless of language.
3. Follow the slot-specific rules exactly. Do not infer or assume information that is not explicit.
4. "partial" is not a failure — it means the item needs expert review.
5. Use exactly three status values: "covered", "partial", or "missing".

Respond ONLY in valid JSON format:
{
  "status": "covered" or "partial" or "missing",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation citing the supporting chunk numbers",
  "evidence_chunk_indices": [1, 2]
}

Evidence citation rules:
- "evidence_chunk_indices" must contain only the 1-based chunk numbers
  that directly support your decision.
- Do not include chunks that are merely related but do not support the conclusion.
- For "covered", include only chunks containing explicit and sufficient evidence.
- For "partial", include only chunks containing the incomplete or ambiguous evidence.
- For "missing", always return an empty list: [].
- Every index must correspond to a chunk shown in the evidence input.
- Never invent a chunk number.

Status definitions:
- "covered"  : Information is explicit, complete, and unambiguous
- "partial"  : Some evidence exists but incomplete, ambiguous, or only fuel types listed without gas names
- "missing"  : No relevant information found

Confidence scale:
- 0.9–1.0: Explicit, clear, unambiguous
- 0.6–0.8: Present but incomplete or requires minor inference
- 0.3–0.5: Tangentially related, does not directly satisfy requirement
- 0.0–0.2: Not present"""

        # ── User prompt ────────────────────────────────────────────────────────
        user_prompt = f"""Requirement: {requirement_name}
Slot to verify: {element}

Slot-specific verification rules:
{slot_rules}

Evidence chunks (check all carefully — evidence may be in Korean or English):
{evidence_text}

Apply the slot-specific rules above strictly.
If in doubt between "covered" and "partial", choose "partial".
Return only the chunk numbers that directly support the selected status.
If status is "missing", return "evidence_chunk_indices": [].
Respond in JSON format only."""

        # Get LLM response
        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
            temperature=0.0
        )

        # Parse JSON response
        try:
            response = response.strip()
            if response.startswith('```'):
                response = response.split('```')[1]
                if response.startswith('json'):
                    response = response[4:]
            response = response.strip()
            result = json.loads(response)

            # Normalize: ensure 'status' field exists
            if 'status' not in result:
                # Backward compat: convert old covered:bool to status
                if result.get('covered') is True:
                    result['status'] = 'covered'
                else:
                    result['status'] = 'missing'

            # Normalize status
            valid_statuses = {'covered', 'partial', 'missing'}
            status = str(result.get('status', 'missing')).lower().strip()

            if status not in valid_statuses:
                status = 'missing'

            result['status'] = status
            result['covered'] = status == 'covered'
            result['partial'] = status == 'partial'

            # Normalize and validate 1-based evidence chunk indices
            raw_indices = result.get('evidence_chunk_indices', [])

            if not isinstance(raw_indices, list):
                raw_indices = []

            validated_indices = []
            seen_indices = set()

            for index in raw_indices:
                try:
                    normalized_index = int(index)
                except (TypeError, ValueError):
                    continue

                if not 1 <= normalized_index <= len(analyzed_chunks):
                    continue

                if normalized_index in seen_indices:
                    continue

                validated_indices.append(normalized_index)
                seen_indices.add(normalized_index)

            # Missing decisions must never expose supporting evidence
            if status == 'missing':
                validated_indices = []

            result['evidence_chunk_indices'] = validated_indices

            selected_evidence = []

            for index in validated_indices:
                chunk = analyzed_chunks[index - 1]
                metadata = chunk.get('metadata', {})

                selected_evidence.append({
                    'chunk_index': index,
                    'chunk_id': chunk.get('chunk_id', ''),
                    'text': chunk.get('text', ''),
                    'page_number': metadata.get('page_num')
                })

            result['selected_evidence'] = selected_evidence

            result['page_numbers'] = [
                item['page_number']
                for item in selected_evidence
                if item.get('page_number') is not None
            ]

        except Exception as e:
            result = {
                'status'       : 'missing',
                'covered'      : False,
                'partial'      : False,
                'confidence': 0.0,
                'reasoning': f'Failed to parse LLM response: {str(e)}',
                'evidence_chunk_indices': [],
                'selected_evidence': [],
                'page_numbers': []
            }

        return result


if __name__ == "__main__":
    print("=" * 80)
    print("LLM VERIFIER V3 TEST (Slot-Aware + Conservative)")
    print("=" * 80)

    verifier = LLMVerifier()

    # Test: base year (common false covered in Heathrow)
    print("\nTest 1: Base year — multi-year table (should be MISSING, not covered)")
    print("-" * 80)
    chunks_no_base_year = [
        {'text': 'Scope 1 GHG emissions (tonnes CO2e): 2021: 120,000 | 2022: 115,000 | 2023: 108,000 | 2024: 102,000'},
        {'text': 'We monitor our carbon footprint and report GHG emissions annually.'},
        {'text': 'Emissions are verified by a third-party auditor each year.'},
    ]
    r1 = verifier.verify_element(
        element="Base year for calculation",
        evidence_chunks=chunks_no_base_year,
        requirement_name="Direct (Scope 1) GHG emissions"
    )
    print(f"Status  : {r1['status']}")
    print(f"Covered : {r1['covered']} | Partial: {r1['partial']}")
    print(f"Confidence: {r1['confidence']:.2f}")
    print(f"Reasoning : {r1['reasoning']}")

    # Test: Korean gases (common false negative in IBK)
    print("\nTest 2: Gases — Korean text (should be PARTIAL, not missing)")
    print("-" * 80)
    chunks_korean_gases = [
        {'text': '직접 온실가스 배출량(Scope 1) 단위: tCO2eq\n- 고정연소(LNG): 1,234\n- 이동연소(휘발유, 경유, LPG): 567\n합계: 1,801'},
        {'text': '보고경계: 직접배출(Scope1), 간접배출(Scope2)'},
    ]
    r2 = verifier.verify_element(
        element="Gases included in calculation (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)",
        evidence_chunks=chunks_korean_gases,
        requirement_name="Direct (Scope 1) GHG emissions"
    )
    print(f"Status  : {r2['status']}")
    print(f"Covered : {r2['covered']} | Partial: {r2['partial']}")
    print(f"Confidence: {r2['confidence']:.2f}")
    print(f"Reasoning : {r2['reasoning']}")

    # Test: consolidation approach Korean
    print("\nTest 3: Consolidation — Korean boundary text (should be PARTIAL)")
    print("-" * 80)
    chunks_korean_consolidation = [
        {'text': '보고경계: 직접배출(Scope1), 간접배출(Scope2)\nScope1: 고정연소(LNG), 이동연소(휘발유, 경유, LPG)\nScope2: 외부 구매 전력\n보고년도: 2023년 1월 1일 ~ 2023년 12월 31일'},
    ]
    r3 = verifier.verify_element(
        element="Consolidation approach (equity share, financial control, operational control)",
        evidence_chunks=chunks_korean_consolidation,
        requirement_name="Direct (Scope 1) GHG emissions"
    )
    print(f"Status  : {r3['status']}")
    print(f"Covered : {r3['covered']} | Partial: {r3['partial']}")
    print(f"Confidence: {r3['confidence']:.2f}")
    print(f"Reasoning : {r3['reasoning']}")

    print("\n" + "=" * 80)
    print("Expected: Test1=missing, Test2=partial, Test3=partial")
    print("=" * 80)
