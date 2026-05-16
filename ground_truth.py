"""
ground_truth.py
GRI 305-1 ground truth for all 11 companies.
Slots a-g align with official GRI 305-1 disclosure requirements.

Slot definitions:
    a = total_scope1_tco2e      (CRITICAL)
    b = gases_included          (CRITICAL)
    c = biogenic_co2
    d = base_year
    e = emission_factors_gwp
    f = consolidation_approach  (CRITICAL)
    g = standards_methodologies

Changelog:
    2026-04-16 — Rebuilt from scratch with 11-company full ground truth (a-g slots)
                 Removed slot6_intensity (GRI 305-4, not 305-1)
                 Added slot_e (GWP), slot_g (standards/methodologies)
"""

# ── Slot definitions ───────────────────────────────────────────────────────────
SLOTS = {
    "slot_a": "Total Scope 1 GHG emissions (metric tonnes CO2e)",
    "slot_b": "Gases included in calculation (CO2, CH4, N2O, etc.)",
    "slot_c": "Biogenic CO2 emissions (if applicable)",
    "slot_d": "Base year and base year emissions",
    "slot_e": "Emission factors and GWP source/version",
    "slot_f": "Consolidation approach (equity share or control)",
    "slot_g": "Standards and methodologies used",
}

CRITICAL_SLOTS = ["slot_a", "slot_b", "slot_f"]
NON_CRITICAL_SLOTS = ["slot_c", "slot_d", "slot_e", "slot_g"]

# ── Coverage decision logic ────────────────────────────────────────────────────
def determine_coverage(slot_results: dict) -> str:
    all_slots = list(SLOTS.keys())
    covered_count = sum(1 for s in all_slots if slot_results.get(s) == "covered")
    missing_count = sum(1 for s in all_slots if slot_results.get(s) == "missing")
    partial_count = sum(1 for s in all_slots if slot_results.get(s) == "partial")

    if covered_count == 0 and partial_count == 0:
        return "missing"

    # critical slot이 missing이면 covered 불가 (partial은 허용)
    any_critical_missing = any(
        slot_results.get(s) == "missing" for s in CRITICAL_SLOTS
    )
    # slot_c(biogenic) missing은 구조적 미해당으로 허용
    # 단 c가 missing이면 나머지 슬롯에서 partial도 1개까지만 허용
    non_biogenic_missing = sum(
        1 for s in all_slots
        if slot_results.get(s) == "missing" and s != "slot_c"
    )
    c_is_missing = slot_results.get("slot_c") == "missing"
    non_biogenic_partial = sum(
        1 for s in all_slots
        if slot_results.get(s) == "partial" and s != "slot_c"
    )

    no_missing_except_biogenic = non_biogenic_missing == 0
    # c=missing이면 partial 1개까지만 허용 (e.g. HyundaiMotor: 0 partial OK)
    # c=covered이면 partial 여러 개 허용 (e.g. Schneider: b partial OK)
    partial_ok = (not c_is_missing) or (non_biogenic_partial <= 1)

    if no_missing_except_biogenic and not any_critical_missing and covered_count >= 4 and partial_ok:
        return "covered"

    return "partial"


# ── Manual ground truth labels (11 companies) ─────────────────────────────────
GROUND_TRUTH = {
    "IBK": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "missing",
        "slot_d": "covered",
        "slot_e": "partial",
        "slot_f": "partial",
        "slot_g": "covered",
        "overall": "partial",
        "review_notes": "b=partial(가스목록 미명시), e=partial(GWP버전 미명시), f=partial(접근법명 미명시)",
    },
    "Shinhan": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "missing",
        "slot_d": "covered",
        "slot_e": "partial",
        "slot_f": "partial",
        "slot_g": "covered",
        "overall": "partial",
        "review_notes": "GHG 상세 데이터는 별도 ESG DATA PACK(p.11)에 분리 공시. 메인 보고서에 배출계수/기준연도/가스종류 텍스트 없음. Page 106 검증의견서에 GHG Protocol+IPCC 2006 명시(g=covered). 멀티 문서 인덱싱 확장 시 개선 가능한 known limitation.",
    },
    "Kepco": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "missing",
        "slot_d": "partial",
        "slot_e": "missing",
        "slot_f": "missing",
        "slot_g": "partial",
        "overall": "partial",
        "review_notes": "한국 공기업. e/f=missing(GWP출처 및 접근법 미명시), g=partial(DNV검증 있으나 표준명 미확인)",
    },
    "Heathrow": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "partial",
        "slot_f": "covered",
        "slot_g": "covered",
        "overall": "covered",
        "review_notes": "e=partial(GWP버전 외부문서 참조만). 나머지 전슬롯 covered.",
    },
    "Schneider": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "covered",
        "slot_f": "covered",
        "slot_g": "covered",
        "overall": "covered",
        "review_notes": "b=partial(SF6만 명시, CH4/N2O 각주). e=covered(IPCC AR6 명시). 전반적으로 우수.",
    },
    "Siemens": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "covered",
        "slot_f": "partial",
        "slot_g": "covered",
        "overall": "covered",
        "review_notes": "f=partial(financial statement 경계 서술, 접근법명 미명시). 나머지 전슬롯 covered.",
    },
    "Samsung": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "missing",
        "slot_d": "partial",
        "slot_e": "covered",
        "slot_f": "partial",
        "slot_g": "covered",
        "overall": "partial",
        "review_notes": "c=missing, d=partial(기준연도 선언 부재), f=partial(접근법명 미명시)",
    },
    "Hyundai": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "missing",
        "slot_d": "covered",
        "slot_e": "covered",
        "slot_f": "covered",
        "slot_g": "covered",
        "overall": "covered",
        "review_notes": "c=missing(biogenic 없음, 자동차 특성). 나머지 전슬롯 covered.",
    },
    "Incheon": {
        "slot_a": "covered",
        "slot_b": "missing",
        "slot_c": "missing",
        "slot_d": "missing",
        "slot_e": "missing",
        "slot_f": "missing",
        "slot_g": "partial",
        "overall": "partial",
        "review_notes": "한국 공항. b/d/e/f=missing. GHG 방법론 공시 매우 미흡.",
    },
    "SC": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "missing",
        "slot_d": "partial",
        "slot_e": "partial",
        "slot_f": "covered",
        "slot_g": "covered",
        "overall": "partial",
        "review_notes": "d=partial(base year 선언 없음), e=partial(GWP버전 미명시), f=covered(financial control 명시)",
    },
    "HSBC": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "missing",
        "slot_d": "covered",
        "slot_e": "partial",
        "slot_f": "covered",
        "slot_g": "covered",
        "overall": "partial",
        "review_notes": "b=partial(CO2+CH4+N2O만, HFCs 미명시), e=partial(GWP버전 외부문서), f=covered(operational control 명시)",
    },
}

# ── Coverage rules reference ───────────────────────────────────────────────────
COVERAGE_RULES = """
GRI 305-1 Coverage Rules:

COVERED if:
  - No missing slots
  - At least 4 slots are covered
  - All critical slots are covered (slot_a, slot_b, slot_f)

PARTIAL if:
  - Some evidence exists
  - But at least one critical slot is partial/missing, OR any slot is missing

MISSING if:
  - Little or no evidence
  - Majority of slots (4+) are missing

Critical slots:
  slot_a: Total Scope 1 GHG emissions  [CRITICAL]
  slot_b: Gases included               [CRITICAL]
  slot_f: Consolidation approach       [CRITICAL]
"""



# ── GRI 305-2 Ground Truth ────────────────────────────────────────────────────
# Slots:
#   slot_a = location_based_scope2   (CRITICAL)
#   slot_b = market_based_scope2     (CRITICAL — missing OK if justified N/A)
#   slot_c = emission_factor_source  (CRITICAL)
#   slot_d = contractual_instruments
#   slot_e = standards_methodologies (CRITICAL)
#
# Overall logic:
#   covered : slot_a + slot_b + slot_c + slot_e all covered
#   partial : slot_a present but some critical slots partial/vague
#   missing : slot_a absent OR (slot_c + slot_e both missing)
#
# Critical insight: numeric data alone ≠ covered
# Companies must distinguish location-based vs market-based AND
# disclose emission factor source + methodology (GRI 305-2 "shall" requirements)

GROUND_TRUTH_305_2 = {
    "IBK": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "missing",
        "slot_d": "missing",
        "slot_e": "partial",
        "overall": "partial",
        "review_notes": "location-based 있음. market-based 불명확. EF source 없음. 방법론 간접 언급만.",
    },
    "Shinhan": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "partial",
        "slot_d": "covered",
        "slot_e": "partial",
        "overall": "partial",
        "review_notes": "location/market 둘 다 있고 REC 언급(d=covered). 국가 전력망 기준(c=partial). 방법론 명시 미흡(e=partial).",
    },
    "Kepco": {
        "slot_a": "covered",
        "slot_b": "missing",
        "slot_c": "missing",
        "slot_d": "missing",
        "slot_e": "missing",
        "overall": "missing",
        "review_notes": "Scope 2 수치 있지만 location/market 구분 없음. GRI 305-2 core requirement 미충족. Numeric data alone does not satisfy disclosure.",
    },
    "Heathrow": {
        "slot_a": "covered",
        "slot_b": "missing",
        "slot_c": "partial",
        "slot_d": "partial",
        "slot_e": "partial",
        "overall": "partial",
        "review_notes": "location-based 있음. market-based 'Not reported' = absence not justification (b=missing). WTT 방법론 언급(e=partial).",
    },
    "Schneider": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "covered",
        "overall": "covered",
        "review_notes": "GHG Protocol Scope 2 Guidance 명시. location/market 둘 다. EF source 명시.",
    },
    "Siemens": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "partial",
        "slot_e": "covered",
        "overall": "covered",
        "review_notes": "location/market 둘 다. contractual 간접 언급(d=partial). methodology 명시.",
    },
    "Samsung": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "covered",
        "overall": "covered",
        "review_notes": "시장기반/지역기반 둘 다. IPCC 가이드라인/ISO 14064 명시. RE100 관련 contractual 있음.",
    },
    "Hyundai": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "partial",
        "slot_d": "missing",
        "slot_e": "partial",
        "overall": "partial",
        "review_notes": "market-based 있음. IPCC 2006 배출계수 언급(c=partial). methodology(GHG Protocol 등) 미명시(e=partial). contractual 없음.",
    },
    "Incheon": {
        "slot_a": "covered",
        "slot_b": "missing",
        "slot_c": "missing",
        "slot_d": "missing",
        "slot_e": "missing",
        "overall": "missing",
        "review_notes": "Scope 2 수치만 있음. location/market 구분 없음. EF source/methodology 없음. Numeric data alone does not satisfy disclosure.",
    },
    "SC": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "partial",
        "slot_e": "covered",
        "overall": "covered",
        "review_notes": "IEA NZE methodology 명시. location/market 둘 다. contractual 간접 언급(d=partial).",
    },
    "HSBC": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "partial",
        "slot_e": "covered",
        "overall": "covered",
        "review_notes": "PCAF 기반 methodology 명시. market-based 있음. contractual 간접 언급(d=partial).",
    },
}

SLOTS_305_2 = {
    "slot_a": "Location-based Scope 2 emissions (tCO2e)",
    "slot_b": "Market-based Scope 2 emissions (tCO2e) or N/A justification",
    "slot_c": "Emission factor source (grid factor)",
    "slot_d": "Contractual instruments (REC, PPA, EAC)",
    "slot_e": "Standards and methodologies (GHG Protocol Scope 2 Guidance)",
}

CRITICAL_SLOTS_305_2 = {"slot_a", "slot_b", "slot_c", "slot_e"}


# ── GRI 305-3 Ground Truth ────────────────────────────────────────────────────
# Slots:
#   slot_a = total_scope3_tco2e        (CRITICAL)
#   slot_b = categories_included       (CRITICAL — explicit list required)
#   slot_c = standards_methodologies   (CRITICAL)
#   slot_d = base_year
#   slot_e = biogenic_co2              (non-critical: 0/11 companies disclose)
#
# Overall logic:
#   covered : slot_a + slot_b + slot_c all covered
#   partial : slot_a present AND slot_b OR slot_c at least partial
#   missing : slot_a absent OR (slot_b AND slot_c both missing)
#
# Key design decisions:
# 1. slot_b requires explicit category list (not activity-level mentions)
# 2. slot_c accepts any named methodology (GHG Protocol, PCAF, ISO 14064, etc.)
# 3. slot_e non-critical: empirically 0/11 companies disclose Scope 3 biogenic
# 4. Financial institutions (Shinhan): ESG Data Pack known limitation
# 5. SC: evaluated on Sustainability report only (consistency with Shinhan)

GROUND_TRUTH_305_3 = {
    "IBK": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "missing",
        "overall": "covered",
        "review_notes": "Page 64: 카테고리 목록(구매한 제품/서비스, 자본재, 출장 등). Page 67: PCAF 기준 금융배출량. d=covered(기준연도 명시).",
    },
    "Shinhan": {
        "slot_a": "missing",
        "slot_b": "missing",
        "slot_c": "missing",
        "slot_d": "missing",
        "slot_e": "missing",
        "overall": "missing",
        "review_notes": "Scope 3 데이터 ESG DATA PACK 별도 문서. 메인 보고서에 없음. Known limitation: multi-document indexing 필요.",
    },
    "Kepco": {
        "slot_a": "covered",
        "slot_b": "missing",
        "slot_c": "missing",
        "slot_d": "missing",
        "slot_e": "missing",
        "overall": "missing",
        "review_notes": "Page 149: Scope 3 수치 있음(194,127,327 tCO2eq). 단 카테고리 구분 없이 기타배출 하나로만 보고. 방법론 없음. slot_b+slot_c 모두 missing → overall missing.",
    },
    "Heathrow": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "covered",
        "slot_d": "missing",
        "slot_e": "missing",
        "overall": "partial",
        "review_notes": "Page 107: GHG Protocol 방법론(c=covered). Page 110: passenger/colleague surface access 등 activity-level 언급. BUT category-level breakdown(Cat 1-15) 없음 → b=partial.",
    },
    "Schneider": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "missing",
        "overall": "covered",
        "review_notes": "Page 44: downstream 85% + 카테고리별 수치. ADEME/IEA emission factors 명시. d=covered(base year 명시).",
    },
    "Siemens": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "missing",
        "overall": "covered",
        "review_notes": "Page 35: 카테고리 3.1~3.5 명시 + spend-based methodology. d=covered(2019 base year).",
    },
    "Samsung": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "missing",
        "overall": "covered",
        "review_notes": "Page 68: 14개 카테고리 수치. ISO 14064 + 제3자 검증. d=covered(기준연도 명시).",
    },
    "Hyundai": {
        "slot_a": "covered",
        "slot_b": "covered",
        "slot_c": "covered",
        "slot_d": "covered",
        "slot_e": "missing",
        "overall": "covered",
        "review_notes": "Page 39-41: 카테고리별 수치 + GHG Protocol Corporate Value Chain + IPCC 2006. d=covered.",
    },
    "Incheon": {
        "slot_a": "missing",
        "slot_b": "missing",
        "slot_c": "missing",
        "slot_d": "missing",
        "slot_e": "missing",
        "overall": "missing",
        "review_notes": "Scope 3 수치 자체 없음. 탄소중립 목표/RE100 언급만 있음.",
    },
    "SC": {
        "slot_a": "missing",
        "slot_b": "missing",
        "slot_c": "missing",
        "slot_d": "missing",
        "slot_e": "missing",
        "overall": "missing",
        "review_notes": "Sustainability 보고서(39p) 단일 기준 평가. Scope 3 데이터 없음. Annual report 포함 시 개선 가능 — Shinhan과 동일한 multi-document known limitation.",
    },
    "HSBC": {
        "slot_a": "covered",
        "slot_b": "partial",
        "slot_c": "partial",
        "slot_d": "missing",
        "slot_e": "missing",
        "overall": "partial",
        "review_notes": "Page 50: financed emissions 언급. supply chain emissions 30.7% reduction 언급. 카테고리 explicit list 없음(b=partial). 방법론 vague(c=partial).",
    },
}

SLOTS_305_3 = {
    "slot_a": "Total Scope 3 GHG emissions (tCO2e)",
    "slot_b": "Categories and activities included (explicit list)",
    "slot_c": "Standards/methodologies used (e.g., GHG Protocol Value Chain, PCAF)",
    "slot_d": "Base year for calculation",
    "slot_e": "Biogenic CO2 emissions (if applicable)",
}

CRITICAL_SLOTS_305_3 = {"slot_a", "slot_b", "slot_c"}

if __name__ == "__main__":
    print("GRI 305-1 Ground Truth Summary (11 companies)")
    print("=" * 60)
    for company, data in GROUND_TRUTH.items():
        print(f"\n{company}: {data['overall'].upper()}")
        for slot_key, slot_label in SLOTS.items():
            status = data[slot_key]
            symbol = {"covered": "COV", "partial": "PAR", "missing": "MIS"}[status]
            print(f"  [{symbol}] {slot_label[:55]}")
        print(f"  Note: {data['review_notes']}")

    print("\n" + "=" * 60)
    print("Verifying coverage logic...")
    all_pass = True
    for company, data in GROUND_TRUTH.items():
        computed = determine_coverage({k: data[k] for k in SLOTS})
        match = computed == data["overall"]
        status = "OK" if match else "MISMATCH"
        print(f"  {company}: computed={computed}, manual={data['overall']} -> {status}")
        if not match:
            all_pass = False
    print("All checks passed!" if all_pass else "Fix mismatches before proceeding.")
