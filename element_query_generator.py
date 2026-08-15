"""
element_query_generator.py
Generate multi-language queries for each element (Base year improved!)
"""

from typing import List, Dict

class ElementQueryGenerator:
    """Generate optimized queries for element matching"""
    
    # Element-specific query templates
    ELEMENT_QUERIES = {
        'gases_included': {
            'en': [
                "gases included CO2 CH4 N2O SF6 NF3 HFC PFC",
                "greenhouse gases calculation methodology",
                "GHG protocol gases carbon dioxide methane"
            ],
            'ko': [
                "온실가스 배출 가스 종류 CO2 CH4 N2O",
                "불소계 가스 SF6 NF3 HFC PFC",
                "산정 기준 가스 포함",
                "온실가스 프로토콜 대상 가스"
            ]
        },
        'total_scope1': {
            'en': [
                "total scope 1 emissions tCO2e",
                "direct GHG emissions metric tons",
                "scope 1 carbon dioxide equivalent"
            ],
            'ko': [
                "scope 1 배출량 총량",
                "직접 배출 tCO2eq",
                "온실가스 배출량 합계"
            ]
        },
        'base_year': {
            'en': [
                "base year emissions",
                "baseline year greenhouse gas emissions",
                "reference year emissions inventory",
                "base year GHG inventory"
            ],
            'ko': [
                "기준연도 배출량",
                "기준 연도 온실가스 배출량",
                "배출량 기준년도",
                "기준연도 설정"
            ]
        },
        'emission_factors': {
            'en': [
                "emission factors source methodology",
                "IPCC emission factors guidelines",
                "GHG protocol emission factors"
            ],
            'ko': [
                "배출계수 출처",
                "IPCC 배출계수",
                "산정 기준 배출계수",
                "IPCC 가이드라인 ISO 14064 기준 적용",
                "국가별 배출계수 산정방법"
            ]
        },
        'consolidation_approach': {
            'en': [
                "consolidation approach equity financial operational control",
                "organizational boundary GHG reporting",
                "control approach scope boundary"
            ],
            'ko': [
                "연결 범위 접근법",
                "지분율 재무적 운영적 통제",
                "조직 경계 설정"
            ]
        },
        'standards_methods': {
            'en': [
                'GHG Protocol standards methodology verification',
                'third-party verification assurance independent',
                'ISO 14064 GHG reporting standard',
                'external assurance sustainability report verification'
            ],
            'ko': [
                'GHG 프로토콜 온실가스 프로토콜',
                '제3자 검증 인증 검증기관',
                'ISO 14064 온실가스 검증',
                '독립된 인증인 검증 성명서',
                '산정 방법론 기준 검증'
            ]
        },
        'biogenic_co2': {
            'en': [
                "biogenic CO2 emissions biomass",
                "biogenic carbon dioxide separate reporting",
                "biomass combustion emissions"
            ],
            'ko': [
                "바이오매스 CO2 배출",
                "생물 기원 이산화탄소",
                "바이오제닉 배출량 별도"
            ]
        },

        'scope2_total': {
            'en': [
                "Scope 2 location-based emissions tCO2e",
                "energy indirect greenhouse gas emissions",
                "location based Scope 2 emissions"
            ],
            'ko': [
                "Scope 2 지역기반 배출량",
                "간접 온실가스 배출량",
                "지역기반 Scope 2 온실가스"
            ]
        },

        'scope2_market': {
            'en': [
                "Scope 2 market-based emissions tCO2e",
                "market based electricity emissions",
                "market-based Scope 2"
            ],
            'ko': [
                "Scope 2 시장기반 배출량",
                "시장기반 온실가스 배출량",
                "시장기반 Scope 2"
            ]
        },

        'scope2_factors': {
            'en': [
                "Scope 2 emission factors GWP source",
                "electricity grid emission factor source",
                "Scope 2 GWP methodology"
            ],
            'ko': [
                "Scope 2 배출계수 출처",
                "전력 배출계수 출처",
                "Scope 2 GWP 산정 기준"
            ]
        },

        'scope2_methodology': {
            'en': [
                "GHG Protocol Scope 2 methodology",
                "Scope 2 standards calculation methodology",
                "Scope 2 calculation tools assumptions"
            ],
            'ko': [
                "Scope 2 산정 방법론",
                "Scope 2 온실가스 산정 기준",
                "Scope 2 계산 도구 가정"
            ]
        },

        'scope3_total': {
            'en': [
                "total Scope 3 emissions tCO2e",
                "gross Scope 3 greenhouse gas emissions",
                "value chain emissions total"
            ],
            'ko': [
                "Scope 3 배출량 총량",
                "Scope 3 온실가스 총배출량",
                "가치사슬 배출량 총계"
            ]
        },

        'scope3_categories': {
            'en': [
                "Scope 3 categories upstream downstream",
                "Scope 3 categories activities included",
                "value chain emission categories"
            ],
            'ko': [
                "Scope 3 카테고리",
                "Scope 3 활동 범주",
                "가치사슬 배출 카테고리"
            ]
        },

        'scope3_factors': {
            'en': [
                "Scope 3 emission factors GWP source",
                "Scope 3 calculation emission factor",
                "Scope 3 GWP methodology"
            ],
            'ko': [
                "Scope 3 배출계수 출처",
                "Scope 3 산정 배출계수",
                "Scope 3 GWP 산정 기준"
            ]
        },

        'scope3_methodology': {
            'en': [
                "GHG Protocol Scope 3 methodology",
                "Corporate Value Chain Scope 3 standard",
                "PCAF Scope 3 methodology"
            ],
            'ko': [
                "Scope 3 산정 방법론",
                "가치사슬 온실가스 산정 기준",
                "Scope 3 PCAF 방법론"
            ]
        }
    }
    
    def match_element_to_category(self, element: str) -> str:
        """Match element description to predefined category"""
        element_lower = element.lower()
        
        if 'gases included' in element_lower or 'co2, ch4' in element_lower:
            return 'gases_included'
        elif 'total scope 1' in element_lower or 'scope 1 emissions in tco2' in element_lower:
            return 'total_scope1'
        elif 'base year' in element_lower:
            return 'base_year'
        elif 'scope 2 emission factors' in element_lower:
            return 'scope2_factors'
        elif 'scope 3 emission factors' in element_lower:
            return 'scope3_factors'
        elif 'emission factors' in element_lower:
            return 'emission_factors'
        elif 'consolidation' in element_lower:
            return 'consolidation_approach'
        elif 'gross location-based scope 2' in element_lower:
            return 'scope2_total'
        elif 'gross market-based scope 2' in element_lower:
            return 'scope2_market'
        elif 'scope 2 emission factors' in element_lower:
            return 'scope2_factors'
        elif 'scope 2 standards' in element_lower:
            return 'scope2_methodology'
        elif 'gross scope 3 emissions' in element_lower:
            return 'scope3_total'
        elif 'scope 3 categories' in element_lower:
            return 'scope3_categories'
        elif 'scope 3 emission factors' in element_lower:
            return 'scope3_factors'
        elif 'scope 3 standards' in element_lower:
            return 'scope3_methodology'
        elif 'biogenic' in element_lower:
            return 'biogenic_co2'
        elif 'standards' in element_lower or 'methodolog' in element_lower or 'verification' in element_lower:
            return 'standards_methods'
        else:
            # Fallback: extract keywords
            return None
    
    def generate_queries(self, element: str) -> List[str]:
        """
        Generate multi-language queries for an element.
        
        Args:
            element: Element description
        
        Returns:
            List of optimized search queries (EN + KO)
        """
        category = self.match_element_to_category(element)
        
        if category and category in self.ELEMENT_QUERIES:
            queries = self.ELEMENT_QUERIES[category]

            # Interleave English and Korean queries so that
            # downstream query caps preserve multilingual coverage.
            combined = []
            max_len = max(
                len(queries['en']),
                len(queries['ko']),
            )

            for i in range(max_len):
                if i < len(queries['en']):
                    combined.append(queries['en'][i])
                if i < len(queries['ko']):
                    combined.append(queries['ko'][i])

            return combined
        else:
            # Fallback: simple keyword extraction
            keywords = self._extract_keywords(element)
            return [keywords]
    
    def _extract_keywords(self, element: str) -> str:
        """Fallback: extract keywords from element"""
        # Remove parentheses content
        import re
        clean = re.sub(r'\([^)]*\)', '', element)
        
        # Extract important words (> 3 chars)
        words = [w for w in clean.split() if len(w) > 3]
        
        return ' '.join(words[:8])

# Test
if __name__ == "__main__":
    print("="*80)
    print("ELEMENT QUERY GENERATOR TEST (BASE YEAR IMPROVED)")
    print("="*80)
    
    generator = ElementQueryGenerator()
    
    test_elements = [
        "Gases included in calculation (CO2, CH4, N2O, HFCs, PFCs, SF6, NF3)",
        "Total Scope 1 emissions in tCO2e",
        "Base year for calculation",  # ← This one!
        "Source of emission factors used"
    ]
    
    for element in test_elements:
        print(f"\nElement: {element}")
        print("-"*80)
        
        queries = generator.generate_queries(element)
        category = generator.match_element_to_category(element)
        
        print(f"Category: {category}")
        print(f"Generated queries ({len(queries)}):")
        for i, q in enumerate(queries, 1):
            # Determine language
            en_count = len(generator.ELEMENT_QUERIES.get(category, {}).get('en', []))
            lang = "EN" if i <= en_count else "KO"
            print(f"  {i}. [{lang}] {q}")
    
    # Test base year specifically
    print("\n" + "="*80)
    print("BASE YEAR QUERIES (DETAILED)")
    print("="*80)
    
    base_year_queries = generator.ELEMENT_QUERIES['base_year']
    
    print("\nEnglish queries:")
    for i, q in enumerate(base_year_queries['en'], 1):
        print(f"  {i}. {q}")
    
    print("\nKorean queries:")
    for i, q in enumerate(base_year_queries['ko'], 1):
        print(f"  {i}. {q}")