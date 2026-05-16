"""
hybrid_search.py
Hybrid search combining semantic + keyword with RRF
"""

from semantic_search import SemanticRetriever
from keyword_search import KeywordRetriever
from typing import List, Dict, Any

def reciprocal_rank_fusion(
    semantic_results: List[Dict],
    keyword_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    Reciprocal Rank Fusion (RRF) algorithm.
    
    Score = sum(1 / (k + rank)) for each result list
    
    Args:
        semantic_results: Results from semantic search
        keyword_results: Results from keyword search
        k: RRF constant (default 60)
    
    Returns:
        Fused and re-ranked results
    """
    
    # Collect all unique chunk IDs
    chunk_scores = {}
    chunk_data = {}
    
    # Add semantic results
    for result in semantic_results:
        chunk_id = result['chunk_id']
        rank = result['rank']
        score = 1.0 / (k + rank)
        
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + score
        chunk_data[chunk_id] = result
    
    # Add keyword results
    for result in keyword_results:
        chunk_id = result['chunk_id']
        rank = result['rank']
        score = 1.0 / (k + rank)
        
        chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + score
        
        # Store data if not already present
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = result
    
    # Sort by fused score
    sorted_ids = sorted(
        chunk_scores.keys(),
        key=lambda x: chunk_scores[x],
        reverse=True
    )
    
    # Format results
    fused_results = []
    for rank, chunk_id in enumerate(sorted_ids, 1):
        result = chunk_data[chunk_id].copy()
        result['rank'] = rank
        result['rrf_score'] = chunk_scores[chunk_id]
        fused_results.append(result)
    
    return fused_results

class HybridRetriever:
    """Hybrid search combining semantic + keyword"""
    
    def __init__(self, collection_name: str = "standards"):
        self.semantic = SemanticRetriever(collection_name)
        self.keyword = KeywordRetriever(collection_name)
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Dict = None
    ) -> List[Dict]:
        """
        Hybrid search with RRF fusion.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter (e.g., {"company_id": "IBK"})
        
        Returns:
            Fused and re-ranked results
        """
        
        # Semantic search (with filter)
        semantic_raw = self.semantic.search(query, n_results=10, where=where)
        semantic_results = self.semantic.format_results(semantic_raw)
        
        # Keyword search (with filter) - FIX!
        keyword_results = self.keyword.search(query, n_results=10, where=where)
        
        # Fuse with RRF
        fused = reciprocal_rank_fusion(semantic_results, keyword_results)
        
        return fused[:n_results]

# Test
if __name__ == "__main__":
    print("="*80)
    print("HYBRID SEARCH (Semantic + Keyword + RRF) - FIXED")
    print("="*80)
    
    hybrid = HybridRetriever("reports")
    
    query = "scope 1 emissions"
    
    # Test WITHOUT filter
    print(f"\nTest 1: Without filter")
    print("-"*80)
    print(f"Query: '{query}'")
    
    results = hybrid.search(query, n_results=5)
    
    print(f"\nReturned {len(results)} results:")
    for result in results:
        company = result.get('metadata', {}).get('company_id', 'N/A')
        print(f"  Rank {result['rank']}: {result['chunk_id'][:50]}... [{company}]")
    
    # Test WITH filter
    print(f"\n\nTest 2: With IBK filter")
    print("-"*80)
    print(f"Query: '{query}'")
    print(f"Filter: company_id=IBK")
    
    results_filtered = hybrid.search(query, n_results=5, where={"company_id": "IBK"})
    
    print(f"\nReturned {len(results_filtered)} results:")
    companies = []
    for result in results_filtered:
        company = result.get('metadata', {}).get('company_id', 'N/A')
        companies.append(company)
        print(f"  Rank {result['rank']}: {result['chunk_id'][:50]}... [{company}]")
    
    # Verify filter
    print(f"\n\nFilter Verification:")
    print("-"*80)
    ibk_count = sum(1 for c in companies if c == "IBK")
    print(f"IBK results: {ibk_count}/{len(companies)}")
    
    if ibk_count == len(companies):
        print(" Filter working correctly!")
    else:
        print(" Filter still not working!")
