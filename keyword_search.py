"""
keyword_search.py
BM25 keyword search implementation with metadata filtering
"""

from rank_bm25 import BM25Okapi
import json
from pathlib import Path
from typing import List, Dict, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data (run once)
# nltk.download('punkt')
# nltk.download('stopwords')

class KeywordRetriever:
    """BM25-based keyword search"""
    
    def __init__(
        self,
        collection_name: str = "standards",
        *,
        chunk_directory: str | None = None,
    ):
        """Load chunks and build BM25 index."""
        
        self.collection_name = collection_name
        self.chunk_directory = chunk_directory
        self.chunks = self._load_chunks()
        
        # Tokenize all documents
        self.tokenized_corpus = [
            self._tokenize(chunk['text']) 
            for chunk in self.chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f" BM25 index built: {len(self.chunks)} documents")
    
    def _load_chunks(self) -> List[Dict]:
        """Load all chunks from JSONL files"""
        chunks = []
        
        if self.chunk_directory is not None:
            chunk_dir = Path(self.chunk_directory)
        elif self.collection_name == "reports":
            chunk_dir = Path("data/chunks/reports")
        else:
            chunk_dir = Path("data/chunks/standards")
        
        for jsonl_file in sorted(chunk_dir.glob("*.jsonl")):
            with jsonl_file.open('r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line)
                    chunks.append(chunk)
        
        return chunks
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short tokens
        stop_words = set(stopwords.words('english'))
        tokens = [
            t for t in tokens 
            if t.isalnum() and t not in stop_words and len(t) > 2
        ]
        
        return tokens
    
    def search(
        self, 
        query: str, 
        n_results: int = 5,
        where: Dict = None
    ) -> List[Dict[str, Any]]:
        """
        BM25 keyword search with optional metadata filtering.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter (e.g., {"company_id": "IBK"})
        
        Returns:
            List of ranked results
        """
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Apply metadata filter BEFORE ranking if provided
        if where:
            # Create filtered indices
            filtered_indices = []
            filtered_scores = []
            
            for idx, chunk in enumerate(self.chunks):
                # Check if chunk matches all filter criteria
                matches = all(
                    chunk['metadata'].get(k) == v 
                    for k, v in where.items()
                )
                
                if matches:
                    filtered_indices.append(idx)
                    filtered_scores.append(scores[idx])
            
            # Sort filtered results by score
            if filtered_indices:
                sorted_pairs = sorted(
                    zip(filtered_indices, filtered_scores),
                    key=lambda x: x[1],
                    reverse=True
                )
                top_pairs = sorted_pairs[:n_results]
                
                # Format results
                results = []
                for rank, (idx, score) in enumerate(top_pairs, 1):
                    results.append({
                        'rank': rank,
                        'chunk_id': self.chunks[idx]['chunk_id'],
                        'text': self.chunks[idx]['text'],
                        'metadata': self.chunks[idx]['metadata'],
                        'score': float(score)
                    })
                
                return results
            else:
                # No matches found
                return []
        
        else:
            # No filter - use all results
            top_indices = scores.argsort()[-n_results:][::-1]
            
            # Format results
            results = []
            for rank, idx in enumerate(top_indices, 1):
                results.append({
                    'rank': rank,
                    'chunk_id': self.chunks[idx]['chunk_id'],
                    'text': self.chunks[idx]['text'],
                    'metadata': self.chunks[idx]['metadata'],
                    'score': float(scores[idx])
                })
            
            return results

# Test
if __name__ == "__main__":
    print("="*80)
    print("KEYWORD SEARCH (BM25) TEST - WITH FILTERING")
    print("="*80)
    
    retriever = KeywordRetriever("reports")
    
    # Test 1: Without filter
    print("\nTest 1: Without filter")
    print("-"*80)
    
    query = "scope 1 emissions"
    results = retriever.search(query, n_results=5)
    
    print(f"Query: '{query}'")
    print(f"Results: {len(results)}")
    for result in results:
        company = result['metadata'].get('company_id', 'N/A')
        print(f"  Rank {result['rank']}: {result['chunk_id'][:50]}... [{company}]")
    
    # Test 2: With filter
    print("\n\nTest 2: With IBK filter")
    print("-"*80)
    
    results_filtered = retriever.search(query, n_results=5, where={"company_id": "IBK"})
    
    print(f"Query: '{query}'")
    print(f"Filter: company_id=IBK")
    print(f"Results: {len(results_filtered)}")
    
    companies = []
    for result in results_filtered:
        company = result['metadata'].get('company_id', 'N/A')
        companies.append(company)
        print(f"  Rank {result['rank']}: {result['chunk_id'][:50]}... [{company}]")
    
    # Verify
    print("\n\nFilter Verification:")
    print("-"*80)
    ibk_count = sum(1 for c in companies if c == "IBK")
    print(f"IBK results: {ibk_count}/{len(companies)}")
    
    if ibk_count == len(companies) and len(companies) > 0:
        print(" Filter working correctly!")
    elif len(companies) == 0:
        print("  No results found with filter")
    else:
        print(" Filter not working!")