"""
semantic_search.py
Clean semantic search implementation
"""

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SemanticRetriever:
    """Simple semantic search retriever"""
    
    def __init__(self, collection_name: str = "standards"):
        """Initialize retriever with ChromaDB collection"""
        self.client = chromadb.PersistentClient(path="data/chromadb")
        self.collection = self.client.get_collection(collection_name)
        self.collection_name = collection_name
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for query text"""
        response = openai_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Semantic search with optional metadata filtering.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Metadata filter (e.g., {"language": "EN"})
        
        Returns:
            Dictionary with ids, documents, metadatas, distances
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def format_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format results into clean structure"""
        formatted = []
        
        for i in range(len(results['ids'][0])):
            formatted.append({
                'rank': i + 1,
                'chunk_id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted

# Test
if __name__ == "__main__":
    retriever = SemanticRetriever("standards")
    
    # Simple query
    results = retriever.search("GRI 305-1 requirements", n_results=3)
    formatted = retriever.format_results(results)
    
    print("="*80)
    print("SEMANTIC SEARCH TEST")
    print("="*80)
    
    for result in formatted:
        print(f"\nRank {result['rank']} (Distance: {result['distance']:.4f})")
        print(f"ID: {result['chunk_id']}")
        print(f"Text: {result['text'][:200]}...")