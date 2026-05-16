"""
generate_embeddings.py
Generate embeddings for all chunks and store in ChromaDB
"""

import chromadb
from pathlib import Path
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_chunks(jsonl_path: Path):
    """Load chunks from JSONL file"""
    chunks = []
    with jsonl_path.open('r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """Get embedding from OpenAI API"""
    try:
        response = openai_client.embeddings.create(
            input=text,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def batch_embed_and_store(
    chunks,
    collection,
    batch_size: int = 100,
    collection_name: str = ""
):
    """
    Generate embeddings for chunks and store in ChromaDB.
    Uses batching to avoid rate limits.
    """
    
    print(f"\n{'='*80}")
    print(f"EMBEDDING: {collection_name}")
    print(f"{'='*80}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Batch size: {batch_size}")
    
    # Prepare data
    ids = []
    documents = []
    embeddings = []
    metadatas = []
    
    # Process in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Batches"):
        batch = chunks[i:i+batch_size]
        
        for chunk in batch:
            # Get embedding
            embedding = get_embedding(chunk['text'])
            
            if embedding is None:
                continue
            
            ids.append(chunk['chunk_id'])
            documents.append(chunk['text'])
            embeddings.append(embedding)
            metadatas.append(chunk['metadata'])
        
        # Add batch to collection
        if len(ids) > 0:
            try:
                collection.add(
                    ids=ids[-len(batch):],
                    embeddings=embeddings[-len(batch):],
                    documents=documents[-len(batch):],
                    metadatas=metadatas[-len(batch):]
                )
            except Exception as e:
                print(f"\nError adding batch: {e}")
        
        # Rate limit protection (OpenAI: 3,000 RPM for tier 1)
        time.sleep(0.5)
    
    print(f"\nStored {len(ids)} embeddings in '{collection_name}'")
    return len(ids)

def embed_all_chunks():
    """Main function to embed all chunks"""
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path="data/chromadb")
    
    print("="*80)
    print("BATCH EMBEDDING - ALL CHUNKS")
    print("="*80)
    
    stats = {}
    
    # 1. Embed Reports
    print("\nREPORTS")
    reports_collection = client.get_collection("reports")
    
    reports_dir = Path("data/chunks/reports")
    all_report_chunks = []
    
    for jsonl_file in sorted(reports_dir.glob("*.jsonl")):
        chunks = load_chunks(jsonl_file)
        all_report_chunks.extend(chunks)
        print(f"   Loaded: {jsonl_file.name} ({len(chunks)} chunks)")
    
    stats['reports'] = batch_embed_and_store(
        all_report_chunks,
        reports_collection,
        batch_size=100,
        collection_name="reports"
    )
    
    # 2. Embed Standards
    print("\nSTANDARDS")
    standards_collection = client.get_collection("standards")
    
    standards_dir = Path("data/chunks/standards")
    all_standard_chunks = []
    
    for jsonl_file in sorted(standards_dir.glob("*.jsonl")):
        chunks = load_chunks(jsonl_file)
        all_standard_chunks.extend(chunks)
        print(f"   Loaded: {jsonl_file.name} ({len(chunks)} chunks)")
    
    stats['standards'] = batch_embed_and_store(
        all_standard_chunks,
        standards_collection,
        batch_size=100,
        collection_name="standards"
    )
    
    # Summary
    print("\n" + "="*80)
    print("EMBEDDING SUMMARY")
    print("="*80)
    print(f"\nReports: {stats['reports']} embeddings")
    print(f"Standards: {stats['standards']} embeddings")
    print(f"Total: {sum(stats.values())} embeddings")
    
    # Verify counts
    print("\n🔍 VERIFICATION")
    print(f"Reports collection: {reports_collection.count()} documents")
    print(f"Standards collection: {standards_collection.count()} documents")
    
    return stats

if __name__ == "__main__":
    stats = embed_all_chunks()