from pathlib import Path
from dotenv import load_dotenv
import os
import time

import fitz  # PyMuPDF
import numpy as np
import faiss
from openai import OpenAI

# --- Load env reliably (works regardless of where you run from) ---
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_KEY, "OPENAI_API_KEY not loaded. Check .env path/contents."

client = OpenAI()

# Sample PDF (SC Sustainability)
PDF_PATH = ROOT / "data/raw/reports/2024_UK_SC_Sustainability_EN.pdf"


def test_pymupdf(pdf_path: Path):
    print("\n[1] PyMuPDF open PDF")
    doc = fitz.open(str(pdf_path))
    print("pages:", doc.page_count)

    page0 = doc[0]
    text0 = page0.get_text("text")
    blocks0 = page0.get_text("blocks")

    print("first page text (first 250 chars):")
    print(text0[:250].replace("\n", " "))

    print("blocks on page 1:", len(blocks0))
    doc.close()


def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    text = text.replace("\n", " ")
    t0 = time.time()
    res = client.embeddings.create(model=model, input=[text])
    dt = (time.time() - t0) * 1000
    vec = res.data[0].embedding
    print(f"embedding latency: {dt:.0f} ms | dim: {len(vec)}")
    return vec


def test_embedding_and_faiss():
    print("\n[2] OpenAI embedding (1–3 calls)")
    v1 = get_embedding("TCFD governance disclosure example.")
    v2 = get_embedding("GRI index and material topics example.")
    v3 = get_embedding("Climate risk management and metrics and targets.")

    print("\n[3] FAISS build + search")
    X = np.array([v1, v2, v3], dtype=np.float32)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    q = np.array([get_embedding("Where is the TCFD governance section?")], dtype=np.float32)
    D, I = index.search(q, k=3)

    print("top indices:", I[0].tolist())
    print("distances:", [float(x) for x in D[0].tolist()])


if __name__ == "__main__":
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}\nUpdate PDF_PATH in test_stack.py")

    test_pymupdf(PDF_PATH)
    test_embedding_and_faiss()
    print("\n smoke test passed")