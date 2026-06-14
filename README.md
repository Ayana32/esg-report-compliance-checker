# ESG Report Compliance Checker

> Automated GRI 305 emissions disclosure verification using hybrid RAG + LLM verification

A production-ready NLP pipeline that checks whether corporate ESG reports comply with GRI 305 (Scope 1, 2, 3 emissions) disclosure requirements. Built with domain-specific slot design, hybrid retrieval, and a three-way coverage classification system grounded in GRI "shall" requirements.

---

## System Architecture

```
ESG Report (PDF)
      │
      ▼
┌─────────────────────┐
│   PDF Chunking      │  PyMuPDF → 500-token chunks with page metadata
│   + Embedding       │  OpenAI text-embedding-3-small → ChromaDB
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Hybrid Retrieval  │  Semantic search (ChromaDB) + BM25 keyword search
│   (RRF Fusion)      │  Reciprocal Rank Fusion → top-25 chunks per slot
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Slot Verifier     │  Slot-specific rules injected into GPT-4o-mini prompt
│   (GPT-4 + Rules)   │  Per-slot verdict: covered / partial / missing
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Overall Judgment  │  Critical slot logic → company-level verdict
│   + Evidence        │  Page-level evidence attribution
└─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF Extraction | PyMuPDF |
| Vector Store | ChromaDB (persistent) |
| Embedding | OpenAI text-embedding-3-small |
| Keyword Search | BM25 (rank-bm25) |
| LLM Verifier | GPT-4o-mini via OpenAI API |
| Frontend | Streamlit |
| Containerization | Docker + docker-compose |
| Language | Python 3.11 |

---

## Key Results

Evaluated on 11 company ESG reports across 3 sectors (Finance, Manufacturing, Infrastructure) and 4 countries (KR, UK, FR, DE).

| Standard | Companies | Slots | Slot Accuracy | FN Rate | FC Rate | Company Accuracy |
|----------|-----------|-------|--------------|---------|---------|-----------------|
| GRI 305-1 (Scope 1) | 11 | 7 | 74.0% | 23.4% | **2.6%** | 4/11 (36.4%) |
| GRI 305-2 (Scope 2) | 11 | 5 | 81.8% | 14.5% | **3.6%** | 8/11 (72.7%) |
| GRI 305-3 (Scope 3) | 11 | 5 | 83.6% | 10.9% | **5.5%** | 8/11 (72.7%) |

**FN = False Negative** (system under-predicts disclosure level)  
**FC = False Covered** (system over-predicts — more critical error; kept low across all standards)

The higher FN rate on 305-1 is attributable to Korean financial/public sector companies (IBK, Shinhan, Kepco, Incheon) that publish detailed GHG methodology in separate data packs not indexed by the current single-document pipeline — documented as a known limitation.

---

## Evaluation Design

### Why three labels?

GRI 305 uses "shall" language for required disclosures but also acknowledges that some information may be genuinely unavailable or not applicable (e.g., biogenic CO2 for financial companies). A binary covered/missing system would force ambiguous cases into one extreme. The three-label system (`covered / partial / missing`) enables:

- **covered**: all "shall" requirements for a slot are met
- **partial**: partial evidence found — requires expert review
- **missing**: no relevant disclosure detected

### Critical slot logic

Not all slots are equal. Slots grounded in GRI "shall" requirements (e.g., total Scope 1 figure, consolidation approach) are marked as **critical**. A company-level verdict of `covered` requires all critical slots to be satisfied, regardless of non-critical slot status.

### GRI 305 Coverage

| Standard | Slots Evaluated | Critical Slots |
|----------|----------------|----------------|
| GRI 305-1 | slot_a (total tCO2e), slot_b (gases), slot_c (biogenic), slot_d (base year), slot_e (emission factors/GWP), slot_f (consolidation), slot_g (standards) | slot_a, slot_b, slot_f |
| GRI 305-2 | slot_a (location-based), slot_b (market-based), slot_c (emission factor source), slot_d (contractual instruments), slot_e (standards) | slot_a, slot_b, slot_c, slot_e |
| GRI 305-3 | slot_a (total tCO2e), slot_b (categories), slot_c (standards), slot_d (base year), slot_e (biogenic) | slot_a, slot_b, slot_c |

---

## Dataset

- **11 companies** across Finance (IBK, Shinhan, HSBC, Standard Chartered), Manufacturing (Hyundai, Samsung, Siemens, Schneider Electric), and Infrastructure (KEPCO, Incheon Airport, Heathrow Airport)
- **7,713 chunks** indexed in ChromaDB
- **4 countries**, **2 languages** (EN, KO)
- Manual ground truth labels constructed per slot per company, grounded in GRI original text

---

## Example Output

```
ESG Compliance Checker — Schneider Electric
Standard: GRI 305-1

Overall Status: ✅ COVERED

slot_a  Total Scope 1 (tCO2e)          COVERED    (conf: 0.90)  📄 p.43, p.44
slot_b  Gases included                  COVERED    (conf: 0.90)  📄 p.39, p.42
slot_c  Biogenic CO2                    COVERED    (conf: 0.75)  📄 p.44
slot_d  Base year                       COVERED    (conf: 0.90)  📄 p.41
slot_e  Emission factors & GWP          COVERED    (conf: 0.90)  📄 p.142
slot_f  Consolidation approach          COVERED    (conf: 0.90)  📄 p.42
slot_g  Standards & methodologies       COVERED    (conf: 0.90)  📄 p.44
```

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/Ayana32/esg-report-compliance-checker.git
cd esg-report-compliance-checker
cp .env.example .env          # add your OPENAI_API_KEY
docker-compose up --build
# → open http://localhost:8501
```

### Local

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
streamlit run streamlit_app.py
```

---

## Known Limitations & Future Work

**Current limitations:**
- Single-document indexing only — companies that publish methodology in separate data packs (Shinhan ESG Data Pack, SC Annual Report) show higher FN rates
- Korean financial/public sector reporting pattern differs from Western ESG disclosure norms, causing retrieval gaps on consolidation and GWP slots
- LLM verifier temperature=0 reduces but does not eliminate output variance on borderline slots

**Future work:**
- Multi-document RAG pipeline (index all disclosure-related documents per company)
- Cross-framework tagging (TCFD Metrics & Targets, ISSB S2 alignment)
- Multi-company comparison dashboard
- Ablation study: chunk size, retrieval method, verifier model comparison

---

## Project Structure

```
esg_compliance_checker/
├── streamlit_app.py              # Streamlit UI
├── compliance_checker_v2_hybrid.py  # Core pipeline
├── llm_verifier.py               # GPT-4 slot verifier
├── hybrid_search.py              # BM25 + ChromaDB fusion
├── element_query_generator.py    # Per-slot query generation
├── ground_truth.py               # Manual labels (305-1/2/3)
├── evaluate.py                   # Evaluation script
├── generate_embeddings.py        # ChromaDB indexing
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── data/
    ├── reports/                  # 11 company PDFs
    ├── chunks/                   # JSONL chunks
    └── evaluation/               # Evaluation reports
```

---

*Built as an NLP portfolio project — RAG pipeline for automated ESG disclosure verification across GRI 305 standards.*
