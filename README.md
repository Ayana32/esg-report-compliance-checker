# ESG Report Compliance Checker

**Hybrid retrieval and LLM verification system for evidence-grounded ESG disclosure search.**

This project checks whether corporate ESG reports satisfy GRI 305 emissions disclosure requirements by combining dense retrieval, BM25 keyword search, Reciprocal Rank Fusion, and GPT-based slot verification.

The system is designed as a search-oriented RAG pipeline: it retrieves page-grounded evidence from long ESG reports, classifies each disclosure slot as `covered`, `partial`, or `missing`, and evaluates performance against manually constructed ground-truth labels.

---

## Key Features

* Hybrid retrieval pipeline combining ChromaDB dense vector search and BM25 keyword search
* Reciprocal Rank Fusion (RRF) for merging semantic and lexical retrieval results
* Slot-level GPT verification grounded in GRI 305 "shall" requirements
* Three-way disclosure classification: `covered`, `partial`, `missing`
* Manual ground-truth evaluation across 11 ESG reports and 7,713 indexed chunks
* Streamlit interface and Docker-based local deployment
* Error analysis using false negative and false covered rates

---

## Live Demo

> 🚧 **Demo coming soon** — a Streamlit Community Cloud demo is being prepared with a limited set of pre-indexed ESG reports (cached/demo mode) to keep API usage and runtime cost controlled.

In the meantime, the app can be run locally via Docker or Streamlit (see [Quick Start](#quick-start)).

---

## System Architecture

```text
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
│   LLM Verifier      │  GPT-4o-mini with slot-specific GRI rules
│   + Slot Rules      │  Per-slot verdict: covered / partial / missing
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

| Layer            | Technology                      |
| ---------------- | ------------------------------- |
| PDF extraction   | PyMuPDF                         |
| Vector store     | ChromaDB                        |
| Embedding        | OpenAI `text-embedding-3-small` |
| Keyword search   | BM25 (`rank-bm25`)              |
| Retrieval fusion | Reciprocal Rank Fusion          |
| LLM verifier     | GPT-4o-mini via OpenAI API      |
| Frontend         | Streamlit                       |
| Containerization | Docker + docker-compose         |
| Language         | Python 3.11                     |

---

## Evaluation Design

### Why three labels?

GRI 305 uses "shall" language for required disclosures, but some information may be genuinely unavailable, not applicable, or disclosed only partially. A binary `covered` / `missing` system would force ambiguous cases into one extreme.

This project therefore uses a three-label system:

* `covered`: all required evidence for the slot is present
* `partial`: some relevant evidence is found, but expert review is needed
* `missing`: no relevant disclosure is detected

### Critical slot logic

Not all disclosure slots are equally important. Slots grounded in core GRI "shall" requirements are marked as critical.

A company-level verdict of `covered` requires all critical slots to be satisfied, regardless of non-critical slot status.

---

## GRI 305 Coverage

| Standard  | Slots Evaluated                                                                                                                                            | Critical Slots                         |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| GRI 305-1 | `slot_a` total tCO2e, `slot_b` gases, `slot_c` biogenic CO2, `slot_d` base year, `slot_e` emission factors/GWP, `slot_f` consolidation, `slot_g` standards | `slot_a`, `slot_b`, `slot_f`           |
| GRI 305-2 | `slot_a` location-based, `slot_b` market-based, `slot_c` emission factor source, `slot_d` contractual instruments, `slot_e` standards                      | `slot_a`, `slot_b`, `slot_c`, `slot_e` |
| GRI 305-3 | `slot_a` total tCO2e, `slot_b` categories, `slot_c` standards, `slot_d` base year, `slot_e` biogenic                                                       | `slot_a`, `slot_b`, `slot_c`           |

---

## Evaluation Results

Evaluation was conducted at the slot level using manually constructed labels grounded in the original GRI 305 disclosure requirements.

| Standard          | Companies | Slots | Slot Accuracy | FN Rate | FC Rate | Company Accuracy |
| ----------------- | --------: | ----: | ------------: | ------: | ------: | ---------------: |
| GRI 305-1 Scope 1 |        11 |     7 |         74.0% |   23.4% |    2.6% |      4/11, 36.4% |
| GRI 305-2 Scope 2 |        11 |     5 |         81.8% |   14.5% |    3.6% |      8/11, 72.7% |
| GRI 305-3 Scope 3 |        11 |     5 |         83.6% |   10.9% |    5.5% |      8/11, 72.7% |

**FN** = False Negative, where the system under-predicts the disclosure level.
**FC** = False Covered, where the system over-predicts coverage. This is the more critical error type for compliance use cases and remained low across all three standards.

The higher FN rate on GRI 305-1 is mainly attributable to Korean financial and public-sector companies, including IBK, Shinhan, KEPCO, and Incheon Airport, which often publish detailed greenhouse gas methodology in separate data packs not indexed by the current single-document pipeline.

---

## Dataset and Ground Truth

* 11 company ESG reports
* 3 sectors: Finance, Manufacturing, Infrastructure
* 4 countries: Korea, United Kingdom, France, Germany
* 2 reporting languages: English and Korean
* 7,713 PDF chunks indexed in ChromaDB
* Manual slot-level ground-truth labels constructed from original GRI 305 requirements

Companies include IBK, Shinhan, HSBC, Standard Chartered, Hyundai, Samsung, Siemens, Schneider Electric, KEPCO, Incheon Airport, and Heathrow Airport.

---

## Example Output

```text
ESG Compliance Checker — Schneider Electric
Standard: GRI 305-1

Overall Status: COVERED

slot_a  Total Scope 1 (tCO2e)          COVERED    conf: 0.90   p.43, p.44
slot_b  Gases included                 COVERED    conf: 0.90   p.39, p.42
slot_c  Biogenic CO2                   COVERED    conf: 0.75   p.44
slot_d  Base year                      COVERED    conf: 0.90   p.41
slot_e  Emission factors & GWP         COVERED    conf: 0.90   p.142
slot_f  Consolidation approach         COVERED    conf: 0.90   p.42
slot_g  Standards & methodologies      COVERED    conf: 0.90   p.44
```

---

## Quick Start

### Option 1: Docker

```bash
git clone https://github.com/Ayana32/esg-report-compliance-checker.git
cd esg-report-compliance-checker
cp .env.example .env
# Add your OPENAI_API_KEY to .env
docker-compose up --build
```

Then open:

```text
http://localhost:8501
```

### Option 2: Local Streamlit

```bash
git clone https://github.com/Ayana32/esg-report-compliance-checker.git
cd esg-report-compliance-checker
pip install -r requirements.txt
export OPENAI_API_KEY=your_key
streamlit run streamlit_app.py
```

---

## Streamlit Deployment

This app can be deployed on Streamlit Community Cloud using the following settings:

```text
Repository: Ayana32/esg-report-compliance-checker
Branch: main
Main file path: streamlit_app.py
Python version: 3.11
```

Required secret:

```toml
OPENAI_API_KEY = "your-api-key"
```

For deployment, the app should use either:

* a limited set of pre-indexed public ESG reports, or
* a demo mode with cached sample outputs

This prevents excessive API usage and keeps the public demo stable.

---

## Project Structure

```text
esg_compliance_checker/
├── streamlit_app.py                 # Streamlit UI
├── compliance_checker_v2_hybrid.py  # Core compliance-checking pipeline
├── llm_verifier.py                  # GPT-based slot verifier
├── hybrid_search.py                 # BM25 + ChromaDB retrieval fusion
├── semantic_search.py               # Dense retrieval
├── keyword_search.py                # BM25 keyword search
├── element_query_generator.py       # Per-slot query generation
├── ground_truth.py                  # Manual labels for GRI 305-1/2/3
├── evaluate.py                      # Evaluation script
├── generate_embeddings.py           # ChromaDB indexing
├── run_and_compare.py               # Experiment runner
├── requirements_loader.py           # GRI requirement loader
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── data/
    ├── reports/                     # Company ESG reports
    ├── chunks/                      # JSONL chunks
    └── evaluation/                  # Evaluation reports
```

---

## Known Limitations

* The current pipeline uses single-document indexing, so companies that publish methodology in separate ESG data packs or annual reports can show higher false negative rates.
* Korean financial and public-sector ESG reporting patterns differ from Western ESG disclosure norms, which can cause retrieval gaps for consolidation and GWP-related slots.
* The LLM verifier uses temperature 0 to reduce variance, but borderline slots may still require expert review.
* The current evaluation focuses on GRI 305 emissions disclosures and does not yet cover other ESG frameworks.

---

## Future Work

* Multi-document RAG pipeline for indexing annual reports, ESG data packs, and methodology appendices together
* Retrieval ablation study comparing BM25-only, dense-only, hybrid RRF, and hybrid RRF + LLM verification
* FastAPI endpoint for programmatic compliance checking
* Cross-framework tagging for TCFD Metrics & Targets and ISSB S2 alignment
* Multi-company comparison dashboard
* Verifier model comparison across GPT-4o-mini and open-source LLMs

---

## Portfolio Context

This project was built as an NLP portfolio project to demonstrate an end-to-end RAG pipeline for long-document search, evidence-grounded verification, and ESG disclosure evaluation.

It connects my ESG consulting background with my current focus on multilingual NLP, retrieval-augmented generation, and evaluation-driven AI systems.
