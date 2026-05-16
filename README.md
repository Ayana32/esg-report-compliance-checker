# esg-report-compliance-checker
ESG Compliance Checker - Chunks Dataset
Generated: 2026-02-21
Version: v3 (Final - Front matter skip, GRI extraction, TCFD tagging)

📊 Dataset Overview
Summary Statistics
MetricValueTotal Chunks3,140Total Tokens1,056,722Avg Tokens/Chunk336.5Reports1,499 chunks (3 PDFs)Standards1,641 chunks (12 PDFs)
Language Distribution
LanguageChunksPercentageEnglish (EN)1,98463.2%Korean (KO)1,15636.8%

📄 Reports Dataset
Location: data/chunks/reports/
Sector Balance
SectorChunksPercentageManufacturing72148.1%Finance55437.0%Infrastructure22414.9%
Report Details
ReportCompanySectorLanguageChunksAvg TokensSchneider ElectricSchneiderManufacturingEN721443IBK BankIBKFinanceKO554429Heathrow AirportHeathrowInfrastructureEN224354
Files: 3 JSONL files (1 per report)

📚 Standards Dataset
Location: data/chunks/standards/
Standard Type Distribution
TypeChunksPercentageK-ESG60236.7%GRI42525.9%TCFD40824.9%SASB20612.6%
Standard Details
GRI (Global Reporting Initiative) - 5 files
StandardChunksAvg TokensGRI 2 General Disclosures141192GRI 1 Foundation95228GRI 3 Material Topics78239GRI 305 Emissions66218GRI 201 Economic Performance45177
Special Features:

GRI disclosure codes extracted (e.g., "GRI 305-1")
12 unique codes identified

TCFD (Task Force on Climate-related Financial Disclosures) - 2 files
StandardChunksAvg TokensTCFD Annex228253TCFD Recommendations180235
Special Features:

TCFD pillar tagging (Governance, Strategy, Risk Management, Metrics & Targets)
Front matter skip (pages 1-3)

SASB (Sustainability Accounting Standards Board) - 4 files
StandardChunksAvg TokensElectric Utilities77223Commercial Banks45206Electronic Manufacturing43196Automobiles41193
Special Features:

Sector-specific tagging
Cover page skip (page 1)

K-ESG (Korean ESG Guidelines) - 1 file
StandardChunksAvg TokensK-ESG Guideline 2021602312
Special Features:

Korean language processing
UI navigation text removal
Guide page skip (page 2)

Files: 12 JSONL files (1 per standard)

Chunk Structure
Each chunk is stored as a JSON object with the following structure:
Reports Chunk Schema
json{
  "chunk_id": "2024_KR_IBKBank_Sustainability_KO_p0012_c0003",
  "text": "chunk content...",
  "metadata": {
    "report_id": "2024_KR_IBKBank_Sustainability_KO",
    "company_id": "IBK",
    "sector": "Finance",
    "language": "KO",
    "page_num": 12,
    "chunk_num": 3,
    "token_count": 450,
    "char_count": 523
  }
}
Standards Chunk Schema
json{
  "chunk_id": "GRI_305_Emissions_2016_EN_p0005_c0002",
  "text": "chunk content...",
  "metadata": {
    "standard_id": "GRI_305_Emissions_2016_EN",
    "standard_type": "GRI",
    "language": "EN",
    "page_num": 5,
    "chunk_num": 2,
    "token_count": 480,
    "char_count": 567,
    "gri_codes": ["305-1", "305-2"]
  }
}

✅ Quality Metrics
Validation Results
CheckStatusDetailsEmpty chunks✅ Pass0 empty chunksDuplicate IDs✅ Pass0 duplicatesMissing metadata✅ PassAll required fields presentVery short chunks✅ Pass23 chunks (<20 tokens, 0.73%)
Token Distribution
MetricValueMinimum2 tokensMaximum502 tokensMean336.5 tokensMedian488 tokens
Note: Very short chunks (<20 tokens) are intentionally kept for important GRI disclosures and section headers.

🛠️ Processing Pipeline
Version History
v3 (Final) - Current

Front matter page skip (TCFD pages 1-3, SASB page 1, K-ESG page 2)
Enhanced GRI code extraction (3-digit only)
Improved TCFD pillar detection
K-ESG UI navigation removal
SASB header boilerplate removal
Empty chunk removal

v2

Copyright/boilerplate removal
GRI code extraction (initial)
TCFD pillar tagging
K-ESG UI text removal (partial)

v1

Basic fixed-size chunking (500 tokens, 50 overlap)
Page-level text extraction
Language detection

Chunking Strategy

Method: Fixed-size token chunking
Size: 500 tokens per chunk
Overlap: 50 tokens between chunks
Tokenizer: tiktoken (gpt-3.5-turbo)

Text Cleaning
Applied to all chunks:

Whitespace normalization
Special character standardization (smart quotes, dashes)
Bullet point normalization
Language-specific fixes:

Korean: Zero-width space removal
English: Hyphenation fix




Usage
Load Chunks (Python)
pythonimport json
from pathlib import Path

# Load a single report
chunks = []
with open('data/chunks/reports/2024_KR_IBKBank_Sustainability_KO.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))

# Load all reports
for jsonl_file in Path('data/chunks/reports').glob('*.jsonl'):
    # process...

# Load all standards
for jsonl_file in Path('data/chunks/standards').glob('*.jsonl'):
    # process...
Filter by Metadata
python# Get all GRI chunks
gri_chunks = [c for c in chunks if c['metadata'].get('standard_type') == 'GRI']

# Get chunks with GRI 305 code
emissions_chunks = [c for c in chunks 
                   if 'gri_codes' in c['metadata'] 
                   and '305' in str(c['metadata']['gri_codes'])]

# Get Korean chunks
korean_chunks = [c for c in chunks if c['metadata']['language'] == 'KO']
```

---

## File Structure
```
data/chunks/
├── README.md                     (this file)
├── reports/
│   ├── 2024_UK_HeathrowAirport_Sustainablity_EN.jsonl
│   ├── 2024_FR_SchneiderElectric_Sustainability_EN.jsonl
│   └── 2024_KR_IBKBank_Sustainability_KO.jsonl
└── standards/
    ├── GRI_1_Foundation_2021_EN.jsonl
    ├── GRI_2_GeneralDisclosures_2021_EN.jsonl
    ├── GRI_3_MaterialTopics_2021_EN.jsonl
    ├── GRI_201_EconomicPerformance_2016_EN.jsonl
    ├── GRI_305_Emissions_2016_EN.jsonl
    ├── TCFD_Recommendations_2017_EN.jsonl
    ├── TCFD_Annex_2021_EN.jsonl
    ├── SASB_CommercialBanks_EN.jsonl
    ├── SASB_Automobiles_EN.jsonl
    ├── SASB_ElectricUtilities_EN.jsonl
    ├── SASB_ElectronicManufacturing_EN.jsonl
    └── KESG_Guideline_2021_KO.jsonl

🔮 Next Steps

Week 3: Parse remaining 8 reports (expand to 11 total)
RAG System: Build retrieval-augmented generation pipeline
Embeddings: Generate vector embeddings for semantic search
Compliance Checker: Implement GRI/TCFD compliance validation


📝 Notes

All chunks validated for quality (no empty chunks, no duplicates)
Metadata fields vary by type (reports vs standards)
GRI codes only available in GRI standard chunks
TCFD pillar tags only in TCFD chunks
Korean text properly encoded (UTF-8)
URLs preserved in text (useful for references)


Created by: ESG Compliance Checker Pipeline v3
Last Updated: 2026-02-21