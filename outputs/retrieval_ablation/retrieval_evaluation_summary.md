# Retrieval Evaluation Summary

## Overview

The retrieval subsystem is evaluated on a manually annotated benchmark for GRI 305-1 disclosure evidence across multilingual sustainability reports.

The benchmark currently contains 18 queries spanning 10 companies and six disclosure slot types.

Retrieval methods compared:

- BM25
- Dense semantic retrieval
- Hybrid retrieval
- Hybrid retrieval + multilingual cross-encoder reranking

Evaluation metrics:

- Hit@5
- Mean Reciprocal Rank (MRR)

## Benchmark Construction

The benchmark was expanded from an initial 9 queries to 12 manually audited queries and subsequently to 18 queries.

Gold evidence is annotated independently of retrieval results using documented relevance criteria. A chunk is considered relevant when it explicitly or materially supports the requested disclosure slot. Multiple complementary chunks may jointly constitute valid gold evidence.

The current 18-query benchmark has the following slot distribution:

| Slot | Queries |
|---|---:|
| Total Scope 1 GHG emissions | 5 |
| Gases included | 2 |
| Biogenic CO2 emissions | 2 |
| Base year | 2 |
| Emission factors / GWP | 3 |
| Consolidation approach | 4 |

The benchmark covers:

- HSBC
- Heathrow
- Hyundai
- IBK
- Incheon International Airport
- KEPCO
- Samsung
- Schneider Electric
- Shinhan
- Siemens

## Results

### Audited 18-query benchmark

| Retrieval method | Hit@5 | MRR |
|---|---:|---:|
| BM25 | 0.833 | 0.669 |
| Semantic | 0.722 | 0.519 |
| Hybrid | 0.944 | 0.727 |
| Hybrid + rerank | **1.000** | **0.815** |

Hybrid retrieval substantially improves over dense retrieval alone, while multilingual cross-encoder reranking provides the strongest overall ranking performance.

On the current manually audited 18-query benchmark, Hybrid + Rerank retrieves at least one relevant evidence chunk in the Top 5 for all queries.

Because the benchmark remains relatively small, these results are treated as diagnostic rather than as a definitive estimate of production performance.

## Annotation Audit

Retrieval evaluation exposed two cases where apparently incorrect retrievals were caused by incomplete gold annotations rather than model failures.

### Samsung — emission factors / GWP

An explicit IPCC AR6 GWP evidence chunk had initially been omitted from the gold set. Evidence-level review showed that the chunk independently satisfied the predefined relevance criteria, so it was added to the annotation.

### HSBC — base year

An apparent Hybrid + Rerank miss returned:

`2024_UK_HSBC_Annual_EN_p0061_c0003`

Inspection showed that the chunk directly reports HSBC's 2019 own-operations baseline table, including Scope 1 and market-based Scope 2 emissions. It was therefore added as complementary gold evidence.

Other retrieved HSBC chunks mentioning 2019 baselines were excluded because they referred to financed emissions or sector portfolios rather than HSBC's operational Scope 1 disclosure.

After this audit, Hybrid + Rerank increased from 0.944 to 1.000 Hit@5 on the 18-query benchmark.

## Failure Analysis

Several recurring retrieval challenges have emerged.

### Scope ambiguity

Terms such as `baseline`, `Scope 1 and 2`, and `financial control` occur in multiple ESG contexts. Retrieval may return financed-emissions or financial-reporting evidence when the query targets operational GHG disclosures.

### Lexical ambiguity

Generic terms such as `financial control` can retrieve governance, audit, and internal-control passages instead of the GHG consolidation approach.

### Numerical and tabular evidence

BM25 remains competitive for queries whose correct evidence is expressed through explicit numerical values, table labels, years, or disclosure terminology.

### Dense retrieval limitations

Dense semantic retrieval currently performs below BM25 and hybrid retrieval on this benchmark. Semantically related sustainability passages can outrank the precise evidence required by the disclosure slot.

### Reranking

Cross-encoder reranking improves the hybrid candidate set substantially, producing the best Hit@5 and MRR of the evaluated configurations.

## Evaluation Integrity

Gold annotations are not modified solely because a retrieved result appears plausible.

For each annotation change:

1. Relevance criteria are defined independently of model ranking.
2. Retrieved candidate text is manually inspected.
3. Only evidence satisfying those criteria is added.
4. Previous evaluation results are preserved.
5. Annotation decisions are documented.

This distinction is important because incomplete gold sets can incorrectly classify valid retrievals as model failures.

## Next Steps

The benchmark will be expanded further before final conclusions are drawn.

Planned evaluation work includes:

- additional company and disclosure-slot coverage
- retrieval failure taxonomy
- chunk-size and overlap ablation
- reranker candidate-set and latency analysis
- end-to-end compliance verification evaluation
