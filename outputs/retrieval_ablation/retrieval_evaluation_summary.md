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

A formal retrieval error analysis was conducted on the reviewed 27-query GRI 305-1 benchmark.

Several recurring retrieval challenges were identified.

### Evidence representation

Numeric and tabular disclosures were not handled consistently by any single first-stage retriever. Some Scope 1 totals were retrieved more effectively by BM25, while others were recovered by semantic retrieval. Hybrid retrieval was more robust across these cases.

Fixed-size chunking can also separate tables, footnotes, and methodological statements across adjacent chunks. This is treated as a plausible contributor rather than an established cause and will be tested through chunking ablation.

### Retrieval ambiguity

Terms such as `baseline`, `base year`, `control`, and `emissions` occur across multiple ESG contexts. This can surface passages about financed emissions, portfolio targets, governance, or other climate disclosures when the query targets operational Scope 1 evidence.

Dense retrieval also returned semantically related but evidentially insufficient passages for several slot-specific queries. This pattern was observed on the benchmark but is not treated as evidence of a general causal limitation of dense retrieval.

### Reranking behaviour

Hybrid + reranking achieved the strongest aggregate performance:

- Hit@5: **0.963**
- MRR: **0.773**
- Relevant evidence retrieved for **26 of 27** queries.

Relative to hybrid retrieval, reranking:

- improved the first relevant rank for **9** queries;
- degraded it for **5** queries;
- left it unchanged for **13** queries;
- recovered **1** Hybrid Top-5 miss;
- caused **0** successful Hybrid queries to become Top-5 misses.

Reranking therefore improved aggregate performance without improving every individual query.

### Genuine unresolved failure

After manual relevance auditing, `Schneider_305-1_slot_d` remained the only Hybrid + rerank Top-5 miss.

Retrieved passages reached the surrounding Schneider Electric climate and base-year context but did not contain the required direct Scope 1 base-year evidence. Because the existing gold chunk remained valid under the predefined annotation criteria, this case was retained as a genuine retrieval failure.

Detailed case-level analysis is documented in
[`retrieval_failure_cases.md`](retrieval_failure_cases.md).

## Evaluation Integrity

Gold annotations are not modified solely because a retrieved result appears plausible.

For each annotation change:

1. Relevance criteria are defined independently of model ranking.
2. Retrieved candidate text is manually inspected.
3. Only evidence satisfying those criteria is added.
4. Previous evaluation results are preserved.
5. Annotation decisions are documented.

This process identified several cases of incomplete relevance annotation during benchmark development. These were classified as evaluation artefacts rather than model failures when the retrieved chunks independently satisfied the existing relevance criteria.

The final reviewed benchmark contains **27 manually audited queries** covering all seven GRI 305-1 disclosure slots across **11 ESG reports**.

## Next Steps

The retrieval benchmark is now sufficiently developed for the planned ablation and end-to-end evaluation work.

Remaining experiments include:

- chunk-size and overlap ablation;
- reranker candidate-set and latency analysis;
- verifier evaluation for `covered`, `partial`, and `missing` decisions;
- small cross-requirement regression checks for GRI 305-2 and GRI 305-3;
- end-to-end retrieval-to-verification evaluation and error analysis.
