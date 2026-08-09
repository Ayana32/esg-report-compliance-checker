# Retrieval Failure Analysis

## Overview

This document analyses retrieval behaviour on a 27-query benchmark covering GRI 305-1 disclosure slots, manually audited across 11 ESG reports. All conclusions are scoped to this benchmark; the sample size is too small to support generalisations beyond it.

**Chunking configuration:** fixed-size chunks of 512 tokens with 64-token overlap, applied uniformly across all documents.

Final Top-5 performance:

| Mode | Hit@5 | MRR |
|---|---:|---:|
| BM25 | 0.852 | 0.705 |
| Semantic | 0.704 | 0.506 |
| Hybrid | 0.926 | 0.680 |
| Hybrid + rerank | **0.963** | **0.773** |

On this benchmark, hybrid retrieval outperformed either BM25 or semantic retrieval alone across both metrics. Adding multilingual cross-encoder reranking improved Hit@5 further and produced the strongest MRR. Given the small benchmark size, individual query outcomes materially affect the aggregate metrics. No significance testing was performed, and results may not hold at larger scale.

Hybrid + reranking retrieved at least one labelled relevant chunk for 26 of 27 queries. The one remaining miss, `Schneider_305-1_slot_d`, was retained as a genuine retrieval failure after manual review.

The analysis separates four issues that can otherwise be conflated:

1. how evidence is represented in ESG reports;
2. ambiguity in retrieval queries;
3. ranking behaviour after candidate retrieval;
4. evaluation artefacts from incomplete relevance annotation.

---

## 1. Evidence Representation Challenges

### 1.1 Numeric and tabular evidence

Several queries depend on values embedded in emissions tables rather than explanatory prose. The fixed-size chunking strategy does not treat tables differently from prose, which may contribute to inconsistent retrieval of tabular evidence.

Representative cases:

- `IBK_305-1_slot_a`: BM25 missed the labelled Scope 1 table; semantic and hybrid retrieval found it, and reranking promoted a relevant chunk to rank 1.
- `Kepco_305-1_slot_a`: BM25 missed the labelled Scope 1 table; semantic, hybrid, and reranked retrieval all succeeded.
- `Incheon_305-1_slot_a`: semantic retrieval returned the canonical emissions table at rank 1; BM25 returned no labelled chunk in the Top 5.
- `Heathrow_305-1_slot_a`: BM25 returned the exact emissions table at rank 1; semantic retrieval did not return either labelled chunk in the Top 5.

Performance varied across companies and document layouts, and no single first-stage retriever dominated every tabular case. Hybrid retrieval was more consistent across these cases than either individual method.

### 1.2 Split and complementary evidence

Some GRI disclosure evidence is distributed across adjacent chunks because a table, footnote, or methodology statement crosses chunk boundaries. This is consistent with the limitations of fixed-size chunking, where boundaries are determined by token count rather than document structure.

Where multiple chunks independently or jointly satisfied the predefined relevance criteria, all valid chunks were included in the gold set. This prevents a correct retrieval from being counted as a failure simply because the evidence spans a boundary.

Future work could explore structure-aware chunking (e.g. splitting at section or table boundaries) to reduce this fragmentation.

### 1.3 Adjacent and near-duplicate evidence

Some reports repeat closely related information across tables, explanatory text, appendices, assurance sections, and ESG data packs.

`Siemens_305-1_slot_a` illustrates this: several chunks from the same reporting region contain closely related Scope 1 information. Semantic and hybrid retrieval prioritised adjacent material; reranking recovered labelled evidence at ranks 3 and 4.

This behaviour is distinct from a complete retrieval failure — the system reaches the correct disclosure region but must discriminate among highly similar neighbouring chunks.

---

## 2. Retrieval Ambiguity

### 2.1 Lexical ambiguity

Terms such as `baseline`, `base year`, `control`, and `emissions` appear across multiple ESG contexts. A query seeking an operational Scope 1 base year can therefore surface passages about financed-emissions baselines, portfolio targets, or supplier targets.

Representative cases:

- `Siemens_305-1_slot_d`: semantic retrieval placed labelled base-year evidence at rank 4; hybrid retrieval promoted it to rank 1.
- `HSBC_305-1_slot_d`: semantic retrieval returned other ESG passages rather than the labelled operational base-year disclosure; hybrid and reranked retrieval recovered the relevant evidence.

### 2.2 Scope ambiguity

A related problem occurs when passages refer to emissions but describe a different reporting scope. This was particularly visible in financial-sector reports, where operational Scope 1 emissions coexist with financed emissions, client emissions, portfolio targets, and Scope 3 disclosures.

For GRI 305-1, relevance requires not only topical similarity but evidence referring specifically to the company's own direct Scope 1 emissions.

### 2.3 Semantically related but evidentially insufficient passages

Dense retrieval frequently returned passages that were topically close to the query but lacked the specific evidence required by the slot. This pattern was observed across several queries; a causal explanation is not established by this benchmark, but it suggests that semantic similarity alone may be insufficient for narrow evidential slots.

Representative cases:

- `Samsung_305-1_slot_e`: semantic retrieval did not return either labelled IPCC/GWP chunk in the Top 5; hybrid and reranked retrieval returned labelled evidence at rank 1.
- `HSBC_305-1_slot_d`, `_slot_f`, and `_slot_b`: semantic retrieval repeatedly preferred ESG Data Pack material over labelled operational-emissions disclosures.
- `Samsung_305-1_slot_g`: semantic retrieval missed the labelled methodology evidence; hybrid retrieval returned it at rank 5; reranking promoted it to rank 1.

---

## 3. Ranking Behaviour

### 3.1 Reranking recovery

The multilingual cross-encoder frequently improved the ordering of relevant candidates already returned by hybrid retrieval.

Examples:

- `IBK_305-1_slot_a`: hybrid rank 2 → reranked rank 1
- `HSBC_305-1_slot_f`: hybrid rank 4 → reranked rank 1
- `Samsung_305-1_slot_g`: hybrid rank 5 → reranked rank 1
- `Heathrow_305-1_slot_a`: hybrid rank 4 → reranked rank 1

Across the full benchmark, hybrid + reranking achieved the highest Hit@5 (0.963) and MRR (0.773).

### 3.2 Reranking degradation

Reranking did not improve every query. In `Schneider_305-1_slot_e`, BM25 placed the labelled evidence at rank 1, hybrid retrieval placed it at rank 2, and hybrid + reranking moved it to rank 5 — a meaningful drop.

Across the 27 queries, reranking improved the first relevant rank for **9** queries, degraded it for **5**, and left it unchanged for **13**. It recovered **1** query or queries that hybrid retrieval had missed within the Top 5, while causing **0** previously successful hybrid query or queries to become Top-5 misses.

The aggregate benefit of reranking is therefore better understood as a benchmark-level tendency rather than a guarantee for every individual query.

---

## 4. Evaluation Artefacts

### 4.1 Incomplete gold annotations

A retrieved chunk can be genuinely relevant even if it was absent from the initial gold set. For this reason, all retrieval misses were manually audited before being classified as model failures.

Several retrieved chunks were found to independently satisfy the predefined relevance criteria despite being absent from the original annotations. Examples include `IBK_305-1_slot_d`, `Heathrow_305-1_slot_a`, and an earlier audit of `HSBC_305-1_slot_d`. These chunks were added because their content satisfied the existing annotation rules — not to improve scores.

Such cases are classified as **evaluation artefacts from incomplete relevance annotation**, not retrieval failures. Without this distinction, a correct retrieval can be incorrectly counted as a model error.

---

## 5. Genuine Unresolved Retrieval Failure

### `Schneider_305-1_slot_d`

This query remained the only Hybrid + rerank Top-5 miss after the gold set was finalised.

The labelled chunk contains the 2021 base-year context together with Schneider Electric's direct Scope 1 emissions value. Across all four retrieval modes, the Top-5 results instead contained closely related material about 2021 base years, Scope 1 and Scope 2 targets, emissions-reduction pathways, Scope 3 baselines, and other nearby climate disclosures — none of which contained the required direct Scope 1 base-year evidence.

Because the existing gold evidence remained valid under the predefined annotation criteria, the benchmark was not modified after this audit. The case is retained as a genuine unresolved retrieval failure.

This failure is particularly informative: the system reaches the correct semantic region but fails to identify the specific chunk needed to answer the slot. It provides a concrete case for subsequent chunking and candidate-set ablations.

---

## Main Findings

Five benchmark-specific conclusions:

1. **Hybrid retrieval outperformed BM25 and semantic retrieval individually on this 27-query benchmark.** Given the small benchmark size, individual query outcomes materially affect the aggregate metrics, and no significance testing was performed.
2. **Dense retrieval tended to return semantically related but evidentially insufficient passages for several slot-specific queries**, though the benchmark does not establish a causal explanation for this pattern.
3. **Tabular, split, and adjacent evidence produced retrieval behaviour that differed from ordinary prose retrieval**, with no single first-stage retriever dominating every case. Fixed-size chunking is one plausible contributor and will be tested in the subsequent chunking ablation.
4. **Multilingual reranking improved aggregate performance but not every individual query.** Relative to hybrid retrieval, it improved the first relevant rank for 9 queries, degraded it for 5, and left it unchanged for 13.
5. **Manual relevance auditing was necessary to separate genuine retrieval failures from incomplete gold annotations.** Without it, correct retrievals would have been counted as model errors.

Overall, these results support hybrid retrieval followed by reranking as the current default configuration for this GRI 305-1 benchmark while motivating further experiments on chunking strategy and candidate-set size. The Schneider base-year failure provides a concrete starting point for that work.
