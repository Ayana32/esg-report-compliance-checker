# Day 2 — Retrieval Gold-Set Expansion and Audit

## Work completed

- Expanded the retrieval gold set from 9 to 12 queries.
- Added examples for KEPCO, Shinhan Financial Group and Samsung Electronics.
- Diagnosed and repaired malformed JSONL in the Incheon Airport chunk file.
- Created model-independent retrieval annotation guidelines.
- Preserved the initial evaluation output before modifying annotations.
- Audited all 12 queries and documented KEEP/REVISE decisions.
- Identified one missing relevant Samsung chunk containing explicit IPCC AR6 GWP values.
- Updated the gold set through a controlled script with backup and validation.
- Re-ran the retrieval ablation using the reviewed gold set.

## Evaluation results

| Mode | Initial Hit@5 | Reviewed Hit@5 | Initial MRR | Reviewed MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.833 | 0.833 | 0.572 | 0.628 |
| Semantic | 0.833 | 0.833 | 0.528 | 0.528 |
| Hybrid | 0.917 | 0.917 | 0.792 | 0.792 |
| Hybrid + reranker | 0.917 | 1.000 | 0.736 | 0.819 |

## Interpretation

Hybrid retrieval with multilingual reranking achieved 100% Hit@5 and the
highest reviewed MRR of 0.819.

The increase followed a model-independent annotation audit. The Samsung gold
set had omitted a chunk that explicitly reported IPCC AR6 GWP values. The
chunk was added because it satisfied the predefined relevance criteria, not
because of its retrieval position or effect on model performance.

## Key engineering lesson

Evaluation quality depends on annotation completeness as well as model quality.
A retrieval system may correctly return relevant evidence while appearing to
fail if the gold set is incomplete. Preserving initial results and maintaining
an annotation decision log makes the correction transparent and reproducible.
