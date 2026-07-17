# Retrieval Evaluation

## Scope

This preliminary retrieval ablation uses nine manually verified
GRI 305-1 evidence queries across IBK, Hyundai, Heathrow, Siemens,
and Schneider Electric.

Each query may contain multiple valid gold chunks because the same
disclosure can appear in a summary table, detailed table, or
methodology section.

## Retrieval Modes

- BM25 keyword retrieval
- Semantic vector retrieval
- Hybrid retrieval using Reciprocal Rank Fusion
- Hybrid retrieval followed by multilingual cross-encoder reranking

Hybrid retrieval uses up to 20 candidates from each underlying
retriever before fusion. The reranker scores the fused candidate
pool and returns the final top-k results.

## Results

| Mode | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|
| BM25 | 0.889 | 1.000 | 0.722 |
| Semantic | 0.889 | 0.889 | 0.537 |
| Hybrid | 0.889 | 0.889 | 0.722 |
| Hybrid + reranker | 1.000 | 1.000 | 0.815 |

## Findings

The multilingual cross-encoder reranker achieved the strongest
performance, reaching 100% Hit@5 and increasing MRR from 0.722 for
hybrid retrieval to 0.815.

BM25 retrieved all gold evidence by rank 10, but failed to retrieve
the IBK Scope 1 evidence within the top five results.

Semantic retrieval failed to retrieve the Siemens Scope 1 total
within the top ten results. It instead returned adjacent and
methodologically related chunks.

The Siemens query demonstrated the value of reranking. Relevant
evidence was present in the larger hybrid candidate pool but was
ranked outside the final hybrid top ten. Cross-encoder reranking
promoted it into the final result set.

## Error Analysis

### Keyword mismatch

The English IBK query did not align well with the Korean wording in
the report, causing BM25 to miss the relevant evidence at top five.

### Chunk-boundary effects

The Siemens Scope 1 table spans adjacent chunks. Related chunks were
retrieved more highly than the chunk containing the exact total.

### Incomplete gold annotations

Manual review identified valid evidence chunks that were initially
missing from the gold set for Siemens and Schneider Electric.
Multiple valid gold chunks are now allowed per query.

### Ranking failure

Some relevant evidence existed in the hybrid candidate pool but was
ranked outside the requested top-k. Cross-encoder reranking improved
this ordering.

## Limitations

The evaluation set contains only nine queries and focuses on GRI
305-1. Results should therefore be treated as preliminary rather
than as a general benchmark.

Future evaluation should include more companies, requirements,
natural user questions, multilingual queries, and independently
reviewed gold annotations.
