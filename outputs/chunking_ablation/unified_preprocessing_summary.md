# Chunking Ablation: Preprocessing and Benchmark Integrity

## Objective

Prepare a reproducible chunk-size ablation in which chunk size and overlap
are the only intended experimental variables.

## Prototype corpus audit

An exact baseline-reproduction check revealed that the original prototype
corpus had heterogeneous post-processing provenance across reports.

Rather than preserve those historical inconsistencies, the experimental
corpora were regenerated from the same raw page extractions using one
deterministic preprocessing pipeline.

## Unified preprocessing pipeline

All four configurations use the same sequence:

1. Read the same page-level raw extraction.
2. Tokenize with `tiktoken` using the `gpt-3.5-turbo` encoding.
3. Apply fixed-size token slicing with approximately 10% overlap.
4. Decode each token slice.
5. Record token count from the decoded text via re-encoding.
6. Apply the same conservative generic text normalization.
7. Remove chunks that become empty after cleaning.
8. Preserve original segmentation IDs for all remaining chunks.

No report-specific content deletion rules are used in the final ablation
pipeline.

## Ablation configurations

| Configuration | Chunk size | Overlap | Total chunks | Mean tokens |
|---|---:|---:|---:|---:|
| `250_25` | 250 | 25 | 14,249 | 228.4 |
| `500_50` | 500 | 50 | 7,703 | 418.6 |
| `750_75` | 750 | 75 | 5,494 | 581.3 |
| `1000_100` | 1000 | 100 | 4,434 | 713.9 |

## Benchmark integrity

The primary benchmark contains 27 manually audited GRI 305-1 queries.

After regenerating the corpora with unified preprocessing:

- every reviewed gold and complementary chunk ID remains present;
- no reviewed evidence shows substantive content changes;
- the reviewed split-evidence structure remains valid.

At 250/25, two benchmark cases require complementary evidence across two
chunks:

- Samsung 305-1 slot b;
- Schneider Electric 305-1 slot d.

The 500/50, 750/75, and 1000/100 configurations preserve single-chunk
evidence for all 27 benchmark queries.

## Methodological decision

The detailed ablation does not attempt to reproduce the historically mixed
post-processing state of the prototype corpus. All experimental conditions
are regenerated from the same raw data using one deterministic pipeline so
that retrieval differences can be attributed to segmentation rather than
preprocessing provenance.

The historical corpus remains unchanged for provenance.

## Next step

Run BM25, semantic, hybrid, and hybrid-plus-reranking retrieval over the four
unified corpora using the same 27-query benchmark and fixed retrieval
settings, then compare Hit@5, MRR, per-query rank changes, evidence
coherence, and corpus-size trade-offs.
