# Chunking Retrieval Ablation

## Objective

This experiment evaluates how chunk size affects retrieval quality for the GRI 305-1 compliance benchmark while holding the remaining retrieval pipeline fixed.

The goal is not to identify the configuration with the highest score on a single metric in isolation. The final chunking configuration is selected based on four considerations:

1. retrieval effectiveness;
2. evidence coherence;
3. reranking stability; and
4. corpus size and downstream processing cost.

The benchmark contains 27 manually audited GRI 305-1 queries covering seven disclosure slots across 11 company reports.

---

## Experimental controls

All four chunking conditions were regenerated from the same raw page extractions using the deterministic preprocessing pipeline established before this experiment.

The evaluated configurations were:

| Configuration | Chunk size | Overlap | Corpus chunks |
| ------------- | ---------: | ------: | ------------: |
| `250_25`      |        250 |      25 |        14,249 |
| `500_50`      |        500 |      50 |         7,703 |
| `750_75`      |        750 |      75 |         5,494 |
| `1000_100`    |       1000 |     100 |         4,434 |

The following components were held fixed across configurations:

* page-level raw source documents;
* tokenizer and chunking semantics;
* preprocessing policy;
* query set;
* company-level metadata filtering;
* BM25 implementation and tokenization;
* semantic embedding model: `text-embedding-3-small`;
* hybrid fusion using Reciprocal Rank Fusion (RRF);
* hybrid candidate depth: 20 results from each retrieval channel;
* cross-encoder reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`;
* evaluation cutoff: Top 5.

Each configuration was independently embedded and stored in its own ChromaDB collection. This prevents semantic retrieval from reusing embeddings generated from a different segmentation condition.

---

## Gold-evidence definition

The gold mappings were manually reviewed before running the final retrieval comparison.

For `500_50`, `750_75`, and `1000_100`, all 27 benchmark queries have at least one independently sufficient single chunk.

For `250_25`, 25 queries have single-chunk evidence and two require complementary evidence distributed across multiple chunks:

* `Samsung_305-1_slot_b`
* `Schneider_305-1_slot_d`

These cases are evaluated using an evidence-completion criterion rather than treating either partial chunk as independently sufficient.

For a single-chunk query:

* Hit@5 is successful when at least one independently sufficient gold chunk occurs in the Top 5.
* Reciprocal rank is based on the highest-ranked sufficient gold chunk.

For a split-evidence query:

* Hit@5 is successful only when all chunks belonging to at least one reviewed complementary evidence group occur in the Top 5.
* The completion rank is the maximum rank among the required chunks in that group.
* Reciprocal rank is calculated from that completion rank.

This prevents small chunks from receiving full credit when retrieval returns only a partial disclosure.

---

## Aggregate retrieval results

### Hit@5

| Configuration |              BM25 |          Semantic |            Hybrid |   Hybrid + rerank |
| ------------- | ----------------: | ----------------: | ----------------: | ----------------: |
| `250_25`      |     0.815 (22/27) |     0.407 (11/27) |     0.704 (19/27) |     0.815 (22/27) |
| **`500_50`**  | **0.852 (23/27)** | **0.667 (18/27)** | **0.926 (25/27)** | **0.963 (26/27)** |
| `750_75`      |     0.815 (22/27) |     0.593 (16/27) |     0.852 (23/27) |     0.852 (23/27) |
| `1000_100`    | **0.889 (24/27)** | **0.704 (19/27)** |     0.889 (24/27) |     0.852 (23/27) |

### MRR

| Configuration |      BM25 | Semantic |    Hybrid | Hybrid + rerank |
| ------------- | --------: | -------: | --------: | --------------: |
| `250_25`      |     0.601 |    0.311 |     0.537 |           0.698 |
| **`500_50`**  |     0.705 |    0.485 | **0.649** |       **0.779** |
| `750_75`      |     0.633 |    0.469 |     0.593 |           0.665 |
| `1000_100`    | **0.781** |    0.477 | **0.652** |           0.639 |

No single configuration dominates every retrieval mode.

In particular, `1000_100` achieves the strongest standalone BM25 performance, with Hit@5 of 0.889 and MRR of 0.781. Therefore, the results do not support a general claim that larger chunks are intrinsically worse for retrieval.

The relevant comparison for the deployed retrieval stack, however, is hybrid retrieval followed by cross-encoder reranking. Under that pipeline, `500_50` achieves both the highest Hit@5 and the highest MRR.

---

## Evidence fragmentation at 250 tokens

The manual gold audit identified two cases in which the `250_25` segmentation divided evidence that remained coherent within a single chunk at larger chunk sizes.

### Samsung 305-1 slot b

The disclosure of included greenhouse gases is divided between two complementary chunks.

Under BM25, both required chunks are retrieved at ranks 1 and 2, so the evidence is complete at rank 2.

Under semantic, hybrid, and hybrid-plus-reranking retrieval, only part of the required evidence is returned within the Top 5. These modes therefore correctly receive no complete-evidence hit for this query.

This distinction matters because crediting either fragment independently would overstate the ability of the retrieval system to supply the complete disclosure.

### Schneider Electric 305-1 slot d

The base-year evidence is also split between two chunks under `250_25`.

None of the four retrieval modes retrieves both required fragments within the Top 5.

This provides direct evidence that the smallest segmentation can create an additional retrieval burden when a disclosure depends on relationships between nearby table elements, such as a year/header and its associated value.

These two cases do not establish that 250-token chunks are generally inferior. They show a narrower result: for this benchmark, smaller segmentation can reduce evidence coherence in disclosures whose interpretation spans adjacent table or list elements.

---

## Hybrid-to-reranker behaviour

The effect of the cross-encoder was also compared with the preceding hybrid ranking.

| Configuration | Hybrid hits | Reranked hits | Recovered misses | Lost hits |
| ------------- | ----------: | ------------: | ---------------: | --------: |
| `250_25`      |          19 |            22 |                4 |         1 |
| **`500_50`**  |      **25** |        **26** |            **1** |     **0** |
| `750_75`      |          23 |            23 |                3 |         3 |
| `1000_100`    |          24 |            23 |                3 |         4 |

At `500_50`, reranking:

* recovers one hybrid miss;
* loses no existing Hybrid Top-5 hits;
* improves the completion rank of nine queries;
* degrades the completion rank of four queries that nevertheless remain successful.

This is the most stable Hit@5 transition among the four configurations.

By contrast, reranking under `750_75` recovers three misses but also loses three existing hits. Under `1000_100`, it recovers three while losing four.

These results show that the effect of the reranker is segmentation-dependent. They do not, by themselves, identify why larger chunks are more frequently demoted.

One possible explanation is that longer chunks contain more competing contextual material, making fine-grained relevance discrimination more difficult for the cross-encoder. The current experiment does not directly isolate or test that mechanism, so contextual noise should be treated as a hypothesis rather than a demonstrated cause.

---

## Queries particularly favoured by 500/50

Under hybrid-plus-reranking, `500_50` succeeds on several queries missed by one or more alternative configurations:

| Query            | 500/50 rank | Other configurations missing |
| ---------------- | ----------: | ---------------------------- |
| Schneider slot f |           2 | 250                          |
| HSBC slot d      |           2 | 250, 750                     |
| HSBC slot f      |           1 | 1000                         |
| HSBC slot b      |           2 | 750, 1000                    |
| Samsung slot b   |           2 | 250                          |
| Schneider slot e |           5 | 250, 750, 1000               |
| Heathrow slot a  |           1 | 750, 1000                    |

This pattern is not confined to one company or one disclosure slot, which makes it less likely that the aggregate advantage of `500_50` is driven by a single duplicated or unusually easy benchmark case.

However, the benchmark contains only 27 queries, so these differences should still be interpreted descriptively rather than as estimates of population-level statistical superiority.

---

## The remaining 500/50 failure

The only Hybrid+Rerank Top-5 miss under `500_50` is:

`Schneider_305-1_slot_d`

The reviewed gold chunk is:

`2024_FR_SchneiderElectric_Sustainability_EN_p0142_c0002`

A deeper diagnostic gives the following ranks:

* BM25 rank: 6
* semantic rank: outside Top 20
* RRF rank: 17
* cross-encoder rerank rank: 24
* fused candidate pool size: 29

This failure should not be described as complete candidate-retrieval failure.

BM25 retrieves the relevant chunk immediately outside the evaluation cutoff at rank 6. The chunk also enters the hybrid candidate pool and appears at RRF rank 17.

The cross-encoder subsequently assigns it a lower relative position, moving it from RRF rank 17 to reranker rank 24.

Importantly, the query was already outside the Top-5 cutoff before reranking. Therefore, the reranker did not create the original Top-5 miss; it further degraded an already low-ranked relevant candidate.

The most precise attribution is:

> The relevant evidence entered the candidate pool, but neither fusion nor reranking promoted it into the Top 5; reranking further demoted the candidate from RRF rank 17 to rank 24.

This case is retained as a genuine failure rather than modifying the gold labels after observing the retrieval output.

---

## Configuration selection

`500_50` is selected as the default chunking configuration for subsequent system experiments.

The selection is based on the following jointly observed properties.

### 1. Best final retrieval effectiveness

Under the intended Hybrid+Rerank pipeline:

* Hit@5 = 26/27 = 0.963
* MRR = 0.779

Both are the highest among the four configurations.

### 2. Complete single-chunk evidence coverage

All 27 benchmark queries retain at least one independently sufficient evidence chunk at `500_50`.

By comparison, `250_25` creates two reviewed split-evidence cases.

### 3. Stable interaction with reranking

The `500_50` hybrid stage retrieves 25/27 queries successfully.

Cross-encoder reranking increases this to 26/27 without converting any existing Hybrid Top-5 hit into a miss.

The other configurations exhibit at least one reranker-induced loss of an existing Top-5 hit.

### 4. Moderate corpus size

`500_50` contains 7,703 chunks.

This is approximately half the number of chunks in `250_25` while still providing substantially stronger final retrieval performance on this benchmark.

`750_75` and `1000_100` produce smaller corpora, but neither matches the final Hybrid+Rerank effectiveness of `500_50`.

---

## Interpretation

The experiment supports a benchmark-specific trade-off rather than a universal optimal chunk size.

Within this ESG disclosure corpus:

* smaller chunks can improve local specificity but can fragment table- or list-based evidence;
* larger chunks can perform strongly under lexical retrieval, as demonstrated by `1000_100` BM25;
* the intermediate `500_50` configuration provides the strongest observed balance across hybrid retrieval effectiveness, evidence coherence, and reranking stability.

The evidence therefore supports selecting `500_50` for the next stages of this system.

It does **not** support the stronger claim that 500-token chunks are generally optimal for ESG documents, RAG systems, or compliance retrieval outside this benchmark.

---

## Limitations

Several limitations constrain the interpretation of the results.

### Benchmark size

The evaluation contains 27 manually audited queries. This provides sufficient resolution for detailed error analysis but is too small to establish broad statistical superiority across ESG reporting domains.

### Benchmark scope

The deep evaluation is limited to GRI 305-1. Adjacent GRI 305 disclosures will be evaluated separately as smoke or regression tests rather than treated as part of the primary chunking benchmark.

### Coupled chunk size and overlap

Overlap is maintained at approximately 10% for each configuration. The experiment therefore evaluates chunk-size/overlap configurations rather than independently estimating the causal effect of chunk size while holding absolute overlap constant.

### Fixed candidate depth

Hybrid candidate depth is fixed at 20 per retrieval channel. A different candidate depth may change the interaction between segmentation and reranking. Candidate-depth sensitivity is addressed in the subsequent reranker ablation.

### Single embedding and reranker models

The conclusions are conditional on `text-embedding-3-small` and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Different retrieval models may interact differently with segmentation.

---

## Decision

**Selected configuration: `500_50`**

This configuration will be carried forward into the reranker candidate-depth, latency, verifier, and end-to-end evaluations.

The selection is based on observed benchmark performance and evidence integrity rather than on reproducing the historical prototype segmentation or optimising a single retrieval metric.

