# Retrieval Gold Annotation Guidelines

## Purpose

This gold set evaluates whether a retrieval method returns evidence that can support an ESG compliance decision for a specific disclosure slot.

The gold set must remain independent of the retrieval method being evaluated. Retrieved results must never be added to the gold set solely because they improve a model's score.

## Unit of Annotation

Each evaluation item contains:

- one company
- one GRI requirement
- one disclosure slot
- one natural-language query
- one or more valid evidence chunks

A gold chunk does not necessarily need to satisfy the entire disclosure slot by itself. Multiple complementary chunks may jointly provide the evidence required for the final compliance decision.

## Relevant Chunk

A chunk is relevant when it contains explicit information that materially supports at least one required component of the target slot.

Examples include:

- a directly reported Scope 1 emissions value
- an explicitly named organisational boundary or consolidation approach
- a stated base year
- a named emission-factor source
- a named GWP framework or version
- explicitly listed greenhouse gases

## Non-Relevant Chunk

A chunk is not relevant when it contains only:

- a GRI or framework index pointing to another page
- a generic statement that emissions are measured or managed
- a target without the requested actual disclosure
- a topic heading without substantive evidence
- information about a different company, reporting boundary or disclosure slot
- a vague mention that does not materially support the compliance decision

## Partial and Complementary Evidence

A chunk may be relevant even when it supports only part of a compound slot.

For example, for:

`Emission factors and GWP source/version`

one chunk may identify the emission-factor methodology while another identifies the GWP framework and version.

Both may be included as valid gold evidence when they materially support the slot.

## Duplicate Evidence

Near-duplicate chunks should not automatically be added.

Include duplicate or repeated chunks only when:

- they are independently valid retrieval targets, and
- retrieving either chunk would provide sufficient useful evidence to the downstream verifier.

Avoid adding repeated index entries or duplicated boilerplate solely to make retrieval evaluation easier.

## Annotation Procedure

For every query:

1. Read the slot definition.
2. Review the candidate chunk without considering its retrieval rank or mode.
3. Decide whether the chunk materially supports the slot.
4. Record all independently valid evidence chunks.
5. Exclude index-only, generic or off-scope chunks.
6. Record the reasoning in the `notes` field.
7. Apply the same standard to every company and retrieval mode.

## Evaluation Integrity

Gold annotations must not be changed merely because:

- BM25 retrieved a non-gold chunk
- semantic retrieval missed an existing gold chunk
- a reranker promoted a new chunk
- adding a chunk would improve Hit@k or MRR

A gold annotation may be corrected only when manual review shows that:

- a valid evidence chunk was previously omitted
- an existing gold chunk does not satisfy the guideline
- the company, page, year or reporting boundary was annotated incorrectly
- the annotation is duplicated or malformed

All corrections should be documented and applied before reporting the final reviewed evaluation results.
