# Adaptive Retrieval POC

## Purpose

This folder contains proof-of-concept implementations for **adaptive context retrieval** - techniques that determine the optimal amount of context (chunk vs heading vs document) to retrieve per query.

## Problem Statement

Chunks are optimized for **finding** relevant content (precise embeddings), but are often insufficient for **answering** questions (lack context). Different queries need different amounts of context:

| Query Type | Example | Optimal Granularity |
|------------|---------|---------------------|
| Factoid | "What port does Redis use?" | Chunk |
| Procedural | "How do I deploy a StatefulSet?" | Heading/Section |
| Explanatory | "How does Kubernetes scheduling work?" | Heading/Document |
| Comparison | "Deployment vs StatefulSet?" | Multiple Headings |

## PLM Context

PLM already stores hierarchical data that is currently **unused** during retrieval:
- `heading_id` for each chunk
- Heading-level embeddings
- Document-level embeddings
- `doc_id` relationships

This POC explores approaches to leverage this existing data.

## Approach Outlines

Each file in this folder describes one approach:

| File | Approach | Priority |
|------|----------|----------|
| `01_reranking.md` | Cross-encoder reranking | P0 |
| `02_auto_merging.md` | Auto-merge chunks to parent | P0 |
| `03_adaptive_rag_classifier.md` | Query complexity classifier | P1 |
| `04_iterative_expansion.md` | Start small, expand if needed | P1 |
| `05_multi_scale_indexing.md` | Index at multiple chunk sizes | P2 |
| `06_crag.md` | Corrective RAG with evaluator | P2 |
| `07_self_rag.md` | LLM self-reflection tokens | P3 |
| `08_recursive_retriever.md` | Follow references to parents | P1 |
| `09_sentence_window.md` | Retrieve sentence, return window | P1 |
| `10_adaptive_k.md` | Dynamic number of documents | P2 |
| `11_cluster_adaptive.md` | Cluster-based selection | P3 |
| `12_adaptive_compression.md` | Variable compression rate | P3 |
| `13_macrag.md` | Multi-scale adaptive context | P3 |
| `14_multi_query.md` | Multiple query variants + merge | P2 |
| `15_parent_child.md` | Search small, return large | P0 |

## Evaluation Criteria

See `evaluation_criteria.md` for metrics to compare approaches.

## Related Work

- `poc/plm_vs_rag_benchmark/` - PLM vs naive RAG comparison
- `poc/chunking_benchmark_v2/` - Chunking strategy experiments
- `docs/UNUSED_SYSTEMS.md` - Analysis of unused PLM data
