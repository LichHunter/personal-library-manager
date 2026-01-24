# Chunking Benchmark V2 - Human Query Findings

**Date**: 2026-01-24  
**Goal**: Test RAG retrieval robustness against human-like query variations

## Executive Summary

Previous benchmark results showing 94-97% coverage were **invalid** due to test conditions being too easy. When tested with realistic documentation and human query variations, even the best strategy achieves only **67.9%** for original queries and drops to **54.7%** for problem/negation queries.

## Why Previous Results Were Invalid

### Problem 1: Documents Too Small

| Metric | Previous Corpus | Realistic Corpus |
|--------|-----------------|------------------|
| Avg document size | 137 words | 3,400 words |
| Documents | 52 | 5 |
| Chunks created | 52-126 | 51-241 |

With 137-word documents and 512-token chunks (~384 words), **each document fit in a single chunk**. This meant:
- Retrieving k=5 chunks = retrieving 5 whole documents
- With 52 documents, we retrieved ~10% of entire corpus per query
- Even random retrieval would perform well

### Problem 2: No Human Query Variations

Previous tests only used "robot queries" - exact keyword-matching questions like:
- "What is the default rate limit for API requests?"

Real users ask:
- **Problem**: "my requests keep getting rejected with 429"
- **Casual**: "api rate limit"  
- **Negation**: "why is the API blocking my requests"

## Human Query Variation Dimensions

| Dimension | Description | Example |
|-----------|-------------|---------|
| **original** | Direct, keyword-matching query | "What is the API rate limit?" |
| **synonym** | Alternative technical terms | "what's the API throttling cap" |
| **problem** | Describe symptom, not solution | "my requests get 429 errors" |
| **casual** | Short, Slack-style query | "api rate limit" |
| **contextual** | Include use case context | "building batch job, need api limits" |
| **negation** | Confusion/why questions | "why is the API blocking me" |

## Test Results: Realistic Corpus

### Corpus Details
- 5 documents, ~3,400 words each (17,000 total)
- 20 queries with 5 human variations each (120 total query variants)
- 53 key facts to find

### Results by Strategy

| Strategy | Chunks | Original | Synonym | Problem | Casual | Contextual | Negation |
|----------|--------|----------|---------|---------|--------|------------|----------|
| MiniLM + 100 tokens | 241 | 62.3% | 52.8% | 37.7% | 54.7% | 54.7% | 39.6% |
| **BGE + 512 tokens** | 51 | **67.9%** | **60.4%** | **54.7%** | 54.7% | **62.3%** | **54.7%** |
| BGE + 200 tokens | 125 | 56.6% | 60.4% | 52.8% | 58.5% | 52.8% | 41.5% |

### Key Findings

1. **Best config (BGE + 512 tokens) achieves only 67.9%** for original queries - far from the 94-97% in previous tests

2. **Human query variations cause 13-20% degradation**:
   - Problem queries: -13.2% (67.9% → 54.7%)
   - Negation queries: -13.2% (67.9% → 54.7%)
   - Synonym queries: -7.5% (67.9% → 60.4%)

3. **Larger chunks perform better** - 512-token chunks (51 total) beat 100-200 token chunks (125-241 total)

4. **BGE embedder outperforms MiniLM** across all dimensions (+5-17% depending on query type)

## Example Failures

### Query: "What is the API rate limit per minute?"

| Dimension | Query | Coverage |
|-----------|-------|----------|
| original | "What is the API rate limit per minute?" | 100% |
| problem | "My API calls are getting blocked, what's the limit?" | **0%** |
| negation | "Why is the API rejecting my requests after 100 calls?" | **0%** |

The embeddings don't connect "getting blocked" → "rate limit documentation".

### Query: "What are the resource requirements for API Gateway?"

| Dimension | Query | Coverage |
|-----------|-------|----------|
| original | "What are the resource requirements for API Gateway?" | 100% |
| synonym | "How much CPU and memory does the API Gateway need?" | **0%** |
| problem | "The API Gateway pods keep getting OOMKilled" | **0%** |

## Implications for Production RAG

1. **Don't trust benchmarks with small documents** - Real documentation has multi-page docs

2. **Test with human query variations** - Robot queries give false confidence

3. **Problem/negation queries need special handling**:
   - Query expansion
   - Hybrid search (BM25 + semantic)
   - Fine-tuned embeddings

4. **Expected real-world performance**: 50-70% retrieval accuracy for human queries with basic semantic search

## Files

```
corpus/
├── realistic_documents/          # 5 docs, ~3400 words each
│   ├── api_reference.md
│   ├── architecture_overview.md
│   ├── deployment_guide.md
│   ├── troubleshooting_guide.md
│   └── user_guide.md
├── corpus_metadata_realistic.json
└── ground_truth_realistic.json   # 20 queries × 6 variations = 120 tests

results/
├── 2026-01-24_134228/           # MiniLM + 100 tokens
├── 2026-01-24_134539/           # BGE + 512 tokens (best)
└── 2026-01-24_134603/           # BGE + 200 tokens
```

## Recommended Next Steps

1. **Implement query expansion** - Generate synonym queries automatically
2. **Add hybrid search** - BM25 catches keyword matches that embeddings miss
3. **Test with reranking** - May help with larger chunk pools
4. **Fine-tune embeddings** - Domain-specific training on problem→solution pairs
