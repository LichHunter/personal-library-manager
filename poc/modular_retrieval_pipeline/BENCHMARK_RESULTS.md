# Benchmark Results

## Results

| Strategy            | Components                                                                                                                                           | Data Ingested                                              | Data Used for Testing                                                   | Success Rate  |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|-------------------------------------------------------------------------|---------------|
| enriched_hybrid_llm (baseline) | FastEnricher, QueryRewriter, Semantic Search, BM25, RRF Fusion                                                                                       | `poc/chunking_benchmark_v2/corpus/kubernetes/` (200 files) | `poc/chunking_benchmark_v2/corpus/needle_questions.json` (20 questions) | 90.0% (18/20) |
| enriched_hybrid_llm (modular)  | KeywordExtractor, EntityExtractor, ContentEnricher, QueryRewriter, QueryExpander, EmbeddingEncoder, BM25Scorer, SimilarityScorer, RRFFuser, Reranker | `poc/chunking_benchmark_v2/corpus/kubernetes/` (200 files) | `poc/chunking_benchmark_v2/corpus/needle_questions.json` (20 questions) | 90.0% (18/20) |

---

## enriched_hybrid_llm

**Flow:**

1. **Indexing:**
   - Document → FastEnricher extracts keywords (YAKE) and entities (spaCy)
   - Creates enriched content with prefix: `"keywords: X, Y, Z | entities: A, B, C"`
   - Enriched content → Embedding encoder → Vector index
   - Original content → BM25 indexer → Lexical index

2. **Retrieval:**
   - User query → QueryRewriter (Claude Haiku) → Rewritten query
   - Rewritten query → Semantic search (embedding similarity) → Semantic results
   - Rewritten query → BM25 search (lexical matching) → BM25 results
   - RRF Fusion merges results with adaptive weights:
     - If query expanded: semantic=3.0, bm25=0.3, k=10
     - If query normal: semantic=1.0, bm25=1.0, k=60
   - Returns top-k results

---

## enriched_hybrid_llm (modular)

**Flow:**

1. **Indexing:**
   - Document → KeywordExtractor (YAKE, n=2, dedupFunc="seqm") → Keywords
   - Document → EntityExtractor (spaCy, 9 entity types) → Entities
   - Keywords + Entities → ContentEnricher → Enriched content with prefix
   - Enriched content → EmbeddingEncoder (sentence-transformers) → Vector index
   - Original content → BM25Scorer (Okapi BM25) → Lexical index

2. **Retrieval:**
   - User query → QueryRewriter (Claude Haiku, 5s timeout) → Rewritten query
   - Rewritten query → QueryExpander (domain terms) → Expanded query (if matched)
   - Expanded query → SimilarityScorer (cosine similarity) → Semantic results
   - Expanded query → BM25Scorer (lexical matching) → BM25 results
   - RRFFuser merges results with adaptive weights:
     - If query expanded: semantic=3.0, bm25=0.3, k=10
     - If query normal: semantic=1.0, bm25=1.0, k=60
   - Reranker (cross-encoder, optional) → Final reranking
   - Returns top-k results
