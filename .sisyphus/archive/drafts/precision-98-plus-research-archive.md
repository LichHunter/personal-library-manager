# Draft: Precision 98%+ Investigation

## Current State
- **Best result**: 88.7% coverage with `enriched_hybrid_llm` (BM25 + semantic + LLM query rewriting)
- **Target**: 98%+ coverage (11.3% gap to close = 6 missed facts)
- **Corpus**: 51 chunks, 20 queries, 53 facts
- **Remaining failures**: Contextual queries (71.7%), negation queries (77.4%)
- **Latency constraint**: Minimize latency within 95%+ coverage

---

## COMPREHENSIVE RESEARCH FINDINGS

### Category 1: RAG-Specific Techniques (Already Explored)

| Technique | Expected Gain | We Have? | Notes |
|-----------|--------------|----------|-------|
| LLM Query Rewriting | +5-7% | YES (88.7%) | Already using Claude Haiku |
| Hybrid BM25+Semantic | +8% | YES | Already using |
| Enrichment (YAKE+spaCy) | +7.5% | YES | Already using |
| Cross-Encoder Reranking | +3-8% | PARTIAL | hybrid_rerank exists |
| Contextual Retrieval | +7-10% | NO | Anthropic technique |
| Multi-Query RRF | +2-5% | EXISTS | Not benchmarked |
| HyDE | +2-4% | EXISTS | Not benchmarked |

### Category 2: Retrieval Granularity (HIGHLY PROMISING)

**Dense X Retrieval / Proposition Chunking** (arXiv 2312.06648)

This is a paradigm shift: instead of retrieving CHUNKS, retrieve ATOMIC FACTS.

| Granularity | Recall@20 | Description |
|-------------|-----------|-------------|
| Passage (500 tokens) | Baseline | What we currently use |
| Sentence | +5% | Finer grained |
| **Proposition** | **+10.1%** | Atomic factoids |

**What is a Proposition?**
- **Atomic**: Cannot be split further
- **Self-contained**: Includes all necessary context
- **Factoid**: One distinct piece of information

**Example**:
```
Chunk: "Redis is used for caching. The TTL is 1 hour. This helps reduce database load."

Propositions:
1. "Redis is used for caching in the system"
2. "The Redis cache TTL is 1 hour"
3. "Redis caching helps reduce database load"
```

**Why this matters for us**:
- We have 53 KEY FACTS to find
- Our benchmark literally counts FACTS retrieved
- Proposition-level retrieval aligns PERFECTLY with our evaluation metric
- +10% improvement would take us from 88.7% to ~97-98%

**Implementation**: Fine-tuned Flan-T5 model for proposition extraction (available at github.com/chentong0/factoid-wiki)

### Category 3: Query Understanding & Decomposition

**ReDI Framework** (Reasoning-enhanced Query Understanding through Decomposition and Interpretation)

Three-stage pipeline:
1. **Decomposition**: Break complex queries into targeted sub-queries
2. **Interpretation**: Enrich each sub-query with semantic interpretations
3. **Retrieval + Fusion**: Retrieve for each, fuse results

**Why this helps**:
- Our hardest queries are indirect/contextual
- "Why can't I schedule every 30 seconds?" → decompose into:
  - "What is the minimum scheduling interval?"
  - "Why is there a minimum scheduling interval?"
  - "What are scheduling constraints?"

### Category 4: Document Structure Understanding

**DISRetrieval** (Discourse Structure for Long Document Retrieval)
- Uses linguistic discourse structure (RST)
- Sentence-level hierarchical representation
- Preserves document flow and relationships

**SEAL** (Structure and Element Aware Learning)
- Leverages HTML/markdown structure
- Element-aware alignment for fine-grained discrimination
- Assigns higher relevance to important elements

**Relevance**: Our documents are technical markdown with clear structure (headings, sections)

### Category 5: Knowledge Graphs & Entity-Based Retrieval

**Graph RAG / KG-Retriever**
- Build knowledge graph from documents
- Link entities across documents
- Graph traversal for retrieval

**Entity-Aware Ranking**
- Extract entities from queries and documents
- Use entity linking to knowledge bases
- Entity co-occurrence for ranking

**Relevance**: Our technical docs have many entities (Redis, PostgreSQL, JWT, RPO, RTO)

### Category 6: Learned Sparse Retrieval

**SPLADE** (Sparse Lexical and Expansion Model)
- Learns sparse vector representations
- Combines lexical matching with semantic understanding
- Better than BM25, compatible with inverted index

**Note**: We tried BMX (similar concept) and it FAILED (-50%). SPLADE might be different but risky.

### Category 7: Failure Analysis Patterns

Common reasons for retrieval failure:
1. **Vocabulary mismatch** → ADDRESSED by LLM rewriting
2. **Context loss in chunking** → ADDRESSABLE by proposition chunking
3. **Semantic similarity limitations** → ADDRESSABLE by entity linking
4. **Multi-hop reasoning required** → ADDRESSABLE by query decomposition
5. **Structure blindness** → ADDRESSABLE by structure-aware indexing

---

## Research Findings

### 1. Contextual Retrieval (Anthropic) - HIGH PRIORITY

**Source**: https://www.anthropic.com/news/contextual-retrieval

**What it does**:
- Prepends chunk-specific explanatory context to each chunk BEFORE embedding
- Uses Claude Haiku with prompt: "Given the document, provide context for this chunk"
- Also applies to BM25 index ("Contextual BM25")

**Expected Improvement**:
- Contextual Embeddings alone: **35% reduction** in retrieval failures
- Contextual Embeddings + Contextual BM25: **49% reduction**
- With reranking: **67% reduction** in failures
- Pass@10 improved from ~87% to ~95%

**Relevance to our case**:
- We already have enrichment (YAKE + spaCy keywords) at chunk level
- Contextual retrieval adds DOCUMENT-LEVEL context to chunks
- Could help with contextual queries (currently 71.7% - lowest)

**Implementation complexity**: MEDIUM
- Need to generate context for each of 51 chunks (one-time cost)
- Claude Haiku is fast and cheap (~$0.00025/chunk)
- Can use prompt caching for efficiency

---

### 2. Cross-Encoder Reranking - HIGH PRIORITY

**Best Models (2024-2025)**:

| Model | Type | Latency | Accuracy | Notes |
|-------|------|---------|----------|-------|
| `BAAI/bge-reranker-v2-m3` | Cross-encoder | ~50ms | High | Multilingual, good for tech docs |
| `BAAI/bge-reranker-large` | Cross-encoder | ~30ms | High | English-focused |
| `ms-marco-MiniLM-L-12-v2` | Cross-encoder | ~15ms | Medium | Fast, good baseline |
| ColBERT v2 | Late-interaction | ~20ms | High | Best precision/speed tradeoff |
| Cohere Rerank | API | ~100ms | Very High | Best quality, costs $0.001/query |

**Our current state**: `hybrid_rerank` exists but only improved problem queries (54.7% → 66.0%)

**Why reranking might help more NOW**:
- We already improved retrieval candidates via LLM query rewriting
- Reranking works best when candidates are already good
- Can stack: LLM rewrite → retrieve → rerank

**Expected Improvement**: +5-10% coverage
- Reranking typically gives 5-15% relative improvement
- More effective on already-good candidates

---

### 3. Multi-Query Retrieval with RRF - MEDIUM PRIORITY

**What it does**:
- Generates 3-5 query variations using LLM
- Retrieves for EACH variation
- Fuses results using Reciprocal Rank Fusion

**We already have**: `multi_query.py` (uses Ollama, 3 variations)

**Why it might help**:
- Different query formulations catch different chunks
- RRF fusion reduces risk of missing relevant chunks
- Addresses vocabulary mismatch from multiple angles

**Expected Improvement**: +3-5%
- Lower than reranking alone
- Diminishing returns with LLM query rewriting

**Implementation**: Already exists - need to benchmark + potentially switch to Claude

---

### 4. HyDE (Hypothetical Document Embeddings) - MEDIUM PRIORITY

**What it does**:
- Generates hypothetical answer document for query
- Embeds the hypothetical document instead of query
- Better semantic matching for natural language queries

**We already have**: `hyde.py` (uses Ollama)

**Why it might help**:
- Bridges gap between query language and document language
- Good for "problem" queries that don't directly match docs

**Potential issue**: May overlap with LLM query rewriting
- Both address vocabulary mismatch
- Might be redundant

**Expected Improvement**: +2-4% (if not already using LLM rewriting)

---

### 5. Reverse HyDE - LOW PRIORITY

**What it does**:
- Generates hypothetical QUESTIONS at INDEX time
- Embeds questions alongside chunks
- Better matching with natural user queries

**We already have**: `reverse_hyde.py` (uses Ollama)

**Why it might NOT help as much**:
- High indexing cost (LLM call per chunk)
- Our queries are already rewritten by LLM
- Overlap with query rewriting approach

**Expected Improvement**: +1-3%

---

### 6. LLM-as-Reranker - EXPERIMENTAL

**What it does**:
- Uses LLM (GPT-4, Claude) to score/rank passages
- Pointwise: Rate each passage 1-10
- Listwise: Rank all passages together
- Groupwise: Compare passages in groups

**Research finding**: LLM rerankers can be 5x faster with optimization

**Expected Improvement**: +5-8%
- High quality but HIGH latency (~500ms+ per query)
- May be overkill for our corpus size

---

### 7. Dynamic Passage Selection (DPS) - EXPERIMENTAL

**What it does**:
- Dynamically selects MINIMAL sufficient set of passages
- Instead of fixed top-K, adapts K per query
- Models inter-passage dependencies

**Expected Improvement**: Unknown - research paper focused

---

## Untested Strategies in Codebase

| Strategy | File | LLM Required | Status | Expected Impact |
|----------|------|--------------|--------|-----------------|
| `hyde` | hyde.py | Yes (Ollama) | Ready | +2-4% |
| `multi_query` | multi_query.py | Yes (Ollama) | Ready | +3-5% |
| `reverse_hyde` | reverse_hyde.py | Yes (Ollama) | Ready | +1-3% |
| `lod` | lod.py | No | Ready* | Unknown |
| `hybrid_rerank` | hybrid_rerank.py | No | Ready | +3-5% |

*LOD requires structured documents

---

## Recommended Testing Order (by Expected ROI)

### Tier 1: High Impact, Proven Techniques

1. **Stack reranking on enriched_hybrid_llm**
   - Expected: 88.7% → 93-95%
   - Implementation: Add `bge-reranker-v2-m3` after LLM retrieval
   - Latency: +50ms

2. **Contextual Embeddings (Anthropic style)**
   - Expected: +5-10% improvement
   - Implementation: Generate context for 51 chunks, re-embed
   - One-time cost: ~$0.01

### Tier 2: Medium Impact, Already Implemented

3. **Benchmark `multi_query` with Claude** (not Ollama)
   - May improve over current LLM rewriting
   - Multiple queries catch different chunks

4. **Benchmark `hyde` strategy**
   - May complement or replace LLM query rewriting
   - Compare coverage vs enriched_hybrid_llm

### Tier 3: Combination Strategies

5. **Ultimate stack: Contextual + LLM rewrite + reranker**
   - Contextual embeddings at index time
   - LLM query rewriting at query time
   - Reranker for final ranking
   - Expected: 95-98%

---

## Decisions Needed

1. **Reranker model choice**: BGE-reranker-v2-m3 vs Cohere API vs ColBERT?
2. **Contextual embeddings**: Implement from scratch or adapt existing enrichment?
3. **Multi-query**: Benchmark existing Ollama version first, or switch to Claude immediately?
4. **Latency budget**: Is 1-2s per query acceptable for 98% target?

---

## Scope Boundaries

### IN SCOPE
- Benchmarking untested strategies (hyde, multi_query, reverse_hyde)
- Implementing contextual embeddings
- Adding reranker to enriched_hybrid_llm
- Creating combination strategies
- Achieving 98%+ coverage

### OUT OF SCOPE
- Changing chunk size or overlap
- Changing embedding model
- Modifying ground truth or queries
- Production deployment

---

## Success Metrics

| Coverage Level | Assessment | Latency Budget |
|----------------|------------|----------------|
| 95%+ | GOOD | <2s |
| 98%+ | TARGET | <3s |
| 100% | STRETCH | Any |

---

## Open Questions

1. Why do contextual queries (71.7%) perform worst?
2. Can we identify the 6 specific missed facts?
3. What's the latency impact of stacking multiple techniques?

---

## SYNTHESIS: Techniques by Expected Impact

### TIER 1: Highest Impact, Novel Approaches

| Technique | Expected Gain | Latency Impact | Implementation |
|-----------|--------------|----------------|----------------|
| **Proposition Chunking** | +10% | One-time at index | Flan-T5 model |
| **Query Decomposition** | +5-8% | +500ms | LLM call |
| **Contextual Retrieval** | +7-10% | One-time at index | Claude Haiku |

### TIER 2: Medium Impact, Already Have

| Technique | Expected Gain | Latency Impact | Status |
|-----------|--------------|----------------|--------|
| Cross-Encoder Reranking | +3-5% | +50ms | hybrid_rerank exists |
| Multi-Query RRF | +2-5% | +800ms | multi_query.py exists |
| HyDE | +2-4% | +500ms | hyde.py exists |

### TIER 3: Research-Phase

| Technique | Expected Gain | Complexity | Status |
|-----------|--------------|------------|--------|
| Entity-Based Retrieval | Unknown | HIGH | Not implemented |
| Graph RAG | +5-15% | VERY HIGH | Not implemented |
| Structure-Aware (SEAL) | +3-5% | MEDIUM | Not implemented |

---

## RECOMMENDED INVESTIGATION PATH

### Phase 1: Low-Hanging Fruit (Existing Strategies)
**Goal**: Benchmark what we already have
1. Test `hyde` strategy
2. Test `multi_query` strategy  
3. Test `hybrid_rerank` with enriched_hybrid_llm
4. Test combinations

**Expected outcome**: Identify if any existing strategy beats 88.7%

### Phase 2: Proposition Chunking (Game Changer)
**Goal**: Implement atomic fact retrieval
1. Extract propositions from 51 chunks using Flan-T5
2. Create proposition-level index (~150-200 propositions)
3. Benchmark proposition retrieval
4. Combine with LLM query rewriting

**Expected outcome**: 95-98% coverage

### Phase 3: Contextual Retrieval (Anthropic)
**Goal**: Add document context to chunks
1. Generate context for each chunk using Claude
2. Re-embed with context
3. Benchmark contextual retrieval

**Expected outcome**: +5-10% additional if Phase 2 insufficient

### Phase 4: Query Decomposition
**Goal**: Handle complex queries
1. Implement query decomposition for contextual/negation queries
2. Multi-hop retrieval with fusion

**Expected outcome**: Handle edge cases

---

## CRITICAL INSIGHT: Our Benchmark is Fact-Based

**Key Realization**: Our ground truth measures SPECIFIC FACTS retrieved (53 facts).

This means:
- Proposition chunking DIRECTLY aligns with our evaluation
- We're not measuring "good enough" retrieval
- We're measuring "did you find this specific piece of information"

**Proposition chunking is the MOST ALIGNED technique with our benchmark methodology.**

---

## Analysis of the 6 Missed Facts (REQUIRES INVESTIGATION)

**CRITICAL**: To reach 98%, we need GROUNDED EVIDENCE about which facts are missed and WHY.

Current state: **Speculation only** - hypotheses below are NOT validated.

Hypotheses (UNVERIFIED):
1. **Fact buried in non-retrieved chunk** → Proposition chunking helps
2. **Query doesn't match fact terminology** → Query rewriting (already doing)
3. **Fact requires multi-hop reasoning** → Query decomposition helps
4. **Fact is entity-based** → Entity linking helps

**BLOCKER**: Before choosing techniques, we MUST:
1. Run enriched_hybrid_llm benchmark with trace logs
2. Identify EXACTLY which 6 facts are missed on original queries
3. For EACH missed fact, determine:
   - Which chunk contains it?
   - What rank did that chunk get?
   - WHY was it ranked low (BM25 fail? Semantic fail? Both?)
4. Categorize root causes with evidence

**Only THEN can we choose the right technique.**

---

## STATUS: RESEARCH PAUSED

**Date**: 2026-01-25
**Reason**: User requested grounded failure analysis before proceeding
**Next action**: Analyze actual benchmark results to identify missed facts

This draft preserved for future reference when implementing techniques.

