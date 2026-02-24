# Draft: PLM Benchmark Framework

## Problem Statement

We need to evaluate PLM's retrieval quality without circular validation:
- Can't trust benchmarks we create ourselves (grading our own homework)
- Generic benchmarks (BEIR) don't match our domain (technical documentation)
- Need external validation with real user data

## Solution: StackOverflow → Documentation Benchmark

### Core Insight

StackOverflow already contains (real_question, relevant_doc) pairs:

```
Real user asks question on SO
    ↓
Answerer links to official docs (kubernetes.io/docs/...)
    ↓
Community upvotes / accepts answer
    ↓
Ground truth: This question → this doc section
```

**Why this is NOT circular validation:**

| Aspect | Who Created It? |
|--------|-----------------|
| Questions | Real SO users (strangers) |
| Relevance signal | Community votes (strangers) |
| Documentation links | SO answerers (strangers) |
| Your system | You |

---

## Data Sources

### Primary: SOTorrent Dataset

**What it is**: Open dataset based on official SO data dump with extracted URLs

**Key tables**:
- `PostVersionUrl`: URLs extracted from post text
- `CommentUrl`: URLs extracted from comments
- `Posts`: Question and answer content

**Access**: 
- Zenodo: https://zenodo.org/record/4415593
- License: CC BY-SA 4.0

**Coverage**: All SO posts through December 2020

### Secondary: StackOverflow Kubernetes Datasets

| Dataset | Size | Source |
|---------|------|--------|
| `mcipriano/stackoverflow-kubernetes-questions` | 30k Q&A | HuggingFace |
| `peterpanpan/stackoverflow-kubernetes-questions` | 22.8k Q&A | HuggingFace |
| `genaidevops/kubernetes-stackoverflow-questions` | 10-100k | HuggingFace |

### Supplementary: External Benchmarks

| Benchmark | Purpose | Use |
|-----------|---------|-----|
| **LoTTE technology** | General tech retrieval validation | Algorithm validation |
| **CQADupStack** | StackExchange retrieval | Baseline comparison |
| **FreshStack** | Technical doc retrieval | Methodology reference |
| **OpsEval** | DevOps/AIOps evaluation | Domain coverage |

---

## Implementation Plan

### Phase 1: Data Extraction

```sql
-- Extract K8s questions with documentation links from SOTorrent
SELECT 
    q.Id as question_id,
    q.Title as question_title,
    q.Body as question_body,
    q.Tags as tags,
    a.Id as answer_id,
    a.Body as answer_body,
    a.Score as answer_score,
    a.AcceptedAnswerId IS NOT NULL as is_accepted,
    urls.Url as doc_url
FROM Posts q
JOIN Posts a ON a.ParentId = q.Id
JOIN PostVersionUrl urls ON urls.PostId = a.Id
WHERE q.Tags LIKE '%kubernetes%'
  AND urls.Url LIKE '%kubernetes.io/docs%'
  AND a.Score >= 5  -- Quality filter: well-received answers
ORDER BY a.Score DESC
```

**Quality filters**:
- Answer score >= 5 (community validated)
- Accepted answer preferred
- Doc URL points to official kubernetes.io/docs

### Phase 2: URL to Chunk Mapping

```python
def map_url_to_chunks(doc_url: str, corpus: Corpus) -> List[str]:
    """
    Map a documentation URL to indexed chunk IDs.
    
    Example:
      doc_url = "https://kubernetes.io/docs/concepts/scheduling/kube-scheduler/"
      → Returns chunk IDs from that page in your corpus
    """
    # Extract path from URL
    path = urlparse(doc_url).path  # "/docs/concepts/scheduling/kube-scheduler/"
    
    # Find chunks with matching source
    matching_chunks = []
    for chunk in corpus.chunks:
        if path in chunk.source_url or path in chunk.source_file:
            matching_chunks.append(chunk.id)
    
    return matching_chunks
```

### Phase 3: Benchmark Dataset Creation

```python
@dataclass
class BenchmarkCase:
    query_id: str
    query: str                    # SO question title + body
    relevant_doc_ids: List[str]   # Chunk IDs from linked docs
    source: str                   # "stackoverflow"
    so_question_id: int
    so_answer_score: int
    is_accepted_answer: bool
    doc_url: str                  # Original documentation URL

def create_benchmark(so_data: List[dict], corpus: Corpus) -> List[BenchmarkCase]:
    benchmark = []
    
    for row in so_data:
        # Map URL to corpus chunks
        relevant_chunks = map_url_to_chunks(row["doc_url"], corpus)
        
        if not relevant_chunks:
            continue  # Skip if doc not in our corpus
        
        benchmark.append(BenchmarkCase(
            query_id=f"so_{row['question_id']}",
            query=f"{row['question_title']}\n\n{row['question_body']}",
            relevant_doc_ids=relevant_chunks,
            source="stackoverflow",
            so_question_id=row["question_id"],
            so_answer_score=row["answer_score"],
            is_accepted_answer=row["is_accepted"],
            doc_url=row["doc_url"]
        ))
    
    return benchmark
```

### Phase 4: Evaluation

```python
def evaluate_plm(plm: HybridRetriever, benchmark: List[BenchmarkCase]) -> dict:
    results = {
        "hit_at_1": 0,
        "hit_at_5": 0,
        "hit_at_10": 0,
        "mrr": 0.0,
        "total": len(benchmark)
    }
    
    for case in benchmark:
        # Run PLM retrieval
        retrieved = plm.retrieve(case.query, k=10)
        retrieved_ids = [r["chunk_id"] for r in retrieved]
        
        # Check hits at various k
        for k in [1, 5, 10]:
            if any(doc_id in retrieved_ids[:k] for doc_id in case.relevant_doc_ids):
                results[f"hit_at_{k}"] += 1
        
        # Calculate MRR
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in case.relevant_doc_ids:
                results["mrr"] += 1.0 / rank
                break
    
    # Normalize
    for key in ["hit_at_1", "hit_at_5", "hit_at_10", "mrr"]:
        results[key] /= results["total"]
    
    return results
```

---

## Metrics

### Primary Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Hit@1** | Is any relevant doc in top 1? | > 30% |
| **Hit@5** | Is any relevant doc in top 5? | > 60% |
| **Hit@10** | Is any relevant doc in top 10? | > 75% |
| **MRR** | Mean Reciprocal Rank | > 0.4 |

### Secondary Metrics (if graded relevance available)

| Metric | Description |
|--------|-------------|
| **NDCG@10** | Ranking quality with position weighting |
| **MAP@10** | Mean Average Precision |
| **Recall@100** | Coverage of all relevant docs |

---

## Validation Tiers

### Tier 1: Automatic (High Volume, Low Cost)

- Use SO answer score as proxy for relevance
- Accepted answers = high confidence
- Score >= 10 = very high confidence

### Tier 2: LLM Validation (Medium Volume, Medium Cost)

```python
def validate_with_llm(question: str, doc_content: str) -> str:
    prompt = f"""
    Question from StackOverflow:
    {question}
    
    Documentation content:
    {doc_content}
    
    Does this documentation answer the question?
    Rate: [Fully Answers | Partially Answers | Doesn't Answer]
    Explain briefly.
    """
    return llm.generate(prompt)
```

**Cost estimate**: ~$0.01 per validation (Claude Haiku)

### Tier 3: Human Validation (Low Volume, High Cost)

- Sample 100-200 cases
- Expert annotators rate relevance
- Establish ground truth for calibration
- Measure inter-annotator agreement

---

## Dataset Splits

### Temporal Split (Recommended)

| Split | Time Period | Purpose |
|-------|-------------|---------|
| **Test** | 2024+ | Final evaluation (no peeking) |
| **Dev** | 2023 | Hyperparameter tuning |
| **Train** | Pre-2023 | If needed for learning |

**Rationale**: Avoids training data contamination in LLMs

### Quality Tiers

| Tier | Criteria | Expected Size |
|------|----------|---------------|
| **Gold** | Accepted answer + score >= 10 | ~500 cases |
| **Silver** | Score >= 5 | ~2,000 cases |
| **Bronze** | Any doc link | ~10,000 cases |

---

## Comparison with Alternatives

### Why Not Just BEIR?

| Aspect | BEIR | SO Benchmark |
|--------|------|--------------|
| Domain match | ❌ Generic | ✅ K8s docs |
| Query style | ❌ Varied | ✅ Troubleshooting |
| Corpus match | ❌ Their docs | ✅ Your docs |
| Who created it | Strangers | Strangers |
| Circular validation risk | None | None |

### Why Not Just Our Needle/Realistic Tests?

| Aspect | Our Tests | SO Benchmark |
|--------|-----------|--------------|
| Who created queries | Us | Strangers |
| Who labeled relevance | Us | Community |
| Circular validation risk | ⚠️ High | ✅ None |
| Domain match | ✅ Perfect | ✅ Perfect |

**Conclusion**: Use BOTH. SO benchmark for validation, our tests for domain coverage.

---

## Implementation Checklist

### Data Collection
- [ ] Download SOTorrent dataset from Zenodo
- [ ] Query for K8s questions with doc links
- [ ] Filter for quality (score >= 5)
- [ ] Extract ~5,000 (question, doc_url) pairs

### Corpus Mapping
- [ ] Build URL → chunk_id mapping for PLM corpus
- [ ] Handle URL variations (trailing slashes, anchors)
- [ ] Log unmapped URLs for corpus gap analysis

### Benchmark Creation
- [ ] Create benchmark dataset in standard format
- [ ] Split into test/dev sets
- [ ] Validate sample with LLM (100 cases)

### Evaluation
- [ ] Implement evaluation metrics (Hit@k, MRR)
- [ ] Run baseline (BM25 only)
- [ ] Run PLM (full hybrid)
- [ ] Compare and report

---

## Expected Outcomes

### Success Criteria

| Metric | BM25 Baseline | PLM Target | SOTA Reference |
|--------|---------------|------------|----------------|
| Hit@5 | ~40% | > 55% | ~60% (CQADupStack) |
| Hit@10 | ~55% | > 70% | ~75% |
| MRR | ~0.30 | > 0.40 | ~0.45 |

### What This Proves

If PLM beats BM25 on SO benchmark:
1. **Algorithm works** — hybrid search adds value
2. **Not overfitting** — tested on external data
3. **Real-world valid** — actual user queries
4. **Domain appropriate** — K8s specific

---

## Future Extensions

### Multi-Domain Expansion

Extend beyond Kubernetes to other documentation:
- Python docs (python-tagged SO questions)
- React docs (react-tagged SO questions)
- AWS docs (aws-tagged SO questions)

### Graded Relevance

Upgrade from binary (found/not found) to graded:
- Use LLM to rate relevance 0-3
- Compute NDCG instead of just Hit@k

### Negative Mining

Add hard negatives for better evaluation:
- Similar docs that DON'T answer the question
- Tests discrimination ability

---

## References

### Datasets
- SOTorrent: https://zenodo.org/record/4415593
- StackOverflow Data Dump: https://archive.org/details/stackexchange
- FreshStack: arXiv:2504.13128

### Methodologies
- BEIR: arXiv:2104.08663
- CoIR: ACL 2025
- CodeAssistBench: arXiv (2026)

### Tools
- ir_datasets: https://ir-datasets.com/
- BEIR library: https://github.com/beir-cellar/beir

---

## Multi-Tier Evaluation Framework (Updated)

### Granularity Levels

All tiers must test THREE retrieval granularities:

| Level | Size | Use Case | Expected Usage |
|-------|------|----------|----------------|
| **Document** | Full doc | "Overview of topic X" | 10% |
| **Heading** | Section (~500-2000 tokens) | "How to configure X" | **70%** |
| **Chunk** | ~200 tokens | "What's the default value of X" | 20% |

**Heading level is primary** — most real queries need section-level context.

---

### Tier Structure

```
Tier 1: Algorithm Validation (BEIR/LoTTE)
    └── "Does our ranking work generally?"
    └── External, no circular validation
    └── Compare vs BM25 baseline

Tier 2: Single-Hop Domain (Updated Existing + SO Direct)
    └── "Does PLM find the right section for a question?"
    └── Needle/Realistic updated with heading-level ground truth
    └── SO questions → K8s docs (direct mapping)

Tier 3: Multi-Hop Decomposed (MuSiQue-style from SO)
    └── "Can PLM support investigation workflows?"
    └── Extract decomposition from SO threads
    └── Each step has heading-level relevance

Tier 4: Self-Built Decomposed
    └── "Does PLM work for OUR specific use cases?"
    └── Manually create from real debugging scenarios
    └── Gold standard, small but high quality
```

---

### Tier 1: Algorithm Validation

**Purpose**: Prove ranking algorithm works

**Source**: BEIR / LoTTE technology / CQADupStack

**Schema**:
```json
{
  "query": "string",
  "relevant_docs": ["doc_id1", "doc_id2"],
  "relevance_grades": {"doc_id1": 2, "doc_id2": 1}
}
```

**Metrics**: NDCG@10, MRR

**Granularity**: N/A (uses BEIR's corpus, not ours)

---

### Tier 2: Single-Hop Domain Retrieval

**Purpose**: Does PLM find the right content for direct questions?

**Sources**:
- Updated needle/realistic tests
- SO questions → K8s docs (from SOTorrent)

**Schema** (multi-level relevance):
```json
{
  "id": "so_12345",
  "query": "How do I check why my pod is pending?",
  "relevant_content": {
    "document": ["scheduling-concepts.md"],
    "heading": ["scheduling-concepts.md#troubleshooting-pending-pods"],
    "chunk": ["scheduling-concepts.md#chunk-47", "scheduling-concepts.md#chunk-48"]
  },
  "source": "stackoverflow",
  "validation": "community_upvoted"
}
```

**Metrics per level**:
- Document: Hit@3, Recall@5
- Heading: Hit@5, MRR
- Chunk: Hit@10, MRR

---

### Tier 3: Multi-Hop from SO (MuSiQue-style)

**Purpose**: Does PLM support multi-step investigation?

**Source**: Extract decomposition from SO threads with multiple answers/steps

**Schema**:
```json
{
  "id": "so_thread_789",
  "original_question": "Ingress not routing traffic to my service",
  "decomposition": [
    {
      "step": 1,
      "question": "How to check ingress controller logs?",
      "intermediate_answer": "kubectl logs -n ingress-nginx",
      "relevant_content": {
        "document": ["ingress-nginx.md"],
        "heading": ["ingress-nginx.md#troubleshooting"],
        "chunk": null
      }
    },
    {
      "step": 2,
      "question": "What does 'no endpoints' error mean in ingress?",
      "intermediate_answer": "Service has no backing pods",
      "relevant_content": {
        "document": ["services.md"],
        "heading": ["services.md#endpoints"],
        "chunk": null
      }
    },
    {
      "step": 3,
      "question": "How to check if pod labels match service selector?",
      "intermediate_answer": "kubectl get pods --show-labels",
      "relevant_content": {
        "document": ["services.md"],
        "heading": ["services.md#selectors"],
        "chunk": null
      }
    }
  ],
  "final_answer": "Pod labels didn't match service selector",
  "source": "stackoverflow_thread",
  "so_thread_id": 12345678
}
```

**Metrics**:
- Per-step retrieval accuracy (did PLM find the right heading for step N?)
- Full-chain accuracy (did PLM find ALL relevant headings?)

---

### Tier 4: Self-Built Gold Standard

**Purpose**: Domain-specific validation we control

**Source**: Manually created from real debugging scenarios

**Schema**: Same as Tier 3

**Characteristics**:
- Created by us from real experiences
- Smaller (50-100 cases)
- Highest quality / most relevant
- Updated as we learn failure modes

---

### Evaluation Matrix

| Tier | Source | Circular Risk | Granularity | Size | Cost |
|------|--------|---------------|-------------|------|------|
| **1: BEIR** | External | None | N/A | Large | Free |
| **2: Single-hop** | SO + Updated existing | Low | Doc/Heading/Chunk | Medium | Free |
| **3: Multi-hop SO** | Extracted from SO | Low | Doc/Heading | Medium | Free |
| **4: Self-built** | Manual | High | Doc/Heading | Small | Time |

---

### Data Structures

```python
@dataclass
class RelevantContent:
    """Multi-level relevance annotation"""
    document: Optional[List[str]] = None      # ["doc.md"]
    heading: Optional[List[str]] = None       # ["doc.md#section-name"]
    chunk: Optional[List[str]] = None         # ["doc.md#chunk-42"]

@dataclass 
class BenchmarkCase:
    id: str
    query: str
    relevant_content: RelevantContent
    source: str  # "beir", "stackoverflow", "needle", "manual"
    
@dataclass
class MultiHopCase:
    id: str
    original_question: str
    decomposition: List[DecompositionStep]
    final_answer: str
    source: str

@dataclass
class DecompositionStep:
    step: int
    question: str
    intermediate_answer: str
    relevant_content: RelevantContent
```

---

### Oracle Analysis: Component vs End-to-End Optimization

**Critical question**: Does best-in-class on each retrieval step result in best-in-class combined search?

**Answer: NO.** Component-level optimality does not guarantee system-level optimality.

#### Why Component Optimization Fails

| Problem | Impact |
|---------|--------|
| **Error correlation** | Both retrievers optimized on same benchmark → fail on same queries → fusion doesn't help |
| **Redundant signal** | Both retrieve similar top-k → RRF adds no lift |
| **Distribution shift** | Reranker trained on BEIR candidates ≠ your pipeline's candidates |
| **Cascade amplification** | First-stage false negatives are unrecoverable downstream |

**Key insight**: Two 85%-accurate retrievers with UNCORRELATED errors beat two 92%-accurate retrievers with correlated errors.

#### Cross-Encoder Transfer Problem

Cross-encoders are sensitive to input distribution:
- Your first stage produces specific hard negatives
- BEIR reranker may struggle with YOUR confusables (Service vs Ingress vs NetworkPolicy)
- **Must evaluate reranker on actual first-stage candidates, not gold/BM25 candidates**

---

### Tier 5: Integration Evaluation (NEW - ESSENTIAL)

Added based on Oracle review. Tests stage interactions, not just capabilities.

#### 5a: Complementarity Analysis

```python
def measure_complementarity(splade_results, semantic_results, k=10):
    """Do retrievers provide uncorrelated signal?"""
    
    overlap = len(set(splade_results[:k]) & set(semantic_results[:k])) / k
    # Low overlap = good complementarity
    
    # Error correlation: do they fail on same queries?
    splade_failures = get_failed_queries(splade_results)
    semantic_failures = get_failed_queries(semantic_results)
    error_correlation = len(splade_failures & semantic_failures) / len(splade_failures | semantic_failures)
    # High error correlation = bad (fusion won't help)
    
    return {
        "overlap_at_k": overlap,
        "error_correlation": error_correlation,
        "fusion_potential": 1 - error_correlation  # Higher = more fusion benefit
    }
```

**Target metrics**:
- Overlap@10: < 50% (retrievers find different relevant docs)
- Error correlation: < 30% (they fail on different queries)

#### 5b: Cascade Evaluation

Evaluate each stage on **actual predecessor output**, not gold data:

```python
def cascade_evaluation(pipeline, test_set):
    """Measure actual contribution of each stage"""
    
    results = {}
    
    for query in test_set:
        # Stage 1: First-stage retrieval
        splade_out = pipeline.splade.retrieve(query, k=100)
        semantic_out = pipeline.semantic.retrieve(query, k=100)
        
        # Stage 2: RRF fusion (on actual stage 1 output)
        rrf_out = pipeline.rrf.fuse(splade_out, semantic_out)
        
        # Stage 3: Reranking (on actual RRF output, not gold)
        rerank_out = pipeline.reranker.rerank(query, rrf_out[:50])
        
        # Measure at each stage
        results[query.id] = {
            "splade_recall@100": recall(splade_out, query.relevant),
            "semantic_recall@100": recall(semantic_out, query.relevant),
            "rrf_mrr@50": mrr(rrf_out, query.relevant),
            "rerank_mrr@10": mrr(rerank_out, query.relevant),
            "end_to_end": hit_at_5(rerank_out, query.relevant)
        }
    
    return aggregate(results)
```

**Key insight**: If first-stage recall@100 is low, reranker can never recover those docs.

#### 5c: Ablation Waterfall

Measure **marginal contribution** of each component:

```python
def ablation_waterfall(pipeline, test_set):
    """What does each component actually contribute?"""
    
    configs = [
        ("full_pipeline", pipeline),
        ("no_reranker", pipeline.without_reranker()),
        ("semantic_only", pipeline.semantic_only()),
        ("splade_only", pipeline.splade_only()),
        ("bm25_baseline", BM25Retriever()),
    ]
    
    results = {}
    for name, config in configs:
        results[name] = evaluate_end_to_end(config, test_set)
    
    # Calculate marginal contributions
    return {
        "reranker_contribution": results["full_pipeline"] - results["no_reranker"],
        "fusion_contribution": results["no_reranker"] - max(results["semantic_only"], results["splade_only"]),
        "semantic_contribution": results["semantic_only"] - results["bm25_baseline"],
        "splade_contribution": results["splade_only"] - results["bm25_baseline"],
    }
```

**This reveals**: Maybe reranker adds 2% but costs 500ms. Is it worth it?

#### 5d: Resilience Testing

Intentionally degrade components to reveal dependencies:

| Degradation | What It Reveals |
|-------------|-----------------|
| Disable SPLADE | Is semantic alone sufficient? |
| Add noise to embeddings | How robust to embedding quality? |
| Reduce reranker quality | Does first stage quality make reranker less critical? |
| Inject query errors | Which downstream components are most affected? |

---

### Updated Evaluation Matrix

| Tier | Tests | Circular Risk | What It Proves |
|------|-------|---------------|----------------|
| **1: BEIR** | Algorithm quality | None | Ranking logic works |
| **2: Single-hop** | Domain retrieval | Low | Finds right content |
| **3: Multi-hop** | Investigation support | Low | Supports workflows |
| **4: Self-built** | Specific scenarios | High | Handles our cases |
| **5: Integration** | Stage interactions | None | **Components compose well** |

---

### Domain-Specific Hard Negatives (Tier 4 Enhancement)

Oracle recommends building explicit confusable pairs:

```json
{
  "query": "How to expose a service externally?",
  "relevant": ["services.md#loadbalancer"],
  "hard_negatives": [
    "services.md#clusterip",       // Similar but wrong type
    "ingress.md#basic-ingress",    // Related but different concept
    "networkpolicy.md#egress"      // Contains 'external' but irrelevant
  ],
  "confusable_reason": "LoadBalancer vs ClusterIP vs Ingress confusion"
}
```

**Method**: Run pipeline on 100 queries, audit failures, codify patterns.

---

### Hyperparameter Tuning Rule

**WRONG**: Tune RRF weights to maximize component metrics
**RIGHT**: Tune RRF weights to maximize END-TO-END metrics

```python
# Find optimal RRF alpha on end-to-end performance
best_alpha = None
best_score = 0

for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pipeline.rrf_weight = alpha
    score = evaluate_full_pipeline(holdout_set)  # End-to-end, not component
    if score > best_score:
        best_score = score
        best_alpha = alpha
```

---

### Context Expansion Placement Question

Current: retrieve → rerank → expand

Alternative: retrieve → expand → rerank (let reranker see full context)

**Needs empirical testing** — expanded context might reveal low-ranked chunk was relevant.

---

### References (Additional)

#### Multi-hop Datasets
- MuSiQue: https://github.com/StonyBrookNLP/musique
- HotpotQA: http://curtis.ml.cmu.edu/datasets/hotpot/
- StrategyQA: https://github.com/eladsegal/strategyqa

---

## Final Design Decisions (2026-02-24 Session)

### Oracle Methodology: Evidence-Based Tier Assignment

**Core Principle**: Don't validate relevance yourself. Capture evidence that *someone else* (SO answerer) validated it, and verify that evidence exists via objective signals.

### Tier Definitions (Confirmed)

| Tier | Trust | Signal Required | Mapping Strategy |
|------|-------|-----------------|------------------|
| **GOLD** | 100% | Fragment anchor OR exact quote match | Single chunk(s) with evidence |
| **SILVER** | ~90% | URL match + (upvotes ≥10 AND accepted) OR (upvotes ≥25) | Chunks from URL with corroboration |
| **BRONZE** | ~75% | URL match + score ≥5 | ALL chunks from linked URL |
| **EXCLUDE** | — | Score <5 OR no URL match | Don't include |

### Provable Signals (GOLD Tier)

1. **Fragment Anchor Match**: `#pod-lifecycle` in URL → chunk heading "Pod Lifecycle"
2. **Exact Quote Match**: Code/text in SO answer found verbatim in chunk
3. **Reciprocal Containment**: ≥20 contiguous words from chunk appear in answer

### Confirmed Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Include ambiguous cases** | YES (BRONZE tier) | Higher volume preferred over strict precision |
| **BRONZE mapping** | All chunks from URL | High recall, filter later by confidence |
| **Confidence metadata** | Detailed | tier, mapping_method, upvotes, is_accepted, quote_match_length, heading_similarity |
| **Quality threshold** | Score ≥ 5 | Community validated |
| **Spot-check** | Skip initially | Trust methodology, validate later if needed |
| **Target corpus** | Kubernetes docs only | Matches existing infrastructure |
| **Test strategy** | Tests after implementation | Add tests after each component |
| **Dataset size** | ~500 GOLD + 2000 SILVER + unlimited BRONZE | Prioritize GOLD quality |
| **Integration** | `src/plm/benchmark/` | Build on existing POC patterns |
| **Logging** | Full traceability | TRACE/DEBUG/INFO/WARN/ERROR per question |

### Traceability Requirement (Critical)

Each question/answer result MUST be traceable in logs:
- What was extracted at each stage
- Which tier was assigned and why
- What signals were detected
- Full audit trail from SO post to final benchmark case

### Metadata Schema (Final)

```python
@dataclass
class BenchmarkCase:
    # Core
    id: str
    query: str                        # SO question title + body
    chunk_ids: list[str]              # Relevant chunk(s)
    
    # Trust signals
    tier: Literal["gold", "silver", "bronze"]
    confidence_score: float           # 0.0-1.0
    
    # Evidence (for traceability)
    mapping_method: Literal["fragment", "quote", "reciprocal", "url_only"]
    fragment_anchor: str | None
    matched_quote: str | None
    quote_match_length: int | None
    heading_similarity_score: float | None
    
    # SO metadata
    so_question_id: int
    so_answer_id: int
    upvotes: int
    is_accepted: bool
    answer_date: date
    doc_url: str
    
    # Chunk metadata
    chunk_source_url: str
    chunk_heading: str | None
    chunk_content_hash: str
    
    # Flags
    multiple_chunks: bool
    page_level_mapping: bool
```

---

## Architecture: Benchmark as API Consumer (2026-02-24)

### Constraint (User Directive)
Benchmarks MUST be separate entities that call the production search service via HTTP API.
They CANNOT instantiate HybridRetriever directly or access internal components.

### Implication: Search Service Enhancement Required

**Current API response** (`/query`):
- Returns only final `score` per result
- No visibility into BM25 vs semantic contribution
- No query transformation details

**Required for traceability** (new `explain=True` mode):

```python
# Per result (when explain=True)
{
  "chunk_id": "...",
  "score": 0.85,
  "debug_info": {
    "bm25_score": 12.34,
    "bm25_rank": 2,
    "semantic_score": 0.78,
    "semantic_rank": 1,
    "rrf_score": 0.85,
    "rerank_score": null,  # if reranking enabled
    "retrieval_stage": "hybrid"
  }
}

# Per query metadata
{
  "metadata": {
    "original_query": "...",
    "rewritten_query": "...",
    "expanded_terms": ["k8s", "container"],
    "retrieval_mode": "hybrid",
    "rrf_k": 60,
    "bm25_weight": 1.0,
    "semantic_weight": 1.0
  },
  "request_id": "a1b2c3d4"
}
```

### Two-Phase Plan Structure

**Phase 0: Search Service Enhancement** (prerequisite for benchmark)
- Add `explain: bool` parameter to `/query` endpoint
- Track BM25/semantic scores in retriever, include in response when explain=True
- Add `X-Request-ID` middleware for correlation
- Include query metadata (original, rewritten, expanded)

**Phase 1+: Benchmark Framework**
- HTTP client calling production API with `explain=True`
- Full traceability via API response (no log parsing needed)
- Metrics calculation from response data

### Benefits of This Architecture

| Benefit | Description |
|---------|-------------|
| **Clean separation** | Benchmark doesn't know retriever internals |
| **Production-representative** | Tests actual API behavior |
| **No log parsing** | All trace data in API response |
| **Reusable** | Any client can use explain mode for debugging |

---
*Draft updated: 2026-02-24*
