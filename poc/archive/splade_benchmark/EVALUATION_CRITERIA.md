# SPLADE Benchmark: Evaluation Criteria

**Goal**: Determine if SPLADE can replace BM25 as the sparse retrieval component, improving technical term matching without regression on other query types.

---

## 1. Problem Statement

Current BM25 implementation fails on technical terminology because:
- Naive tokenization treats "SubjectAccessReview" as single token
- No semantic expansion (can't connect "k8s" to "kubernetes")
- No learned term weighting (all terms weighted by TF-IDF only)

**Hypothesis**: SPLADE's learned sparse representations will improve retrieval for technical terms by:
1. Expanding queries with semantically related terms
2. Learning domain-appropriate term weights
3. Preserving exact match signals while adding semantic understanding

---

## 2. Baseline Measurements Required

### 2.1 Current System Performance

Measure current BM25 component in isolation (not hybrid):

| Benchmark | Metric | Purpose |
|-----------|--------|---------|
| Informed (25 queries) | MRR@10 | Primary - technical terms |
| Informed (25 queries) | Recall@10 | Coverage check |
| Informed (25 queries) | Hit@1 | Precision check |
| Needle (20 queries) | MRR@10 | Regression check |
| Realistic (50 queries) | MRR@10 | Generalization check |

### 2.2 BM25-Only Baseline

Must isolate BM25 performance separate from hybrid (RRF) to fairly compare:

| Measurement | Why Needed |
|-------------|------------|
| BM25-only MRR on informed | Direct comparison target for SPLADE |
| BM25-only MRR on needle | Ensure SPLADE doesn't regress |
| BM25 query latency | Baseline for latency comparison |
| BM25 index size | Baseline for storage comparison |

### 2.3 Hybrid System Baseline (for context)

Record current hybrid (BM25 + semantic + RRF) performance:
- Use results from convex fusion POC as reference
- Informed MRR: 0.636 (RRF baseline)
- Needle MRR: 0.842 (RRF baseline)

---

## 3. What Must Be Tested

### 3.1 SPLADE Model Selection

Test multiple SPLADE variants to find best fit:

| Model | Size | Description | Test Priority |
|-------|------|-------------|---------------|
| splade-cocondenser-ensembledistil | 110M | Best official model, distilled | **Primary** |
| splade-cocondenser-selfdistil | 110M | Self-distilled variant | Secondary |
| splade-v3 | 110M | Latest release | Secondary |

**Selection criteria**:
- Best MRR on informed queries
- Acceptable latency (<100ms query encoding)
- Reasonable index size (<3x BM25)

### 3.2 SPLADE as BM25 Replacement

Test SPLADE in place of BM25 within current architecture:

| Configuration | Description |
|---------------|-------------|
| SPLADE-only | Pure SPLADE retrieval (no semantic) |
| SPLADE + Semantic (RRF) | Replace BM25 with SPLADE in hybrid |
| SPLADE + Semantic (Convex) | Test if SPLADE + convex fusion works better |

### 3.3 Document Encoding Strategies

| Strategy | Description | Tradeoff |
|----------|-------------|----------|
| Full document | Encode entire chunk | Most context, slower |
| Title + first N tokens | Encode title and beginning | Faster, may miss details |
| Enriched content | Use current enriched format | Test if enrichment helps SPLADE |
| Raw content | No enrichment | Isolate SPLADE benefit |

### 3.4 Query Encoding Strategies

| Strategy | Description |
|----------|-------------|
| Raw query | Direct query encoding |
| Expanded query | Apply existing query expansion before SPLADE |
| SPLADE expansion only | Let SPLADE handle all expansion |

### 3.5 Technical Term Specific Tests

Create focused test set for technical terminology:

| Category | Example Queries | Expected Behavior |
|----------|-----------------|-------------------|
| CamelCase API types | "SubjectAccessReview", "PodSecurityPolicy" | Exact match + semantic expansion |
| Abbreviations | "k8s", "HPA", "RBAC" | Should expand to full terms |
| Multi-word phrases | "webhook token authenticator" | Should match phrase and components |
| Hyphenated terms | "cluster-admin", "kube-apiserver" | Handle hyphenation properly |
| Dotted paths | "kubernetes.io/hostname" | Handle path-like terms |

---

## 4. Metrics to Collect

### 4.1 Retrieval Quality Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| MRR@10 | Mean Reciprocal Rank at k=10 | Primary quality metric |
| Recall@10 | Fraction of relevant docs in top 10 | Coverage metric |
| Hit@1 | Is rank 1 correct? | Precision metric |
| Hit@5 | Is correct doc in top 5? | Looser precision |
| NDCG@10 | Normalized Discounted Cumulative Gain | Graded relevance |

### 4.2 Efficiency Metrics

| Metric | Definition | Threshold |
|--------|------------|-----------|
| Query encoding latency | Time to encode query with SPLADE | <100ms |
| Document encoding latency | Time to encode one chunk | <100ms |
| Total index build time | Time to encode all chunks | <1 hour |
| Index size | Disk space for SPLADE index | <3x BM25 |
| Memory usage | RAM during retrieval | <2x BM25 |
| Retrieval latency | Time from query to results | <50ms |

### 4.3 Sparsity Metrics

| Metric | Definition | Why It Matters |
|--------|------------|----------------|
| Average non-zero terms (query) | How many terms activated per query | Affects query speed |
| Average non-zero terms (doc) | How many terms activated per doc | Affects index size |
| Term overlap | Fraction of query terms found in top docs | Measures expansion effectiveness |

### 4.4 Per-Query Analysis

For each query, record:
- Query text and ground truth
- SPLADE rank of ground truth vs BM25 rank
- Top expansion terms added by SPLADE
- Score of ground truth document
- Whether technical term was expanded correctly

---

## 5. Test Protocol

### Phase 1: Environment Setup

**Objective**: Prepare SPLADE models and infrastructure

**Tasks**:
1. Install SPLADE dependencies (transformers, torch)
2. Download pre-trained SPLADE model
3. Verify model loads and produces sparse vectors
4. Set up sparse vector storage format
5. Create inverted index builder for SPLADE vectors

**Acceptance**: Model produces valid sparse vectors for sample queries

### Phase 2: Baseline Collection

**Objective**: Establish BM25-only baselines

**Tasks**:
1. Run BM25-only retrieval on informed queries
2. Run BM25-only retrieval on needle queries
3. Run BM25-only retrieval on realistic queries
4. Record latency and index size
5. Document per-query results

**Acceptance**: BM25 baselines documented for all benchmarks

### Phase 3: Document Encoding

**Objective**: Encode all chunks with SPLADE

**Tasks**:
1. Load all chunks from existing index
2. Encode each chunk with SPLADE
3. Build inverted index from sparse vectors
4. Measure encoding time and index size
5. Verify index integrity (all chunks indexed)

**Acceptance**: All chunks encoded, index built, size within 3x BM25

### Phase 4: SPLADE-Only Evaluation

**Objective**: Test SPLADE as standalone retriever

**Tasks**:
1. Run SPLADE retrieval on informed queries
2. Run SPLADE retrieval on needle queries
3. Run SPLADE retrieval on realistic queries
4. Compare to BM25 baselines
5. Analyze per-query improvements/regressions

**Acceptance**: Results for all benchmarks, comparison to baselines

### Phase 5: Hybrid Integration

**Objective**: Test SPLADE + Semantic hybrid

**Tasks**:
1. Implement SPLADE + Semantic with RRF fusion
2. Run hybrid on all benchmarks
3. Compare to current BM25 + Semantic hybrid
4. Test with convex fusion (using learned alpha from POC 1)
5. Identify best hybrid configuration

**Acceptance**: Hybrid results for all benchmarks

### Phase 6: Technical Term Deep Dive

**Objective**: Analyze SPLADE behavior on technical terminology

**Tasks**:
1. Create focused test set of technical term queries
2. Run SPLADE and analyze expansion terms
3. Verify CamelCase terms are handled correctly
4. Check abbreviation expansion
5. Document failure cases

**Acceptance**: Technical term analysis report

### Phase 7: Latency Profiling

**Objective**: Ensure SPLADE meets latency requirements

**Tasks**:
1. Profile query encoding time (cold and warm)
2. Profile retrieval time
3. Compare end-to-end latency vs BM25
4. Identify optimization opportunities
5. Test batch encoding if applicable

**Acceptance**: Latency within acceptable bounds (<100ms query encoding)

---

## 6. Success Criteria

### 6.1 Primary Success (PASS)

All of the following must be true:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Informed MRR improvement | >10% over BM25-only | Meaningful improvement on target queries |
| No Needle regression | MRR within 95% of BM25 | Don't break what works |
| No Realistic regression | MRR within 95% of BM25 | Generalization check |
| Query encoding latency | <100ms | Acceptable for production |
| Index size | <3x BM25 | Reasonable storage overhead |

**If PASS**: Recommend SPLADE as BM25 replacement.

### 6.2 Partial Success (PARTIAL)

Some improvement but with caveats:

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| 5-10% improvement on informed | Below primary threshold | May still be worthwhile |
| Regression on one benchmark | <10% regression | Acceptable tradeoff |
| Latency 100-150ms | Above ideal | Consider caching/optimization |
| Index size 3-5x | Above ideal | May need compression |

**If PARTIAL**: Document tradeoffs, consider for specific use cases.

### 6.3 Failure (FAIL)

Hypothesis rejected:

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| No improvement on informed | SPLADE <= BM25 | SPLADE not beneficial |
| Regression >10% on any benchmark | Significant quality loss | Not acceptable |
| Latency >200ms | Too slow for interactive use | Need different approach |
| Index size >5x | Storage prohibitive | Need compression first |

**If FAIL**: Document findings, consider alternatives (fine-tuned SPLADE, different model).

---

## 7. Comparison Points

### 7.1 SPLADE vs BM25 (Direct Replacement)

| Aspect | BM25 | SPLADE | Expected Winner |
|--------|------|--------|-----------------|
| CamelCase handling | Poor | Good | SPLADE |
| Abbreviation expansion | None | Learned | SPLADE |
| Exact match | Good | Good | Tie |
| Query latency | ~0ms | ~50ms | BM25 |
| Index size | 1x | 2-3x | BM25 |
| Maintenance | None | Model updates | BM25 |

### 7.2 SPLADE+Semantic vs BM25+Semantic (Hybrid)

| Aspect | BM25+Semantic | SPLADE+Semantic | Expected Winner |
|--------|---------------|-----------------|-----------------|
| Technical terms | Moderate | Good | SPLADE hybrid |
| Natural language | Good | Good | Tie |
| Query latency | ~20ms | ~70ms | BM25 hybrid |
| Complexity | Low | Medium | BM25 hybrid |

### 7.3 Expected Per-Query Type Performance

| Query Type | BM25 | SPLADE | Hybrid (SPLADE+Sem) |
|------------|------|--------|---------------------|
| CamelCase API names | Poor | Good | Excellent |
| Abbreviations | Poor | Moderate | Good |
| Natural language | Good | Good | Excellent |
| Exact phrases | Good | Good | Good |
| Semantic concepts | Poor | Moderate | Excellent |

---

## 8. Risk Factors and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Domain mismatch (MS MARCO vs K8s) | Medium | High | Test on technical term subset first |
| Latency too high | Low | High | Profile early, consider caching |
| Index size explosion | Medium | Medium | Monitor sparsity, adjust FLOPS regularization |
| Expansion noise | Medium | Medium | Analyze expansion terms, filter low-weight |
| Model loading time | Low | Low | Load once, keep in memory |
| GPU required | Low | Medium | CPU inference possible, just slower |

---

## 9. Out of Scope

The following are explicitly NOT part of this POC:

| Item | Reason | Deferred To |
|------|--------|-------------|
| Fine-tuning SPLADE | Too complex for initial POC | Future POC if needed |
| SPLADE + ColBERT hybrid | Different architecture | Separate POC |
| Mistral-SPLADE | Requires larger model | Future exploration |
| Production deployment | This is evaluation only | After POC succeeds |
| Index compression | Optimization step | After baseline established |

---

## 10. Deliverables

### 10.1 Required Outputs

| Deliverable | Format | Purpose |
|-------------|--------|---------|
| BM25 baseline results | JSON | Comparison reference |
| SPLADE results (all configs) | JSON | Main evaluation data |
| Per-query analysis | CSV/JSON | Detailed breakdown |
| Latency measurements | JSON | Performance data |
| Technical term analysis | Markdown | Qualitative findings |
| Expansion term samples | Markdown | Show what SPLADE learns |
| Final recommendation | Markdown | Actionable conclusion |

### 10.2 Visualization Requirements

1. **MRR comparison bar chart**: BM25 vs SPLADE vs Hybrid across benchmarks
2. **Per-query scatter plot**: BM25 rank vs SPLADE rank
3. **Latency breakdown**: Query encoding, retrieval, total
4. **Index size comparison**: BM25 vs SPLADE
5. **Expansion term examples**: Word clouds or lists for sample queries

---

## 11. Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Setup | 0.5 days | None |
| Phase 2: Baselines | 0.5 days | Phase 1 |
| Phase 3: Encoding | 1 day | Phase 1 |
| Phase 4: SPLADE-only | 1 day | Phase 2, 3 |
| Phase 5: Hybrid | 1 day | Phase 4 |
| Phase 6: Technical terms | 0.5 days | Phase 4 |
| Phase 7: Latency | 0.5 days | Phase 4 |
| Documentation | 1 day | All phases |
| **Total** | **6-7 days** | |

---

## 12. Decision Framework

After completing evaluation, use this framework:

### If SPLADE-only > BM25-only by >10%:
→ Strong signal that SPLADE helps
→ Proceed to test SPLADE + Semantic hybrid
→ If hybrid also improves, recommend production integration

### If SPLADE-only ≈ BM25-only (within 5%):
→ SPLADE may not be worth the complexity
→ Check if technical term subset shows improvement
→ Consider fine-tuning before dismissing

### If SPLADE-only < BM25-only:
→ Domain mismatch likely
→ Analyze failure cases
→ Consider fine-tuning or alternative models

### If latency unacceptable:
→ Test CPU vs GPU tradeoffs
→ Consider query caching
→ Evaluate if async encoding is feasible

---

## 13. References

- SPLADE paper: arXiv:2107.05720
- SPLADE-v3 paper: arXiv:2403.06789
- BEIR benchmark: beir.ai
- Current BM25 implementation: `src/plm/search/components/bm25.py`
- Convex fusion results: `poc/convex_fusion_benchmark/RESULTS.md`
- Plan document: `.sisyphus/plans/informed-query-improvements.md`

---

*Document version: 1.0*
*Created: 2026-02-22*
