# SPLADE Benchmark POC: Implementation TODO

**Estimated Duration**: 6-7 days  
**Priority**: 2 (after Convex Fusion POC)  
**Predecessor**: Convex Fusion POC (COMPLETE - PARTIAL)

---

## Phase 1: Environment Setup (Day 1 - Morning)

### 1.1 Project Structure
- [ ] Create `pyproject.toml` with dependencies:
  - transformers (SPLADE model)
  - torch (inference)
  - numpy (vector operations)
  - scipy (sparse matrices)
  - pandas (analysis)
  - matplotlib (visualization)
  - tqdm (progress bars)
- [ ] Create `.venv` and run `uv sync`
- [ ] Verify CUDA availability (optional, CPU works too)
- [ ] Create `artifacts/` subdirectory for outputs
- [ ] Create `artifacts/plots/` for visualizations

### 1.2 Data Infrastructure
- [ ] Create symlink to `poc/plm_vs_rag_benchmark/test_db/` (existing PLM index)
- [ ] Create symlink to `poc/plm_vs_rag_benchmark/corpus/` (source documents)
- [ ] Locate benchmark query files:
  - Informed queries (25)
  - Needle queries (20)
  - Realistic queries (50 sample)
- [ ] Verify ground truth document IDs are accessible

### 1.3 Model Setup
- [ ] Download SPLADE model: `naver/splade-cocondenser-ensembledistil`
- [ ] Verify model loads successfully
- [ ] Test encoding on sample text
- [ ] Measure model load time
- [ ] Document GPU vs CPU inference options

### 1.4 Validation Checkpoint
- [ ] Encode sample query and verify sparse vector output
- [ ] Confirm non-zero terms are reasonable (10-100 terms typical)
- [ ] Verify term IDs map to vocabulary correctly
- [ ] Document any setup issues encountered

---

## Phase 2: Baseline Collection (Day 1 - Afternoon)

### 2.1 BM25-Only Baseline Extraction
- [ ] Create `baseline_bm25.py` module
- [ ] Load existing BM25 index from PLM
- [ ] Run BM25-only retrieval (no semantic, no RRF) on:
  - [ ] All 25 informed queries
  - [ ] All 20 needle queries
  - [ ] 50 sampled realistic queries
- [ ] Record per-query results:
  - Query text
  - Ground truth document ID
  - BM25 rank of ground truth
  - BM25 score of ground truth
  - Top-10 retrieved document IDs

### 2.2 Baseline Metrics Calculation
- [ ] Calculate MRR@10 for each benchmark
- [ ] Calculate Recall@10 for each benchmark
- [ ] Calculate Hit@1 and Hit@5 for each benchmark
- [ ] Compare to known hybrid baselines (from convex fusion POC)
- [ ] Document any discrepancies

### 2.3 BM25 Efficiency Baseline
- [ ] Measure BM25 query latency (average over 100 queries)
- [ ] Measure BM25 index size on disk
- [ ] Document memory usage during retrieval

### 2.4 Save Baselines
- [ ] Save results to `artifacts/baseline_bm25.json`
- [ ] Include per-query details
- [ ] Include aggregate metrics
- [ ] Include efficiency measurements

---

## Phase 3: SPLADE Document Encoding (Day 2)

### 3.1 Encoding Pipeline Setup
- [ ] Create `splade_encoder.py` module
- [ ] Implement document encoding function:
  - Input: document text
  - Output: sparse vector (dict of term_id: weight)
- [ ] Implement batch encoding for efficiency
- [ ] Add progress bar for encoding

### 3.2 Chunk Loading
- [ ] Load all chunks from existing PLM index
- [ ] Count total chunks to encode
- [ ] Verify chunk content is accessible
- [ ] Document chunk format (raw vs enriched)

### 3.3 Encoding Execution
- [ ] Encode all chunks with SPLADE
- [ ] Track encoding time per chunk
- [ ] Monitor memory usage during encoding
- [ ] Handle any encoding errors gracefully
- [ ] Save sparse vectors to intermediate format

### 3.4 Encoding Analysis
- [ ] Calculate average non-zero terms per document
- [ ] Calculate total unique terms across all documents
- [ ] Identify most common expansion terms
- [ ] Compare to BM25 vocabulary size

### 3.5 Index Building
- [ ] Create `splade_index.py` module
- [ ] Build inverted index from sparse vectors:
  - Term → list of (doc_id, weight) pairs
- [ ] Implement efficient retrieval function
- [ ] Save index to disk
- [ ] Measure index size

### 3.6 Encoding Checkpoint
- [ ] Save encoding stats to `artifacts/encoding_stats.json`:
  - Total chunks encoded
  - Encoding time (total and per-chunk)
  - Average sparsity (non-zero terms)
  - Index size on disk
- [ ] Verify index integrity (all chunks searchable)

---

## Phase 4: SPLADE-Only Evaluation (Day 3)

### 4.1 Query Encoding
- [ ] Implement query encoding function
- [ ] Test on sample queries
- [ ] Measure query encoding latency
- [ ] Verify query sparse vectors are reasonable

### 4.2 Retrieval Implementation
- [ ] Implement SPLADE retrieval function:
  - Input: query sparse vector
  - Output: ranked list of (doc_id, score) pairs
- [ ] Use dot product scoring (sum of matching term weights)
- [ ] Return top-k results (k=50 for reranking compatibility)

### 4.3 Run SPLADE-Only Evaluation
- [ ] Run SPLADE retrieval on informed queries (25)
- [ ] Run SPLADE retrieval on needle queries (20)
- [ ] Run SPLADE retrieval on realistic queries (50)
- [ ] Record per-query results:
  - SPLADE rank of ground truth
  - SPLADE score of ground truth
  - Top-10 retrieved documents
  - Query encoding time

### 4.4 SPLADE vs BM25 Comparison
- [ ] Calculate MRR@10 for each benchmark
- [ ] Calculate improvement over BM25 baseline:
  - Informed: target >10% improvement
  - Needle: target no regression (within 95%)
  - Realistic: target no regression (within 95%)
- [ ] Calculate Recall@10 and Hit@1 changes
- [ ] Identify queries where SPLADE wins
- [ ] Identify queries where SPLADE loses

### 4.5 Per-Query Analysis
- [ ] For each query, record:
  - BM25 rank vs SPLADE rank
  - Rank improvement (positive = SPLADE better)
  - Whether query contains technical terms
- [ ] Categorize queries:
  - Improved (>3 rank positions better)
  - Unchanged (within 3 positions)
  - Regressed (worse by any amount)
- [ ] Analyze patterns in improvements/regressions

### 4.6 Save SPLADE-Only Results
- [ ] Save to `artifacts/splade_only_results.json`
- [ ] Include per-query details
- [ ] Include aggregate metrics
- [ ] Include comparison to BM25

---

## Phase 5: Hybrid Integration (Day 4)

### 5.1 SPLADE + Semantic with RRF
- [ ] Create `hybrid_splade.py` module
- [ ] Implement SPLADE + Semantic hybrid:
  - Get SPLADE results (top-50)
  - Get Semantic results (top-50)
  - Apply RRF fusion (k=60)
- [ ] Use same RRF parameters as current production

### 5.2 Run Hybrid Evaluation
- [ ] Run SPLADE+Semantic hybrid on informed queries
- [ ] Run SPLADE+Semantic hybrid on needle queries
- [ ] Run SPLADE+Semantic hybrid on realistic queries
- [ ] Record per-query results

### 5.3 Compare to BM25+Semantic Hybrid
- [ ] Load BM25+Semantic baseline (from convex fusion POC)
- [ ] Calculate improvement for each benchmark
- [ ] Identify if SPLADE hybrid outperforms BM25 hybrid

### 5.4 Test Alternative Fusion Methods
- [ ] Test SPLADE + Semantic with convex fusion:
  - Use optimal alpha from convex fusion POC (0.36)
  - Test if SPLADE changes optimal alpha
- [ ] Compare RRF vs convex fusion with SPLADE
- [ ] Document best configuration

### 5.5 Save Hybrid Results
- [ ] Save to `artifacts/hybrid_splade_results.json`
- [ ] Include all configurations tested
- [ ] Include comparison to BM25 hybrid

---

## Phase 6: Technical Term Deep Dive (Day 5 - Morning)

### 6.1 Create Technical Term Test Set
- [ ] Extract queries with known technical terms:
  - CamelCase: "SubjectAccessReview", "PodSecurityPolicy"
  - Abbreviations: "RBAC", "HPA", "k8s"
  - Multi-word: "webhook token authenticator"
  - Hyphenated: "kube-apiserver", "cluster-admin"
  - Dotted: "kubernetes.io/hostname"
- [ ] Ensure ground truth exists for each
- [ ] Document expected SPLADE behavior

### 6.2 Analyze SPLADE Expansions
- [ ] For each technical term query:
  - [ ] Get SPLADE sparse vector
  - [ ] Extract top-20 expansion terms (by weight)
  - [ ] Verify expansion makes semantic sense
  - [ ] Check if related Kubernetes terms appear
- [ ] Document expansion patterns

### 6.3 Compare Term Handling
- [ ] For each technical term query:
  - [ ] Compare BM25 rank vs SPLADE rank
  - [ ] Analyze why SPLADE helps (or doesn't)
  - [ ] Check if expansion introduces noise

### 6.4 Failure Case Analysis
- [ ] Identify queries where SPLADE fails
- [ ] Analyze expansion terms for failures
- [ ] Determine if failures are due to:
  - Missing domain terms in SPLADE vocabulary
  - Incorrect expansions
  - Over-expansion (too many terms)
- [ ] Document recommendations for improvement

### 6.5 Save Technical Term Analysis
- [ ] Save to `artifacts/technical_term_analysis.json`
- [ ] Include expansion terms for each query
- [ ] Include success/failure categorization
- [ ] Write human-readable summary in `artifacts/technical_term_report.md`

---

## Phase 7: Latency Profiling (Day 5 - Afternoon)

### 7.1 Query Encoding Latency
- [ ] Measure cold start latency (first query after model load)
- [ ] Measure warm latency (subsequent queries)
- [ ] Run 100 queries and calculate:
  - Mean latency
  - P50 latency
  - P95 latency
  - P99 latency
- [ ] Compare GPU vs CPU latency (if GPU available)

### 7.2 Retrieval Latency
- [ ] Measure index lookup time
- [ ] Measure scoring time
- [ ] Measure sorting/ranking time
- [ ] Calculate total retrieval latency

### 7.3 End-to-End Latency
- [ ] Measure query encoding + retrieval combined
- [ ] Compare to BM25 end-to-end latency
- [ ] Compare to current hybrid system latency

### 7.4 Throughput Testing
- [ ] Measure queries per second (single-threaded)
- [ ] Test batch query encoding if applicable
- [ ] Estimate production capacity

### 7.5 Memory Profiling
- [ ] Measure memory usage during model load
- [ ] Measure memory usage during encoding
- [ ] Measure memory usage during retrieval
- [ ] Compare to BM25 memory usage

### 7.6 Save Latency Results
- [ ] Save to `artifacts/latency_profile.json`
- [ ] Include all latency measurements
- [ ] Include memory measurements
- [ ] Include throughput estimates

---

## Phase 8: Alternative Configurations (Day 6 - Morning)

### 8.1 Test Different SPLADE Models
- [ ] Test `splade-cocondenser-selfdistil`:
  - Encode sample documents
  - Run on informed queries
  - Compare to primary model
- [ ] Test `splade-v3` if available:
  - Encode sample documents
  - Run on informed queries
  - Compare to primary model
- [ ] Document model comparison results

### 8.2 Test Document Encoding Variations
- [ ] Test encoding with enriched content:
  - Use current enrichment (keywords | entities)
  - Compare to raw content encoding
- [ ] Test encoding with title prefix:
  - Prepend document title
  - Compare retrieval quality
- [ ] Document best encoding strategy

### 8.3 Test Query Encoding Variations
- [ ] Test with existing query expansion:
  - Apply current expander before SPLADE
  - Check if it helps or hurts
- [ ] Test SPLADE expansion only:
  - Disable existing expander
  - Rely on SPLADE for expansion
- [ ] Document best query strategy

### 8.4 Save Configuration Comparison
- [ ] Save to `artifacts/configuration_comparison.json`
- [ ] Include all tested configurations
- [ ] Identify best overall configuration

---

## Phase 9: Visualization (Day 6 - Afternoon)

### 9.1 Create Comparison Charts
- [ ] Create `visualize.py` script
- [ ] Plot 1: MRR comparison bar chart
  - BM25-only vs SPLADE-only vs Hybrid (BM25) vs Hybrid (SPLADE)
  - Grouped by benchmark (Informed, Needle, Realistic)
- [ ] Plot 2: Per-query rank comparison scatter
  - X-axis: BM25 rank
  - Y-axis: SPLADE rank
  - Diagonal = no change, below = SPLADE better

### 9.2 Create Latency Charts
- [ ] Plot 3: Latency comparison bar chart
  - Query encoding, retrieval, total
  - BM25 vs SPLADE
- [ ] Plot 4: Latency distribution histogram
  - Query encoding latency distribution

### 9.3 Create Index Size Chart
- [ ] Plot 5: Index size comparison
  - BM25 index vs SPLADE index
  - Show ratio

### 9.4 Create Expansion Visualization
- [ ] Plot 6: Sample expansion term display
  - Show query and top expansion terms
  - Color by weight
- [ ] Alternative: Word cloud for expansion terms

### 9.5 Save Visualizations
- [ ] Save all plots to `artifacts/plots/`
- [ ] Create summary figure for RESULTS.md

---

## Phase 10: Documentation (Day 7)

### 10.1 Create Results Document
- [ ] Create `RESULTS.md` from template structure
- [ ] Fill in execution summary:
  - Start/end dates
  - Executor
  - Overall status (PASS/PARTIAL/FAIL)

### 10.2 Document Primary Findings
- [ ] Fill in hypothesis verdict with evidence
- [ ] Fill in primary metrics table:
  - Informed MRR: BM25 vs SPLADE vs target
  - Needle MRR: BM25 vs SPLADE vs target
  - Realistic MRR: BM25 vs SPLADE vs target
- [ ] Document improvement percentages
- [ ] Note statistical significance if calculated

### 10.3 Document Efficiency Findings
- [ ] Fill in latency comparison table
- [ ] Fill in index size comparison
- [ ] Document memory usage
- [ ] Note any concerns

### 10.4 Document Technical Term Findings
- [ ] Summarize expansion term analysis
- [ ] Highlight successful expansions
- [ ] Document failure cases
- [ ] Provide examples

### 10.5 Write Recommendations
- [ ] If PASS: Document production integration path
- [ ] If PARTIAL: Document conditions and caveats
- [ ] If FAIL: Document reasons and alternatives
- [ ] Suggest follow-up work if needed

### 10.6 Update README
- [ ] Add results summary to README.md
- [ ] Update with final verdict
- [ ] Add reproduction instructions

### 10.7 Update Plan Document
- [ ] Update `.sisyphus/plans/informed-query-improvements.md`
- [ ] Mark SPLADE POC as complete
- [ ] Add key findings to POC Status Summary
- [ ] Update priority list for remaining POCs

---

## Phase 11: Final Verdict (Day 7 - End)

### 11.1 Success Criteria Check
- [ ] Check primary criteria:
  - [ ] Informed MRR improvement >10%? YES/NO
  - [ ] Needle MRR within 95% of BM25? YES/NO
  - [ ] Realistic MRR within 95% of BM25? YES/NO
  - [ ] Query latency <100ms? YES/NO
  - [ ] Index size <3x BM25? YES/NO
- [ ] Determine overall verdict: PASS / PARTIAL / FAIL

### 11.2 Document Decision
- [ ] Write clear recommendation in RESULTS.md
- [ ] If PASS: Outline integration steps
- [ ] If PARTIAL: Document acceptable use cases
- [ ] If FAIL: Recommend next steps (fine-tuning? different model?)

### 11.3 Cleanup
- [ ] Remove debug code
- [ ] Verify all artifacts are saved
- [ ] Verify all plots are generated
- [ ] Test reproduction steps work

---

## Checkpoints

| Checkpoint | Artifact | Acceptance Criteria |
|------------|----------|---------------------|
| End of Phase 1 | Model loaded | SPLADE produces valid sparse vectors |
| End of Phase 2 | `baseline_bm25.json` | BM25 baselines for all benchmarks |
| End of Phase 3 | SPLADE index | All chunks encoded, index built |
| End of Phase 4 | `splade_only_results.json` | SPLADE-only results for all benchmarks |
| End of Phase 5 | `hybrid_splade_results.json` | Hybrid results for all benchmarks |
| End of Phase 6 | `technical_term_analysis.json` | Technical term analysis complete |
| End of Phase 7 | `latency_profile.json` | All latency measurements |
| End of Phase 9 | `artifacts/plots/` | All visualizations created |
| End of Phase 10 | `RESULTS.md` | Complete, no TODOs |

---

## Risk Mitigation Checkpoints

- [ ] **After Phase 1**: If model fails to load, check CUDA/memory, try CPU-only
- [ ] **After Phase 3**: If encoding too slow (>2 hours), consider batch size tuning
- [ ] **After Phase 4**: If no improvement on informed, analyze expansions before proceeding
- [ ] **After Phase 5**: If hybrid worse than SPLADE-only, investigate fusion weights
- [ ] **After Phase 7**: If latency >150ms, document and note optimization opportunities

---

## Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| PLM index | `poc/plm_vs_rag_benchmark/test_db/` | Available |
| Benchmark queries | `poc/plm_vs_rag_benchmark/` | Available |
| Baseline metrics | `poc/convex_fusion_benchmark/` | Available |
| SPLADE model | HuggingFace | Download needed |

---

## Notes

- Do NOT modify production code in `src/plm/` during this POC
- All implementations should be self-contained in `poc/splade_benchmark/`
- If retriever modification needed, create local wrapper
- Document all assumptions and deviations from plan
- Focus on evaluation, not production optimization

---

*Created: 2026-02-22*
*Based on: EVALUATION_CRITERIA.md*
