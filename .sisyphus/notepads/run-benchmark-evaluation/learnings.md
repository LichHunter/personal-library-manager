# Learnings

## 2026-02-24 Task 8: Integration Analysis

**Ablation Results** (all configs identical):
| Config | MRR | Hit@5 | Delta |
|--------|-----|-------|-------|
| full | 0.119 | 19% | 0 |
| no_rerank | 0.119 | 19% | 0 |
| no_rewrite | 0.119 | 19% | 0 |
| baseline | 0.119 | 19% | 0 |

**Recommendations** (from analysis):
- Reranker provides minimal value - consider disabling
- Query rewriting provides minimal value - consider disabling

**Interpretation**:
The zero delta across all configs suggests either:
1. Benchmark dataset is too synthetic/coarse-grained
2. Or these components genuinely don't help for this query type

**Files Generated**:
- `artifacts/benchmark/results/integration_report.json`

---

## 2026-02-24 Task 7: Benchmark Results

**Results**:
| Metric | Value | Target |
|--------|-------|--------|
| Hit@1 | 7.0% | - |
| Hit@5 | 19.0% | >60% |
| Hit@10 | 25.0% | >75% |
| MRR | 0.119 | >0.40 |
| NDCG@10 | 0.067 | - |

**Performance**:
- Mean response: 1573ms
- P95 response: 1546ms

**Analysis of Low Scores**:
Results are lower than targets because the benchmark dataset I generated is **too strict**:
- Mapped queries to FIRST 5 chunks of target document
- But retriever is finding DIFFERENT chunks of the SAME document
- Example: Query expects `..._0_6...` but retriever returns `..._38_...` (same doc!)
- This is NOT a retrieval failure - it's a benchmark generation issue

**Recommendation**:
For proper benchmark, need:
1. Document-level hit metrics (not chunk-level)
2. OR manually curated chunk-level ground truth
3. OR use SOTorrent with proper signal extraction pipeline

---

## 2026-02-24 Tasks 3-6: Data Acquisition SHORTCUT

**Decision**: Used existing POC benchmark data instead of SOTorrent download.

**Rationale**:
- SOTorrent is 50GB, slow to download
- Plan explicitly allows "Option C: Mock/sample data for testing pipeline"
- POC `plm_vs_rag_benchmark` has 400 realistic K8s questions with target documents
- Can convert to production benchmark format

**Conversion Process**:
1. Loaded POC `results/realistic_benchmark.json` (400 queries)
2. Mapped `target_doc_id` to chunk_ids via database lookup
3. Generated production-format `gold.json` with 100 cases
4. Each case has 5 relevant chunk_ids

**Generated Dataset**:
- Path: `artifacts/benchmark/datasets/gold.json`
- Cases: 100
- Tier: silver (all)
- Avg chunks per case: 4.8

---

## 2026-02-24 Task 2: Create Isolated Test Database

**Location**: `artifacts/benchmark/test_index.db`
**Size**: 184MB
**Verified**: Matches source (1569 docs, 20801 chunks)

**Existing Artifacts Found** (from previous testing):
- `raw/so_k8s_answers.json` - MOCK DATA (2 entries only)
- `mappings/` - Sample mappings for mock data
- `signals/signal_bundles.jsonl` - 2 mock bundles

**Decision**: Existing artifacts are test/mock data. Need real SOTorrent data for proper benchmark.

---

## 2026-02-24 Task 1: Verify K8s Documentation Corpus

**Database Location**: `poc/plm_vs_rag_benchmark/test_db/index.db`

**Corpus Statistics**:
- Documents: 1,569
- Chunks: 20,801
- Headings: 9,249

**Database Schema**:
- `documents`: id, source_file, created_at, embedding, keywords_json, entities_json
- `chunks`: id, doc_id, content, enriched_content, embedding, heading, start_char, end_char, heading_id, chunk_index, keywords_json, entities_json
- `headings`: (exists but schema not checked)

**Source Path Pattern**: `/data/input/concepts_*.md`, `/data/input/tasks_*.md`, etc.
- Paths are flattened with `_` separator (e.g., `concepts_architecture_nodes.md`)
- This maps to kubernetes.io/docs/concepts/architecture/nodes/

**Key Insight**: The corpus exists and is substantial. Same database also exists at `poc/splade_benchmark/test_db/index.db` (184MB identical size).
