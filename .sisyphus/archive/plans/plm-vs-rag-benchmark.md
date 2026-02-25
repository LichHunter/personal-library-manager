# PLM vs Baseline RAG Benchmark

## TL;DR

> **Quick Summary**: Compare PLM's sophisticated hybrid retrieval against LangChain RAG using 3 variants to isolate which features (enrichment vs RRF/BM25/expansion) drive improvement.
> 
> **Three Variants**:
> 1. **PLM Full** — Complete system (BM25 + semantic + RRF + enrichment + expansion)
> 2. **Baseline + Enriched** — LangChain + FAISS + PLM's enriched chunks
> 3. **Baseline Naive** — LangChain + FAISS + raw chunks
>
> **Deliverables**:
> - PLM running on COPY of existing database (original untouched)
> - LangChain baseline supporting both raw and enriched modes
> - Chunk extractor to pull identical chunks from PLM SQLite
> - Benchmark results on 20 needle + 400 realistic questions
> - Attribution analysis: what % improvement comes from each feature
> 
> **Estimated Effort**: Medium (6-10 hours)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Setup → Extract Chunks → Run All 3 Variants → Analyze
> 
> **Key Optimizations**:
> - Copy existing PLM database (no re-ingestion)
> - Extract chunks from SQLite (identical chunk boundaries across all variants)
> - Oracle-reviewed for statistical rigor

---

## Context

### Original Request
Create comprehensive benchmark comparing PLM against baseline RAG systems. Must start PLM on a NEW/clean database without deleting current production database.

### Interview Summary
**Key Decisions**:
- Use out-of-the-box LangChain RAG (no custom chunking/enrichment) as baseline
- This proves PLM's sophistication provides real value vs naive implementation
- Run both question sets: needle (20, validation) + realistic (400, comprehensive)
- Document all preprocessing differences explicitly in report
- **OPTIMIZATION**: Copy existing PLM database instead of re-ingesting (user requested)

**Research Findings**:
- PLM achieves 90% pass rate on needle questions, 40.75% Hit@5 on realistic
- LangChain baseline expected to underperform on vocabulary mismatch (no query expansion)
- Academic research confirms hybrid BM25+semantic beats pure dense retrieval
- MCP-Bench and RAGAS provide industry-standard evaluation frameworks

### Metis Review
**Identified Gaps** (addressed):
- Chunking fidelity: Extract PLM's actual chunks from SQLite (Oracle recommendation)
- Content enrichment: Test with 3 variants to isolate enrichment's contribution
- Database isolation: Copy existing PLM DB to temp directory; baseline uses temp directory
- Configuration matching: Embedding model must be identical (BGE-base-en-v1.5)
- Ingestion time: User requested copying existing DB instead of re-ingesting (saves ~10-15 min)

### Oracle Review
**Critical Finding**: Enrichment asymmetry—PLM embeds enriched content, baseline embeds raw content. Cannot attribute wins to specific features.

**Applied Recommendations**:
1. ✅ Add "Baseline + Enriched" variant to isolate RRF/BM25/expansion from enrichment
2. ✅ Extract PLM's exact chunks from SQLite instead of re-chunking
3. ✅ Verify LangChain FAISS uses normalized embeddings (L2 normalization)
4. ✅ Log query expansion triggers
5. ✅ Add bootstrap confidence intervals for MRR difference

---

## Work Objectives

### Core Objective
Empirically prove PLM's hybrid retrieval system outperforms naive out-of-the-box RAG, AND isolate which features (enrichment vs RRF/BM25/expansion) contribute to the improvement.

### Concrete Deliverables
- `poc/plm_vs_rag_benchmark/` — New POC directory with all benchmark code
- `poc/plm_vs_rag_benchmark/baseline_langchain.py` — LangChain RAG implementation (supports both raw and enriched modes)
- `poc/plm_vs_rag_benchmark/chunk_extractor.py` — Extract chunks + enriched content from PLM's SQLite
- `poc/plm_vs_rag_benchmark/benchmark_runner.py` — Unified benchmark script for all 3 variants
- `poc/plm_vs_rag_benchmark/results/` — JSON results and markdown report
- `docs/PLM_VS_RAG_BENCHMARK_RESULTS.md` — Final research report (optional)

### Three Comparison Variants

| Variant | Description | What It Tests |
|---------|-------------|---------------|
| **PLM Full** | Complete system: BM25 + semantic + RRF + enrichment + expansion | "Does full PLM beat naive RAG?" |
| **Baseline + Enriched** | LangChain + FAISS + PLM's enriched chunks | "How much does enrichment alone help?" |
| **Baseline Naive** | LangChain + FAISS + raw chunks | "What's the naive RAG baseline?" |

**Attribution Analysis**:
- PLM vs Baseline-Naive = Total improvement
- Baseline-Enriched vs Baseline-Naive = Enrichment contribution
- PLM vs Baseline-Enriched = RRF + BM25 + expansion contribution

### Definition of Done
- [ ] PLM runs on copy of production database (original untouched)
- [ ] All 3 variants use identical chunk boundaries (extracted from PLM SQLite)
- [ ] All 3 variants answer same 420 questions (20 needle + 400 realistic)
- [ ] Metrics calculated: MRR, Hit@1, Hit@5, Recall@5, LLM grades
- [ ] Bootstrap 95% CI on MRR differences
- [ ] Report documents: methodology, variant comparison, attribution analysis, conclusions

### Must Have
- Fresh database for PLM (not production)
- Identical embedding model (BGE-base-en-v1.5)
- Identical document corpus (Kubernetes docs)
- Identical query sets (needle + realistic)
- Metrics: MRR, Hit@k, pass rates
- LLM-based grading for answer quality

### Must NOT Have (Guardrails)
- ❌ DO NOT modify or delete production database at `plm-search-index` Docker volume
- ❌ DO NOT use `--strategy all` (drops important metrics)
- ❌ DO NOT replicate PLM's enrichment in baseline (intentional difference)
- ❌ DO NOT use different embedding models between systems
- ❌ DO NOT skip the needle validation phase before realistic benchmark

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES (pytest, existing benchmark scripts)
- **Automated tests**: Tests-after (verify benchmark produces valid output)
- **Framework**: pytest for verification scripts

### Agent-Executed QA Scenarios (MANDATORY)

Every task includes specific verification scenarios using Bash commands.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Create POC directory structure
├── Task 2: Implement LangChain baseline
└── Task 3: Create unified benchmark runner

Wave 2 (After Wave 1):
├── Task 4: Ingest corpus into PLM (fresh DB)
├── Task 5: Ingest corpus into baseline
└── (parallel: both ingest same docs)

Wave 3 (After Wave 2):
├── Task 6: Run needle benchmark (both systems)
└── Task 7: Validate metrics and sanity check

Wave 4 (After Wave 3):
├── Task 8: Run realistic benchmark (both systems)
└── Task 9: Generate comparison report

Critical Path: Task 1 → Task 4 → Task 6 → Task 8 → Task 9
Parallel Speedup: ~40% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4, 5 | None |
| 2 | 1 | 5 | 3 |
| 3 | 1 | 6, 8 | 2 |
| 4 | 1 | 6 | 2, 3, 5 |
| 5 | 1, 2 | 6 | 3, 4 |
| 6 | 3, 4, 5 | 7 | None |
| 7 | 6 | 8 | None |
| 8 | 7 | 9 | None |
| 9 | 8 | None | None |

---

## TODOs

### Task 1: Create POC Directory Structure

**What to do**:
- Create `poc/plm_vs_rag_benchmark/` directory
- Create `pyproject.toml` with dependencies (langchain, faiss-cpu, sentence-transformers)
- Create `README.md` with POC documentation template
- Create subdirectories: `results/`, `corpus/` (symlink to existing)

**Must NOT do**:
- Do not create new corpus - symlink to existing `poc/chunking_benchmark_v2/corpus/kubernetes`

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: []
- **Reason**: Simple file creation, no complex logic

**Parallelization**:
- **Can Run In Parallel**: NO (first task)
- **Blocks**: Tasks 2, 3, 4, 5

**References**:
- `poc/modular_retrieval_pipeline/pyproject.toml` — Example POC pyproject structure
- `poc/README.md` — POC guidelines and README template

**Acceptance Criteria**:

```
Scenario: POC directory structure created
  Tool: Bash
  Steps:
    1. ls -la poc/plm_vs_rag_benchmark/
    2. Assert: pyproject.toml exists
    3. Assert: README.md exists
    4. Assert: results/ directory exists
    5. ls -la poc/plm_vs_rag_benchmark/corpus/
    6. Assert: kubernetes symlink points to ../chunking_benchmark_v2/corpus/kubernetes
  Expected Result: All files and directories present
  Evidence: Directory listing captured
```

**Commit**: YES (groups with 2, 3)
- Message: `feat(benchmark): add PLM vs RAG benchmark POC structure`
- Files: `poc/plm_vs_rag_benchmark/`

---

### Task 2: Implement Chunk Extractor

**What to do**:
- Create `poc/plm_vs_rag_benchmark/chunk_extractor.py`
- Extract chunks from PLM's SQLite database
- For each chunk, extract:
  - `chunk_id` — Unique identifier
  - `doc_id` — Parent document ID
  - `content` — Raw chunk text
  - `enriched_content` — Content with keywords/entities prepended
  - `embedding` — Pre-computed embedding (optional, for verification)
- Save to JSON for both baseline variants to use
- This ensures identical chunk boundaries across all 3 variants

**Must NOT do**:
- Do not re-chunk documents
- Do not modify the original database

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: []
- **Reason**: Simple SQLite query and JSON export

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Task 3)
- **Blocks**: Task 5
- **Blocked By**: Task 1, Task 4 (needs copied DB)

**References**:
- `src/plm/search/storage/sqlite.py` — SQLiteStorage schema (chunks table has `content`, `enriched_content`)

**Acceptance Criteria**:

```
Scenario: Chunk extractor exports all chunks
  Tool: Bash
  Steps:
    1. cd poc/plm_vs_rag_benchmark
    2. python chunk_extractor.py --db-path /tmp/plm-benchmark-copy/index.db --output chunks.json
    3. python -c "
       import json
       with open('chunks.json') as f:
           chunks = json.load(f)
       assert len(chunks) > 5000, f'Expected 5000+ chunks, got {len(chunks)}'
       assert 'content' in chunks[0]
       assert 'enriched_content' in chunks[0]
       assert 'doc_id' in chunks[0]
       print(f'OK: Extracted {len(chunks)} chunks')
       "
  Expected Result: 5000+ chunks extracted with content and enriched_content
  Evidence: chunks.json created, count printed
```

**Commit**: YES (groups with 1, 3)
- Message: `feat(benchmark): add chunk extractor for PLM SQLite`
- Files: `poc/plm_vs_rag_benchmark/chunk_extractor.py`

---

### Task 3: Implement LangChain Baseline RAG

**What to do**:
- Create `poc/plm_vs_rag_benchmark/baseline_langchain.py`
- Implement `BaselineLangChainRAG` class with TWO modes:
  - `ingest_chunks(chunks: list[dict], use_enriched: bool)` — Load pre-extracted chunks
  - `retrieve(query: str, k: int = 5) -> list[dict]` — Return top-k chunks
  - `close()` — Cleanup resources
- When `use_enriched=True`: Embed `enriched_content` field
- When `use_enriched=False`: Embed `content` field (raw)
- Use `BAAI/bge-base-en-v1.5` embedding model with `normalize_embeddings=True`
- Use FAISS with **L2 normalization** (critical for cosine similarity)
- Return chunks with `doc_id`, `content`, `score` fields

**Must NOT do**:
- Do not implement query expansion
- Do not implement BM25 (pure semantic baseline)
- Do not re-chunk (use extracted chunks)

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: []
- **Reason**: Straightforward implementation following existing patterns

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Task 2)
- **Blocks**: Task 5
- **Blocked By**: Task 1

**References**:
- `poc/modular_retrieval_pipeline/components/embedding_encoder.py` — PLM's embedding (use same model + normalization)
- LangChain FAISS: https://python.langchain.com/docs/integrations/vectorstores/faiss/
- **Critical**: Use `normalize_L2=True` or pre-normalize embeddings for cosine similarity

**Acceptance Criteria**:

```
Scenario: Baseline RAG supports both raw and enriched modes
  Tool: Bash
  Preconditions: POC directory exists, chunks.json exists
  Steps:
    1. cd poc/plm_vs_rag_benchmark && uv sync
    2. python -c "
       from baseline_langchain import BaselineLangChainRAG
       import json
       
       # Load test chunks
       with open('chunks.json') as f:
           chunks = json.load(f)[:100]  # Use subset for speed
       
       # Test raw mode
       rag_raw = BaselineLangChainRAG()
       rag_raw.ingest_chunks(chunks, use_enriched=False)
       results_raw = rag_raw.retrieve('kubernetes pod', k=5)
       assert len(results_raw) == 5
       print(f'OK: Raw mode works, top score: {results_raw[0][\"score\"]:.3f}')
       
       # Test enriched mode
       rag_enriched = BaselineLangChainRAG()
       rag_enriched.ingest_chunks(chunks, use_enriched=True)
       results_enriched = rag_enriched.retrieve('kubernetes pod', k=5)
       assert len(results_enriched) == 5
       print(f'OK: Enriched mode works, top score: {results_enriched[0][\"score\"]:.3f}')
       "
  Expected Result: Both modes work, scores are valid floats
  Evidence: stdout captured
```

```
Scenario: Baseline uses normalized embeddings (cosine similarity)
  Tool: Bash
  Steps:
    1. grep -n "normalize" poc/plm_vs_rag_benchmark/baseline_langchain.py
  Expected Result: normalize_embeddings=True or normalize_L2 found
  Evidence: grep output showing normalization
```

**Commit**: YES (groups with 1, 2)
- Message: `feat(benchmark): add LangChain baseline RAG with raw/enriched modes`
- Files: `poc/plm_vs_rag_benchmark/baseline_langchain.py`

---

### Task 3: Create Unified Benchmark Runner

**What to do**:
- Create `poc/plm_vs_rag_benchmark/benchmark_runner.py`
- Implement unified benchmark that:
  - Creates fresh databases for both systems using `tempfile.TemporaryDirectory()`
  - Ingests same corpus into both
  - Runs same queries through both
  - Calculates metrics using existing `RetrievalMetrics` class
  - Optionally runs LLM grading using existing `RetrievalGrader`
  - Saves results to JSON and generates markdown report
- CLI interface: `python benchmark_runner.py --questions needle --no-llm-grade` (fast)
- CLI interface: `python benchmark_runner.py --questions all --llm-grade` (full)

**Must NOT do**:
- Do not touch production database
- Do not hardcode paths - use Path objects relative to script location

**Recommended Agent Profile**:
- **Category**: `unspecified-low`
- **Skills**: []
- **Reason**: Integration work combining existing components

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Task 2)
- **Blocks**: Tasks 6, 8
- **Blocked By**: Task 1

**References**:
- `poc/modular_retrieval_pipeline/benchmark.py` — Existing benchmark structure (copy patterns)
- `poc/modular_retrieval_pipeline/utils/metrics.py` — `RetrievalMetrics` class
- `poc/modular_retrieval_pipeline/components/retrieval_grader.py` — `RetrievalGrader` class
- `poc/chunking_benchmark_v2/corpus/needle_questions.json` — Question format

**Acceptance Criteria**:

```
Scenario: Benchmark runner has correct CLI interface
  Tool: Bash
  Steps:
    1. cd poc/plm_vs_rag_benchmark
    2. python benchmark_runner.py --help
    3. Assert: --questions argument documented
    4. Assert: --llm-grade flag documented
    5. Assert: --output argument documented
  Expected Result: Help text shows all expected arguments
  Evidence: Help output captured
```

```
Scenario: Benchmark runner imports correctly
  Tool: Bash
  Steps:
    1. cd poc/plm_vs_rag_benchmark
    2. python -c "from benchmark_runner import run_benchmark; print('OK: imports work')"
  Expected Result: "OK: imports work" printed
  Evidence: stdout captured
```

**Commit**: YES (groups with 1, 2)
- Message: `feat(benchmark): add unified PLM vs RAG benchmark runner`
- Files: `poc/plm_vs_rag_benchmark/benchmark_runner.py`

---

### Task 4: Copy Existing PLM Database for Benchmark

**What to do**:
- **OPTIMIZATION**: Copy existing production database instead of re-ingesting 1,569 docs
- Locate existing PLM database (check common paths):
  - Docker volume: `docker volume inspect docker_plm-search-index`
  - Local dev: Check for `index.db` or `plm.db` in project
  - MCP service: Check running `plm-search` MCP for its INDEX_PATH
- Copy database files to temp directory:
  - SQLite database (`*.db`)
  - BM25 index files (`*.bm25s` or similar)
- Initialize `HybridRetriever` pointing to the copy
- Verify document count matches expected (~1,569 docs)

**Why this approach**:
- Saves ~10-15 minutes of ingestion time
- Uses exact same indexed data as production
- Still keeps original database untouched (working on copy)

**Must NOT do**:
- Do not modify the original database files
- Do not use original paths directly (always copy first)

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: []
- **Reason**: Simple file copy and verification

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 2 (with Task 5)
- **Blocks**: Task 6
- **Blocked By**: Task 1

**References**:
- `src/plm/search/retriever.py` — `HybridRetriever` class
- `src/plm/search/storage/sqlite.py` — SQLiteStorage with `db_path` parameter
- `docker/docker-compose.search.yml` — Docker volume location

**Acceptance Criteria**:

```
Scenario: Find existing PLM database
  Tool: Bash
  Steps:
    1. # Check Docker volume
    2. docker volume inspect docker_plm-search-index 2>/dev/null && echo "DOCKER_VOLUME_EXISTS"
    3. # Check common local paths
    4. find . -name "*.db" -path "*/index/*" 2>/dev/null | head -5
    5. # Check MCP plm-search status
    6. curl -s http://localhost:8000/status 2>/dev/null || echo "MCP_NOT_RUNNING"
  Expected Result: At least one database location found
  Evidence: Path to existing database captured
```

```
Scenario: Copy database to temp directory
  Tool: Bash
  Steps:
    1. # Assuming database found at $DB_PATH
    2. mkdir -p /tmp/plm-benchmark-copy
    3. cp -r $DB_PATH/* /tmp/plm-benchmark-copy/
    4. ls -la /tmp/plm-benchmark-copy/
  Expected Result: Database files copied successfully
  Evidence: Directory listing shows .db and index files
```

```
Scenario: PLM loads from copied database
  Tool: Bash
  Steps:
    1. python -c "
       import sys
       sys.path.insert(0, 'src')
       from plm.search.retriever import HybridRetriever
       
       retriever = HybridRetriever(
           db_path='/tmp/plm-benchmark-copy/index.db',
           bm25_index_path='/tmp/plm-benchmark-copy'
       )
       # Verify documents loaded
       status = retriever.get_status()
       print(f'Documents: {status.get(\"total_documents\", \"unknown\")}')
       print(f'Chunks: {status.get(\"total_chunks\", \"unknown\")}')
       assert status.get('total_documents', 0) > 1000, 'Expected 1000+ docs'
       print('OK: PLM loaded from copy')
       "
  Expected Result: "OK: PLM loaded from copy" with ~1,569 documents
  Evidence: Document/chunk count printed

Scenario: Original database untouched
  Tool: Bash
  Steps:
    1. # Record mtime before benchmark
    2. stat -c %Y $ORIGINAL_DB_PATH > /tmp/db_mtime_before
    3. # ... run benchmark ...
    4. stat -c %Y $ORIGINAL_DB_PATH > /tmp/db_mtime_after
    5. diff /tmp/db_mtime_before /tmp/db_mtime_after
  Expected Result: mtimes identical (no modification)
  Evidence: diff shows no changes
```

**Commit**: NO (code is in Task 3)

---

### Task 5: Load Extracted Chunks into Both Baseline Variants

**What to do**:
- Load chunks from `chunks.json` (extracted in Task 2)
- Create TWO baseline instances:
  - **Baseline-Naive**: `ingest_chunks(chunks, use_enriched=False)` — embeds `content`
  - **Baseline-Enriched**: `ingest_chunks(chunks, use_enriched=True)` — embeds `enriched_content`
- Both use identical chunk boundaries (from PLM's SQLite)
- Both use same embedding model (BGE-base-en-v1.5) with normalization

**Why this approach**:
- Fair comparison: All 3 variants use identical chunks
- Oracle recommendation: Isolates retrieval algorithm from preprocessing

**Must NOT do**:
- Do not re-chunk or re-process documents
- Do not use different embedding model

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: []
- **Reason**: Simple loading using LangChain

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 2 (after chunk extraction)
- **Blocks**: Task 6
- **Blocked By**: Tasks 2, 3 (needs chunk extractor and baseline implementation)

**References**:
- `poc/plm_vs_rag_benchmark/chunks.json` — Extracted chunks from Task 2
- `poc/plm_vs_rag_benchmark/baseline_langchain.py` — Baseline implementation from Task 3

**Acceptance Criteria**:

```
Scenario: Both baseline variants load same chunk count
  Tool: Bash
  Steps:
    1. python -c "
       import json
       from baseline_langchain import BaselineLangChainRAG
       
       with open('chunks.json') as f:
           chunks = json.load(f)
       
       # Load raw baseline
       rag_raw = BaselineLangChainRAG()
       rag_raw.ingest_chunks(chunks, use_enriched=False)
       
       # Load enriched baseline
       rag_enriched = BaselineLangChainRAG()
       rag_enriched.ingest_chunks(chunks, use_enriched=True)
       
       print(f'Raw baseline: {len(chunks)} chunks loaded')
       print(f'Enriched baseline: {len(chunks)} chunks loaded')
       print('OK: Both baselines loaded identical chunks')
       "
  Expected Result: Both baselines load same number of chunks
  Evidence: stdout captured
```

**Commit**: NO (code is in Task 3)

---

### Task 6: Run Needle Benchmark (Validation) — All 3 Variants

**What to do**:
- Run benchmark with `--questions needle --no-llm-grade` (fast validation)
- Uses 20 needle questions from `needle_questions.json`
- Run ALL 3 VARIANTS:
  - PLM Full (copy of production DB)
  - Baseline-Enriched (LangChain + enriched chunks)
  - Baseline-Naive (LangChain + raw chunks)
- Calculate MRR, Hit@1, Hit@5, Recall@5 for all 3 systems
- Verify pipeline works before expensive realistic benchmark
- Save results to `results/needle_benchmark.json`

**Must NOT do**:
- Do not run LLM grading yet (save API costs for full benchmark)

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: []
- **Reason**: Running existing benchmark script

**Parallelization**:
- **Can Run In Parallel**: NO
- **Sequential**: Must complete before Task 7
- **Blocks**: Task 7
- **Blocked By**: Tasks 4, 5

**References**:
- `poc/chunking_benchmark_v2/corpus/needle_questions.json` — 20 questions
- `poc/modular_retrieval_pipeline/utils/metrics.py` — Metric calculations

**Acceptance Criteria**:

```
Scenario: Needle benchmark produces results for all 3 variants
  Tool: Bash
  Preconditions: All ingestion complete
  Steps:
    1. cd poc/plm_vs_rag_benchmark
    2. python benchmark_runner.py --questions needle --no-llm-grade --output results/needle_benchmark.json
    3. python -c "
       import json
       with open('results/needle_benchmark.json') as f:
           r = json.load(f)
       # Verify all 3 variants have results
       assert 'plm' in r, 'Missing PLM results'
       assert 'baseline_enriched' in r, 'Missing Baseline-Enriched results'
       assert 'baseline_naive' in r, 'Missing Baseline-Naive results'
       # Verify metrics exist for all
       for variant in ['plm', 'baseline_enriched', 'baseline_naive']:
           assert 'mrr' in r[variant], f'Missing MRR for {variant}'
           assert 'hit_at_1' in r[variant], f'Missing Hit@1 for {variant}'
           assert 'hit_at_5' in r[variant], f'Missing Hit@5 for {variant}'
           assert 0 <= r[variant]['mrr'] <= 1, f'Invalid MRR for {variant}'
       print('All 3 variants have valid results')
       "
  Expected Result: All 3 variants have valid MRR, Hit@k metrics
  Evidence: results/needle_benchmark.json saved
```

```
Scenario: Results show expected ranking (PLM >= Enriched >= Naive)
  Tool: Bash
  Steps:
    1. python -c "
       import json
       with open('results/needle_benchmark.json') as f:
           r = json.load(f)
       
       plm_mrr = r['plm']['mrr']
       enriched_mrr = r['baseline_enriched']['mrr']
       naive_mrr = r['baseline_naive']['mrr']
       
       print(f'PLM MRR:              {plm_mrr:.3f}')
       print(f'Baseline-Enriched MRR: {enriched_mrr:.3f}')
       print(f'Baseline-Naive MRR:    {naive_mrr:.3f}')
       print()
       print(f'Enrichment contribution: {enriched_mrr - naive_mrr:+.3f}')
       print(f'RRF/BM25/expansion:      {plm_mrr - enriched_mrr:+.3f}')
       print(f'Total PLM improvement:   {plm_mrr - naive_mrr:+.3f}')
       "
  Expected Result: PLM >= Baseline-Enriched >= Baseline-Naive (usually)
  Evidence: Attribution breakdown printed
       assert 0 <= r['baseline']['mrr'] <= 1
       print(f'PLM MRR: {r[\"plm\"][\"mrr\"]:.3f}')
       print(f'Baseline MRR: {r[\"baseline\"][\"mrr\"]:.3f}')
       print('OK: Needle benchmark valid')
       "
  Expected Result: Both systems have valid MRR, Hit@k metrics
  Evidence: results/needle_benchmark.json saved, metrics printed
```

```
Scenario: PLM expected to outperform baseline
  Tool: Bash
  Steps:
    1. python -c "
       import json
       with open('results/needle_benchmark.json') as f:
           r = json.load(f)
       plm_mrr = r['plm']['mrr']
       baseline_mrr = r['baseline']['mrr']
       delta = plm_mrr - baseline_mrr
       print(f'PLM MRR: {plm_mrr:.3f}')
       print(f'Baseline MRR: {baseline_mrr:.3f}')
       print(f'Delta: {delta:+.3f}')
       if delta > 0:
           print('PASS: PLM outperforms baseline')
       else:
           print('WARN: PLM does not outperform baseline (investigate)')
       "
  Expected Result: PLM MRR > Baseline MRR (expected delta: +0.1 to +0.3)
  Evidence: Comparison output captured
```

**Commit**: NO (results are artifacts, not code)

---

### Task 7: Validate Metrics and Sanity Check

**What to do**:
- Review needle benchmark results
- Verify PLM outperforms baseline (expected: MRR +10-30%)
- Check for anomalies (e.g., both systems scoring 0)
- If validation fails, investigate before proceeding to expensive realistic benchmark

**Must NOT do**:
- Do not proceed to realistic benchmark if needle benchmark shows anomalies

**Recommended Agent Profile**:
- **Category**: `quick`
- **Skills**: []
- **Reason**: Review and validation

**Parallelization**:
- **Can Run In Parallel**: NO
- **Sequential**: Gate before expensive benchmark
- **Blocks**: Task 8
- **Blocked By**: Task 6

**References**:
- `results/needle_benchmark.json` — Results from Task 6

**Acceptance Criteria**:

```
Scenario: Sanity check passes
  Tool: Bash
  Steps:
    1. python -c "
       import json
       with open('results/needle_benchmark.json') as f:
           r = json.load(f)
       
       # Check for anomalies
       anomalies = []
       
       # Both systems should find at least some results
       if r['plm']['hit_at_5'] < 10:
           anomalies.append('PLM Hit@5 suspiciously low')
       if r['baseline']['hit_at_5'] < 5:
           anomalies.append('Baseline Hit@5 suspiciously low')
       
       # MRR should be positive
       if r['plm']['mrr'] == 0:
           anomalies.append('PLM MRR is 0 (broken)')
       if r['baseline']['mrr'] == 0:
           anomalies.append('Baseline MRR is 0 (broken)')
       
       if anomalies:
           print('ANOMALIES DETECTED:')
           for a in anomalies:
               print(f'  - {a}')
           print('INVESTIGATE BEFORE PROCEEDING')
           exit(1)
       else:
           print('OK: Sanity check passed')
       "
  Expected Result: "OK: Sanity check passed"
  Evidence: stdout captured
```

**Commit**: NO

---

### Task 8: Run Realistic Benchmark (Full) — All 3 Variants with Bootstrap CI

**What to do**:
- Run benchmark with `--questions realistic --llm-grade` (full benchmark)
- Uses 400 realistic questions (vocabulary mismatch test)
- Run ALL 3 VARIANTS with LLM grading:
  - PLM Full
  - Baseline-Enriched
  - Baseline-Naive
- Calculate MRR, Hit@1, Hit@5, Recall@5 for all 3 systems
- Calculate **bootstrap 95% confidence intervals** on MRR differences
- Log query expansion triggers (how often PLM's expansion fires)
- Expect ~$9-15 API cost (400 queries × 3 variants × grading)
- Save results to `results/realistic_benchmark.json`
- Generate per-question breakdown and attribution analysis

**Must NOT do**:
- Do not skip LLM grading (key for answer quality comparison)
- Do not skip confidence intervals (Oracle recommendation)

**Recommended Agent Profile**:
- **Category**: `unspecified-low`
- **Skills**: []
- **Reason**: Extended benchmark run, moderate complexity

**Parallelization**:
- **Can Run In Parallel**: NO
- **Sequential**: After validation
- **Blocks**: Task 9
- **Blocked By**: Task 7

**References**:
- `poc/chunking_benchmark_v2/corpus/kubernetes/realistic_questions.json` — 400 questions
- `poc/modular_retrieval_pipeline/components/retrieval_grader.py` — LLM grading
- Bootstrap CI: Use `numpy.random.choice` with 1000 samples

**Acceptance Criteria**:

```
Scenario: Realistic benchmark completes for all 3 variants
  Tool: Bash
  Preconditions: Needle benchmark passed sanity check
  Steps:
    1. cd poc/plm_vs_rag_benchmark
    2. time python benchmark_runner.py --questions realistic --llm-grade --output results/realistic_benchmark.json
    3. python -c "
       import json
       with open('results/realistic_benchmark.json') as f:
           r = json.load(f)
       
       # Verify all 3 variants have results
       for variant in ['plm', 'baseline_enriched', 'baseline_naive']:
           assert r[variant]['total_queries'] == 400, f'{variant} missing queries'
           assert 'mrr' in r[variant]
           assert 'avg_llm_grade' in r[variant]
       
       print('=== Realistic Benchmark Results (400 queries) ===')
       print()
       for variant in ['plm', 'baseline_enriched', 'baseline_naive']:
           print(f'{variant}:')
           print(f'  MRR: {r[variant][\"mrr\"]:.3f}')
           print(f'  Hit@5: {r[variant][\"hit_at_5\"]:.1f}%')
           print(f'  Avg Grade: {r[variant][\"avg_llm_grade\"]:.1f}/10')
           print()
       print('OK: All 3 variants benchmarked')
       "
  Expected Result: 400 queries × 3 variants processed
  Evidence: results/realistic_benchmark.json saved
```

```
Scenario: Bootstrap confidence intervals calculated
  Tool: Bash
  Steps:
    1. python -c "
       import json
       with open('results/realistic_benchmark.json') as f:
           r = json.load(f)
       
       # Check CI fields exist
       assert 'mrr_ci_95' in r.get('statistics', {}), 'Missing bootstrap CI'
       ci = r['statistics']['mrr_ci_95']
       
       print('=== 95% Confidence Intervals on MRR Differences ===')
       print(f'PLM vs Naive:    {ci[\"plm_vs_naive\"][0]:.3f} to {ci[\"plm_vs_naive\"][1]:.3f}')
       print(f'PLM vs Enriched: {ci[\"plm_vs_enriched\"][0]:.3f} to {ci[\"plm_vs_enriched\"][1]:.3f}')
       print(f'Enriched vs Naive: {ci[\"enriched_vs_naive\"][0]:.3f} to {ci[\"enriched_vs_naive\"][1]:.3f}')
       "
  Expected Result: CI bounds for all pairwise comparisons
  Evidence: CI values printed
```

```
Scenario: Attribution analysis shows feature contributions
  Tool: Bash
  Steps:
    1. python -c "
       import json
       with open('results/realistic_benchmark.json') as f:
           r = json.load(f)
       
       plm = r['plm']['mrr']
       enriched = r['baseline_enriched']['mrr']
       naive = r['baseline_naive']['mrr']
       
       total_improvement = plm - naive
       enrichment_contribution = enriched - naive
       rrf_bm25_contribution = plm - enriched
       
       print('=== Feature Attribution Analysis ===')
       print(f'Total PLM improvement over naive: {total_improvement:+.3f} MRR')
       print(f'  - Enrichment contribution:      {enrichment_contribution:+.3f} ({enrichment_contribution/total_improvement*100:.0f}%)')
       print(f'  - RRF/BM25/expansion:           {rrf_bm25_contribution:+.3f} ({rrf_bm25_contribution/total_improvement*100:.0f}%)')
       "
  Expected Result: Attribution breakdown with percentages
  Evidence: Feature contributions printed
       delta = plm_hit5 - baseline_hit5
       
       print(f'Hit@5 Comparison:')
       print(f'  PLM: {plm_hit5:.1f}%')
       print(f'  Baseline: {baseline_hit5:.1f}%')
       print(f'  Delta: {delta:+.1f}%')
       
       if delta > 5:
           print('STRONG: PLM significantly outperforms baseline')
       elif delta > 0:
           print('POSITIVE: PLM outperforms baseline')
       else:
           print('UNEXPECTED: Baseline matches or beats PLM (investigate)')
       "
  Expected Result: PLM Hit@5 > Baseline Hit@5 (expected delta: +5-20%)
  Evidence: Comparison output captured
```

**Commit**: NO (results are artifacts)

---

### Task 9: Generate Comparison Report

**What to do**:
- Create `results/BENCHMARK_REPORT.md` with comprehensive analysis
- Include:
  - Executive summary with headline metrics
  - Methodology section (corpus, questions, metrics)
  - Known differences (chunking, enrichment, BM25)
  - Results tables (needle + realistic)
  - Per-category breakdown (if available)
  - Statistical significance notes
  - Conclusions and recommendations
- Optionally copy to `docs/PLM_VS_RAG_BENCHMARK_RESULTS.md`

**Must NOT do**:
- Do not cherry-pick results — report both successes and failures
- Do not omit known differences section

**Recommended Agent Profile**:
- **Category**: `writing`
- **Skills**: []
- **Reason**: Technical writing and analysis

**Parallelization**:
- **Can Run In Parallel**: NO
- **Sequential**: Final task
- **Blocks**: None (final)
- **Blocked By**: Task 8

**References**:
- `poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md` — Example report format
- `docs/RESEARCH.md` — Project research documentation style

**Acceptance Criteria**:

```
Scenario: Report contains required sections
  Tool: Bash
  Steps:
    1. cat poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md
    2. grep -c "## Executive Summary" poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md
    3. grep -c "## Methodology" poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md
    4. grep -c "## Known Differences" poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md
    5. grep -c "## Results" poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md
    6. grep -c "## Conclusions" poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md
  Expected Result: All required sections present (grep returns 1 for each)
  Evidence: Section headers found
```

```
Scenario: Report includes numeric results
  Tool: Bash
  Steps:
    1. grep -E "MRR|Hit@|NDCG" poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md | head -20
  Expected Result: Metric names and values present
  Evidence: grep output showing metrics
```

**Commit**: YES
- Message: `docs(benchmark): add PLM vs RAG benchmark results and analysis`
- Files: `poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1, 2, 3 | `feat(benchmark): add PLM vs RAG benchmark POC` | poc/plm_vs_rag_benchmark/ | uv sync succeeds |
| 9 | `docs(benchmark): add PLM vs RAG benchmark results` | results/BENCHMARK_REPORT.md | Report exists |

---

## Success Criteria

### Verification Commands
```bash
# 1. POC structure exists
ls -la poc/plm_vs_rag_benchmark/
# Expected: pyproject.toml, baseline_langchain.py, benchmark_runner.py, results/, corpus/

# 2. Benchmark ran successfully
cat poc/plm_vs_rag_benchmark/results/realistic_benchmark.json | python -m json.tool | head -30
# Expected: Valid JSON with plm and baseline sections

# 3. PLM outperforms baseline
python -c "
import json
with open('poc/plm_vs_rag_benchmark/results/realistic_benchmark.json') as f:
    r = json.load(f)
print(f'PLM wins by {r[\"plm\"][\"hit_at_5\"] - r[\"baseline\"][\"hit_at_5\"]:.1f}% on Hit@5')
"

# 4. Report generated
wc -l poc/plm_vs_rag_benchmark/results/BENCHMARK_REPORT.md
# Expected: >= 100 lines
```

### Final Checklist
- [ ] Production database untouched
- [ ] Both systems ingested same ~1,569 documents
- [ ] 20 needle questions benchmarked
- [ ] 400 realistic questions benchmarked
- [ ] MRR, Hit@k metrics calculated for both
- [ ] LLM grades calculated for both
- [ ] Report documents methodology and known differences
- [ ] PLM demonstrates measurable improvement over baseline

---

## Appendix: Expected Results (Hypothesis)

Based on prior benchmarks and research:

### Three-Way Comparison (Needle Questions - 20 queries)

| Metric | PLM Full | Baseline-Enriched | Baseline-Naive |
|--------|----------|-------------------|----------------|
| Hit@5 | ~90% | ~80% | ~70% |
| MRR | ~0.85 | ~0.70 | ~0.60 |

### Three-Way Comparison (Realistic Questions - 400 queries)

| Metric | PLM Full | Baseline-Enriched | Baseline-Naive |
|--------|----------|-------------------|----------------|
| Hit@5 | ~41% | ~32% | ~25% |
| MRR | ~0.27 | ~0.20 | ~0.15 |
| Avg LLM Grade | ~7.5/10 | ~6.8/10 | ~6.0/10 |

### Expected Attribution Analysis

| Contribution | Realistic MRR Delta | % of Total |
|--------------|---------------------|------------|
| **Total PLM improvement** | +0.12 | 100% |
| Enrichment alone | +0.05 | ~40% |
| RRF + BM25 + expansion | +0.07 | ~60% |

**Hypothesis**: RRF/BM25/expansion contributes MORE than enrichment because:
1. BM25 catches exact keyword matches that pure semantic misses
2. Query expansion helps with vocabulary mismatch
3. RRF fusion balances lexical and semantic signals

**PLM advantages that should show**:
1. BM25 + semantic hybrid catches exact keyword matches baseline misses
2. Query expansion adds domain synonyms
3. RRF fusion balances lexical and semantic signals
4. Content enrichment improves embedding quality

**Known baseline limitations**:
1. Pure semantic retrieval fails on exact keyword queries
2. No query expansion = vocabulary mismatch failures
3. All variants use identical chunks (no chunking difference)
