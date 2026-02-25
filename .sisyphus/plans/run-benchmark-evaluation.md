# Run Benchmark Evaluation

## TL;DR

> **Quick Summary**: Verify K8s corpus exists, create isolated test database, build ground truth from SOTorrent, run benchmark evaluation on current PLM implementation.
>
> **Deliverables**:
> - Verified K8s documentation corpus
> - Isolated test database copy
> - Generated GOLD/SILVER/BRONZE benchmark datasets
> - Benchmark results (Hit@k, MRR, NDCG)
>
> **Estimated Effort**: Medium
> **Parallel Execution**: NO - sequential pipeline
> **Critical Path**: Verify corpus → Copy DB → Get SOTorrent → Build GT → Run eval

---

## TODOs

- [x] 1. Verify K8s Documentation Corpus

  **What to do**:
  - Find the production index database (likely `INDEX_PATH` env var or `./search-index/index.db`)
  - Query the database to check what documents exist
  - Verify kubernetes.io/docs content is present
  - Count documents and chunks to ensure reasonable corpus size
  
  **Verification Commands**:
  ```bash
  # Find database
  find . -name "index.db" -o -name "*.db" 2>/dev/null
  
  # Check corpus content (adjust path as needed)
  sqlite3 <db_path> "SELECT COUNT(*) FROM documents;"
  sqlite3 <db_path> "SELECT COUNT(*) FROM chunks;"
  sqlite3 <db_path> "SELECT DISTINCT source_file FROM documents LIMIT 20;"
  ```
  
  **If corpus missing or empty**:
  - Need to ingest K8s docs first (out of scope for this plan)
  - Stop and report gap to user
  
  **Acceptance Criteria**:
  - [ ] Database file located
  - [ ] Documents table has >100 documents
  - [ ] Chunks table has >1000 chunks
  - [ ] Source files include kubernetes.io paths or k8s markdown files

  **Commit**: NO

---

- [x] 2. Create Isolated Test Database

  **What to do**:
  - Copy production database to `artifacts/benchmark/test_index.db`
  - Verify copy is readable and complete
  - Set environment variable for benchmark commands to use test DB
  
  **Commands**:
  ```bash
  mkdir -p artifacts/benchmark
  cp <production_db_path> artifacts/benchmark/test_index.db
  
  # Verify copy
  sqlite3 artifacts/benchmark/test_index.db "SELECT COUNT(*) FROM chunks;"
  ```
  
  **Why isolated copy**:
  - Benchmark operations are read-only, but isolation is safer
  - Allows running tests without affecting production search
  - Can be deleted after evaluation
  
  **Acceptance Criteria**:
  - [ ] Copy created at `artifacts/benchmark/test_index.db`
  - [ ] Row counts match production

  **Commit**: NO (artifact, not code)

---

- [x] 3. Acquire SOTorrent Data

  **What to do**:
  - Check if SOTorrent data already exists locally
  - If not, download from Zenodo: https://zenodo.org/record/4415593
  - Extract relevant tables (Posts, PostVersionUrl)
  - Note: Full dataset is ~50GB, may need subset
  
  **Options**:
  
  A) **Full SOTorrent** (recommended for complete benchmark):
  ```bash
  # Download and extract (large, ~50GB)
  wget https://zenodo.org/record/4415593/files/sotorrent-2020-12-31.7z
  7z x sotorrent-2020-12-31.7z
  ```
  
  B) **Pre-filtered subset** (if available):
  - Check if K8s-filtered extract already exists
  - Much smaller, faster to process
  
  C) **Mock/sample data** (for testing pipeline only):
  - Create small sample JSON with known K8s Q&A pairs
  - Useful for validating pipeline before full run
  
  **Acceptance Criteria**:
  - [ ] SOTorrent data accessible (SQLite or extracted files)
  - [ ] Can query for K8s questions with doc links

  **Commit**: NO (data acquisition)

---

- [x] 4. Extract SO Data for K8s

  **What to do**:
  - Run SO extractor to pull K8s questions with doc links
  - Filter for answer score >= 5
  - Output to `artifacts/benchmark/raw/so_k8s_answers.json`
  
  **Command**:
  ```bash
  plm-benchmark extract-so \
    --db <sotorrent_db_path> \
    --output artifacts/benchmark/raw/so_k8s_answers.json
  ```
  
  **Expected output**:
  - JSON file with 5000+ Q&A pairs
  - Each entry has: question_id, question_title, answer_body, doc_url, etc.
  
  **Acceptance Criteria**:
  - [ ] Output file created
  - [ ] Contains 1000+ entries (minimum viable)
  - [ ] All entries have kubernetes.io/docs URLs

  **Commit**: NO (artifact)

---

- [x] 5. Build URL-to-Chunk Mappings

  **What to do**:
  - Generate mappings from K8s doc URLs to chunk IDs in our corpus
  - Use the test database copy
  
  **Command**:
  ```bash
  # This may be part of signal extraction or separate
  # Check benchmark CLI for mapping command
  plm-benchmark build-mappings \
    --corpus-db artifacts/benchmark/test_index.db \
    --output artifacts/benchmark/mappings/
  ```
  
  **Output files**:
  - `url_to_chunks.json`
  - `anchor_to_heading.json`
  - `unmapped_urls.log` (for gap analysis)
  
  **Acceptance Criteria**:
  - [ ] Mapping files created
  - [ ] >50% of SO URLs map to corpus chunks
  - [ ] Unmapped URLs logged for review

  **Commit**: NO (artifact)

---

- [x] 6. Extract Signals and Generate Benchmark Cases

  **What to do**:
  - Run full signal extraction → generation → verification → assembly pipeline
  - This produces GOLD/SILVER/BRONZE datasets
  
  **Commands**:
  ```bash
  # Step 1: Extract signals
  plm-benchmark extract-signals \
    --so-data artifacts/benchmark/raw/so_k8s_answers.json \
    --mappings artifacts/benchmark/mappings/ \
    --corpus-db artifacts/benchmark/test_index.db \
    --output artifacts/benchmark/signals/
  
  # Step 2: Generate cases (uses LLM)
  plm-benchmark generate \
    --signals artifacts/benchmark/signals/signal_bundles.jsonl \
    --output artifacts/benchmark/generated/
  
  # Step 3: Verify (code-based, deterministic)
  plm-benchmark verify \
    --generated artifacts/benchmark/generated/ \
    --corpus-db artifacts/benchmark/test_index.db \
    --output artifacts/benchmark/verified/
  
  # Step 4: Assemble final datasets
  plm-benchmark assemble \
    --verified artifacts/benchmark/verified/verified_passed.jsonl \
    --regenerated artifacts/benchmark/verified/regenerated_passed.jsonl \
    --signals artifacts/benchmark/signals/signal_bundles.jsonl \
    --output artifacts/benchmark/datasets/
  ```
  
  **Expected output**:
  - `artifacts/benchmark/datasets/gold.json` (≥400 cases)
  - `artifacts/benchmark/datasets/silver.json` (≥1500 cases)
  - `artifacts/benchmark/datasets/bronze.json` (≥5000 cases)
  - `artifacts/benchmark/statistics_report.json`
  
  **Note**: Generation step requires LLM (PLM_LLM_MODEL env var)
  
  **Acceptance Criteria**:
  - [ ] All dataset files created
  - [ ] GOLD has ≥100 cases (may be lower than target initially)
  - [ ] Quarantine rate <5%
  - [ ] Statistics report shows reasonable pass rates

  **Commit**: NO (artifacts)

---

- [x] 7. Run Benchmark Evaluation

  **What to do**:
  - Start search service pointing to test database
  - Run benchmark against each tier
  - Collect and analyze results
  
  **Commands**:
  ```bash
  # Terminal 1: Start search service with test DB
  INDEX_PATH=artifacts/benchmark/test_index.db \
  uvicorn plm.search.service.app:app --port 8000
  
  # Terminal 2: Run benchmarks
  plm-benchmark evaluate \
    --dataset artifacts/benchmark/datasets/gold.json \
    --k 10 \
    --url http://localhost:8000 \
    --output artifacts/benchmark/results/gold_results.json
  
  plm-benchmark evaluate \
    --dataset artifacts/benchmark/datasets/silver.json \
    --k 10 \
    --url http://localhost:8000 \
    --output artifacts/benchmark/results/silver_results.json
  ```
  
  **Expected metrics**:
  | Metric | Target (GOLD) | Target (SILVER) |
  |--------|---------------|-----------------|
  | Hit@5 | >60% | >50% |
  | Hit@10 | >75% | >65% |
  | MRR | >0.40 | >0.35 |
  
  **Acceptance Criteria**:
  - [ ] Results JSON files created
  - [ ] Metrics are within reasonable range (not 0% or 100%)
  - [ ] Per-query traces available for debugging

  **Commit**: NO (results are artifacts)

---

- [x] 8. Run Integration Analysis (Optional)

  **What to do**:
  - Analyze component contributions
  - Understand BM25 vs semantic retriever performance
  
  **Command**:
  ```bash
  plm-benchmark-integration analyze-integration \
    --dataset artifacts/benchmark/datasets/gold.json \
    --analysis all \
    --url http://localhost:8000 \
    --output artifacts/benchmark/results/integration_report.json
  ```
  
  **Provides**:
  - Complementarity analysis (do retrievers find different docs?)
  - Ablation results (what does each component contribute?)
  - Recommendations for tuning
  
  **Acceptance Criteria**:
  - [ ] Integration report generated
  - [ ] Error correlation <40% (good fusion potential)

  **Commit**: NO (analysis artifact)

---

## Summary

After completing all steps, you'll have:

1. **Verified corpus** - Confirmed K8s docs are indexed
2. **Isolated test environment** - Safe to experiment
3. **Ground truth datasets** - GOLD/SILVER/BRONZE with audit trails
4. **Benchmark results** - Hit@k, MRR, NDCG on real user questions
5. **Integration insights** - Component-level analysis

**To execute**: Run `/start-work` to begin.
