# PLM Benchmark Framework

## TL;DR

> **Quick Summary**: Build an externally-validated benchmark framework using StackOverflow data to evaluate PLM retrieval quality. Uses a multi-agent pipeline with deterministic code verification (not LLM reviewers) to ensure data quality. The benchmark calls the production search service via HTTP API, with full traceability.
>
> **Deliverables**:
> - Enhanced search service API with `explain=True` mode returning score breakdowns
> - Multi-agent dataset generation pipeline with:
>   - Deterministic signal extraction (code, parallel)
>   - Parallel LLM generators for query/quote formulation
>   - Code-based verifier (no LLM, deterministic)
>   - Regeneration loop with structured feedback
>   - Full audit metadata on every case
> - ~500 GOLD + 2000 SILVER + unlimited BRONZE benchmark cases
> - Benchmark runner that consumes the API and calculates Hit@k, MRR, NDCG metrics
> - Integration evaluation suite (complementarity, cascade, ablation analysis)
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES - 5 waves (including sub-waves for 6a-6e)
> **Critical Path**: Phase 0 (API) → Phase 1 (SO data) → Phase 2a-e (multi-agent generation) → Phase 3 (benchmark runner)
> **Trust Guarantee**: Verification is CODE, not LLM. Same input = same output. No reviewer variance.

---

## Context

### Original Request
Build a benchmark framework for evaluating PLM retrieval quality using external validation (StackOverflow data). The benchmark must generate Q&A pairs we can 100% trust without manual verification. Must be implemented as separate entity calling production search service via HTTP API.

### Interview Summary
**Key Discussions**:
- Oracle's "Evidence-Based Tier Assignment" methodology adopted for trustworthy data
- Include BRONZE tier (ambiguous cases) for higher volume
- All chunks from URL for BRONZE mapping
- Detailed confidence metadata for later filtering
- Skip spot-check initially, trust methodology
- Full traceability requirement: each question traceable through logs and API responses

**Research Findings**:
- Current search service returns only final score, no breakdown
- Need to add `explain=True` mode to API before benchmark can trace decisions
- Existing POC infrastructure has reusable metrics (Hit@k, MRR) and question schema
- SOTorrent dataset available at Zenodo with extracted URLs from SO posts

### Metis Review
**Identified Gaps** (addressed):
- URL → chunk_id mapping: Build from ingested docs, match URL path to source_file
- SOTorrent acquisition: Specify download, storage, SQL queries
- Edge cases: Multiple URLs per answer, redirects, versioned URLs
- Scope boundaries: No multi-domain, no LLM verification tier, no live SO sync

---

## Work Objectives

### Core Objective
Create a reproducible benchmark framework that evaluates PLM retrieval quality using externally-validated StackOverflow → Kubernetes documentation pairs, with full traceability of every benchmark result through the production API.

### Concrete Deliverables
- `src/plm/search/service/app.py` enhanced with `explain=True` parameter
- `src/plm/search/retriever.py` modified to track and return score breakdown
- `src/plm/benchmark/` module with data extraction, tier assignment, and evaluation
- `artifacts/benchmark/` directory with GOLD/SILVER/BRONZE JSON datasets
- CLI commands: `plm-benchmark extract`, `plm-benchmark generate`, `plm-benchmark evaluate`

### Definition of Done
- [ ] `curl /query?explain=true` returns `debug_info` with bm25_score, semantic_score, rrf_score
- [ ] `plm-benchmark generate --output artifacts/benchmark/` produces tier-labeled dataset
- [ ] `plm-benchmark evaluate --dataset artifacts/benchmark/gold.json` outputs Hit@5, MRR, NDCG
- [ ] Every benchmark result traceable via `request_id` in API response

### Must Have
- Benchmark calls production API via HTTP (no direct retriever instantiation)
- Evidence-based tier assignment (GOLD/SILVER/BRONZE with provable signals)
- Score breakdown in API response when explain=True
- Kubernetes documentation corpus only
- Detailed logging (TRACE/DEBUG/INFO/WARN/ERROR per question)

### Must NOT Have (Guardrails)
- NO direct HybridRetriever instantiation in benchmark code
- NO LLM-based relevance verification tier (deferred)
- NO multi-domain support (Python, AWS, etc.) in v1
- NO live StackOverflow API integration
- NO modification to production SQLite schema
- NO human spot-check workflow (trust methodology initially)

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks are verifiable by running commands or calling the API.

### Test Decision
- **Infrastructure exists**: YES (pytest in project)
- **Automated tests**: YES (tests after implementation)
- **Framework**: pytest

### Agent-Executed QA Scenarios (MANDATORY)

Every task includes verification scenarios executable by the agent.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Search service explain mode enhancement
└── Task 2: SOTorrent data acquisition and extraction

Wave 2 (After Wave 1):
├── Task 3: URL-to-chunk mapping builder
├── Task 4: Tier assignment engine
└── Task 5: Quote/fragment matching utilities

Wave 3a (After Wave 2):
└── Task 6a: Deterministic signal extraction

Wave 3b (After 6a):
└── Task 6b: Parallel LLM generator agents (spawns N parallel workers)

Wave 3c (After 6b):
└── Task 6c: Deterministic verifier (code-based quality gate)

Wave 3d (After 6c):
└── Task 6d: Regeneration loop with structured feedback

Wave 3e (After 6d):
└── Task 6e: Final assembly and audit metadata

Wave 4 (After Wave 3e + Task 1):
├── Task 7: Benchmark runner (API consumer)
└── Task 8: Integration evaluation suite

Wave 5 (After Wave 4):
└── Task 9: Tests and documentation
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 7, 8 | 2 |
| 2 | None | 3, 4, 5 | 1 |
| 3 | 2 | 6a | 4, 5 |
| 4 | 2 | 6a | 3, 5 |
| 5 | 2 | 6a | 3, 4 |
| 6a | 3, 4, 5 | 6b | None |
| 6b | 6a | 6c | None (internal parallelism) |
| 6c | 6b | 6d | None |
| 6d | 6c | 6e | None (internal parallelism) |
| 6e | 6d | 7 | None |
| 7 | 1, 6e | 8 | None |
| 8 | 7 | 9 | None |
| 9 | 8 | None | None |

### Multi-Agent Parallelism Details

**Task 6b (Generator Agents):**
- Spawns N parallel LLM agents (recommended: 4-8)
- Each agent processes batch of ~500 SignalBundles
- Total parallelism: N × batch_size concurrent

**Task 6d (Regeneration Loop):**
- Failed cases processed in parallel
- Fresh agent per retry to avoid stuck patterns
- Maximum 3 retries per case

---

## TODOs

### Phase 0: Search Service Enhancement

- [x] 1. Add Explain Mode to Search Service API

  **What to do**:
  - Add `explain: bool = False` parameter to `QueryRequest` model in `app.py`
  - Create `DebugInfo` Pydantic model (per-result debug info)
  - Create `QueryMetadata` Pydantic model (per-query metadata)
  - Add optional `debug_info` field to `QueryResult` model
  - Add optional `metadata` and `request_id` fields to `QueryResponse` model
  - Modify `HybridRetriever.retrieve()` to accept `explain` parameter
  - Track scores at each retrieval stage when explain=True
  - Add `X-Request-ID` middleware for request correlation

  **Schema Specifications**:

  DebugInfo (per result):
  | Field | Type | Description |
  |-------|------|-------------|
  | bm25_score | float or null | Raw BM25/SPLADE score. Null if sparse disabled. |
  | semantic_score | float or null | Cosine similarity. Null if semantic disabled. |
  | bm25_rank | int or null | 0-indexed position in sparse results. Null if not in top-N. |
  | semantic_rank | int or null | 0-indexed position by semantic score. Null if not in top-N. |
  | rrf_score | float | Final RRF fusion score. Always present. |
  | rerank_score | float or null | Cross-encoder score. Null if use_rerank=False. |
  | retrieval_stage | string | One of: "rrf", "rerank". Which stage produced final ranking. |

  QueryMetadata (per query):
  | Field | Type | Description |
  |-------|------|-------------|
  | original_query | string | The raw query text |
  | rewritten_query | string or null | LLM-rewritten query. Null if use_rewrite=False. |
  | expanded_terms | list of strings | Terms added by QueryExpander. Empty list if none. |
  | retrieval_mode | string | One of: "hybrid", "splade_only". Based on semantic.enabled config. |
  | rrf_k | int | RRF k parameter used (60 default, 10 if expanded) |
  | bm25_weight | float | Weight applied to sparse scores |
  | semantic_weight | float | Weight applied to semantic scores |

  **Score Collection Points** (where to capture scores):
  - BM25/SPLADE scores: From `sparse_results` list after `sparse_retriever.search()` call
  - Semantic scores: From `sem_scores` array after `np.dot(embeddings, query_emb)`
  - RRF scores: From `rrf_scores` dict after fusion loop
  - Rerank scores: From reranker output if use_rerank=True

  **Edge Case Handling**:
  - When `config.semantic.enabled=False`: Set semantic_score=null, semantic_rank=null, retrieval_mode="splade_only"
  - When `use_rerank=False`: Set rerank_score=null, retrieval_stage="rrf"
  - When `use_rerank=True`: Set retrieval_stage="rerank", populate rerank_score

  **Middleware Behavior**:
  - If client sends `X-Request-ID` header: use that value
  - If no header: generate UUID4
  - Always return `X-Request-ID` in response headers
  - Include `request_id` field in QueryResponse when explain=True

  **Return Structure from retrieve()**:
  - When explain=False: Return `list[dict]` as before (no breaking change)
  - When explain=True: Return `tuple[list[dict], ExplainData]` where ExplainData contains metadata and per-chunk debug_info

  **Must NOT do**:
  - Change default API behavior (explain=False by default)
  - Modify production SQLite schema
  - Add external dependencies

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Modifies core retrieval path, requires careful tracking of scores through pipeline
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 7, 8 (benchmark runner needs explain mode)
  - **Blocked By**: None

  **References**:
  - `src/plm/search/service/app.py` - FastAPI endpoints to modify
  - `src/plm/search/retriever.py:586-687` - RRF fusion where scores are computed
  - `src/plm/search/retriever.py:628` - sparse_results from sparse_retriever.search()
  - `src/plm/search/retriever.py:655-668` - sem_scores and rrf_scores calculation
  - `src/plm/search/retriever.py:630-646` - SPLADE-only mode (semantic disabled)

  **Acceptance Criteria**:

  **Tests (after implementation):**
  - [ ] Test file created: `tests/search/test_explain_mode.py`
  - [ ] Test: explain=False returns no debug_info field
  - [ ] Test: explain=True returns debug_info with all required fields
  - [ ] Test: request_id header flows through to response
  - [ ] `pytest tests/search/test_explain_mode.py` → PASS

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Explain mode returns score breakdown
    Tool: Bash (curl)
    Preconditions: Search service running on localhost:8000 with indexed docs
    Steps:
      1. curl -s -X POST http://localhost:8000/query \
           -H "Content-Type: application/json" \
           -d '{"query": "kubernetes pod", "k": 5, "explain": true}'
      2. Parse JSON response
      3. Assert: results[0].debug_info exists
      4. Assert: results[0].debug_info.bm25_score is number
      5. Assert: results[0].debug_info.semantic_score is number
      6. Assert: results[0].debug_info.rrf_score is number
      7. Assert: metadata.retrieval_mode exists
    Expected Result: Full score breakdown in response
    Evidence: Response body saved to .sisyphus/evidence/task-1-explain-mode.json

  Scenario: Explain mode disabled returns clean response
    Tool: Bash (curl)
    Preconditions: Search service running
    Steps:
      1. curl -s -X POST http://localhost:8000/query \
           -H "Content-Type: application/json" \
           -d '{"query": "kubernetes pod", "k": 5, "explain": false}'
      2. Assert: results[0].debug_info does not exist
      3. Assert: metadata does not exist
    Expected Result: No debug fields in production response
    Evidence: Response body saved

  Scenario: Request ID correlation
    Tool: Bash (curl)
    Preconditions: Search service running
    Steps:
      1. curl -s -X POST http://localhost:8000/query \
           -H "Content-Type: application/json" \
           -H "X-Request-ID: test-correlation-123" \
           -d '{"query": "test", "k": 3, "explain": true}'
      2. Assert: response contains request_id field
      3. Assert: request_id equals "test-correlation-123"
    Expected Result: Request ID flows through
    Evidence: Response headers and body captured
  ```

  **Commit**: YES
  - Message: `feat(search): add explain mode to query API for benchmark traceability`
  - Files: `src/plm/search/service/app.py`, `src/plm/search/retriever.py`
  - Pre-commit: `pytest tests/search/test_explain_mode.py`

---

### Phase 1: SO Data Extraction

- [x] 2. SOTorrent Data Acquisition and Extraction

  **What to do**:
  - Download SOTorrent dataset from Zenodo (https://zenodo.org/record/4415593)
  - Identify which tables/files contain the data needed (PostVersionUrl, Posts, Comments)
  - Write SQL or data processing script to extract Kubernetes-related questions
  - Filter for answers that link to kubernetes.io/docs URLs
  - Apply quality threshold: answer score >= 5
  - Extract fields: question_id, question_title, question_body, answer_id, answer_body, answer_score, is_accepted, doc_url
  - Store extracted data as JSON in `artifacts/benchmark/raw/so_k8s_answers.json`
  - Log statistics: total questions found, unique doc URLs, score distribution

  **Schema Specifications**:

  SOTorrent Tables Used:
  | Table | Purpose | Key Columns |
  |-------|---------|-------------|
  | Posts | Question and answer content | Id, ParentId, Title, Body, Score, PostTypeId, AcceptedAnswerId, Tags, CreationDate |
  | PostVersionUrl | Extracted URLs from posts | PostId, Url, PostBlockVersionId |

  Output File Schema (so_k8s_answers.json):
  | Field | Type | Description |
  |-------|------|-------------|
  | question_id | int | SO question post ID |
  | question_title | string | Question title (max 150 chars) |
  | question_body | string | Question HTML body |
  | question_tags | list[string] | Tags like ["kubernetes", "docker"] |
  | answer_id | int | SO answer post ID |
  | answer_body | string | Answer HTML body |
  | answer_score | int | Answer upvotes (≥5 after filtering) |
  | is_accepted | bool | Whether this is the accepted answer |
  | doc_url | string | The kubernetes.io/docs URL extracted |
  | answer_date | string | ISO8601 date of answer creation |

  **Kubernetes Identification Rules**:
  - Tag-based: `Tags LIKE '%kubernetes%'` OR `Tags LIKE '%k8s%'`
  - URL-based: `Url LIKE '%kubernetes.io/docs%'`
  - Both conditions must be met (K8s question WITH K8s doc link)

  **URL Extraction Rules**:
  - Extract from PostVersionUrl table (pre-extracted by SOTorrent)
  - If multiple K8s doc URLs in one answer, create separate entries
  - Normalize URLs: remove trailing slashes, preserve fragments (#anchor)
  - Handle both http:// and https:// prefixes

  **Quality Thresholds**:
  | Threshold | Value | Rationale |
  |-----------|-------|-----------|
  | Minimum answer score | ≥5 | Community validated |
  | Include accepted answers | YES | Higher trust signal |
  | Duplicate handling | Keep all | Different URLs = different mappings |

  **Database Setup** (if using SQLite for processing):
  - Download: `sotorrent-2020-12-*.7z` from Zenodo
  - Extract and load into SQLite: ~50GB uncompressed
  - Index: `CREATE INDEX idx_posts_parent ON Posts(ParentId)`
  - Index: `CREATE INDEX idx_urls_postid ON PostVersionUrl(PostId)`

  **Must NOT do**:
  - Store entire SOTorrent dataset in repo (too large)
  - Use live StackOverflow API
  - Include non-Kubernetes domains

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Data processing task requiring SQL/pandas, large dataset handling
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 3, 4, 5 (need SO data to process)
  - **Blocked By**: None

  **References**:
  - SOTorrent: https://zenodo.org/record/4415593 - Dataset download
  - SOTorrent schema documentation - Table structures
  - `artifacts/benchmark/raw/` - Output directory (create if needed)
  - Draft SQL in `.sisyphus/drafts/benchmark-framework.md` lines 76-96

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: SO data extraction produces valid output
    Tool: Bash (python/jq)
    Preconditions: SOTorrent data downloaded
    Steps:
      1. Run extraction script
      2. Check artifacts/benchmark/raw/so_k8s_answers.json exists
      3. jq 'length' artifacts/benchmark/raw/so_k8s_answers.json
      4. Assert: length >= 5000 (minimum viable dataset)
      5. jq '.[0] | keys' - verify required fields present
      6. Assert: fields include question_id, question_title, answer_score, doc_url
    Expected Result: JSON file with 5000+ Q&A pairs
    Evidence: File stats and sample records logged

  Scenario: Quality filtering applied correctly
    Tool: Bash (jq)
    Preconditions: Extraction complete
    Steps:
      1. jq '[.[] | select(.answer_score < 5)] | length' artifacts/benchmark/raw/so_k8s_answers.json
      2. Assert: result equals 0 (no low-score answers)
      3. jq '[.[] | .doc_url | select(contains("kubernetes.io/docs"))] | length' 
      4. Assert: equals total count (all URLs are k8s docs)
    Expected Result: All entries meet quality threshold
    Evidence: Filter validation output
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add SOTorrent data extraction for K8s questions`
  - Files: `src/plm/benchmark/extraction/`, `artifacts/benchmark/raw/`
  - Pre-commit: Verify output file exists and has expected structure

---

- [x] 3. URL-to-Chunk Mapping Builder

  **What to do**:
  - Read the indexed corpus from SQLite database (INDEX_PATH/index.db)
  - Extract all chunks with their doc_id, source_file, heading, start_char, end_char
  - Build mapping from URL path segments to doc_ids
  - Handle URL normalization (trailing slashes, query params, anchors)
  - Create heading-to-anchor slug converter (## Pod Security → #pod-security)
  - Build anchor-to-heading_id mapping for fragment matching
  - Store mappings as JSON in `artifacts/benchmark/mappings/`
  - Log unmapped URLs for corpus gap analysis

  **Schema Specifications**:

  Input: Corpus SQLite Schema (read-only):
  | Table | Columns Used |
  |-------|--------------|
  | chunks | id, doc_id, content, heading, start_char, end_char |
  | documents | id, source_file, title, url |

  Output Files:
  | File | Schema | Description |
  |------|--------|-------------|
  | `url_to_docid.json` | `{url_path: [doc_id, ...]}` | URL path → list of doc IDs |
  | `url_to_chunks.json` | `{url_path: [chunk_id, ...]}` | URL path → list of chunk IDs |
  | `anchor_to_heading.json` | `{anchor_slug: [{doc_id, heading_id, heading_text}, ...]}` | Fragment → heading mappings |
  | `unmapped_urls.log` | One URL per line with reason | Corpus gap analysis |

  **URL Transformation Rules**:
  ```
  Input:  https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/
  Step 1: Extract path → /docs/concepts/workloads/pods/pod-lifecycle/
  Step 2: Remove trailing slash → /docs/concepts/workloads/pods/pod-lifecycle
  Step 3: Split fragment if present → path=/docs/.../pod-lifecycle, fragment=None
  Step 4: Look up in url_to_docid mapping
  ```

  **Heading-to-Anchor Slug Conversion**:
  | Heading | Anchor Slug | Rule Applied |
  |---------|-------------|--------------|
  | `## Pod Lifecycle` | `pod-lifecycle` | Lowercase, spaces→hyphens |
  | `## Running Pods` | `running-pods` | Lowercase, spaces→hyphens |
  | `## CronJob Spec` | `cronjob-spec` | Preserve case boundaries |
  | `## API Server Flags` | `api-server-flags` | Multi-word handling |

  **Slugification Algorithm**:
  1. Remove markdown heading prefix (`##`, `###`)
  2. Convert to lowercase
  3. Replace spaces with hyphens
  4. Remove special characters except hyphens
  5. Collapse multiple hyphens to single

  **Edge Case Handling**:
  | Case | Handling |
  |------|----------|
  | URL with query params | Strip query string before lookup |
  | URL with multiple anchors | Use first anchor only |
  | Versioned URLs (`/v1.25/docs/...`) | Map to version-agnostic if possible |
  | Redirected URLs (301/302) | Log as unmapped (don't follow redirects) |
  | URL not in corpus | Log to unmapped_urls.log with reason |

  **Must NOT do**:
  - Modify the SQLite database
  - Assume URL format without validation

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward data transformation, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5)
  - **Blocks**: Task 6 (dataset generation needs mappings)
  - **Blocked By**: Task 2 (needs SO data to know which URLs to map)

  **References**:
  - `src/plm/search/storage/sqlite.py` - Storage interface for reading chunks
  - Existing chunk schema: id, doc_id, content, heading, start_char, end_char
  - Draft mapping logic in `.sisyphus/drafts/benchmark-framework.md` lines 103-123

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: URL mapping covers corpus
    Tool: Bash (python/jq)
    Preconditions: Corpus indexed in INDEX_PATH
    Steps:
      1. Run mapping builder
      2. jq 'keys | length' artifacts/benchmark/mappings/url_to_docid.json
      3. Assert: length >= 500 (reasonable corpus size)
      4. Spot-check: pick 10 random URLs from SO data, verify they map to chunks
    Expected Result: Mapping file with 500+ URL paths
    Evidence: Mapping stats logged

  Scenario: Anchor-to-heading mapping works
    Tool: Bash
    Preconditions: Mappings generated
    Steps:
      1. Check artifacts/benchmark/mappings/anchor_to_heading.json exists
      2. Test known anchor: jq '.["pod-lifecycle"]' should return heading_id
      3. Verify heading slugification is consistent
    Expected Result: Fragment anchors map to heading IDs
    Evidence: Sample anchor lookups logged
  ```

  **Commit**: YES (groups with Task 4, 5)
  - Message: `feat(benchmark): add URL-to-chunk mapping builder`
  - Files: `src/plm/benchmark/mapping/`

---

- [x] 4. Tier Assignment Engine

  **What to do**:
  - Implement Oracle's Evidence-Based Tier Assignment methodology
  - Define tier criteria as configuration (YAML or Python constants)
  - GOLD tier: Fragment anchor match OR exact quote match (≥30 chars) OR reciprocal containment (≥20 words)
  - SILVER tier: URL match AND ((upvotes ≥10 AND accepted) OR (upvotes ≥25) OR (multiple_answers_link_same_url ≥2))
  - BRONZE tier: URL match AND score ≥5 (all chunks from URL)
  - EXCLUDE: score <5 OR no URL match
  - Store tier assignment SEPARATELY from signals (enable re-scoring later)
  - Log every tier decision with full evidence trail (JSON-serializable)
  - Make thresholds configurable via environment or config file

  **Schema Specifications**:

  Input: TierAssignmentInput dataclass:
  | Field | Type | Source |
  |-------|------|--------|
  | so_answer_id | int | From Task 2 extraction |
  | url_match | bool | True if URL maps to corpus (Task 3) |
  | fragment_anchor | str or None | From URL if present |
  | fragment_matches_heading | bool | From Task 3 anchor mapping |
  | quote_matches | list[QuoteMatch] | From Task 5 quote matcher |
  | reciprocal_matches | list[ReciprocalMatch] | From Task 5 containment detector |
  | upvotes | int | From SO data |
  | is_accepted | bool | From SO data |
  | multiple_answers_same_url | int | Count of other answers linking same URL |

  Output: TierAssignment dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | tier | Literal["gold", "silver", "bronze", "exclude"] | Assigned tier |
  | tier_reason | string | Human-readable reason (e.g., "quote_match_42_chars") |
  | confidence_score | float | 0.0-1.0 based on signal strength |
  | signals_detected | list[string] | All signals found (may have multiple) |
  | primary_signal | string | The signal that determined tier |
  | evidence | dict | Full evidence for auditability |

  **Tier Decision Logic (Deterministic Priority)**:
  ```
  IF fragment_matches_heading:
      RETURN GOLD, reason="fragment_anchor_match"
  ELIF any quote_match with length >= 30:
      RETURN GOLD, reason=f"quote_match_{max_length}_chars"
  ELIF any reciprocal_match with words >= 20:
      RETURN GOLD, reason=f"reciprocal_containment_{word_count}_words"
  ELIF url_match AND ((upvotes >= 10 AND is_accepted) OR upvotes >= 25):
      RETURN SILVER, reason="url_high_trust"
  ELIF url_match AND multiple_answers_same_url >= 2:
      RETURN SILVER, reason="url_corroborated"
  ELIF url_match AND upvotes >= 5:
      RETURN BRONZE, reason="url_community_validated"
  ELSE:
      RETURN EXCLUDE, reason="insufficient_signal"
  ```

  **Configuration Schema** (tier_config.yaml):
  ```yaml
  gold:
    quote_min_length: 30
    reciprocal_min_words: 20
  silver:
    upvotes_with_accepted: 10
    upvotes_alone: 25
    corroboration_count: 2
  bronze:
    min_upvotes: 5
  ```

  **Confidence Score Calculation**:
  | Signal | Base Confidence | Modifiers |
  |--------|-----------------|-----------|
  | fragment_anchor_match | 1.0 | N/A |
  | quote_match | 0.9 | +0.1 if length > 50 |
  | reciprocal_containment | 0.85 | +0.05 per 10 extra words |
  | url_high_trust | 0.75 | +0.1 if both upvotes≥10 AND accepted |
  | url_corroborated | 0.70 | +0.05 per extra corroborating answer |
  | url_community_validated | 0.60 | +0.02 per upvote above 5 (max 0.75) |

  **Must NOT do**:
  - Hardcode thresholds without configuration option
  - Lose evidence trail (must be reproducible)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core business logic with multiple conditions, needs careful implementation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 5)
  - **Blocks**: Task 6
  - **Blocked By**: Task 2

  **References**:
  - Oracle methodology in `.sisyphus/drafts/benchmark-framework.md` lines 810-880
  - Tier definitions confirmed in draft lines 820-827

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: GOLD tier requires provable signal
    Tool: Bash (python)
    Preconditions: Tier engine implemented
    Steps:
      1. Create test case with fragment anchor match
      2. Run tier assignment
      3. Assert: tier == "GOLD"
      4. Assert: evidence includes "fragment_match": true
      5. Create test case with URL only (no fragment, no quote)
      6. Run tier assignment
      7. Assert: tier != "GOLD"
    Expected Result: GOLD only for provable signals
    Evidence: Test case results logged

  Scenario: Tier decision logging is complete
    Tool: Bash
    Preconditions: Sample data processed
    Steps:
      1. Process 10 SO answers through tier engine
      2. For each result, verify log contains:
         - so_answer_id
         - detected_signals (fragment, quote, etc.)
         - applied_tier
         - confidence_score
      3. Assert: all 10 have complete trace
    Expected Result: Full audit trail per decision
    Evidence: Log output captured
  ```

  **Commit**: YES (groups with Task 3, 5)
  - Message: `feat(benchmark): add evidence-based tier assignment engine`
  - Files: `src/plm/benchmark/tier/`

---

- [x] 5. Quote and Fragment Matching Utilities

  **What to do**:
  - Build HTML parser for SO answer bodies (BeautifulSoup)
  - Extract text from `<code>`, `<pre>`, `<blockquote>` tags
  - Implement exact quote matcher: find ≥30 char matches between SO answer and chunk content
  - Implement reciprocal containment detector: find ≥20 contiguous word overlaps
  - Implement fragment anchor normalizer: handle variations (#pod-lifecycle, #pod_lifecycle, etc.)
  - Handle edge cases: HTML entities, code formatting, markdown artifacts
  - Return match evidence: matched_text, match_length, match_position

  **Schema Specifications**:

  QuoteMatch dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | matched_text | string | The exact text that matched |
  | match_length | int | Character count of match |
  | source_type | Literal["code", "blockquote", "prose"] | Where in SO answer |
  | chunk_id | string | Which chunk contains the match |
  | chunk_offset | int | Character offset in chunk content |
  | answer_offset | int | Character offset in SO answer |

  ReciprocalMatch dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | matched_words | list[string] | The contiguous word sequence |
  | word_count | int | Number of words matched |
  | chunk_id | string | Which chunk contains the match |
  | direction | Literal["chunk_in_answer", "answer_in_chunk"] | Direction of containment |

  **HTML Extraction Rules**:
  | Tag | Extraction Method | Priority |
  |-----|-------------------|----------|
  | `<code>` | Extract inner text, preserve whitespace | HIGH (likely technical content) |
  | `<pre>` | Extract inner text, preserve formatting | HIGH |
  | `<blockquote>` | Extract inner text | MEDIUM (often doc quotes) |
  | `<p>`, `<div>` | Extract inner text | LOW |

  **Text Normalization for Matching**:
  | Step | Before | After |
  |------|--------|-------|
  | 1. HTML decode | `&lt;` | `<` |
  | 2. Collapse whitespace | `a  b\n c` | `a b c` |
  | 3. Strip outer whitespace | ` text ` | `text` |
  | 4. Lowercase (for comparison only) | `Kubectl` | `kubectl` |

  **Generic Text Blacklist** (never match these):
  - "run the following command"
  - "for more information"
  - "see the documentation"
  - "as shown below"
  - "for example"
  - Any string < 20 characters that appears in >10% of chunks

  **Fragment Normalization Rules**:
  | Input Variation | Normalized Form |
  |-----------------|-----------------|
  | `#pod-lifecycle` | `pod-lifecycle` |
  | `#pod_lifecycle` | `pod-lifecycle` |
  | `#PodLifecycle` | `pod-lifecycle` |
  | `#pod--lifecycle` | `pod-lifecycle` |
  | `#pod lifecycle` | `pod-lifecycle` |

  **Edge Case Handling**:
  | Case | Handling |
  |------|----------|
  | Nested code in blockquote | Extract both separately |
  | Markdown in HTML | Strip markdown syntax before matching |
  | Unicode in code | Normalize to ASCII equivalents where possible |
  | Empty tags | Skip silently |

  **Must NOT do**:
  - Use regex for HTML parsing (use BeautifulSoup)
  - Match generic text like "run the following command" (minimum length threshold)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Utility functions, well-defined inputs/outputs
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Task 6
  - **Blocked By**: Task 2

  **References**:
  - Oracle's signal definitions in draft
  - BeautifulSoup documentation for HTML parsing

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Quote matching finds exact code blocks
    Tool: Bash (python)
    Preconditions: Matching utilities implemented
    Steps:
      1. Create SO answer with: "<code>kubectl get pods --all-namespaces</code>"
      2. Create chunk containing: "kubectl get pods --all-namespaces"
      3. Run quote matcher
      4. Assert: match found with length >= 30
      5. Assert: matched_text equals "kubectl get pods --all-namespaces"
    Expected Result: Exact code match detected
    Evidence: Match result logged

  Scenario: Reciprocal containment with paraphrase
    Tool: Bash (python)
    Preconditions: Utilities implemented
    Steps:
      1. Chunk: "A Pod is the smallest deployable unit in Kubernetes..."
      2. SO answer: "As the docs say, a Pod is the smallest deployable unit in Kubernetes..."
      3. Run containment detector
      4. Assert: match found with >= 20 words
    Expected Result: Paraphrase detection works
    Evidence: Match details logged
  ```

  **Commit**: YES (groups with Task 3, 4)
  - Message: `feat(benchmark): add quote and fragment matching utilities`
  - Files: `src/plm/benchmark/matching/`

---

### Phase 2: Multi-Agent Dataset Generation Pipeline

> **ARCHITECTURE PRINCIPLE**: Verification is CODE, not LLM.
> 
> Tier signals (fragment match, quote ≥30 chars, upvotes ≥10) are deterministic.
> Same input = same output. No "reviewer variance."
> 
> LLMs only do creative work: query formulation, quote finding, evidence writing.
> All LLM outputs are verified post-hoc by deterministic code.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2a: DETERMINISTIC SIGNAL EXTRACTION (Code only, parallel)            │
├─────────────────────────────────────────────────────────────────────────────┤
│ For each SO post: extract URL, fragment, upvotes, is_accepted, chunk_ids   │
│ Compute max_possible_tier from signals                                      │
│ Output: SignalBundle per post (deterministic, stateless)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ GENERATOR 1      │     │ GENERATOR 2      │     │ GENERATOR N      │
│ (LLM Agent)      │     │ (LLM Agent)      │     │ (LLM Agent)      │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ Tasks:           │     │ Tasks:           │     │ Tasks:           │
│ 1. Query formula │     │ 1. Query formula │     │ 1. Query formula │
│ 2. Quote finding │     │ 2. Quote finding │     │ 2. Quote finding │
│ 3. Evidence text │     │ 3. Evidence text │     │ 3. Evidence text │
│                  │     │                  │     │                  │
│ Self-checks:     │     │ Self-checks:     │     │ Self-checks:     │
│ • query ≥5 words │     │ • query ≥5 words │     │ • query ≥5 words │
│ • quote verbatim │     │ • quote verbatim │     │ • quote verbatim │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2c: DETERMINISTIC VERIFIER (CODE ONLY — NOT AN LLM)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ verify(case, corpus) → VerificationResult                                   │
│   • chunk_exists: all chunk_ids in corpus?                                  │
│   • quote_exists: matched_quote in chunk content?                           │
│   • quote_length: len(quote) >= 30?                                         │
│   • tier_matches: tier == calculate_tier(signals)?                          │
│   • query_valid: 5 <= word_count(query) <= 100?                             │
│                                                                             │
│ CRITICAL: Pure Python. No LLM. No judgment. Deterministic.                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
             ┌──────────┐              ┌─────────────────┐
             │ PASS     │              │ FAIL            │
             │ → Output │              │ → Regenerate    │
             └──────────┘              └────────┬────────┘
                                                │
           ┌────────────────────────────────────┼────────────────────────────┐
           ▼                                    ▼                            ▼
    retry < 3?                          same_failure × 2?              retry >= 3?
    → Regenerate with                   → UNMAPPABLE                   → QUARANTINE
      structured feedback                 (valid outcome)               (human review)
```

---

- [x] 6a. Deterministic Signal Extraction Pipeline

  **What to do**:
  - Build stateless extraction function that processes one SO post at a time
  - For each SO post, extract:
    - `extracted_url`: The kubernetes.io/docs URL from answer
    - `url_fragment`: The #anchor if present (None otherwise)
    - `answer_upvotes`: Integer score
    - `is_accepted`: Boolean
    - `chunk_ids`: List of chunk IDs this URL maps to (using Task 3 mappings)
    - `chunk_contents`: Actual text content of mapped chunks (for quote matching)
  - Compute `max_possible_tier`:
    - If fragment present → can be GOLD (via fragment match)
    - If upvotes ≥10 AND accepted → can be SILVER (high trust)
    - If upvotes ≥5 → can be BRONZE
    - Else → EXCLUDE
  - Store as `SignalBundle` dataclass with all raw signals
  - Implement parallel processing with ThreadPoolExecutor (8 workers)
  - Hash the corpus version at start (ensure all verification uses same corpus)
  - Output: `artifacts/benchmark/signals/signal_bundles.jsonl` (one JSON per line)

  **Schema Specifications**:

  Input File:
  - Path: `artifacts/benchmark/raw/so_k8s_answers.json` (from Task 2)
  - Format: JSON array of SOAnswer objects

  SignalBundle dataclass (output per SO answer):
  | Field | Type | Description |
  |-------|------|-------------|
  | bundle_id | string | UUID for this bundle |
  | so_question_id | int | Original SO question ID |
  | so_answer_id | int | Original SO answer ID |
  | question_title | string | SO question title |
  | question_body | string | SO question body (HTML) |
  | answer_body | string | SO answer body (HTML) |
  | extracted_url | string | The K8s doc URL |
  | url_fragment | string or None | The #anchor if present |
  | answer_upvotes | int | Answer score |
  | is_accepted | bool | Whether accepted answer |
  | answer_date | string | ISO8601 date |
  | chunk_ids | list[string] | Chunk IDs this URL maps to |
  | chunk_contents | list[string] | Text content of each chunk |
  | quote_matches | list[QuoteMatch] | From Task 5 quote matcher |
  | reciprocal_matches | list[ReciprocalMatch] | From Task 5 containment detector |
  | fragment_matches_heading | bool | Whether fragment maps to heading |
  | max_possible_tier | Literal["gold", "silver", "bronze", "exclude"] | Upper bound on achievable tier |
  | extraction_timestamp | string | ISO8601 when extracted |
  | corpus_version_hash | string | SHA256 of corpus state |

  Output Files:
  | File | Format | Description |
  |------|--------|-------------|
  | `signal_bundles.jsonl` | JSONL (one bundle per line) | All extracted bundles |
  | `unmappable.log` | Text, one entry per line | Posts that couldn't be mapped |
  | `extraction_stats.json` | JSON object | Summary statistics |

  extraction_stats.json Schema:
  | Field | Type | Description |
  |-------|------|-------------|
  | total_posts_processed | int | Total SO posts input |
  | bundles_created | int | Successfully extracted bundles |
  | unmappable_count | int | Posts with no chunk mapping |
  | gold_potential | int | Bundles with max_tier=gold |
  | silver_potential | int | Bundles with max_tier=silver |
  | bronze_potential | int | Bundles with max_tier=bronze |
  | excluded_count | int | Posts excluded (score<5 or no URL match) |
  | corpus_version_hash | string | Hash used for this extraction |
  | extraction_duration_seconds | float | Total processing time |

  **Corpus Version Hash Calculation**:
  ```
  hash = SHA256(
    sorted([chunk.id + chunk.content_hash for chunk in corpus])
  )
  ```
  This ensures any corpus change invalidates previous extractions.

  **Must NOT do**:
  - Use any LLM in this step (pure code extraction)
  - Modify signals after extraction (immutable)
  - Process posts that map to zero chunks (exclude immediately)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Deterministic data transformation, no LLM, embarrassingly parallel
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (first step in dataset generation)
  - **Blocks**: Task 6b
  - **Blocked By**: Tasks 3, 4, 5

  **References**:
  - Parallel processing pattern: `src/plm/extraction/fast/cli.py:267-278` (ThreadPoolExecutor)
  - Task 3 URL mappings: `artifacts/benchmark/mappings/`
  - Task 4 tier calculation logic

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Signal extraction is deterministic
    Tool: Bash (python)
    Preconditions: SO data and mappings ready
    Steps:
      1. Run signal extraction twice on same input
      2. Compute hash of output file both times
      3. Assert: hashes are identical (deterministic)
    Expected Result: Same input produces same output
    Evidence: Hash comparison logged

  Scenario: Parallel extraction produces complete output
    Tool: Bash
    Preconditions: 5000+ SO posts available
    Steps:
      1. Run: plm-benchmark extract-signals --workers 8
      2. Count lines in signal_bundles.jsonl
      3. Assert: line count >= 4000 (allowing ~20% unmappable)
      4. Verify no duplicate so_post_ids
    Expected Result: All mappable posts extracted
    Evidence: Extraction stats in logs

  Scenario: Unmappable posts are logged correctly
    Tool: Bash
    Preconditions: Some posts have URLs not in corpus
    Steps:
      1. Check unmappable.log exists
      2. Verify each line has: so_post_id, url, reason
      3. Assert: reasons are one of [no_chunks, invalid_url, score_below_5]
    Expected Result: Clear audit trail for exclusions
    Evidence: Unmappable log sample
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add deterministic signal extraction pipeline`
  - Files: `src/plm/benchmark/extraction/signals.py`

---

- [x] 6b. Parallel LLM Generator Agents

  **What to do**:
  - Design generator agent prompt that produces benchmark cases from SignalBundles
  - Agent receives: SO question title/body, answer text, chunk contents, signals
  - Agent outputs:
    - `query`: Natural language question (reformulated from SO, 5-100 words)
    - `matched_quote`: Exact text from chunk that answers the question (≥30 chars for GOLD, None for BRONZE)
    - `evidence_text`: Brief explanation of why this chunk answers the query
    - `reasoning`: Agent's reasoning for the mapping (for audit)
  - Implement self-checks INSIDE the generator prompt:
    - Query must be 5-100 words
    - If providing quote, it must appear verbatim in chunk content
    - Quote must be ≥30 characters if claiming GOLD tier
  - If self-check fails, agent retries internally (up to 2 times) before returning
  - Split work by SO post ID ranges (e.g., 0-500, 501-1000, etc.)
  - Use ThreadPoolExecutor to run N generator agents in parallel
  - Each agent processes a batch of SignalBundles
  - Output: `artifacts/benchmark/generated/batch_{N}.jsonl`

  **Schema Specifications**:

  Input File:
  - Path: `artifacts/benchmark/signals/signal_bundles.jsonl` (from Task 6a)
  - Format: JSONL, one SignalBundle per line

  LLM Provider Configuration:
  | Setting | Value | Rationale |
  |---------|-------|-----------|
  | Provider | Anthropic Claude | Consistent with project |
  | Model | claude-sonnet (via PLM_LLM_MODEL) | Balance of quality/cost |
  | Temperature | 0.3 | Low variance for consistency |
  | Max tokens | 1024 | Sufficient for structured output |

  GeneratedCase dataclass (output per bundle):
  | Field | Type | Description |
  |-------|------|-------------|
  | case_id | string | UUID for this case |
  | bundle_id | string | Reference to source SignalBundle |
  | query | string | Reformulated natural language question (5-100 words) |
  | matched_quote | string or None | Exact verbatim quote from chunk (≥30 chars for GOLD) |
  | evidence_text | string | 1-2 sentence explanation of relevance |
  | reasoning | string | LLM's reasoning for the mapping (audit trail) |
  | chunk_ids | list[string] | Chunks this query should retrieve |
  | tier_from_signals | Literal["gold", "silver", "bronze"] | Tier based on signals (NOT LLM decision) |
  | generation_timestamp | string | ISO8601 when generated |
  | generator_model | string | Model used (e.g., "claude-3-sonnet") |
  | internal_retries | int | Number of self-check retries (0-2) |

  **Generator Prompt Template**:
  ```
  You are generating a benchmark question for evaluating a documentation search system.

  ## Source Information
  - Original SO Question: {question_title}
  - SO Question Body: {question_body}
  - SO Answer (linked to docs): {answer_body}
  - Documentation Chunk Content: {chunk_contents}
  - Available Signals: {signals_summary}

  ## Your Task
  1. Create a NATURAL LANGUAGE QUESTION that:
     - Captures the user's information need from the SO question
     - Is reformulated (not a copy of the SO title)
     - Is 5-100 words long
     - Would be answered by the documentation chunk

  2. If the answer contains text from the documentation chunk:
     - Find an EXACT quote (≥30 chars) that appears VERBATIM in the chunk
     - Copy it character-for-character

  3. Write a brief EVIDENCE explanation (1-2 sentences) of why this chunk answers the question.

  ## Self-Check (MUST pass before returning)
  - [ ] Query is 5-100 words
  - [ ] If quote provided, verify it appears EXACTLY in chunk content
  - [ ] If quote provided for GOLD tier, verify length ≥ 30 characters

  ## Output Format (JSON)
  {
    "query": "...",
    "matched_quote": "..." or null,
    "evidence_text": "...",
    "reasoning": "..."
  }
  ```

  **Batch Processing Configuration**:
  | Setting | Value | Rationale |
  |---------|-------|-----------|
  | Workers | 4-8 | Balance parallelism vs API rate limits |
  | Batch size | 500 | ~10 batches for 5000 bundles |
  | Rate limit | 10 req/sec | Stay under API limits |
  | Retry on API error | 3 times | Handle transient failures |

  Output Files:
  | File | Format | Description |
  |------|--------|-------------|
  | `batch_{N}.jsonl` | JSONL | Generated cases for batch N |
  | `generation_stats.json` | JSON | Per-batch statistics |
  | `generation_errors.log` | Text | Failed generations with errors |

  **Must NOT do**:
  - Let generator decide tier (tier comes from signals, verified by code)
  - Accept quotes that don't exist in chunk (self-check must catch this)
  - Process SignalBundles with max_possible_tier == EXCLUDE

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: LLM orchestration, prompt engineering, parallel batch processing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (this task spawns parallel agents)
  - **Parallel Group**: Wave 3 (after 6a)
  - **Blocks**: Task 6c
  - **Blocked By**: Task 6a

  **References**:
  - V6 pipeline batch processing: `src/plm/extraction/slow/hybrid_ner/pipeline.py`
  - Generator prompt design: study SO question → query reformulation
  - POC-1 ground truth generation: `poc/poc-1-llm-extraction-guardrails/generate_ground_truth.py`

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Parallel generation completes all batches
    Tool: Bash
    Preconditions: Signal bundles ready
    Steps:
      1. Run: plm-benchmark generate --workers 4 --batch-size 500
      2. Count batch files in artifacts/benchmark/generated/
      3. Assert: all expected batch files exist
      4. Count total cases across all batches
      5. Assert: total >= 4000
    Expected Result: All batches generated
    Evidence: Batch completion stats

  Scenario: Generator self-checks work
    Tool: Bash (python)
    Preconditions: Generate sample batch
    Steps:
      1. For 10 random cases, verify:
         - query has 5-100 words
         - if matched_quote exists, it appears in chunk_content
         - if matched_quote exists, len >= 30
      2. Assert: all 10 pass self-check criteria
    Expected Result: Self-checks enforced
    Evidence: Validation results logged

  Scenario: Generator produces natural queries
    Tool: Bash (python)
    Preconditions: Batch generated
    Steps:
      1. Sample 20 queries from output
      2. Verify none are exact copies of SO question title
      3. Verify all read as natural questions (not keyword lists)
    Expected Result: Queries are reformulated, not copied
    Evidence: Sample queries logged for review
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add parallel LLM generator agents`
  - Files: `src/plm/benchmark/generation/generator.py`, `src/plm/benchmark/generation/prompts.py`

---

- [x] 6c. Deterministic Verifier (Code-Based Quality Gate)

  **What to do**:
  - Build `verify()` function that checks all generated cases
  - Verification is PURE PYTHON (no LLM, no judgment)
  - Checks to implement:
    ```python
    def verify(case: GeneratedCase, corpus: ChunkCorpus) -> VerificationResult:
        failures = []
        
        # Check 1: All chunks exist in corpus
        for cid in case.chunk_ids:
            if cid not in corpus:
                failures.append(("chunk_exists", cid, "not found"))
        
        # Check 2: Quote exists in chunk content (if provided)
        if case.matched_quote:
            chunk_text = " ".join(corpus[cid].content for cid in case.chunk_ids)
            # Normalize whitespace for matching
            normalized_quote = normalize_whitespace(case.matched_quote)
            normalized_chunk = normalize_whitespace(chunk_text)
            if normalized_quote not in normalized_chunk:
                failures.append(("quote_exists", case.matched_quote[:40], "not in chunks"))
        
        # Check 3: Quote length (for GOLD tier claims)
        if case.matched_quote and case.tier == "gold":
            if len(case.matched_quote) < 30:
                failures.append(("quote_length", len(case.matched_quote), "< 30 chars"))
        
        # Check 4: Tier matches signals (deterministic!)
        expected_tier = calculate_tier_from_signals(
            has_fragment=bool(case.fragment_anchor),
            has_valid_quote=case.matched_quote and len(case.matched_quote) >= 30,
            upvotes=case.upvotes,
            is_accepted=case.is_accepted
        )
        if case.tier != expected_tier:
            failures.append(("tier_match", expected_tier, case.tier))
        
        # Check 5: Query validity
        word_count = len(case.query.split())
        if not (5 <= word_count <= 100):
            failures.append(("query_length", word_count, "outside 5-100 range"))
        
        return VerificationResult(
            passed=len(failures) == 0,
            failures=failures,
            case_id=case.id
        )
    ```
  - Run verifier on ALL generated cases
  - Output three files:
    - `verified_passed.jsonl`: Cases that passed all checks
    - `verified_failed.jsonl`: Cases that failed (for regeneration)
    - `verification_report.json`: Summary statistics
  - Log every check result for auditability

  **Schema Specifications**:

  Input Files:
  - Path: `artifacts/benchmark/generated/batch_*.jsonl` (from Task 6b)
  - Format: JSONL, one GeneratedCase per line

  VerificationResult dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | case_id | string | The case being verified |
  | passed | bool | True if all checks passed |
  | failures | list[VerificationFailure] | List of failed checks |
  | checks_run | list[string] | All checks executed |
  | verification_timestamp | string | ISO8601 when verified |

  VerificationFailure dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | check_name | string | One of: chunk_exists, quote_exists, quote_length, tier_match, query_length |
  | expected | string | What was expected |
  | found | string | What was actually found |
  | recoverable | bool | Whether regeneration could fix this |

  **Check Definitions (All Mandatory)**:
  | Check | Pass Condition | Failure Recoverable? |
  |-------|----------------|---------------------|
  | chunk_exists | All chunk_ids exist in corpus | NO (corpus gap) |
  | quote_exists | matched_quote found verbatim in chunk content | YES (find different quote) |
  | quote_length | len(matched_quote) >= 30 if tier is GOLD | YES (find longer quote) |
  | tier_match | claimed tier matches calculated tier from signals | YES (accept correct tier) |
  | query_length | 5 <= word_count(query) <= 100 | YES (rephrase query) |

  Output Files:
  | File | Format | Description |
  |------|--------|-------------|
  | `verified_passed.jsonl` | JSONL | Cases passing all checks |
  | `verified_failed.jsonl` | JSONL | Cases with failures (for regeneration) |
  | `verification_report.json` | JSON | Summary statistics |

  verification_report.json Schema:
  | Field | Type | Description |
  |-------|------|-------------|
  | total_cases | int | Total cases verified |
  | passed_count | int | Cases that passed all checks |
  | failed_count | int | Cases with at least one failure |
  | pass_rate | float | passed_count / total_cases |
  | failure_breakdown | dict[str, int] | Count per check_name |
  | recoverable_failures | int | Failures that can be fixed by regeneration |
  | unrecoverable_failures | int | Failures due to corpus gaps |
  | verification_duration_seconds | float | Total processing time |

  **Must NOT do**:
  - Use ANY LLM for verification (deterministic code only)
  - Skip any check (all are mandatory)
  - Modify cases during verification (read-only)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Pure Python, deterministic, no external dependencies
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (embarrassingly parallel per case)
  - **Parallel Group**: Wave 3 (after 6b)
  - **Blocks**: Task 6d
  - **Blocked By**: Task 6b

  **References**:
  - Oracle verifier design in consultation notes
  - Text normalization patterns in codebase

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Verifier is deterministic
    Tool: Bash (python)
    Preconditions: Generated cases available
    Steps:
      1. Run verifier twice on same input
      2. Compare verification_report.json from both runs
      3. Assert: identical pass/fail counts
      4. Assert: identical failure reasons for same case IDs
    Expected Result: Same input produces same output
    Evidence: Hash of both reports

  Scenario: Verifier catches quote mismatches
    Tool: Bash (python)
    Preconditions: Inject test case with invalid quote
    Steps:
      1. Create case with matched_quote that doesn't exist in chunk
      2. Run verifier
      3. Assert: case appears in verified_failed.jsonl
      4. Assert: failure reason is ("quote_exists", ...)
    Expected Result: Invalid quotes rejected
    Evidence: Failure record logged

  Scenario: Verifier catches tier mismatches
    Tool: Bash (python)
    Preconditions: Inject test case with wrong tier
    Steps:
      1. Create case claiming GOLD but only has URL match (no quote/fragment)
      2. Run verifier
      3. Assert: case appears in verified_failed.jsonl
      4. Assert: failure reason is ("tier_match", expected="bronze", found="gold")
    Expected Result: Tier inflation rejected
    Evidence: Failure record logged

  Scenario: Pass rate is reasonable
    Tool: Bash
    Preconditions: Full verification run complete
    Steps:
      1. Check verification_report.json
      2. Calculate pass_rate = passed / total
      3. Assert: pass_rate >= 0.70 (at least 70% pass first time)
      4. If pass_rate < 0.70, log warning for prompt tuning
    Expected Result: Majority pass first verification
    Evidence: Verification report stats
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add deterministic code-based verifier`
  - Files: `src/plm/benchmark/verification/verifier.py`

---

- [x] 6d. Regeneration Loop with Structured Feedback

  **What to do**:
  - Build regeneration orchestrator that processes failed cases
  - For each failed case, create structured feedback for regeneration:
    ```python
    def create_regeneration_prompt(original_case, failures) -> str:
        lines = ["Previous attempt failed verification:"]
        for check, expected, found in failures:
            if check == "quote_exists":
                lines.append(f"- Quote '{found}...' not found in chunk content.")
                lines.append("  FIX: Find a different quote that appears VERBATIM in the chunk.")
            elif check == "quote_length":
                lines.append(f"- Quote only {found} chars, need ≥30 for GOLD tier.")
                lines.append("  FIX: Find a longer quote, or don't claim quote match.")
            elif check == "tier_match":
                lines.append(f"- Claimed tier '{found}' but signals only support '{expected}'.")
                lines.append(f"  FIX: Accept tier '{expected}' based on available signals.")
            elif check == "query_length":
                lines.append(f"- Query has {found} words, need 5-100.")
                lines.append("  FIX: Expand or condense the query.")
        return "\n".join(lines)
    ```
  - Implement termination conditions:
    - `retry_count >= MAX_RETRIES (3)` → QUARANTINE (needs human review)
    - `same_failure repeated twice` → UNMAPPABLE (fundamental impossibility)
    - `no_chunks_for_url in failures` → UNMAPPABLE (corpus gap)
  - Track retry count per case
  - Use fresh generator agent for each retry (avoid stuck patterns)
  - Output final categories:
    - `regenerated_passed.jsonl`: Cases that passed after retry
    - `unmappable.jsonl`: Cases that can't be mapped (valid outcome)
    - `quarantine.jsonl`: Cases needing human review (should be <5%)

  **Schema Specifications**:

  Input File:
  - Path: `artifacts/benchmark/signals/verified_failed.jsonl` (from Task 6c)
  - Format: JSONL, one failed GeneratedCase with VerificationResult per line

  RegenerationAttempt dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | case_id | string | Original case ID |
  | attempt_number | int | 1, 2, or 3 |
  | previous_failures | list[VerificationFailure] | Why previous attempt failed |
  | feedback_prompt | string | Structured feedback given to LLM |
  | new_case | GeneratedCase | The regenerated case |
  | verification_result | VerificationResult | Result of verifying new case |
  | timestamp | string | ISO8601 when attempted |

  **Termination Conditions (Decision Tree)**:
  ```
  IF failures contain "chunk_exists" (corpus gap):
      → UNMAPPABLE, reason="corpus_gap"
  ELIF retry_count >= 3:
      → QUARANTINE, reason="max_retries_exceeded"
  ELIF same failure appeared in last 2 attempts:
      → UNMAPPABLE, reason="persistent_failure_{check_name}"
  ELIF verification passed:
      → PASSED
  ELSE:
      → RETRY with structured feedback
  ```

  **Structured Feedback Templates**:
  | Failure | Feedback Template |
  |---------|-------------------|
  | quote_exists | "Quote '{quote[:40]}...' NOT FOUND in chunk. Find a DIFFERENT quote that appears VERBATIM. Here is the actual chunk content: {chunk_content}" |
  | quote_length | "Quote is only {length} chars. GOLD tier requires ≥30 chars. Either find a longer quote or accept SILVER/BRONZE tier." |
  | tier_match | "You claimed '{claimed}' tier but signals only support '{expected}'. Accept the lower tier." |
  | query_length | "Query has {count} words. Must be 5-100 words. {'Expand' if count < 5 else 'Condense'} the query." |

  Output Files:
  | File | Format | Description |
  |------|--------|-------------|
  | `regenerated_passed.jsonl` | JSONL | Cases that passed after retry |
  | `unmappable.jsonl` | JSONL | Cases with fundamental mapping issues |
  | `quarantine.jsonl` | JSONL | Cases needing human review |
  | `regeneration_log.jsonl` | JSONL | All attempts with full history |

  regeneration_stats.json Schema:
  | Field | Type | Description |
  |-------|------|-------------|
  | total_failed_input | int | Failed cases from 6c |
  | regenerated_passed | int | Cases fixed by regeneration |
  | unmappable_count | int | Fundamentally unmappable |
  | quarantine_count | int | Sent to human review |
  | recovery_rate | float | regenerated_passed / total_failed_input |
  | quarantine_rate | float | quarantine_count / (total from 6b) |
  | attempts_distribution | dict[int, int] | Count per attempt number |

  **Quality Target**:
  - Quarantine rate < 5% of total generated cases
  - Recovery rate > 50% of failed cases

  **Must NOT do**:
  - Retry more than 3 times per case (infinite loop prevention)
  - Use same generator for retry (use fresh agent)
  - Ignore unmappable as failure (it's a valid outcome)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Orchestration logic, retry management, multi-agent coordination
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (retries are independent)
  - **Parallel Group**: Wave 3 (after 6c)
  - **Blocks**: Task 6e
  - **Blocked By**: Task 6c

  **References**:
  - Oracle regeneration design in consultation notes
  - V6 pipeline retry patterns

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Regeneration improves pass rate
    Tool: Bash
    Preconditions: Initial verification complete with failures
    Steps:
      1. Run regeneration loop
      2. Count cases in regenerated_passed.jsonl
      3. Compare to initial failed count
      4. Assert: at least 50% of failed cases now pass
    Expected Result: Regeneration recovers most failures
    Evidence: Before/after stats logged

  Scenario: Termination conditions work
    Tool: Bash
    Preconditions: Regeneration complete
    Steps:
      1. Check no case was retried more than 3 times (grep retry_count in logs)
      2. Check quarantine.jsonl exists
      3. Assert: quarantine count < 5% of total
      4. Check unmappable.jsonl has valid reasons
    Expected Result: No infinite loops, clear outcomes
    Evidence: Retry count distribution logged

  Scenario: Structured feedback is specific
    Tool: Bash (grep logs)
    Preconditions: Some cases were regenerated
    Steps:
      1. Find regeneration prompts in logs
      2. Verify prompts include specific failure (e.g., "Quote 'xyz...' not found")
      3. Verify prompts include specific fix instruction
    Expected Result: Actionable feedback in prompts
    Evidence: Sample regeneration prompts logged
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add regeneration loop with structured feedback`
  - Files: `src/plm/benchmark/regeneration/orchestrator.py`

---

- [x] 6e. Final Assembly and Audit Metadata

  **What to do**:
  - Merge all passed cases: verified_passed + regenerated_passed
  - Split into tier-specific files:
    - `artifacts/benchmark/datasets/gold.json`
    - `artifacts/benchmark/datasets/silver.json`
    - `artifacts/benchmark/datasets/bronze.json`
  - Add comprehensive audit metadata to each case
  - Generate comprehensive statistics report:
    - Total cases per tier
    - Pass-first-time rate
    - Retry rate and success rate
    - Unmappable rate and reasons
    - Quarantine rate (should be <5%)
    - Signal distribution (fragment vs quote vs URL-only)
  - Validate final datasets meet targets:
    - GOLD: ≥400 cases
    - SILVER: ≥1500 cases
    - BRONZE: ≥5000 cases
  - Implement CLI: `plm-benchmark assemble --output datasets/`

  **Schema Specifications**:

  Input Files:
  - `artifacts/benchmark/signals/verified_passed.jsonl` (from Task 6c)
  - `artifacts/benchmark/signals/regenerated_passed.jsonl` (from Task 6d)

  BenchmarkCaseAudit dataclass (COMPLETE FIELD LIST):
  | Field | Type | Category | Description |
  |-------|------|----------|-------------|
  | id | string | Provenance | UUID for this benchmark case |
  | so_question_id | int | Provenance | Original SO question ID |
  | so_answer_id | int | Provenance | Original SO answer ID |
  | extraction_timestamp | string | Provenance | ISO8601 when signals extracted |
  | generation_timestamp | string | Provenance | ISO8601 when case generated |
  | corpus_version_hash | string | Provenance | SHA256 of corpus at extraction |
  | extracted_url | string | Raw Signal | The K8s doc URL from answer |
  | url_fragment | string or None | Raw Signal | The #anchor if present |
  | answer_upvotes | int | Raw Signal | Answer score from SO |
  | answer_is_accepted | bool | Raw Signal | Whether accepted answer |
  | answer_date | string | Raw Signal | ISO8601 date of answer |
  | chunk_ids | list[string] | Raw Signal | Chunk IDs mapped from URL |
  | tier | Literal["gold", "silver", "bronze"] | Tier Assignment | Assigned tier |
  | tier_reason | string | Tier Assignment | Why this tier (e.g., "quote_match_42_chars") |
  | confidence_score | float | Tier Assignment | 0.0-1.0 confidence |
  | signals_detected | list[string] | Tier Assignment | All signals found |
  | query | string | LLM Generated | The benchmark query (5-100 words) |
  | evidence_text | string | LLM Generated | Explanation of relevance |
  | matched_quote | string or None | LLM Generated | Verbatim quote if GOLD |
  | reasoning | string | LLM Generated | LLM's mapping reasoning |
  | verification_passed_first_try | bool | Verification | True if passed Task 6c |
  | retry_count | int | Verification | 0-3, attempts needed |
  | verification_checks | list[tuple[str, bool]] | Verification | All checks and results |
  | final_status | Literal["accepted"] | Verification | Always "accepted" in final dataset |
  | relevant_chunk_ids | list[string] | Benchmark | Chunks that answer query (= chunk_ids) |

  Output Files:
  | File | Format | Description |
  |------|--------|-------------|
  | `datasets/gold.json` | JSON array | GOLD tier cases with full audit |
  | `datasets/silver.json` | JSON array | SILVER tier cases with full audit |
  | `datasets/bronze.json` | JSON array | BRONZE tier cases with full audit |
  | `statistics_report.json` | JSON object | Comprehensive statistics |

  statistics_report.json Schema:
  | Field | Type | Description |
  |-------|------|-------------|
  | generation_date | string | ISO8601 when assembled |
  | corpus_version_hash | string | Corpus used |
  | total_cases | int | Sum across all tiers |
  | cases_per_tier | dict[str, int] | {"gold": N, "silver": N, "bronze": N} |
  | pass_first_time_rate | float | % passing Task 6c directly |
  | retry_rate | float | % needing regeneration |
  | retry_success_rate | float | % of retries that succeeded |
  | unmappable_rate | float | % unmappable (valid exclusion) |
  | quarantine_rate | float | % quarantined (target <5%) |
  | signal_distribution | dict | {"fragment_anchor": N, "quote_match": N, "url_only": N} |
  | tier_reason_breakdown | dict | Counts per tier_reason |
  | average_confidence | dict[str, float] | Per-tier average confidence |
  | unique_so_questions | int | Unique SO questions |
  | unique_doc_urls | int | Unique K8s doc URLs |

  **Dataset Size Targets**:
  | Tier | Minimum | Target | Maximum |
  |------|---------|--------|---------|
  | GOLD | 400 | 500 | No limit |
  | SILVER | 1500 | 2000 | No limit |
  | BRONZE | 5000 | 10000 | No limit |

  If targets not met, log warning with gap analysis.

  **Must NOT do**:
  - Include cases without full audit metadata
  - Include quarantine cases in final datasets
  - Lose any provenance information

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Data assembly, JSON manipulation, no complex logic
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final step in Wave 3)
  - **Blocks**: Task 7
  - **Blocked By**: Task 6d

  **References**:
  - BenchmarkCase schema in draft
  - Existing dataset formats in `poc/chunking_benchmark_v2/corpus/`

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Final datasets meet size targets
    Tool: Bash
    Preconditions: Assembly complete
    Steps:
      1. jq 'length' artifacts/benchmark/datasets/gold.json
      2. Assert: >= 400
      3. jq 'length' artifacts/benchmark/datasets/silver.json
      4. Assert: >= 1500
      5. jq 'length' artifacts/benchmark/datasets/bronze.json
      6. Assert: >= 5000
    Expected Result: All tier targets met
    Evidence: Dataset stats in .sisyphus/evidence/task-6e-final-stats.json

  Scenario: Audit metadata is complete
    Tool: Bash (jq)
    Preconditions: Datasets assembled
    Steps:
      1. Sample 10 cases from gold.json
      2. For each, verify ALL audit fields present:
         - so_question_id, so_answer_id
         - extraction_timestamp, corpus_version_hash
         - tier, tier_reason
         - verification_passed_first_try, retry_count
         - relevant_chunk_ids (non-empty)
      3. Assert: all 10 have complete metadata
    Expected Result: Full audit trail in every case
    Evidence: Sample cases logged

  Scenario: No quarantine cases in final output
    Tool: Bash (jq)
    Preconditions: Datasets assembled
    Steps:
      1. Search all dataset files for final_status != "accepted"
      2. Assert: count == 0
    Expected Result: Only accepted cases in final datasets
    Evidence: Validation output

  Scenario: Statistics report is comprehensive
    Tool: Bash
    Preconditions: Assembly complete
    Steps:
      1. Check artifacts/benchmark/statistics_report.json exists
      2. Verify it contains:
         - total_cases, cases_per_tier
         - pass_first_time_rate, retry_rate
         - unmappable_rate, quarantine_rate
         - signal_distribution
      3. Assert: quarantine_rate < 0.05 (5%)
    Expected Result: Comprehensive stats for quality assessment
    Evidence: Report contents logged
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add final assembly with audit metadata`
  - Files: `src/plm/benchmark/assembly/`, `artifacts/benchmark/datasets/`
  - Pre-commit: Verify all dataset files exist with correct structure

---

### Phase 3: Benchmark Runner

- [x] 7. Benchmark Runner (API Consumer)

  **What to do**:
  - Build HTTP client that calls production search service at configurable URL
  - Load benchmark dataset (gold/silver/bronze JSON)
  - For each benchmark case:
    - POST to /query with `explain: true`
    - Capture full response including debug_info and metadata
    - Calculate Hit@k by checking if any relevant_chunk_id in top-k results
    - Calculate rank of first relevant chunk for MRR
    - Log full trace: query, expected chunks, retrieved chunks, scores, timing
  - Calculate aggregate metrics: Hit@1, Hit@5, Hit@10, MRR, NDCG@10
  - Support filtering by tier, confidence threshold
  - Implement CLI: `plm-benchmark evaluate --dataset gold.json --k 10 --url http://localhost:8000`
  - Output results as JSON with per-query and aggregate metrics

  **Schema Specifications**:

  Input File (Benchmark Dataset):
  - Path: `artifacts/benchmark/datasets/{tier}.json` (from Task 6e)
  - Format: JSON array of BenchmarkCaseAudit objects

  CLI Arguments:
  | Argument | Type | Default | Description |
  |----------|------|---------|-------------|
  | --dataset | path | (required) | Path to benchmark JSON |
  | --url | string | http://localhost:8000 | Search service URL |
  | --k | int | 10 | Number of results to retrieve |
  | --output | path | stdout | Output file for results |
  | --tier-filter | string | None | Filter to specific tier |
  | --min-confidence | float | 0.0 | Minimum confidence threshold |
  | --concurrency | int | 4 | Parallel HTTP requests |
  | --timeout | int | 30 | Request timeout in seconds |

  PerQueryResult dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | case_id | string | Benchmark case ID |
  | query | string | The query sent |
  | expected_chunk_ids | list[string] | Ground truth chunks |
  | retrieved_chunk_ids | list[string] | Chunks returned by API |
  | hit_at_1 | bool | Relevant chunk in top 1? |
  | hit_at_5 | bool | Relevant chunk in top 5? |
  | hit_at_10 | bool | Relevant chunk in top 10? |
  | first_relevant_rank | int or None | Rank of first relevant (1-indexed), None if not found |
  | reciprocal_rank | float | 1/rank or 0 if not found |
  | ndcg_at_10 | float | NDCG@10 for this query |
  | request_id | string | API request ID for traceability |
  | response_time_ms | float | API response latency |
  | debug_info | list[DebugInfo] | Per-result score breakdown |
  | api_metadata | QueryMetadata | Query transformation details |

  BenchmarkResults dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | run_id | string | UUID for this benchmark run |
  | run_timestamp | string | ISO8601 when run |
  | dataset_path | string | Input dataset used |
  | service_url | string | Search service URL |
  | k | int | K value used |
  | total_queries | int | Number of queries run |
  | hit_at_1 | float | Proportion with hit in top 1 |
  | hit_at_5 | float | Proportion with hit in top 5 |
  | hit_at_10 | float | Proportion with hit in top 10 |
  | mrr | float | Mean Reciprocal Rank |
  | ndcg_at_10 | float | Mean NDCG@10 |
  | mean_response_time_ms | float | Average API latency |
  | p95_response_time_ms | float | 95th percentile latency |
  | per_query_results | list[PerQueryResult] | Detailed per-query results |

  **Metric Definitions**:
  | Metric | Formula | Description |
  |--------|---------|-------------|
  | Hit@k | any(relevant in retrieved[:k]) | Binary: found or not |
  | MRR | mean(1/rank) over all queries | Rank of first relevant |
  | NDCG@k | DCG@k / IDCG@k | Normalized discounted gain |
  | DCG@k | sum(rel_i / log2(i+1)) for i in 1..k | Discounted cumulative gain |

  Output File:
  - Format: JSON (BenchmarkResults object)
  - Location: `--output` path or stdout

  **Must NOT do**:
  - Instantiate HybridRetriever directly (must call HTTP API)
  - Skip explain mode (need traceability)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: HTTP client, metrics calculation, logging orchestration
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 3)
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 1, 6

  **References**:
  - Existing metrics: `poc/convex_fusion_benchmark/metrics.py` (MRR, Recall@k, Hit@k)
  - Existing benchmark pattern: `poc/chunking_benchmark_v2/benchmark_realistic_questions.py`
  - API endpoint: `/query` with explain=true

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Benchmark runner produces metrics
    Tool: Bash
    Preconditions: Search service running, dataset generated
    Steps:
      1. plm-benchmark evaluate --dataset artifacts/benchmark/datasets/gold.json --k 10 --url http://localhost:8000
      2. Check output JSON exists
      3. Assert: output contains "hit_at_1", "hit_at_5", "hit_at_10", "mrr"
      4. Assert: all metrics are floats between 0 and 1
      5. Assert: output contains "per_query_results" array
    Expected Result: Valid metrics JSON
    Evidence: Output saved to .sisyphus/evidence/task-7-benchmark-results.json

  Scenario: Full traceability in results
    Tool: Bash (jq)
    Preconditions: Benchmark run complete
    Steps:
      1. jq '.per_query_results[0]' output.json
      2. Assert: contains "query", "expected_chunks", "retrieved_chunks"
      3. Assert: contains "debug_info" from API response
      4. Assert: contains "request_id"
      5. Assert: contains "bm25_contribution", "semantic_contribution" per result
    Expected Result: Each query fully traceable
    Evidence: Sample query trace logged
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add benchmark runner with API consumption`
  - Files: `src/plm/benchmark/runner/`
  - Pre-commit: Unit tests pass

---

- [x] 8. Integration Evaluation Suite (Tier 5)

  **What to do**:
  - Implement complementarity analysis: measure overlap and error correlation between BM25 and semantic
  - Implement cascade evaluation: measure contribution at each pipeline stage
  - Implement ablation waterfall: marginal contribution of each component
  - Use explain mode to get component-level scores
  - Calculate: overlap@10, error_correlation, fusion_potential
  - Calculate: per-stage recall@100, MRR improvement at each stage
  - Calculate: reranker_contribution, fusion_contribution, semantic_contribution
  - Implement CLI: `plm-benchmark analyze-integration --dataset gold.json`
  - Output detailed report with recommendations

  **Schema Specifications**:

  CLI Arguments:
  | Argument | Type | Default | Description |
  |----------|------|---------|-------------|
  | --dataset | path | (required) | Path to benchmark JSON |
  | --analysis | string | "all" | One of: complementarity, cascade, ablation, all |
  | --url | string | http://localhost:8000 | Search service URL |
  | --output | path | stdout | Output file for report |

  **Analysis Types**:

  1. **Complementarity Analysis**:
  
  ComplementarityResult dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | overlap_at_10 | float | % of results in both BM25 and semantic top-10 |
  | overlap_at_50 | float | % overlap at larger k |
  | bm25_unique_hits | int | Relevant docs found only by BM25 |
  | semantic_unique_hits | int | Relevant docs found only by semantic |
  | error_correlation | float | % of queries where both fail |
  | fusion_potential | float | 1 - error_correlation (higher = better) |

  Target values:
  - overlap_at_10 < 0.5 (good diversity)
  - error_correlation < 0.3 (uncorrelated errors)

  2. **Cascade Evaluation**:

  CascadeResult dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | bm25_recall_at_100 | float | BM25 alone recall |
  | semantic_recall_at_100 | float | Semantic alone recall |
  | rrf_recall_at_50 | float | After fusion |
  | rrf_mrr | float | MRR after fusion |
  | rerank_mrr | float | MRR after reranking (if enabled) |
  | stage_contributions | dict | Delta per stage |

  This measures actual pipeline behavior, not theoretical.

  3. **Ablation Waterfall**:

  AblationConfig dataclass:
  | Config Name | Description |
  |-------------|-------------|
  | full_pipeline | All components enabled |
  | no_reranker | Disable reranking |
  | no_query_expansion | Disable term expansion |
  | semantic_only | Disable BM25/SPLADE |
  | bm25_only | Disable semantic |
  | no_rewrite | Disable query rewriting |

  AblationResult dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | config_name | string | Which configuration |
  | hit_at_5 | float | Hit@5 for this config |
  | mrr | float | MRR for this config |
  | delta_vs_full | float | Difference from full pipeline |

  **Ablation Implementation Note**:
  Since benchmark must use HTTP API, ablation requires multiple API calls with different parameters:
  - `use_rerank=false` for no_reranker
  - `use_rewrite=false` for no_rewrite
  - For semantic_only/bm25_only, need API parameter support (may require Task 1 extension)

  IntegrationReport dataclass:
  | Field | Type | Description |
  |-------|------|-------------|
  | run_timestamp | string | ISO8601 |
  | dataset_used | string | Dataset path |
  | complementarity | ComplementarityResult | Retriever diversity |
  | cascade | CascadeResult | Stage contributions |
  | ablation | list[AblationResult] | Per-config results |
  | recommendations | list[string] | Actionable insights |

  **Recommendation Logic**:
  - If error_correlation > 0.4: "Consider adding retriever with different failure modes"
  - If reranker_contribution < 0.02: "Reranker adds minimal value, consider disabling for latency"
  - If semantic_only > bm25_only: "Semantic retrieval is primary contributor"
  - If overlap_at_10 > 0.7: "Retrievers are too similar, low fusion benefit"

  **Must NOT do**:
  - Modify retrieval pipeline for testing (test production config)
  - Skip any Tier 5 analysis type

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: Complex statistical analysis, requires understanding of retrieval theory
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 3)
  - **Blocks**: Task 9
  - **Blocked By**: Task 7

  **References**:
  - Oracle's Tier 5 design in `.sisyphus/drafts/benchmark-framework.md` lines 629-796
  - Complementarity analysis: lines 636-658
  - Cascade evaluation: lines 659-691
  - Ablation waterfall: lines 699-721

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Complementarity analysis produces metrics
    Tool: Bash
    Preconditions: Benchmark runner complete
    Steps:
      1. plm-benchmark analyze-integration --dataset gold.json --analysis complementarity
      2. Assert: output contains "overlap_at_10" (float 0-1)
      3. Assert: output contains "error_correlation" (float 0-1)
      4. Assert: output contains "fusion_potential" (float 0-1)
      5. Assert: overlap_at_10 < 0.7 (reasonable complementarity)
    Expected Result: Complementarity metrics calculated
    Evidence: Analysis output saved

  Scenario: Ablation waterfall shows marginal contributions
    Tool: Bash
    Preconditions: Full benchmark data
    Steps:
      1. plm-benchmark analyze-integration --dataset gold.json --analysis ablation
      2. Assert: output contains "full_pipeline_mrr"
      3. Assert: output contains "semantic_only_mrr"
      4. Assert: output contains "bm25_only_mrr"
      5. Assert: output contains "reranker_contribution" (delta)
    Expected Result: Clear marginal contribution per component
    Evidence: Ablation report saved
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add Tier 5 integration evaluation suite`
  - Files: `src/plm/benchmark/integration/`

---

### Phase 4: Tests and Documentation

- [x] 9. Tests and Documentation

  **What to do**:
  - Write unit tests for all benchmark components:
    - URL mapping: test normalization edge cases
    - Tier assignment: test each tier threshold
    - Quote matching: test with real SO HTML samples
    - Metrics calculation: verify Hit@k, MRR formulas
  - Write integration test: end-to-end from SO data to metrics
  - Create README in `src/plm/benchmark/README.md` with:
    - Architecture overview
    - CLI usage examples
    - Configuration options
    - Interpreting results
  - Update AGENTS.md with benchmark module entry

  **Must NOT do**:
  - Skip edge case tests
  - Document features that don't exist

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation-heavy task with test writing
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Final (Wave 4)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 8

  **References**:
  - Existing test patterns: `tests/search/`
  - Existing docs: `src/plm/search/AGENTS.md`

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All tests pass
    Tool: Bash
    Preconditions: All implementation complete
    Steps:
      1. pytest tests/benchmark/ -v
      2. Assert: exit code 0
      3. Assert: >= 20 tests run
      4. Assert: no failures
    Expected Result: Full test coverage passes
    Evidence: pytest output captured

  Scenario: Documentation is complete
    Tool: Bash
    Preconditions: README written
    Steps:
      1. Check src/plm/benchmark/README.md exists
      2. Assert: contains "## Usage" section
      3. Assert: contains "## Configuration" section
      4. Assert: contains CLI examples for all commands
    Expected Result: Usable documentation
    Evidence: README contents verified
  ```

  **Commit**: YES
  - Message: `docs(benchmark): add tests and documentation`
  - Files: `tests/benchmark/`, `src/plm/benchmark/README.md`
  - Pre-commit: `pytest tests/benchmark/`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(search): add explain mode to query API` | app.py, retriever.py | pytest tests/search/test_explain_mode.py |
| 2 | `feat(benchmark): add SOTorrent data extraction` | src/plm/benchmark/extraction/ | File exists with 5000+ entries |
| 3, 4, 5 | `feat(benchmark): add mapping, tier, and matching utilities` | src/plm/benchmark/{mapping,tier,matching}/ | Unit tests |
| 6a | `feat(benchmark): add deterministic signal extraction` | src/plm/benchmark/extraction/signals.py | Deterministic output verified |
| 6b | `feat(benchmark): add parallel LLM generator agents` | src/plm/benchmark/generation/ | Batches generated |
| 6c | `feat(benchmark): add code-based verifier` | src/plm/benchmark/verification/ | Deterministic verification |
| 6d | `feat(benchmark): add regeneration loop` | src/plm/benchmark/regeneration/ | <5% quarantine rate |
| 6e | `feat(benchmark): add final assembly with audit` | src/plm/benchmark/assembly/, artifacts/ | Datasets exist with full audit |
| 7 | `feat(benchmark): add benchmark runner` | src/plm/benchmark/runner/ | Produces metrics |
| 8 | `feat(benchmark): add integration evaluation` | src/plm/benchmark/integration/ | Analysis runs |
| 9 | `docs(benchmark): add tests and documentation` | tests/benchmark/, README.md | pytest passes |

---

## Success Criteria

### Verification Commands
```bash
# API explain mode works
curl -X POST localhost:8000/query -d '{"query":"test","k":5,"explain":true}' | jq '.results[0].debug_info'

# Dataset sizes meet targets
jq 'length' artifacts/benchmark/datasets/gold.json  # >= 400
jq 'length' artifacts/benchmark/datasets/silver.json  # >= 1500

# Benchmark produces metrics
plm-benchmark evaluate --dataset gold.json --k 10 | jq '.mrr'  # float 0-1

# All tests pass
pytest tests/benchmark/ -v  # 0 failures
```

### Final Checklist
- [ ] Explain mode returns score breakdown (bm25, semantic, rrf)
- [ ] GOLD tier contains only provably-correct pairs
- [ ] Every benchmark result traceable to SO source
- [ ] Metrics match existing POC calculation methods
- [ ] No direct retriever instantiation in benchmark code
- [ ] Verification is code-based (deterministic, no LLM)
- [ ] Quarantine rate < 5% (most cases pass verification)
- [ ] All audit metadata present on every case
- [ ] All 13 tasks (1-5, 6a-6e, 7-9) complete with passing tests
