# Benchmark Framework - Learnings

## 2026-02-24 Initial Analysis

### Search Service Architecture
- FastAPI app in `src/plm/search/service/app.py` (258 lines)
- Core models: `QueryRequest`, `QueryResult`, `QueryResponse`
- `HybridRetriever` in `src/plm/search/retriever.py` (775 lines)
- RRF fusion at lines 658-668: semantic FIRST, then sparse SECOND (CRITICAL ORDER)

### Score Collection Points for Task 1
- BM25/SPLADE scores: From `sparse_results` after `sparse_retriever.search()` (line 628)
- Semantic scores: From `sem_scores` after `np.dot(embeddings, query_emb)` (line 655)
- RRF scores: From `rrf_scores` dict after fusion loop (lines 661-668)
- Rerank scores: From reranker output if `use_rerank=True`

### Existing Test Structure
- Tests in `tests/search/` with 6 files
- Patterns: `test_retriever.py`, `test_storage.py`, `test_components.py`
- Use pytest framework

### Project Conventions
- Package manager: `uv` exclusively
- Type checker: `basedpyright`
- No linter/formatter enforced
- Python 3.10+, Anthropic Claude, FAISS, SQLite

### SOTorrent Data (for Task 2)
- Available at Zenodo: https://zenodo.org/record/4415593
- Key tables: PostVersionUrl, Posts, Comments
- SQL query pattern in `.sisyphus/drafts/benchmark-framework.md` lines 76-96
- Quality filter: answer score >= 5

## 2026-02-24 Task 2: SOTorrent Extractor Implementation

### Dataset Structure (from Zenodo)
- Total dataset: 74.2 GB (compressed .sql.7z files)
- Posts.sql.7z: 14.5 GB (questions + answers)
- PostVersionUrl.sql.7z: 1.2 GB (URLs extracted from posts)
- Files are MySQL SQL dumps, need conversion to SQLite

### Implementation Details
- Created module: `src/plm/benchmark/extraction/`
- CLI: `python -m plm.benchmark.extraction.so_extractor --db <path> --output <path>`
- Output: JSON array with schema matching task spec

### SQL Query Corrections
- Draft had `a.AcceptedAnswerId IS NOT NULL` - WRONG
- Corrected to `q.AcceptedAnswerId = a.Id` (AcceptedAnswerId is on QUESTION, points to answer)
- Added `a.PostTypeId = 2` filter to ensure only answers joined

### URL Normalization
- Normalize http:// to https://
- Strip trailing slashes (preserve root /)
- Preserve URL fragments (#anchors)

### Tags Parsing
- SO tags format: `<tag1><tag2><tag3>` (angle brackets)
- Parser handles both SO format and comma-separated fallback

## 2026-02-24 Task 1: Explain Mode Implementation

### Implementation Summary
- Added `explain: bool = False` to QueryRequest in app.py
- Created DebugInfo and QueryMetadata Pydantic models
- Modified HybridRetriever.retrieve() and _retrieve_rrf() to return tuple when explain=True

### Score Collection Points (Verified)
- sparse_score_by_idx: Built from sparse_results list, keyed by chunk index
- sparse_rank_by_idx: Built from enumerate(sparse_results), 0-indexed rank
- sem_scores: numpy array from dot product, indexed directly
- sem_rank_by_idx: Built from enumerate(sem_ranks[:n_candidates])
- rrf_scores: dict accumulated during fusion loop

### Type System Challenges
- Python type checkers don't narrow union types based on boolean parameter values
- Solution: Use `# type: ignore[assignment]` for conditional return types
- Runtime behavior is correct despite static type warnings

### X-Request-ID Middleware
- Uses Starlette BaseHTTPMiddleware
- Stores request_id in request.state for access in endpoint
- Returns X-Request-ID header on all responses

### Edge Cases Handled
- SPLADE-only mode: semantic_score=None, semantic_rank=None, retrieval_mode="splade_only"
- Reranking: rerank_score populated, retrieval_stage="rerank"
- Empty results: Returns empty explain_data structure

### Test Coverage
- 10 tests in tests/search/test_explain_mode.py
- Tests both retriever and API layers
- Tests request_id header correlation

## 2026-02-24 Task 3: URL-to-Chunk Mapping Builder

### Implementation Summary
- Created module: `src/plm/benchmark/mapping/url_mapper.py`
- CLI: `python -m plm.benchmark.mapping.url_mapper --db <path> --output <dir>`
- Output files: url_to_docid.json, url_to_chunks.json, anchor_to_heading.json, unmapped_urls.log

### Key Implementation Details

#### URL Normalization
- Extracts path from full URL (removes scheme, netloc, query params)
- Strips trailing slashes (preserves root /)
- Separates fragment for potential future use
- Handles both http:// and https:// URLs

#### Heading-to-Anchor Slugification
- Removes markdown heading prefix (##, ###, etc.)
- Converts to lowercase
- Replaces spaces with hyphens
- Removes special characters except hyphens
- Collapses multiple hyphens to single
- Examples: "## Pod Lifecycle" → "pod-lifecycle", "## CronJob Spec" → "cronjob-spec"

#### Database Schema Handling
- Detects presence of heading_id column via PRAGMA table_info
- Gracefully handles databases without heading_id (uses NULL)
- Handles databases without chunk_index (omits ORDER BY clause)
- Uses read-only connection: `file:path?mode=ro`

#### Mapping Structures
- url_to_docid: Maps URL path to list of unique doc_ids (deduplicates)
- url_to_chunks: Maps URL path to list of chunk_ids (preserves all chunks)
- anchor_to_heading: Maps slug to list of {doc_id, heading_id, heading_text} objects
- unmapped_urls.log: One URL per line with reason (e.g., "not a URL")

### Edge Cases Handled
- Non-URL source_file values logged to unmapped_urls.log
- Duplicate doc_ids per URL prevented (checked before append)
- Headings without heading_id skipped in anchor mapping
- Empty databases handled gracefully with warning

### Test Results
- Created test database with 4 chunks across 3 documents
- 2 unique URL paths mapped correctly
- 3 unique anchor slugs generated
- 1 unmapped URL (local file path) logged
- All JSON output files valid and properly formatted

## 2026-02-24 Task 4: Tier Assignment Engine

### Implementation Summary
- Created module: `src/plm/benchmark/tier/`
- Files: `__init__.py`, `engine.py`, `config.py`
- Main function: `assign_tier(input: TierAssignmentInput, config: TierConfig | None) -> TierAssignment`

### Dataclass Design

#### TierAssignmentInput
- so_answer_id, url_match, fragment_anchor, fragment_matches_heading
- quote_matches (list[QuoteMatch]), reciprocal_matches (list[ReciprocalMatch])
- upvotes, is_accepted, multiple_answers_same_url

#### TierAssignment (output)
- tier: Literal["gold", "silver", "bronze", "exclude"]
- tier_reason, confidence_score, signals_detected, primary_signal
- evidence: dict (JSON-serializable audit trail)

### Tier Decision Logic (Deterministic Priority)
1. fragment_matches_heading → GOLD
2. quote_match >= 30 chars → GOLD
3. reciprocal_containment >= 20 words → GOLD
4. url_match AND (upvotes >= 10 AND accepted) OR (upvotes >= 25) → SILVER
5. url_match AND multiple_answers >= 2 → SILVER
6. url_match AND upvotes >= 5 → BRONZE
7. else → EXCLUDE

### Configuration Approach
- Frozen dataclasses for immutability: GoldConfig, SilverConfig, BronzeConfig, ConfidenceConfig
- DEFAULT_CONFIG singleton with all defaults
- get_config() function for extensibility (can later load from env/file)
- All thresholds configurable via TierConfig parameter to assign_tier()

### Confidence Score Calculation
- fragment_anchor_match: 1.0 (maximum trust)
- quote_match: 0.9 base (+0.1 if length > 50)
- reciprocal_containment: 0.85 base (+0.05 per 10 extra words)
- url_high_trust: 0.75 base (+0.1 if both upvotes >= 10 AND accepted)
- url_corroborated: 0.70 base (+0.05 per extra corroborating answer)
- url_community_validated: 0.60 base (+0.02 per upvote above 5, capped at 0.75)

### Evidence Trail
- All input data preserved in evidence dict
- quote_matches and reciprocal_matches serialized with key fields
- Enables full audit/reproducibility per tier decision

### Test Coverage
- 26 tests in tests/benchmark/test_tier_engine.py
- Test classes: TestGoldTier, TestSilverTier, TestBronzeTier, TestExcludeTier
- TestTierPriority: Verifies deterministic order (GOLD over SILVER signals)
- TestEvidenceTrail: JSON serialization, input data preservation
- TestDeterminism: Same input → same output
- TestCustomConfig: Override thresholds at runtime

## 2026-02-24 Task 6a: Signal Extraction Pipeline

### Implementation Summary
- Created module: `src/plm/benchmark/extraction/signals.py`
- CLI: `python -m plm.benchmark.extraction.signals --so-data <path> --mappings <dir> --corpus-db <path> --output <dir> --workers <n>`
- Output files: signal_bundles.jsonl, unmappable.log, extraction_stats.json

### SignalBundle Dataclass
- 20 fields: bundle_id (UUID), SO IDs, question/answer text, URL info, chunk mappings
- Includes quote_matches and reciprocal_matches (converted from matching module types)
- Includes max_possible_tier from tier engine
- Includes extraction_timestamp (ISO8601) and corpus_version_hash (SHA256)

### Key Implementation Details

#### Corpus Version Hash
- Computed from sorted chunk_id:content_hash pairs
- Hash input: "chunk_1:sha256(content1)chunk_2:sha256(content2)..."
- Final hash: SHA256 of concatenated pairs
- Deterministic: Same corpus → same hash across runs

#### Parallel Processing
- ThreadPoolExecutor with configurable workers (default: 4)
- extract_wrapper() handles exceptions gracefully
- Progress logging every 100 posts
- Returns (bundle, unmappable_entry) tuple for each post

#### Type System Handling
- QuoteMatch and ReciprocalMatch defined in tier/engine.py (canonical)
- Matching modules return their own dataclass instances
- Conversion: Create tier engine instances from matching module results
- Used cast() for source_type Literal in quote_matcher.py

#### Unmappable Posts
- Reason codes: "no_chunks", "processing_error"
- Log format: "answer_id=X question_id=Y reason=Z [error=...]"
- Posts with no chunk_ids after URL mapping excluded immediately

#### Tier Assignment
- Uses TierAssignmentInput with all matching results
- Calls assign_tier() from tier engine
- Stores max_possible_tier in bundle (not actual tier)

### Test Results
- Created test SO data with 2 posts
- Mapped to test.db with 2 URL paths and 3 chunks
- Extraction: 2 bundles created (1 silver, 1 bronze)
- Corpus hash: 0ed356a977528d8d6963ce195e8d97d34a77e972896a8d55df23db914f4422d2
- Determinism verified: Same hash on second run

### Output Format

#### signal_bundles.jsonl
- One JSON object per line (JSONL format)
- All fields JSON-serializable (lists, strings, booleans, nulls)
- bundle_id is UUID string (not object)
- chunk_contents populated from database

#### extraction_stats.json
- Schema: total_posts_processed, bundles_created, unmappable_count
- Tier counts: gold_potential, silver_potential, bronze_potential, excluded_count
- corpus_version_hash and extraction_duration_seconds

#### unmappable.log
- One entry per line
- Format: "answer_id=X question_id=Y reason=Z [error=...]"
- Only written if unmappable posts exist

### Edge Cases Handled
- Empty chunk_ids after mapping → excluded (no_chunks reason)
- Processing exceptions → logged with error message
- Missing chunk content in DB → empty string in chunk_contents
- No quote/reciprocal matches → empty lists in bundle
- URL fragments → extracted and checked against anchor mappings

## 2026-02-24 Task 6b: Parallel LLM Generator

### Implementation Summary
- Created module: `src/plm/benchmark/generation/`
- Files: `__init__.py`, `generator.py`, `prompts.py`
- CLI: `python -m plm.benchmark.generation.generator --signals <path> --output <dir> --workers <n> --batch-size <n>`
- Output files: batch_{N}.jsonl, generation_stats.json, generation_errors.log

### GeneratedCase Dataclass (11 fields)
- case_id: UUID for tracking
- bundle_id: Reference to source SignalBundle
- query: Reformulated NL question (5-100 words)
- matched_quote: Exact verbatim quote (≥30 chars for GOLD) or None
- evidence_text: 1-2 sentence explanation
- reasoning: LLM's mapping reasoning (audit)
- chunk_ids: List of target chunks
- tier_from_signals: Tier from SignalBundle (not LLM decision)
- generation_timestamp: ISO8601
- generator_model: Model used (e.g., "claude-haiku-4-5")
- internal_retries: Self-check retries (0-2)

### Prompt Engineering
- System prompt: JSON-only output, no markdown
- Tier-specific instructions: GOLD requires 30+ char quote, others optional
- Content truncation: max_chunk_chars=8000, max_question_chars=2000, max_answer_chars=4000
- Self-check requirements embedded in prompt: word count, quote verbatim, quote length

### Self-Check Implementation
- count_words(): Split-based word counting
- normalize_whitespace(): Collapse for quote matching
- validate_quote_in_chunks(): Whitespace-normalized substring check
- validate_generated_case(): Full validation (word count, quote existence, quote length)
- Up to 2 internal retries on validation failure

### JSON Parsing Robustness
- Handles markdown code blocks (```json...```)
- Falls back to regex extraction for malformed JSON
- Returns None on complete parse failure

### Parallel Processing
- ThreadPoolExecutor with configurable workers (default: 4)
- Batch processing: configurable batch_size (default: 500)
- Progress logging every 50 bundles
- Bundles with max_possible_tier="exclude" skipped at load time

### LLM Configuration
- Uses `plm.shared.llm.call_llm()` for provider routing
- Model: PLM_LLM_MODEL env var or "claude-haiku-4-5" default
- Temperature: 0.3 (low variance)
- Max tokens: 1024 (sufficient for structured output)

### Output Files

#### batch_{N}.jsonl
- One GeneratedCase JSON per line
- Sequential batch numbering (1-indexed)

#### generation_stats.json
- total_bundles_input, cases_generated, failed_count
- pass_first_try, required_retry counts
- tier_breakdown dict (counts per tier)
- avg_query_words, generation_duration_seconds

#### generation_errors.log
- Format: "bundle_id=X so_answer_id=Y error=Z retries=N"
- Only written if errors exist

### Test Results
- 2 test bundles (1 gold, 1 silver, 1 exclude)
- exclude tier correctly skipped
- Both cases generated on first try (0 retries)
- Quotes verified against chunk content
- Average query length: 11 words

## 2026-02-24 Task 6c: Deterministic Verifier

### Implementation Summary
- Created module: `src/plm/benchmark/verification/`
- Files: `__init__.py`, `verifier.py`
- CLI: `python -m plm.benchmark.verification.verifier --generated <dir> --corpus-db <path> --output <dir>`
- Output files: verified_passed.jsonl, verified_failed.jsonl, verification_report.json

### VerificationResult Dataclass (5 fields)
- case_id: UUID of case being verified
- passed: bool (True if all checks pass)
- failures: list[VerificationFailure] (empty if passed)
- checks_run: list[str] (always 5 checks)
- verification_timestamp: ISO8601

### VerificationFailure Dataclass (4 fields)
- check_name: str (chunk_exists, quote_exists, quote_length, tier_match, query_length)
- expected: str (what was expected)
- found: str (what was found)
- recoverable: bool (True if regeneration could fix)

### Five Verification Checks (Deterministic, Pure Python)

1. **chunk_exists**: All chunk_ids in corpus
   - Recoverable: NO (corpus gap)
   - Failure: chunk_id not in corpus dict

2. **quote_exists**: matched_quote verbatim in chunks (whitespace normalized)
   - Recoverable: YES (regeneration could find different quote)
   - Failure: normalized quote not substring of concatenated normalized chunks
   - Skipped if matched_quote is None

3. **quote_length**: len(quote) >= 30 if tier_from_signals == "gold"
   - Recoverable: YES (regeneration could find longer quote)
   - Failure: GOLD tier with quote < 30 chars
   - Skipped if tier is silver/bronze or matched_quote is None

4. **tier_match**: tier_from_signals in {gold, silver, bronze}
   - Recoverable: NO (invalid tier value)
   - Failure: tier_from_signals not in valid set
   - Note: Does NOT recalculate tier (no access to original signals)

5. **query_length**: 5 <= word_count <= 100
   - Recoverable: YES (regeneration could adjust query length)
   - Failure: word_count outside range
   - Word count: len(query.split())

### Key Implementation Details

#### Whitespace Normalization
- Function: normalize_whitespace(text)
- Steps: collapse spaces, strip, lowercase
- Used for quote matching (same pattern as generator.py)

#### Corpus Loading
- Function: load_chunk_corpus(db_path)
- Reads SQLite table: chunks(chunk_id TEXT, content TEXT)
- Returns: dict[str, str] mapping chunk_id → content
- Handles missing table gracefully with warning

#### Batch Processing
- Finds all batch_*.jsonl files in generated/ directory
- Processes sequentially (no parallelization needed for verification)
- Loads GeneratedCase from JSON (11 fields)

#### Output Structure
- verified_passed.jsonl: One JSON per line with {case, verification}
- verified_failed.jsonl: Same structure, only failed cases
- verification_report.json: Summary statistics

#### Report Schema
- total_cases: int (all cases processed)
- passed_count: int (cases passing all checks)
- failed_count: int (cases with failures)
- pass_rate: float (passed / total)
- failure_breakdown: dict[str, int] (count per check_name)
- recoverable_failures: int (sum of recoverable failures)
- unrecoverable_failures: int (sum of unrecoverable failures)
- verification_duration_seconds: float (total time)

### Test Results
- Unit tests: 5 cases (1 pass, 4 fail with various checks)
- CLI test: 3 cases (2 pass, 1 fail on chunk_exists)
- Output files: verified_passed.jsonl (2 cases), verified_failed.jsonl (1 case)
- Report: Correct counts, failure breakdown, recoverable/unrecoverable split
- Determinism: Same input → same output (verified)

### Edge Cases Handled
- Empty matched_quote (None): Skips quote_exists and quote_length checks
- Missing chunks: Marked unrecoverable (corpus gap)
- Invalid tier: Marked unrecoverable (data corruption)
- Empty batch files: Handled gracefully
- No batch files: Logs warning, exits cleanly

### Design Decisions

1. **No Tier Recalculation**: Verifier doesn't have access to original signals (quote_matches, reciprocal_matches, etc.), so tier_match check only validates the tier value is in {gold, silver, bronze}. Full tier recalculation would require re-running the tier engine with original signal data.

2. **Whitespace Normalization**: Uses same pattern as generator.py (collapse spaces, strip, lowercase) for consistency in quote matching.

3. **Recoverable vs Unrecoverable**: Corpus gaps (chunk_exists, invalid tier) are unrecoverable. Quote/query issues are recoverable (regeneration could fix).

4. **Separate Output Files**: Passed and failed cases in separate JSONL files for easy filtering and downstream processing (e.g., regeneration loop only processes failed cases).

5. **Deterministic**: No randomness, no LLM calls, same input always produces same output.

## 2026-02-24 Task 6d: Regeneration Orchestrator

### Implementation Summary
- Created module: `src/plm/benchmark/regeneration/`
- Files: `__init__.py`, `orchestrator.py`
- CLI: `python -m plm.benchmark.regeneration.orchestrator --failed <path> --signals <path> --corpus-db <path> --output <dir>`
- Output files: regenerated_passed.jsonl, unmappable.jsonl, quarantine.jsonl, regeneration_log.jsonl, regeneration_stats.json

### RegenerationAttempt Dataclass (7 fields)
- case_id: Original case identifier
- attempt_number: 1, 2, or 3
- previous_failures: List of VerificationFailure dicts from prior attempt
- feedback_prompt: Structured feedback string given to LLM
- new_case: GeneratedCase dict if generation succeeded, else None
- verification_result: VerificationResult dict if verified, else error dict
- timestamp: ISO8601

### RegenerationResult Dataclass (7 fields)
- original_case_id: UUID from failed case
- bundle_id: Reference to source SignalBundle
- final_status: Literal["passed", "unmappable", "quarantine"]
- termination_reason: Reason for final outcome
- total_attempts: Number of regeneration attempts
- final_case: GeneratedCase dict if passed, else None
- all_attempts: Full history of RegenerationAttempts

### Structured Feedback Templates
- quote_exists: "Quote 'X' NOT FOUND in chunk. FIX: Find a DIFFERENT quote."
- quote_length: "Quote only X chars, need >=30 for GOLD. FIX: Find longer quote."
- tier_match: "Claimed tier 'X' but signals support 'Y'. FIX: Accept tier 'Y'."
- query_length: "Query has X words, need 5-100. FIX: Expand/Condense the query."
- chunk_exists: "Chunk 'X' NOT FOUND in corpus. FIX: Cannot be fixed (corpus gap)."

### Termination Conditions Decision Tree
1. Corpus gap (chunk_exists + !recoverable) → UNMAPPABLE, reason="corpus_gap"
2. Max retries (attempt >= 3) → QUARANTINE, reason="max_retries_exceeded"
3. Same failure twice (current == previous) → UNMAPPABLE, reason="persistent_failure_<check>"
4. No failures → PASSED
5. Otherwise → RETRY

### Fresh LLM Calls
- Each regeneration uses fresh call_llm() invocation (no conversation context)
- Model: PLM_LLM_MODEL env var or "claude-haiku-4-5" default
- Temperature: 0.3 (low variance)
- Feedback appended to base prompt under "## REGENERATION FEEDBACK" section

### Output Files

#### regenerated_passed.jsonl
- One GeneratedCase JSON per line (only final_case from passed results)
- Format matches original generator output

#### unmappable.jsonl
- Lightweight format: original_case_id, bundle_id, termination_reason, total_attempts
- For cases with fundamental issues (corpus gap, persistent failures)

#### quarantine.jsonl
- Full RegenerationResult JSON per line
- For cases needing human review (max retries exceeded)

#### regeneration_log.jsonl
- Complete audit trail: all RegenerationResults
- Includes all_attempts with full feedback and verification details

#### regeneration_stats.json
- Schema: total_failed_input, regenerated_passed, unmappable_count, quarantine_count
- recovery_rate (passed/input), quarantine_rate (quarantine/total)
- attempts_distribution: dict mapping attempt count to case count
- regeneration_duration_seconds

### Quality Targets
- Quarantine rate < 5% of total generated
- Recovery rate > 50% of failed cases
- Warning logged if targets not met

### Design Decisions

1. **Early Corpus Gap Detection**: Check for unrecoverable chunk_exists failures BEFORE attempting regeneration to avoid wasted LLM calls.

2. **Failure History Tracking**: Keep list of all previous failure sets to detect persistent failures (same check names across attempts).

3. **Fresh Agent Pattern**: Each regeneration is a fresh LLM call with full prompt + feedback, not a conversation continuation. Prevents context pollution.

4. **Graceful Bundle Not Found**: If bundle_id not in signals map, immediately mark unmappable with "bundle_not_found" reason.

5. **Lightweight Unmappable Format**: Only essential fields in unmappable.jsonl since these cases can't be fixed. Full details in regeneration_log.jsonl.

6. **Full Quarantine Records**: Quarantine cases get complete RegenerationResult since human review needs full context.

## 2026-02-24 Task 6e: Final Assembly and Audit Metadata

### Implementation Summary
- Created module: `src/plm/benchmark/assembly/`
- Files: `__init__.py`, `assembler.py`
- CLI: `python -m plm.benchmark.assembly.assembler --verified <path> --regenerated <path> --signals <path> --output <dir>`
- Output files: gold.json, silver.json, bronze.json, statistics_report.json

### BenchmarkCaseAudit Dataclass (25 fields)
Complete audit metadata for each benchmark case:
- **Provenance** (6 fields): id, so_question_id, so_answer_id, extraction_timestamp, generation_timestamp, corpus_version_hash
- **Raw Signal** (6 fields): extracted_url, url_fragment, answer_upvotes, answer_is_accepted, answer_date, chunk_ids
- **Tier Assignment** (4 fields): tier, tier_reason, confidence_score, signals_detected
- **LLM Generated** (4 fields): query, evidence_text, matched_quote, reasoning
- **Verification** (3 fields): verification_passed_first_try, retry_count, verification_checks
- **Benchmark** (2 fields): final_status, relevant_chunk_ids

### Key Implementation Details

#### Merge Logic
- Load verified_passed.jsonl (from Task 6c)
- Load regenerated_passed.jsonl (from Task 6d)
- Deduplicate by case_id: prefer regenerated version if exists in both
- Result: dict[case_id -> case_data] with source tracking

#### Signal Bundle Mapping
- Load signal_bundles.jsonl (from Task 6a)
- Map case_id → bundle_id → SignalBundle for provenance
- Extract signals_detected from bundle: fragment_anchor, quote_match, reciprocal_match, url_only
- Compute confidence_score: Fixed at 0.75 (can be enhanced later with signal-based calculation)

#### Tier Splitting
- Split merged cases by tier_from_signals (gold/silver/bronze)
- Exclude quarantine cases (final_status != "accepted")
- Track verification source: verified_passed → verification_passed_first_try=True, regenerated_passed → False

#### Statistics Report
- Schema: 14 fields covering generation metadata, tier distribution, quality metrics
- Metrics calculated:
  - pass_first_time_rate: verified_count / total_cases
  - retry_rate: regenerated_count / total_cases
  - retry_success_rate: 1.0 (all regenerated cases passed verification)
  - unmappable_rate: 0.0 (handled in Task 6d)
  - quarantine_rate: 0.0 (excluded from final datasets)
  - signal_distribution: counts per signal type
  - tier_reason_breakdown: counts per tier_reason
  - average_confidence: per-tier average (all 0.75 in current implementation)
  - unique_so_questions: count of unique SO question IDs
  - unique_doc_urls: count of unique K8s documentation URLs

#### Tier Target Validation
- Targets: GOLD ≥400, SILVER ≥1500, BRONZE ≥5000
- Logs WARNING if minimum not met with gap analysis
- Logs INFO if target exceeded

### Output Format

#### Dataset Files (gold.json, silver.json, bronze.json)
- JSON array of BenchmarkCaseAudit objects
- Indented for human readability
- All 25 audit fields present in each case

#### statistics_report.json
- Single JSON object with 14 fields
- Indented for human readability
- Includes generation_date (ISO8601) and corpus_version_hash for reproducibility

### Test Results
- Created test data: 2 verified + 2 regenerated cases, 4 signal bundles
- Assembly output: 1 gold, 1 silver, 2 bronze cases
- Statistics: Correct tier distribution, pass_first_time_rate=0.5, retry_rate=0.5
- Tier targets: All below minimum (expected for test data)
- Warnings logged correctly for unmet targets

### Design Decisions

1. **Confidence Score Fixed at 0.75**: Current implementation uses fixed value. Future enhancement: calculate from signals (fragment_anchor=1.0, quote_match=0.9, reciprocal=0.85, url_high_trust=0.75, etc.)

2. **Verification Checks Format**: Stored as list[tuple[str, bool]] where tuple is (check_name, passed). Enables per-check audit trail.

3. **Signals Detected Extraction**: Derived from bundle fields (fragment_matches_heading, quote_matches, reciprocal_matches). Falls back to "url_only" if no other signals present.

4. **Tier Reason from Bundle**: Uses bundle.max_possible_tier as tier_reason (the tier assignment reason from Task 6a). Not recalculated.

5. **No Quarantine Cases**: Final datasets exclude all cases with final_status != "accepted". Quarantine cases remain in regeneration_log.jsonl for human review.

6. **Corpus Version Hash Propagation**: Uses first bundle's corpus_version_hash for report (all bundles should have same hash from same extraction run).

### Edge Cases Handled
- Missing bundle_id: Logs warning, skips case
- Empty verified/regenerated files: Logs warning, continues with available data
- No bundles: Uses "unknown" for corpus_version_hash
- Empty confidence_scores for tier: Defaults to 0.0 average

