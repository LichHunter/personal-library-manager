# Learnings - Realistic Questions Benchmark

## [2026-01-27] Task 1: Path Mapping
- Simple string replace works for 97.6% of kubefix dataset
- Path pattern: `/content/en/docs/X/Y.md` → `X_Y.md`
- 2,502/2,563 questions have matching docs in corpus

## [2026-01-27] Task 2: Transformation & Quality
- TRANSFORMATION_PROMPT_V2 has 8 concrete examples
- Claude Haiku used for transformation (cheap, fast)
- 5 quality heuristics: originality, phrasing, conciseness, realism, not "What is"
- Retry logic: 3 attempts with 1s delay
- JSON parsing handles markdown code blocks

## [2026-01-27] Task 3: Prompt Iteration Test
- Tested on 20 diverse samples
- Successful transforms: 0/20
- Pass rate: 0/20 (0.0%)

## [2026-01-27] Task 3: Prompt Iteration Test
- Tested on 20 diverse samples
- Successful transforms: 0/20
- Pass rate: 0/20 (0.0%)

## [2026-01-27] Task 4: Generate Realistic Questions
- Implemented `generate_realistic_questions(n=200, output_path=None)` function
- Function loads kubefix dataset and filters to 2,502 valid questions with matching docs
- Samples exactly n questions with seed=42 for reproducibility
- Transforms each question using Claude Haiku with progress logging every 20
- Evaluates quality using 5 heuristics (originality, phrasing, conciseness, realism, not "What is")
- Saves output to JSON with metadata: source, model, prompt_version, total, high_quality count
- Added `--generate N` CLI flag to main()
- Output structure: metadata + questions array with all required fields
- Code verified: syntax check passed, imports successful, CLI flag registered
- Ready for execution with ANTHROPIC_API_KEY environment variable

## [2026-01-27] Task 5: Retrieval Benchmark Implementation
- Implemented `run_retrieval_benchmark(questions_file=None)` function
- Loads full K8s corpus from `corpus/kubernetes/` (1,569 docs expected)
- Chunks with MarkdownSemanticStrategy (target=400, min=50, max=800)
- Initializes enriched_hybrid_llm strategy with BGE embedder
- Indexes all chunks (expects ~8000+ chunks)
- For each question, retrieves top-5 and checks if expected doc_id appears
- Records: hit@1, hit@5, rank, latency_ms for each query
- Calculates summary metrics: hit_at_1 rate, hit_at_5 rate, MRR
- Saves results to timestamped folder: `results/realistic_<YYYY-MM-DD_HHMMSS>/retrieval_results.json`
- Added `--run-benchmark` and `--questions-file PATH` CLI flags
- Code verified: syntax check passed, CLI help shows new flags
- Ready for execution (requires questions file from Task 4)

## [2026-01-27] Task 3: Prompt Iteration Test
- Tested on 20 diverse samples
- Successful transforms: 20/20
- Pass rate: 15/20 (75.0%)
- Common issues: q1 doesn't start with problem language (5)

## [2026-01-27] Task 3: Prompt Iteration Test
- Tested on 20 diverse samples
- Successful transforms: 20/20
- Pass rate: 19/20 (95.0%)
- Common issues: q1 doesn't start with problem language (1)

## [2026-01-27] Prompt Improvement: Problem-Oriented Phrasing

### Change Made
Strengthened `TRANSFORMATION_PROMPT_V2` guidelines to emphasize problem language:
- Added explicit requirement: "MUST start with problem language: how, why, can't, my, is, does, getting, error, when, which"
- Added explicit avoidance list: "AVOID: Starting with imperatives (mount, block, create), desires (want to, need to), or advice-seeking (best practices)"

### Results
- **Pass rate: 95% (19/20)** ✅ (target: ≥80%)
- Improvement: +20% from baseline 75% (15/20)
- Only 1 failure: "restore my database from an old backup" (missing problem language prefix)

### Key Insight
The explicit prohibition of imperative verbs and desire-seeking language was crucial. The original prompt had weak guidance ("Avoid starting with 'What is'") which wasn't strong enough. The new explicit list of forbidden patterns (mount, block, create, want to, need to, best practices) directly addresses the failing pattern.

### Recommendation
This prompt version is production-ready for realistic question generation. The 95% pass rate provides high-quality training data for retrieval benchmarks.

## [2026-01-27] Task 4: Generate Questions
- Generated 200 questions
- High quality: 198 (99.0%)
- Output: /home/fujin/Code/personal-library-manager/poc/chunking_benchmark_v2/corpus/kubernetes/realistic_questions.json

## [2026-01-27] Task 5: Retrieval Benchmark
- Corpus: 1569 docs, 7269 chunks
- Hit@1: 20.25%, Hit@5: 40.75%, MRR: 0.2740
- Results: /home/fujin/Code/personal-library-manager/poc/chunking_benchmark_v2/results/realistic_2026-01-27_110955/retrieval_results.json

## [2026-01-27] Task 6: Failure Analysis Report
- Report generated for: results/realistic_2026-01-27_110955
- Total failures: 237
- Categories: VOCABULARY_MISMATCH=237, RANKING_ERROR=0, CHUNKING_ISSUE=0, EMBEDDING_BLIND=0

## [2026-01-27] Task 6: Failure Analysis Report
- Report generated for: results/realistic_2026-01-27_110955
- Total failures: 237
- Categories: VOCABULARY_MISMATCH=237, RANKING_ERROR=0, CHUNKING_ISSUE=0, EMBEDDING_BLIND=0
