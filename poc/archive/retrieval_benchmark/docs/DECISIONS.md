# Decision Log

This document tracks design decisions made during the development of the retrieval benchmark framework. Each decision includes context, options considered, the choice made, and rationale.

---

## DEC-001: Separate POC for Benchmark Framework

**Date**: 2026-01-21

**Context**: We need a framework to benchmark retrieval strategies. Should this extend `poc/test_data/` or be a new POC?

**Options**:
1. Extend `poc/test_data/` - Add benchmark code to existing POC
2. New POC `poc/retrieval_benchmark/` - Separate concerns

**Decision**: Option 2 - New POC

**Rationale**:
- Clear separation of concerns (data generation vs benchmarking)
- Different dependencies (test_data is lightweight, benchmark needs vector DBs)
- Easier to understand and maintain
- test_data can be used by other projects without benchmark overhead

---

## DEC-002: Primary Vector Database

**Date**: 2026-01-21

**Context**: Which vector database to use as the primary implementation?

**Options**:
1. ChromaDB - Popular, mature, good Python API
2. LanceDB - Newer, columnar, fast
3. sqlite-vec - SQLite extension, simple
4. Qdrant - Feature-rich, client-server
5. In-memory only - NumPy + brute force

**Decision**: ChromaDB as primary, with in-memory baseline and sqlite-vec as secondary

**Rationale**:
- ChromaDB has the best documentation and community support
- In-memory baseline needed for comparison (no DB overhead)
- sqlite-vec aligns with our eventual production choice (SQLite-based)
- LanceDB kept as future option if we need better performance
- Qdrant overkill for local-first use case

---

## DEC-003: RAPTOR Implementation

**Date**: 2026-01-21

**Context**: Should we port our existing `poc/raptor_test/` code or reimplement?

**Options**:
1. Port existing code - Faster, but carries technical debt
2. Clean reimplementation - More work, but cleaner integration

**Decision**: Option 2 - Clean reimplementation

**Rationale**:
- Existing code was exploratory, not designed for this framework
- Need to fit the `RetrievalStrategy` protocol
- Opportunity to fix issues found during testing
- Cleaner code = easier to debug when comparing strategies

---

## DEC-004: Execution Model

**Date**: 2026-01-21

**Context**: Should benchmark configurations run in parallel or sequentially?

**Options**:
1. Parallel execution - Faster overall
2. Sequential execution - Simpler, more predictable

**Decision**: Option 2 - Sequential execution

**Rationale**:
- Avoids race conditions in LLM calls (Ollama)
- Prevents GPU memory contention for embeddings
- Easier to debug failures
- More predictable timing measurements
- Parallel can be added later if needed

---

## DEC-005: Output Format

**Date**: 2026-01-21

**Context**: What format for benchmark results?

**Options**:
1. JSON only - Rich structure, harder to compare
2. CSV only - Easy to compare, less structure
3. Both JSON and CSV - Best of both worlds
4. Database (SQLite) - Queryable, but overkill

**Decision**: Option 2 - CSV as primary format

**Rationale**:
- Easy to open in spreadsheet for comparison
- Simple to parse programmatically
- Sufficient structure for our needs
- Can always generate JSON from CSV if needed
- Three separate CSVs (index_stats, search_results, summary) provide needed granularity

---

## DEC-006: LLM Provider

**Date**: 2026-01-21

**Context**: Which LLM provider for strategies that need LLM?

**Options**:
1. Ollama only - Local, free, fits 8GB VRAM
2. OpenAI API - Better quality, costs money
3. Multiple providers - More flexibility, more complexity

**Decision**: Option 1 - Ollama only (for now)

**Rationale**:
- Local-first aligns with project goals
- 8GB VRAM constraint rules out large API models anyway
- Can test multiple models (llama3.2:3b, llama3.2:1b, mistral:7b)
- API providers can be added later if needed

---

## DEC-007: Consistency Testing

**Date**: 2026-01-21

**Context**: How to measure result consistency across multiple runs?

**Options**:
1. Fixed number of runs (e.g., always 3)
2. Configurable runs (1, 3, or 5)
3. Run until confidence interval

**Decision**: Option 2 - Configurable runs per query

**Rationale**:
- Different experiments need different precision levels
- Quick tests: 1 run (fast iteration)
- Baseline: 3 runs (reasonable confidence)
- Full: 5 runs (high confidence)
- Configuration-driven keeps the code simple

---

## DEC-008: Embedding Model Selection

**Date**: 2026-01-21

**Context**: Which embedding models to support/test?

**Options**:
1. Single model (all-MiniLM-L6-v2) - Simple
2. Multiple SentenceTransformer models - Compare quality
3. Multiple providers (SBERT + Ollama + OpenAI) - Maximum flexibility

**Decision**: Option 2 - Multiple SentenceTransformer models

**Rationale**:
- SentenceTransformers is fast and local
- Can compare quality vs speed tradeoffs:
  - all-MiniLM-L6-v2: Fast, 384 dims
  - all-mpnet-base-v2: Better quality, 768 dims
- Ollama embeddings not needed (SBERT is better for this use case)
- OpenAI adds cost and latency

---

## DEC-009: Document Structure for LOD Strategies

**Date**: 2026-01-21

**Context**: How should LOD strategies represent document hierarchy?

**Options**:
1. Fixed 3-level hierarchy (doc → section → chunk)
2. Dynamic depth based on document structure
3. Configurable levels

**Decision**: Option 1 - Fixed 3-level hierarchy

**Rationale**:
- Wikipedia articles have consistent structure (title → headings → content)
- Simpler implementation and comparison
- Dynamic depth adds complexity without clear benefit for our test data
- Can be extended later if needed for different document types

---

## DEC-010: Chunk Overlap Strategy

**Date**: 2026-01-21

**Context**: How to handle chunk boundaries?

**Options**:
1. No overlap - Simple, but loses context at boundaries
2. Fixed overlap (e.g., 50 tokens) - Preserves some context
3. Sentence-aware chunking - Respects sentence boundaries

**Decision**: Option 2 - Fixed overlap, configurable per strategy

**Rationale**:
- Balance between simplicity and context preservation
- Overlap helps queries that span chunk boundaries
- Keep it simple for now; sentence-aware can be added later
- Different strategies can use different overlap values

---

## DEC-011: Ground Truth Source

**Date**: 2026-01-21

**Context**: Where does ground truth come from?

**Options**:
1. Generate with this framework
2. Use ground truth from test_data POC
3. Manual curation

**Decision**: Option 2 - Use existing ground truth from test_data

**Rationale**:
- test_data already has ground truth generation
- Separation of concerns (data generation vs benchmarking)
- Avoids duplicating LLM-based Q&A generation
- This framework focuses on evaluation, not data creation

---

## DEC-012: Metric Calculation Approach

**Date**: 2026-01-21

**Context**: How to calculate metrics across runs?

**Options**:
1. Aggregate all runs together
2. Calculate per-run, then average
3. Both (report both individual and aggregate)

**Decision**: Option 2 - Calculate per-run, then average

**Rationale**:
- Allows calculating consistency (variance across runs)
- Standard statistical approach
- Individual run data preserved in search_results.csv
- Summary.csv contains aggregated metrics

---

## DEC-013: Search Result Ranking

**Date**: 2026-01-21

**Context**: How to rank results when searching across multiple levels (LOD)?

**Options**:
1. Concatenate and re-rank by score
2. Preserve level order (docs first, then sections, then chunks)
3. Interleave based on normalized scores

**Decision**: Option 1 - Concatenate and re-rank by score

**Rationale**:
- Consistent with flat strategy (fair comparison)
- Score reflects relevance regardless of level
- Simpler evaluation (just check top-k)
- Level information preserved in metadata for analysis

---

## DEC-014: Content Quality Evaluation Metrics

**Date**: 2026-01-21

**Context**: Current evaluation only checks if correct document/section IDs are in top-k results. We need to evaluate if the actual content of retrieved chunks is useful for answering the question. This must work across all retrieval strategies (flat, RAPTOR, LOD).

**Options**:
1. LLM-as-judge - Ask LLM to evaluate each retrieval (RAGAS, TruLens style)
2. Deterministic only - String matching, token overlap, ROUGE scores
3. Embedding similarity only - Cosine similarity between expected and retrieved
4. Tiered approach - Fast deterministic metrics + optional semantic metrics

**Decision**: Option 4 - Tiered approach with deterministic baseline and optional semantic metrics

**Rationale**:
- LLM-as-judge is expensive and slow for large-scale benchmarking
- Pure string matching misses semantic similarity (paraphrasing)
- Tiered approach gives flexibility: fast iteration with Tier 1, deeper analysis with Tier 2
- Ground truth already has `evidence` field with exact quotes - leverage this
- Embedding similarity reuses existing infrastructure (SentenceTransformers)

**Implementation**:

Tier 1 (Deterministic, No Additional Cost):
| Metric | Description |
|--------|-------------|
| `evidence_recall` | % of ground truth evidence quotes found in retrieved chunks |
| `answer_token_overlap` | Jaccard similarity between answer keywords and retrieved content |
| `rouge_l` | ROUGE-L F1 score between expected answer and combined retrieved context |

Tier 2 (Semantic, Embedding Cost):
| Metric | Description |
|--------|-------------|
| `max_evidence_similarity` | Max cosine similarity between any evidence quote and any retrieved chunk |
| `answer_context_similarity` | Cosine similarity between expected answer embedding and combined context embedding |

**Data Flow**:
```
Input:
  - question: str
  - expected_answer: str
  - evidence: list[str]  # exact quotes from ground truth
  - retrieved_chunks: list[str]  # actual retrieved content

Output:
  - ContentMetrics dataclass with all scores
```

**Consequences**:
- Enables meaningful comparison beyond "right document found"
- Tier 1 runs fast enough for every query evaluation
- Tier 2 can be enabled via config flag for detailed analysis
- May need to add `rouge-score` dependency for ROUGE-L
- Future: Can add LLM-based evaluation as Tier 3 for sample-based validation

---

## DEC-015: Human-Readable Report Format

**Date**: 2026-01-21

**Context**: CSV output is hard to read for humans. Need a format for quick analysis and sharing results.

**Options**:
1. Markdown tables only
2. HTML report with charts
3. Rich console output (terminal tables)
4. Markdown report file

**Decision**: Option 4 - Markdown report file (`report.md`) alongside CSVs

**Rationale**:
- Markdown renders nicely on GitHub and in editors
- No additional dependencies (vs HTML/charts)
- Persistent file (vs console-only output)
- Can include formatted tables, code blocks, and hierarchical structure
- CSVs retained for programmatic analysis

**Report Structure**:
```
# Benchmark Report

## Summary
[Table: Strategy comparison with key metrics]

## Index Statistics  
[Table: Documents, chunks, vectors, timing per strategy]

## Query Results
[Per-query breakdown showing:]
  - Question (what was asked)
  - Expected answer, document, sections
  - Results (ranks, timing)
  - Retrieved chunks with content preview
```

**Consequences**:
- Easy to review benchmark results without tooling
- Can be committed to repo for historical comparison
- Report generation adds minimal overhead (~10ms)

---

## Template for New Decisions

```markdown
## DEC-XXX: [Title]

**Date**: YYYY-MM-DD

**Context**: [What problem or question prompted this decision?]

**Options**:
1. [Option 1] - [Brief description]
2. [Option 2] - [Brief description]
3. [Option 3] - [Brief description]

**Decision**: Option N - [Choice made]

**Rationale**:
- [Reason 1]
- [Reason 2]
- [Reason 3]

**Consequences**:
- [What this enables]
- [What this prevents or makes harder]
- [Future considerations]
```
