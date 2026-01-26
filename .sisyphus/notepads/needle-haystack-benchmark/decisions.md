# Decisions Made During Needle-in-Haystack Benchmark

## [2026-01-26T20:20:00Z] Task: Complete needle-haystack benchmark

### Decision 1: Manual Grading by Agent (Not Automated)

**Context**: The plan specified "manual grading by executing agent" rather than automated LLM API calls.

**Decision**: Agent (Sisyphus/Atlas) manually read all 20 questions and their retrieved chunks, then assigned scores 1-10 with detailed reasoning.

**Rationale**:
- Provides deeper insights than automated metrics
- Allows for nuanced evaluation of answer quality
- Reveals patterns in retrieval failures
- More accurate than simple "needle_found" boolean

**Outcome**: Successfully graded all 20 questions with detailed reasoning, revealing that automated "needle_found" (95%) was misleading compared to actual answer quality (90% pass rate).

---

### Decision 2: Use enriched_hybrid_llm Strategy (Not Comparison)

**Context**: The plan specified testing a single strategy, not comparing multiple strategies.

**Decision**: Benchmark only `enriched_hybrid_llm` strategy with:
- BM25 + semantic embeddings (BGE-base)
- YAKE + spaCy keyword enrichment
- Claude Haiku query rewriting
- RRF fusion (k=60)

**Rationale**:
- Focus on validating the best-performing strategy from previous benchmarks
- Avoid complexity of multi-strategy comparison
- Provide deep analysis of one strategy's strengths/weaknesses

**Outcome**: Strategy achieved VALIDATED status (90% pass rate), with clear identification of strengths (conceptual questions) and weaknesses (version/date lookups).

---

### Decision 3: 20 Questions from Single Needle Document

**Context**: Need to generate realistic questions that test retrieval quality.

**Decision**: Generate 20 human-like questions from a single needle document (Kubernetes Topology Manager), with mix of:
- 5 problem-based questions
- 5 how-to questions
- 5 conceptual questions
- 5 fact-lookup questions

**Rationale**:
- Single needle provides cleaner signal (all questions answerable from same doc)
- Human-like language tests real-world usage patterns
- Mix of question types provides comprehensive coverage
- 20 questions balances thoroughness with feasibility

**Outcome**: Questions successfully tested different aspects of retrieval. Strategy excelled at problem/how-to/conceptual (100% pass) but struggled with fact-lookup (60% pass).

---

### Decision 4: Top-5 Retrieval (Not Top-10)

**Context**: Need to decide how many chunks to retrieve per query.

**Decision**: Retrieve top-5 chunks per query.

**Rationale**:
- Realistic for production use (users won't read 10+ chunks)
- Sufficient for most questions (most answers in top 1-2)
- Keeps evaluation manageable

**Outcome**: Top-5 was sufficient for 18/20 questions. Two failures (Q6, Q12) were due to semantic search limitations, not insufficient k.

---

### Decision 5: MarkdownSemanticStrategy Chunking

**Context**: Need to choose chunking strategy for indexing.

**Decision**: Use `MarkdownSemanticStrategy(target=400, min=50, max=800)`.

**Rationale**:
- Preserves document structure (headers, sections)
- Semantic boundaries improve chunk coherence
- 400-token target balances context and precision
- Proven effective in previous benchmarks

**Outcome**: Chunking worked well - 200 docs → 1,030 chunks, with 14 chunks from needle document. Most answers found in single chunks.

---

### Decision 6: Grading Rubric (1-10 Scale)

**Context**: Need consistent scoring system for manual grading.

**Decision**: Use 1-10 scale with clear rubric:
- 9-10: Verbatim or nearly verbatim answer
- 7-8: Concept present, different wording
- 5-6: Partial answer, missing details
- 3-4: Tangentially related only
- 1-2: Completely irrelevant

Pass threshold: ≥7/10

**Rationale**:
- Granular enough to distinguish quality levels
- Clear criteria reduce subjectivity
- Pass threshold (7) ensures answer is actually useful

**Outcome**: Rubric was easy to apply consistently. Clear distinction between perfect (10), good (7-8), and failed (<7) retrievals.

---

### Decision 7: Verdict Thresholds

**Context**: Need to determine overall benchmark verdict.

**Decision**: Use thresholds:
- VALIDATED: ≥75% pass rate
- INCONCLUSIVE: 50-74% pass rate
- INVALIDATED: <50% pass rate

**Rationale**:
- 75% threshold is realistic for production use
- Allows for some failures while maintaining quality
- Aligns with industry standards for retrieval systems

**Outcome**: Strategy achieved 90% pass rate → VALIDATED verdict. Clear recommendation for production use with awareness of limitations.

---

### Decision 8: Comprehensive Report Structure

**Context**: Need to present findings in actionable format.

**Decision**: Include in final report:
- Executive summary with verdict
- Configuration details
- Per-question analysis (all 20)
- Aggregate metrics
- Failure analysis
- Recommendations

**Rationale**:
- Executive summary for quick understanding
- Per-question details for debugging
- Failure analysis identifies improvement areas
- Recommendations guide future work

**Outcome**: Report provides complete picture of strategy performance, with clear strengths, weaknesses, and actionable recommendations.
