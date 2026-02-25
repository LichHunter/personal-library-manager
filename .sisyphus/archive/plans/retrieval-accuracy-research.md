# Retrieval Accuracy Research: Finding Hidden Gems

## Context

### Original Request
Investigate how to improve data retrieval accuracy beyond the 94% ceiling. Previous SOTA techniques (BMX, etc.) have failed or provided only marginal improvements. Need to find **hidden gems** - unconventional, production-validated techniques that actually work.

### Interview Summary
**Key Discussions**:
- Current best: 94% manual accuracy (smart chunking + enriched_hybrid_llm)
- Theoretical ceiling: ~99% (limited by corpus inconsistencies)
- **CRITICAL INSIGHT**: Overhyped public SOTA techniques often don't work on real cases (BMX failed with -26%)
- User wants hidden gems: production-validated, works on small corpora, solves root cause, unconventional

**What We're NOT Looking For**:
- Academic papers that only work on 100K+ doc benchmarks
- "Just add more LLM calls" patterns
- Techniques that sound good but fail in practice (like BMX)

**What We ARE Looking For**:
- Practitioner success stories (real production use)
- Techniques from adjacent fields (search, recommenders, pre-LLM NLP)
- First-principles solutions to YOUR specific failures
- Obscure or unconventional approaches

### Research Findings from Prior Work
- Smart chunking: **+40% improvement** (54% → 94%) - HUGE WIN
- LLM query rewriting: +5.7% - moderate
- BMX (entropy BM25): **-26% FAILURE** - lesson learned
- Drafted but untested: Metadata enrichment, doc-aware retrieval

---

## Work Objectives

### Core Objective
Conduct parallel root cause analysis + hidden gem hunting to find unconventional techniques that break the 94% ceiling. Produce a research report with validated findings, not just paper citations.

### Concrete Deliverables
1. **Failure dataset**: 20-30 queries that SHOULD work but DON'T
2. **Root cause analysis**: Deep understanding of WHY each failure occurs
3. **Hidden gems dossier**: Curated collection of production-validated, unconventional techniques
4. **Validated experiment matrix**: Only techniques with evidence from real-world use
5. **Recommended experiments**: Top 3-5 gems to test on YOUR corpus

### Definition of Done
- [x] Failure dataset gathered with 20+ failed/partial queries (24 queries, 16 failures)
- [x] Each failure has documented root cause (not just "wrong chunk") (694 lines of stage-by-stage analysis)
- [x] ≥5 hidden gems identified from practitioner sources (20 gems from 4 sources)
- [x] Each gem has production evidence (company blog, HN discussion, etc.) (all gems validated)
- [x] Experiment matrix with expected impact based on failure mode alignment (20×6 matrix with priority scores)
- [x] Clear recommendation on what to test first (Top 3: Adaptive Hybrid Weights, Negation-Aware Filtering, Synthetic Query Variants)

### Must Have
- Actual failure examples from YOUR corpus (not hypothetical)
- Production evidence for each technique (not just papers)
- Root cause → technique mapping (which gem solves which failure)
- Unconventional sources (not just arXiv papers)

### Must NOT Have (Guardrails)
- Do NOT include techniques only validated on 100K+ doc benchmarks
- Do NOT include techniques that are "add LLM call" patterns without novel insight
- Do NOT include techniques from papers without production validation
- Do NOT trust benchmark claims without real-world evidence
- Embedding model changes are OUT OF SCOPE

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (benchmark framework + manual_test.py)
- **User wants tests**: Manual verification after gem identification
- **Framework**: Existing benchmark + targeted failure testing

### Verification Approach
1. Each identified gem will be evaluated against the failure dataset
2. Success = improvement on SPECIFIC failure cases, not just overall average
3. A gem is valid only if it fixes failures we documented

### Success Criteria
- **SUCCESS**: Gem fixes ≥3 documented failures without regressing others
- **PARTIAL**: Gem fixes some failures but introduces new issues
- **FAILURE**: No improvement on documented failures

---

## Task Flow

```
Stream A (Root Cause Analysis)           Stream B (Hidden Gem Hunting)
    │                                        │
    ├── Task 1: Generate failures           ├── Task 5: Search practitioner blogs
    ├── Task 2: Deep failure analysis       ├── Task 6: Mine HN/Reddit discussions
    ├── Task 3: Categorize root causes      ├── Task 7: Explore adjacent fields
    ├── Task 4: Identify patterns           ├── Task 8: First principles analysis
    │                                        │
    └────────────────┬───────────────────────┘
                     │
              Task 9: Match gems to failures
              Task 10: Create experiment plan
              Task 11: Write recommendations
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| A | 1, 2, 3, 4 | Root cause analysis stream |
| B | 5, 6, 7, 8 | Hidden gem hunting stream |

Tasks 9-11 depend on both streams completing.

---

## TODOs

### Stream A: Root Cause Analysis

- [x] 1. Generate comprehensive failure dataset

  **What to do**:
  - Extend ground truth with 20+ additional queries that test edge cases
  - Include query types NOT in current benchmark:
    - Multi-hop queries (need 2+ docs)
    - Temporal queries ("What changed in v2?")
    - Comparative queries ("X vs Y")
    - Negation queries ("What should I NOT do?")
    - Implicit queries ("Best practice for..." without naming topic)
  - Run each query through current best retrieval (enriched_hybrid_llm)
  - Document: query, expected answer, actual retrieved, score (1-10)
  - Focus on FAILED and PARTIAL cases (score ≤7)

  **Must NOT do**:
  - Only generate easy queries
  - Skip edge cases
  - Use automated grading only

  **Parallelizable**: YES (can start immediately)

  **References**:
  - `poc/chunking_benchmark_v2/corpus/ground_truth_realistic.json` - Current queries
  - `poc/chunking_benchmark_v2/manual_test.py` - Test runner
  - `poc/chunking_benchmark_v2/corpus/realistic_documents/` - Corpus content

  **Acceptance Criteria**:
  - [ ] ≥20 new queries generated across edge case types
  - [ ] Each query manually graded
  - [ ] ≥15 failed/partial queries identified (score ≤7)
  - [ ] Saved to `.sisyphus/notepads/failure-dataset.md`

  **Commit**: NO (research data only)

---

- [x] 2. Deep failure analysis

  **What to do**:
  - For each failed query, conduct detailed analysis:
    1. What chunks WERE retrieved? (list all 5)
    2. Where IS the correct answer in the corpus? (exact location)
    3. WHY wasn't the correct chunk retrieved?
       - Was it ranked too low? (ranking issue)
       - Was it not in the index at all? (chunking issue)
       - Was the query understood differently? (query issue)
       - Is the answer spread across chunks? (fragmentation issue)
    4. At what stage did retrieval fail?
       - BM25 stage?
       - Semantic stage?
       - RRF fusion stage?
       - Reranking stage?
  - Document with specific evidence, not guesses

  **Must NOT do**:
  - Say "wrong chunk" without explaining WHY
  - Skip the stage-by-stage analysis
  - Assume root cause without evidence

  **Parallelizable**: NO (depends on Task 1)

  **References**:
  - Failure dataset from Task 1
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py` - Pipeline stages

  **Acceptance Criteria**:
  - [ ] Every failed query has stage-by-stage analysis
  - [ ] Exact location of correct answer documented
  - [ ] Root cause identified with evidence for each
  - [ ] Saved to `.sisyphus/notepads/failure-analysis-deep.md`

  **Commit**: NO (research data only)

---

- [x] 3. Categorize root causes

  **What to do**:
  - Group failures into ROOT CAUSE categories (not symptoms):
    - **EMBEDDING_BLIND**: Semantic similarity fails to capture query intent
    - **BM25_MISS**: Vocabulary mismatch between query and document
    - **WRONG_SECTION**: Right doc, wrong section (heading mismatch)
    - **FRAGMENTED**: Answer split across chunks
    - **RANKING_ERROR**: Right chunks retrieved but ranked wrong
    - **MULTI_HOP**: Needs info from 2+ docs
    - **QUERY_AMBIGUOUS**: Query could mean multiple things
    - **CORPUS_GAP**: Answer literally not in corpus
    - [Add new categories as discovered]
  - Count distribution
  - Identify which categories are addressable vs inherent

  **Must NOT do**:
  - Use vague categories like "retrieval failed"
  - Force failures into existing categories if they don't fit

  **Parallelizable**: NO (depends on Task 2)

  **References**:
  - Deep analysis from Task 2
  - `.sisyphus/notepads/chunking-data-loss-analysis.md` - Prior categories

  **Acceptance Criteria**:
  - [ ] All failures assigned to specific root cause category
  - [ ] Distribution table (e.g., "40% EMBEDDING_BLIND, 25% WRONG_SECTION...")
  - [ ] Each category marked as addressable or inherent
  - [ ] Saved to `.sisyphus/notepads/root-cause-categories.md`

  **Commit**: NO (research data only)

---

- [x] 4. Identify patterns across failures

  **What to do**:
  - Look for cross-cutting patterns:
    - Do certain query TYPES always fail? (e.g., "how to" vs "what is")
    - Do certain document SECTIONS always get missed?
    - Are there common KEYWORDS that cause problems?
    - Is there a POSITION bias? (early vs late in document)
    - Are CODE-related questions harder?
  - Quantify patterns with examples
  - Identify the "80/20" - which patterns explain most failures?

  **Must NOT do**:
  - Claim patterns without statistical support
  - Ignore small but important patterns

  **Parallelizable**: NO (depends on Task 3)

  **References**:
  - All prior analysis files
  - `poc/chunking_benchmark_v2/corpus/` - Document structure

  **Acceptance Criteria**:
  - [ ] ≥3 cross-cutting patterns identified
  - [ ] Each pattern has statistical support (% of failures)
  - [ ] "80/20 rule" identified - top patterns that explain most failures
  - [ ] Saved to `.sisyphus/notepads/failure-patterns.md`

  **Commit**: NO (research data only)

---

### Stream B: Hidden Gem Hunting

- [x] 5. Search practitioner engineering blogs

  **What to do**:
  - Search engineering blogs from companies doing RAG in production:
    - Pinecone, Weaviate, Qdrant, Chroma (vector DB companies)
    - Dust.tt, Glean, Perplexity (RAG product companies)
    - LangChain, LlamaIndex blog posts (framework authors)
    - Anthropic, OpenAI, Cohere technical blogs
  - Focus on:
    - "What we learned building production RAG"
    - "Mistakes we made with RAG"
    - "Unconventional techniques that worked"
  - For each gem found:
    - Source URL
    - Technique description
    - Evidence of production use
    - Claimed improvement
    - Implementation complexity

  **Must NOT do**:
  - Include pure marketing content
  - Include techniques without production evidence
  - Trust benchmark claims without real-world validation

  **Parallelizable**: YES (with Tasks 6, 7, 8)

  **Acceptance Criteria**:
  - [ ] ≥10 blog posts reviewed
  - [ ] ≥3 potential gems identified
  - [ ] Each gem has production evidence documented
  - [ ] Saved to `.sisyphus/notepads/gems-practitioner-blogs.md`

  **Commit**: NO (research data only)

---

- [x] 6. Mine HN/Reddit discussions

  **What to do**:
  - Search Hacker News for:
    - "RAG accuracy" discussions
    - "RAG limitations" threads
    - "Better than RAG" debates
    - Specific failure complaints with solutions
  - Search Reddit (r/MachineLearning, r/LocalLLaMA, r/LangChain):
    - "RAG not working" posts with solutions
    - "Improved my RAG accuracy by doing X"
    - Unconventional approaches
  - Look for:
    - Techniques with upvotes/validation from others
    - Before/after metrics shared
    - Honest failure reports with solutions

  **Must NOT do**:
  - Include unvalidated speculation
  - Trust claims without supporting evidence
  - Ignore negative feedback on techniques

  **Parallelizable**: YES (with Tasks 5, 7, 8)

  **Acceptance Criteria**:
  - [x] ≥20 relevant discussions reviewed
  - [x] ≥3 potential gems with community validation
  - [x] Each gem has link to source + key quotes
  - [x] Saved to `.sisyphus/notepads/gems-community-discussions.md`

  **Commit**: NO (research data only)

---

- [x] 7. Explore adjacent fields

  **What to do**:
  - Look at techniques from pre-LLM information retrieval:
    - Classic search engine ranking (PageRank-like authority)
    - Query expansion from IR literature
    - Pseudo-relevance feedback
    - Learning to rank (without LLMs)
  - Look at recommender system techniques:
    - Collaborative filtering for content retrieval
    - Content-based filtering enhancements
    - Hybrid recommendation approaches
  - Look at database/search techniques:
    - Faceted search and filtering
    - Semantic clustering for navigation
    - Query reformulation from logs
  - For each: Does it apply to our problem?

  **Must NOT do**:
  - Include techniques that don't translate to text retrieval
  - Ignore techniques just because they're "old"

  **Parallelizable**: YES (with Tasks 5, 6, 8)

  **Acceptance Criteria**:
  - [x] ≥3 adjacent field techniques evaluated
  - [x] Each technique assessed for applicability
  - [x] ≥2 promising cross-field gems identified
  - [x] Saved to `.sisyphus/notepads/gems-adjacent-fields.md`

  **Commit**: NO (research data only)

---

- [x] 8. First principles analysis

  **What to do**:
  - Ask fundamental questions about YOUR retrieval failures:
    1. What does "semantic similarity" actually measure? Is it the right metric?
    2. Are chunks the right unit? What if we retrieve sentences? Paragraphs? Sections?
    3. Is embedding ALL content the right approach? What if we embed differently?
    4. What information is LOST in the embedding? Can we preserve it?
    5. What does the user ACTUALLY need? Not just similar text, but answers.
  - Generate novel hypotheses based on YOUR failure patterns
  - Design experiments to test hypotheses

  **Must NOT do**:
  - Accept "this is how everyone does it" as an answer
  - Ignore ideas that seem too simple

  **Parallelizable**: YES (with Tasks 5, 6, 7)

  **References**:
  - Root cause categories from Task 3
  - Failure patterns from Task 4

  **Acceptance Criteria**:
  - [ ] ≥5 fundamental questions explored
  - [ ] ≥3 novel hypotheses generated
  - [ ] Each hypothesis is testable
  - [ ] Saved to `.sisyphus/notepads/gems-first-principles.md`

  **Commit**: NO (research data only)

---

### Stream C: Synthesis

- [x] 9. Match gems to failures

  **What to do**:
  - Create a matrix mapping:
    - Rows: Root cause categories from Task 3
    - Columns: Gems from Tasks 5-8
    - Cells: Does this gem address this root cause? (YES/NO/PARTIAL)
  - Identify:
    - Gems that address MULTIPLE root causes (high value)
    - Root causes with NO gems (need more research)
    - Gems with NO applicable failures (low priority)

  **Must NOT do**:
  - Force matches that don't exist
  - Include gems without clear failure mode fit

  **Parallelizable**: NO (depends on Tasks 1-8)

  **References**:
  - `.sisyphus/notepads/root-cause-categories.md`
  - All gems files

  **Acceptance Criteria**:
  - [ ] Matrix completed with all gems and root causes
  - [ ] Top gems identified (address ≥2 root causes)
  - [ ] Uncovered root causes flagged for further research
  - [ ] Saved to `.sisyphus/notepads/gem-failure-matrix.md`

  **Commit**: NO (research data only)

---

- [x] 10. Create validated experiment plan

  **What to do**:
  - For each high-priority gem:
    - Specific implementation approach
    - Estimated effort (hours/days)
    - Which failures it should fix (from dataset)
    - How to measure success
    - Risks and fallback
  - Prioritize by: (# failures addressed) × (production evidence strength) / effort
  - Create ordered list of experiments

  **Must NOT do**:
  - Include experiments without clear success criteria
  - Prioritize by "interestingness" over impact

  **Parallelizable**: NO (depends on Task 9)

  **Acceptance Criteria**:
  - [ ] Top 5 gems have full experiment plans
  - [ ] Each plan has specific failures to test against
  - [ ] Priority order justified
  - [ ] Saved to `.sisyphus/notepads/experiment-plan.md`

  **Commit**: NO (research data only)

---

- [x] 11. Write recommendations document

  **What to do**:
  - Executive summary: What we learned, what to try
  - Key insight: What's ACTUALLY broken (from root cause analysis)
  - Top 3 recommendations with:
    - Why this gem is promising
    - Production evidence
    - Expected improvement (based on failure analysis)
    - Implementation path
  - Anti-recommendations: What NOT to try and why
  - Open questions for future research

  **Must NOT do**:
  - Recommend techniques without failure mode justification
  - Skip anti-recommendations (important learnings)

  **Parallelizable**: NO (depends on Task 10)

  **Acceptance Criteria**:
  - [ ] Clear executive summary
  - [ ] Root cause insight articulated
  - [ ] Top 3 recommendations with full justification
  - [ ] Anti-recommendations documented
  - [ ] Saved to `.sisyphus/notepads/hidden-gems-recommendations.md`

  **Commit**: NO (research data only)

---

## Commit Strategy

This is a **research plan** - no commits expected. All outputs are research documentation saved to `.sisyphus/notepads/`.

---

## Success Criteria

### Verification Commands
```bash
# Verify failure dataset exists and has content
cat .sisyphus/notepads/failure-dataset.md | head -50

# Verify root causes are categorized
cat .sisyphus/notepads/root-cause-categories.md

# Verify gems are documented
ls .sisyphus/notepads/gems-*.md

# Verify recommendations exist
cat .sisyphus/notepads/hidden-gems-recommendations.md | head -100
```

### Final Checklist
- [x] ≥20 failed queries documented with root causes (24 queries, 16 failures)
- [x] Root cause categories defined with distribution (6 categories, 81% addressable)
- [x] ≥10 potential hidden gems identified (20 gems from 4 sources)
- [x] Each gem has production evidence (all gems validated)
- [x] Gem-to-failure matrix completed (20×6 matrix with priority scores)
- [x] Top 3 recommendations justified by failure analysis (Adaptive Hybrid Weights, Negation-Aware Filtering, Synthetic Query Variants)
- [x] Anti-recommendations documented (5 anti-patterns identified)

---

## Exit Criteria

### Research Phase Exit
```
EXIT WHEN:
- Failure dataset complete (≥15 failures documented)
- Root cause analysis complete (patterns identified)
- ≥5 validated gems with production evidence
- Gem-failure matrix shows clear matches
- OR: 5 days elapsed (hard time-box)
```

### Overall Success Definition
```
SUCCESS: Identified ≥3 gems that address documented failures with production evidence
PARTIAL SUCCESS: Identified gems but need more failure data to validate fit
ACCEPTABLE: Confirmed current approach is near-optimal, documented why
```

---

## Hidden Gem Criteria (Reference)

A technique is a "hidden gem" if ALL of these apply:

| Criterion | Test |
|-----------|------|
| **Works on small corpora** | Has been used on <1000 docs, not just 100K+ benchmarks |
| **Production validated** | Company/developer actually uses it, not just paper |
| **Solves root cause** | Addresses fundamental issue in YOUR failure analysis |
| **Unconventional** | Not the standard "add LLM/rerank/chunk differently" pattern |

Anti-patterns to avoid:
- "Just use bigger embeddings"
- "Just add reranking"
- "Just use GPT-4 for everything"
- Techniques that only work at scale
