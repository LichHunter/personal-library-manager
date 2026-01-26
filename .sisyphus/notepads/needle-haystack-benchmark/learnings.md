# Learnings from Needle-in-Haystack Benchmark

## [2026-01-26T20:20:00Z] Task: Complete needle-haystack benchmark

### What Worked Well

1. **Manual Grading Approach**
   - Reading all 20 questions and their retrieved chunks manually provided deep insights
   - Grading rubric (1-10 scale) was clear and easy to apply consistently
   - Detailed reasoning for each score helped identify patterns in failures

2. **enriched_hybrid_llm Strategy Performance**
   - Excellent on conceptual questions (100% pass rate)
   - Strong on how-to and problem-solving questions (100% pass rate)
   - Good semantic understanding of human-like queries
   - 90% overall pass rate (18/20) validates the strategy

3. **Question Generation**
   - Human-like questions (casual language, problem-based) tested real-world usage
   - Mix of question types (problem, how-to, conceptual, fact-lookup) provided comprehensive coverage
   - Questions successfully avoided documentation language

4. **Benchmark Structure**
   - 200 K8s docs provided realistic haystack size
   - Single needle document (17.7KB, 2556 words) had good semantic variety
   - Top-5 retrieval was sufficient for most questions

### Issues Encountered

1. **Version/Date Fact Lookup Failures**
   - Q6: "Which k8s version made topology manager GA?" - Failed (4/10)
   - Q12: "When did prefer-closest-numa-nodes become GA?" - Complete failure (2/10)
   - Root cause: Semantic search doesn't match numerical facts well
   - Version numbers lack semantic context for embeddings

2. **Chunk Ranking Issues**
   - Some questions had answers in positions 4-5 rather than 1-2
   - Q11, Q14, Q18 required reading multiple chunks
   - RRF fusion (k=60) may need tuning

3. **Latency**
   - Average 1,022ms per query
   - Claude Haiku query rewriting accounts for ~960ms
   - Acceptable for batch/offline, not for real-time

### Key Insights

1. **Semantic Search Strengths**
   - Excels at conceptual understanding and relationships
   - Handles vocabulary mismatch well (casual → technical)
   - Good at finding explanations and procedures

2. **Semantic Search Weaknesses**
   - Struggles with specific numerical facts (versions, dates)
   - Needs keyword matching for exact fact lookups
   - May benefit from hybrid approach: semantic for concepts, keyword for facts

3. **Manual Grading Value**
   - Automated "needle_found" metric (95%) was misleading
   - Manual grading (90% pass) more accurately reflects answer quality
   - Reading chunks reveals retrieval quality issues not visible in metrics

### Recommendations for Future Work

1. **Improve Version/Date Handling**
   - Add metadata extraction for version numbers and dates
   - Create separate index for factual lookups
   - Use exact keyword matching for version queries

2. **Optimize Chunk Ranking**
   - Experiment with RRF k parameter (currently 60)
   - Consider cross-encoder reranking for top-10 → top-5
   - Analyze why some answers appear in positions 4-5

3. **Reduce Latency**
   - Cache common query rewrites
   - Use faster LLM (Haiku-3.5) or reduce timeout
   - Consider async query rewriting

4. **Enhance Enrichment**
   - Add version number extraction to YAKE/spaCy
   - Preserve hyphenated technical terms
   - Include feature names in enrichment

### Patterns and Conventions

1. **Grading Rubric**
   - 9-10: Verbatim or nearly verbatim answer
   - 7-8: Concept present, different wording
   - 5-6: Partial answer, missing details
   - 3-4: Tangentially related only
   - 1-2: Completely irrelevant

2. **Question Types**
   - Problem-based: "My X is broken, why?"
   - How-to: "How do I configure Y?"
   - Conceptual: "What's the difference between A and B?"
   - Fact-lookup: "What version/flag/value?"

3. **Benchmark Metrics**
   - Pass threshold: ≥7/10
   - Verdict thresholds:
     - VALIDATED: ≥75% pass rate
     - INCONCLUSIVE: 50-74% pass rate
     - INVALIDATED: <50% pass rate

### Files Created

1. `corpus/needle_selection.json` - Needle document metadata
2. `corpus/needle_questions.json` - 20 human-like questions
3. `benchmark_needle_haystack.py` - Benchmark script
4. `results/needle_haystack_retrieval.json` - Raw retrieval results (173KB)
5. `results/needle_haystack_graded.md` - Manual grading with reasoning (21KB)
6. `results/needle_haystack_report.md` - Comprehensive analysis (17KB)

### Commit

- Hash: `11d8c52`
- Message: `feat(benchmark): complete needle-haystack benchmark for enriched_hybrid_llm`
- Files: 6 files, 2,231 insertions
