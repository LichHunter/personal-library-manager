# Automated vs Manual Evaluation: A Comprehensive Analysis

**Date**: 2026-01-25  
**Author**: Investigation Session  
**Context**: RAG Retrieval Benchmark Validation

---

## Executive Summary

| Evaluation Method | Score | What It Measures |
|-------------------|-------|------------------|
| **Automated Benchmark** | 88.7% | "88.7% of key fact strings were found in retrieved chunks" |
| **Manual Test (Haiku grading)** | 72% (7.2/10) | "72% of questions got usable answers (LLM evaluation)" |
| **Manual Test (Human grading)** | 54% (5.4/10) | "54% of questions got usable answers (human evaluation)" |

**Key Insight**: The gap is not an error - these methods measure fundamentally different things. String presence ≠ Answer quality.

---

## Table of Contents

1. [What the Automated Benchmark Measures](#1-automated-benchmark)
2. [What Manual Testing Measures](#2-manual-testing)
3. [The Five Fundamental Differences](#3-fundamental-differences)
4. [Concrete Evidence from Testing](#4-concrete-evidence)
5. [The Haiku vs Human Grading Gap](#5-haiku-vs-human-gap)
6. [Visual Comparison](#6-visual-comparison)
7. [Why This Matters](#7-why-this-matters)
8. [Recommendations](#8-recommendations)

---

## 1. Automated Benchmark

### The Algorithm

**File**: `poc/chunking_benchmark_v2/run_benchmark.py:152-169`

```python
def exact_match(fact: str, text: str) -> bool:
    """Exact substring match."""
    return fact.lower() in text.lower()

def fuzzy_match(fact: str, text: str) -> bool:
    """Word-level fuzzy match."""
    text_lower = text.lower()
    fact_lower = fact.lower()

    if fact_lower in text_lower:
        return True

    # If all words from the fact appear in the text
    words = fact_lower.split()
    if len(words) >= 2 and all(w in text_lower for w in words):
        return True

    return False
```

### Evaluation Process

1. **Concatenate all retrieved chunks** into one big string:
   ```python
   combined_text = " ".join(c.content for c in retrieved)
   ```

2. **For each "key fact"** in the ground truth, check if the fact string exists:
   ```python
   found_facts = [f for f in key_facts if match_fn(f, combined_text)]
   ```

3. **Count binary hits**: Fact found = 1, Fact not found = 0

### Example Ground Truth Entry

```json
{
  "id": "realistic_001",
  "original_query": "What is the API rate limit per minute?",
  "key_facts": [
    "100 requests per minute per authenticated user",
    "20 requests per minute for unauthenticated requests"
  ]
}
```

### What "88.7% Coverage" Actually Means

- **Total key facts**: 53 (across all 20 queries)
- **Facts found**: 47
- **Coverage**: 47/53 = **88.7%**

**It does NOT mean**: "88.7% of questions were correctly answered"  
**It DOES mean**: "88.7% of fact strings appeared somewhere in the retrieved text"

---

## 2. Manual Testing

### The Algorithm

**File**: `poc/chunking_benchmark_v2/manual_test.py:392-520`

```python
def grade_result(question: dict, retrieval_result: dict) -> dict:
    """Grade using Claude Haiku with 1-10 rubric."""
    
    prompt = f"""You are grading a RAG retrieval system's response.

QUESTION: {query}
EXPECTED ANSWER: {expected_answer}

RETRIEVED CONTENT:
{chunks_text}

GRADING RUBRIC:
10: Perfect - All requested information retrieved, directly answers question
9: Excellent - Complete answer with minor irrelevant content  
8: Very Good - Answer present but buried in some noise
7: Good - Core answer present, missing some supporting details
6: Adequate - Partial answer, enough to be useful
5: Borderline - Some relevant info but misses key point
4: Poor - Tangentially related content only
3: Very Poor - Mostly irrelevant, hint of topic
2: Bad - Almost entirely irrelevant
1: Failed - No relevant content retrieved

Grade this retrieval. Output JSON:
{{"score": N, "explanation": "...", "verdict": "PASS|PARTIAL|FAIL"}}
"""
    
    response = call_llm(prompt, model="claude-haiku", timeout=120)
    # Parse and return score
```

### What It Evaluates

1. **Semantic understanding**: Does the content ANSWER the question?
2. **Usability**: Can a human extract the answer from this text?
3. **Context quality**: Is the answer clear or buried in noise?
4. **Completeness**: Is the full answer present or just fragments?
5. **Correctness**: Is the retrieved information CORRECT?

### What "72% Average Score" Means

- **10 questions tested**
- **Each graded 1-10** on answer quality
- **Average**: 7.2/10 = 72%

**It means**: "On average, retrieved content was 72% useful for answering the questions"

---

## 3. Fundamental Differences

### Difference 1: Binary vs Graded Evaluation

| Automated | Manual |
|-----------|--------|
| Found = 100%, Not Found = 0% | 1-10 quality scale |
| No middle ground | Partial credit exists |
| "100 connections" found → PASS | "100 connections" present but buried → 70% |

**Example**: Database connections question
- **Ground truth fact**: "100 connections"
- **Retrieved text**: "...system supports up to 2,000 concurrent connections... PostgreSQL configured with 100 connection pool..."
- **Automated**: ✅ PASS (string "100 connection" found via fuzzy match)
- **Manual**: 6/10 (Information is confusing - two different numbers present)

---

### Difference 2: Context Blindness vs Context Awareness

| Automated | Manual |
|-----------|--------|
| Ignores WHERE the fact appears | Evaluates if fact is in relevant context |
| Fact in Chunk 5 = Fact in Chunk 1 | Fact buried = lower score |
| No penalty for surrounding noise | Heavy penalty for noise |

**Example**: JWT token expiration
- **Ground truth fact**: "3600 seconds"
- **Retrieved Chunk 5**: "Cache TTL strategies: Short-lived data: 5-15 minutes (session tokens), Medium-lived: 1 hour (workflow definitions)..."
- **Automated**: ✅ PASS (found "1 hour" which matches "3600 seconds" concept)
- **Manual**: ❌ 3/10 (The "1 hour" refers to workflow definitions, NOT JWT tokens!)

---

### Difference 3: String Matching vs Semantic Understanding

| Automated | Manual |
|-----------|--------|
| `"100 requests".lower() in text.lower()` | "Does this mean 100 requests per minute for authenticated users?" |
| Can't distinguish 100 vs 2000 | Catches wrong numbers |
| Negations ignored | Negations evaluated |

**Example**: Connection limits
- **Ground truth fact**: "100 connections with PgBouncer"
- **Retrieved text**: "Database supports 2,000 concurrent connections"
- **Automated**: ⚠️ Might PASS (words "connections" found)
- **Manual**: ❌ 2/10 (WRONG answer - 2000 ≠ 100)

---

### Difference 4: Truncation Impact

| Automated | Manual |
|-----------|--------|
| Fact found anywhere = PASS | Truncated answer = lower score |
| Doesn't care if answer is cut off | Penalizes incomplete information |
| Fragment is enough | Must be usable |

**Example**: Authentication methods
- **Ground truth fact**: "API Keys, OAuth 2.0, JWT Tokens"
- **Retrieved Chunk 1**: "CloudFlow API Reference... supports authentication including API Keys, OAuth 2.0, and JWT To..." (truncated at 512 tokens)
- **Automated**: ✅ PASS (all three terms found)
- **Manual**: ⚠️ 4/10 (Answer is cut off, can't see full details)

---

### Difference 5: Position Ignorance vs Rank Awareness

| Automated | Manual |
|-----------|--------|
| Combines all 5 chunks equally | Chunk 1 more valuable than Chunk 5 |
| Top-5 treated as one big document | Lower-ranked chunks = more work to find answer |
| No weighting by relevance | Implicitly weights by position |

**Example**: Rate limits
- Fact in Chunk 1: **Automated** ✅ PASS, **Manual** ✅ 10/10
- Same fact in Chunk 5: **Automated** ✅ PASS, **Manual** ⚠️ 7/10 (buried)

---

## 4. Concrete Evidence

### Questions Where Both Methods Agreed (3/10)

| Question | Automated | Manual | Why Agreement |
|----------|-----------|--------|---------------|
| API rate limit | PASS | 10/10 | Exact answer in Chunk 1, perfectly clear |
| 429 error handling | PASS | 9/10 | Code examples with exact pattern |
| Error handling | PASS | 9/10 | Exact answer with YAML example |

**Pattern**: When the answer is **clear, complete, and in top chunks**, both methods agree.

---

### Questions Where Methods Disagreed (7/10)

| Question | Automated | Manual (Human) | Gap | Root Cause |
|----------|-----------|----------------|-----|------------|
| JWT token validity | PASS | 3/10 | -70% | String found but in WRONG context (cache TTL, not JWT) |
| Auth methods | PASS | 4/10 | -60% | Answer truncated mid-sentence |
| Concurrent executions | PASS | 2/10 | -80% | Wrong metric retrieved (500/sec vs 100 concurrent) |
| DB connection limits | PASS | 2/10 | -80% | Wrong number (2000 vs 100) |
| Recovery objectives | PASS | 6/10 | -40% | RTO correct, RPO WRONG (24h vs 1h) |
| Auth methods (v2) | PASS | 7/10 | -30% | Missing RS256 detail |
| Troubleshoot failures | PASS | 1/10 | -90% | Only TOC header, no actual instructions |

**Pattern**: Automated passes when strings exist; Manual fails when **context is wrong, answer is incorrect, or information is unusable**.

---

## 5. Haiku vs Human Gap

Even within manual testing, there's a grading gap:

| Grader | Average Score | Verdict |
|--------|---------------|---------|
| Haiku (LLM) | 7.2/10 (72%) | INCONCLUSIVE |
| Human | 5.4/10 (54%) | INVALIDATED |

**Bias**: Haiku over-grades by **+1.8 points** on average

### Why Haiku Over-Grades

1. **Gives credit for tangential relevance**
   - Human: "This chunk is about workflows, not JWT tokens" → 3/10
   - Haiku: "Tangentially related to authentication" → 5/10

2. **Misses factual errors**
   - Human: "2000 connections ≠ 100 connections, this is WRONG" → 2/10
   - Haiku: "Contains database connection information" → 6/10

3. **Too generous with partial information**
   - Human: "Only the TOC was retrieved, no actual troubleshooting steps" → 1/10
   - Haiku: "Some troubleshooting context present" → 5/10

4. **Doesn't penalize truncation enough**
   - Human: "Answer cut off mid-word, unusable" → 4/10
   - Haiku: "Core information present" → 8/10

---

## 6. Visual Comparison

### What 88.7% Coverage Looks Like

```
Query: "What is the JWT token expiration?"

Ground Truth Key Facts:
  ✓ "3600 seconds" - FOUND (in cache TTL section)
  ✓ "max 3600 seconds from iat" - FOUND (partial match)
  ✓ "All tokens expire after 3600 seconds" - FOUND (fuzzy match)

Automated Result: 3/3 = 100% ✓
```

### What 54% Quality Looks Like (Same Query)

```
Query: "What is the JWT token expiration?"

Retrieved Chunks:
  Chunk 1: API Reference overview (generic, no expiry time)
  Chunk 2: JWT token validation (mentions JWT but no expiry)
  Chunk 4: JWT time diagnostics (no expiry stated)
  Chunk 5: "Cache TTL: 1 hour for workflow definitions" ← WRONG CONTEXT!

Human Evaluation:
  - Is the question answered? NO
  - Is the information usable? NO
  - Is it correct? The "1 hour" refers to WORKFLOWS, not JWTs!

Manual Result: 3/10 ✗
```

---

## 7. Why This Matters

### The Automated Benchmark is Misleading Because:

1. **String presence ≠ Answer quality**
   - A fact can be "present" but in wrong context, truncated, or contradicted

2. **It can't detect wrong answers**
   - If docs say both "100 connections" and "2000 connections", finding either is a PASS

3. **No usability assessment**
   - A fact buried in the 5th chunk surrounded by noise is counted same as a clear answer

4. **Fuzzy matching creates false positives**
   - `all words in fact appear in text` matches many irrelevant passages

### The Manual Test is More Realistic Because:

1. **Mimics real user experience**
   - "Can I actually answer my question with this content?"

2. **Catches wrong information**
   - Numbers, contexts, negations all evaluated

3. **Penalizes poor UX**
   - Truncated, buried, or noisy answers get lower scores

4. **Graded scale captures nuance**
   - Partial answers get partial credit

---

## 8. Recommendations

### For Benchmark Improvement

#### Option 1: Add Position Weighting

```python
weights = {0: 1.0, 1: 0.9, 2: 0.7, 3: 0.5, 4: 0.3}  # By chunk rank
score = sum(weights[i] for i, chunk in enumerate(chunks) if fact in chunk)
```

#### Option 2: Use LLM-Based Fact Verification

```python
def verify_fact(fact, chunks):
    prompt = f"""Is '{fact}' clearly and correctly stated in this text? 
    Score 0-1:
    - 1.0: Fact is clearly stated and easy to find
    - 0.5: Fact is present but buried/truncated
    - 0.0: Fact is not present or wrong
    
    Text: {chunks}
    """
    return call_llm(prompt)
```

#### Option 3: Use Recall@K with Context

Instead of just "is fact present?", check:
- Is fact in top-1 chunk? (most valuable) → weight 1.0
- Is fact in top-3 chunks? (good) → weight 0.7
- Is fact in top-5 chunks? (acceptable) → weight 0.5

### For Production RAG Systems

1. **Don't trust automated benchmarks alone**
   - Always validate with human/LLM quality assessment

2. **Measure what matters**
   - User satisfaction > String matching

3. **Consider the full pipeline**
   - Retrieval is only useful if the answer is usable

4. **Test with real user queries**
   - Synthetic benchmarks miss real-world complexity

---

## Conclusion

The 88.7% automated benchmark and the 54-72% manual scores are **both correct** - they just measure different things:

| Metric | What It Measures | Score | Use Case |
|--------|------------------|-------|----------|
| Automated (88.7%) | "Can the fact STRING be found in retrieved text?" | High | Fast iteration, regression testing |
| Manual/Haiku (72%) | "Can an LLM extract a useful answer?" | Medium | Quality assessment, bias detection |
| Manual/Human (54%) | "Can a human actually answer the question?" | Low | Ground truth, production readiness |

**The gap exists because finding a string is much easier than providing a usable, correct answer.**

For real-world RAG applications, the **54-72% quality score** is more predictive of user experience than the 88.7% string-matching score.

---

## Appendix: Test Data

### Full Results from Manual Testing (10 Questions)

| # | Question | Automated | Haiku | Human | Gap (Auto-Human) |
|---|----------|-----------|-------|-------|------------------|
| 1 | API rate limit | PASS | 10/10 | 10/10 | 0% |
| 2 | JWT token validity | PASS | 5/10 | 3/10 | -70% |
| 3 | Auth methods | PASS | 8/10 | 4/10 | -60% |
| 4 | 429 error handling | PASS | 9/10 | 9/10 | 0% |
| 5 | Concurrent executions | PASS | 5/10 | 2/10 | -80% |
| 6 | Error handling | PASS | 9/10 | 10/10 | 0% |
| 7 | DB connection limits | PASS | 6/10 | 2/10 | -80% |
| 8 | Recovery objectives | PASS | 8/10 | 6/10 | -40% |
| 9 | Auth methods (v2) | PASS | 7/10 | 7/10 | -30% |
| 10 | Troubleshoot failures | PASS | 5/10 | 1/10 | -90% |

**Averages**:
- Automated: 100% (all PASS)
- Haiku: 7.2/10 (72%)
- Human: 5.4/10 (54%)

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-25  
**Related Files**:
- `poc/chunking_benchmark_v2/run_benchmark.py` - Automated benchmark
- `poc/chunking_benchmark_v2/manual_test.py` - Manual testing tool
- `.sisyphus/notepads/manual-testing-interface/automated-vs-manual-testing-analysis.md` - Original analysis
- `.sisyphus/notepads/manual-testing-interface/manual-grading-analysis.md` - Human grading details
