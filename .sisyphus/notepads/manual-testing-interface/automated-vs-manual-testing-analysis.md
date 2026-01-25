# Why Automated Testing (88.7%) vs Manual Testing (72%) Differ

## Executive Summary

**Automated Benchmark**: 88.7% coverage
**Manual Testing (Haiku grading)**: 7.2/10 (72%)
**Difference**: ~16.7 percentage points

**Root Cause**: The two tests measure FUNDAMENTALLY DIFFERENT THINGS.

---

## What Each Test Actually Measures

### Automated Benchmark (88.7%)

**Method**: String matching of "key_facts" in retrieved chunks

**How it works**:
```python
def exact_match(fact: str, text: str) -> bool:
    return fact.lower() in text.lower()

def fuzzy_match(fact: str, text: str) -> bool:
    # Check if all words from fact appear in text
    words = fact_lower.split()
    if len(words) >= 2 and all(w in text_lower for w in words):
        return True
```

**Example** (Query: "What is the API rate limit?"):
- **Key facts to find**: 
  - "100 requests per minute per authenticated user"
  - "20 requests per minute for unauthenticated requests"
- **Test**: Does the combined text of top-5 chunks contain these strings?
- **Result**: If BOTH strings are found → 100% coverage for this query

**What it measures**: "Is the exact fact STRING present in retrieved text?"

### Manual Testing with Haiku Grading (72%)

**Method**: LLM evaluates if retrieved content ANSWERS the question

**How it works**:
```
Prompt to Haiku:
- Here's the QUESTION
- Here's the EXPECTED ANSWER
- Here are the RETRIEVED CHUNKS
- Grade 1-10: Does the retrieved content answer the question?
```

**Example** (Query: "What is the API rate limit?"):
- **Question**: What is the API rate limit?
- **Expected**: 100 requests per minute per authenticated user, 20 for unauthenticated
- **Retrieved chunks**: [actual text from retrieval]
- **Haiku evaluates**: Can a human read these chunks and answer the question?

**What it measures**: "Can the retrieved content be USED to answer the question?"

---

## The Fundamental Difference

| Aspect | Automated (88.7%) | Manual/Haiku (72%) |
|--------|-------------------|-------------------|
| **What it checks** | String presence | Answer usability |
| **Threshold** | Substring match | Semantic understanding |
| **Truncation impact** | None (fact can be anywhere) | High (truncated = unusable) |
| **Context matters** | No | Yes |
| **Partial answers** | Binary (found/not found) | Graded (1-10) |
| **Quality assessment** | None | Full evaluation |

---

## Concrete Examples of Discrepancy

### Example 1: JWT Token Expiration

**Ground Truth Key Facts**:
- "3600 seconds"
- "max 3600 seconds from iat"
- "All tokens expire after 3600 seconds"

**Automated Test**:
- Search for "3600 seconds" in retrieved chunks
- If found → PASS (100% for this fact)
- Even if buried in unrelated context → PASS

**Manual Test Result**: 5/10 (PARTIAL)
- Haiku's explanation: "While the retrieved content discusses JWT tokens and authentication in CloudFlow, no specific documentation chunk directly states the token expiration time of 3600 seconds."
- The STRING might be present, but the CONTEXT doesn't clearly answer the question

**Why the difference**:
- Automated: "3600" found in text → PASS
- Manual: "The information isn't clearly presented as 'JWT tokens expire after 3600 seconds'" → PARTIAL

### Example 2: Authentication Methods

**Ground Truth Key Facts**:
- "API Keys"
- "OAuth 2.0"
- "JWT Tokens"

**Automated Test**:
- Search for each string
- All three found → 100% coverage

**Manual Test Result**: 8/10 (PASS, but not perfect)
- Haiku's explanation: "The full details are cut off in the retrieved text, the core information is present. Some additional context about authentication is found in the troubleshooting guide, providing supporting evidence. The information is somewhat buried and requires a bit of parsing"

**Why the difference**:
- Automated: All strings present → 100%
- Manual: Information is there but truncated/buried → 80%

### Example 3: Database Connection Limits

**Ground Truth Key Facts**:
- "100 connections" (hypothetical)
- "PgBouncer"

**Automated Test**:
- Search for "100 connections" and "PgBouncer"
- If both found → 100%

**Manual Test Result**: 6/10 (PARTIAL)
- Haiku's explanation: "The retrieved content contains a database connection reference showing '2,000 concurrent connections', but this does not specifically match the expected PostgreSQL with 100 connection limit and PgBouncer detail"

**Why the difference**:
- Automated: Might find "100" and "connections" separately → could falsely PASS
- Manual: "2,000 connections ≠ 100 connections" → correctly identifies WRONG answer

---

## The Core Problem

### Automated Benchmark is TOO LENIENT

**Lenient string matching**:
```python
# This passes even if the numbers are wrong context:
"The system handles 100 users and has connection pooling"
# ↑ Contains "100" and "connection" → might falsely match "100 connections"
```

**No quality assessment**:
- Found string = 100% for that fact
- No partial credit
- No penalty for noise

**Ignores context**:
- "Not 100 requests" would still match "100 requests"
- Negations not handled

### Manual Testing is MORE REALISTIC

**Semantic evaluation**:
- Does the retrieved content ACTUALLY answer the question?
- Is the answer clear or buried in noise?
- Is it truncated/incomplete?

**Quality grading**:
- Perfect answer = 10
- Buried in noise = 7-8
- Partial answer = 5-6
- Wrong answer = 1-3

**Context awareness**:
- Can distinguish between "100 connections" and "2,000 connections"
- Recognizes when information is cut off

---

## Quantifying the Difference

### Where Automated Passes but Manual Fails

| Question | Automated | Manual | Gap | Reason |
|----------|-----------|--------|-----|--------|
| JWT expiration | PASS (string found) | 5/10 | -50% | String present but not as clear answer |
| Concurrent executions | PASS (partial match) | 5/10 | -50% | Different metric retrieved |
| DB connections | PASS (words found) | 6/10 | -40% | Wrong number (2000 vs 100) |
| Troubleshooting | PASS (header found) | 5/10 | -50% | Only TOC, no actual instructions |

### Where Both Agree

| Question | Automated | Manual | Reason |
|----------|-----------|--------|--------|
| API rate limit | PASS | 10/10 | Exact answer in Chunk 1 |
| Error handling | PASS | 9/10 | Code example with exact pattern |
| 429 handling | PASS | 9/10 | Code snippet + explanation |

---

## Conclusion

### Why 88.7% ≠ 72%

1. **Different metrics**:
   - Automated: Binary string presence (found/not found)
   - Manual: Graded answer quality (1-10)

2. **Different thresholds**:
   - Automated: Substring anywhere = PASS
   - Manual: Must be usable as an answer

3. **Context blindness**:
   - Automated: Ignores whether the fact is in relevant context
   - Manual: Evaluates if a human could use it to answer

4. **Truncation impact**:
   - Automated: Doesn't care if answer is cut off
   - Manual: Truncated answer = lower score

### Which is More Accurate?

**Manual testing (72%) is more representative of real-world RAG performance.**

Why:
- Users don't search for exact strings
- Users need USABLE answers, not just "fact present somewhere"
- Truncated/buried information is less valuable
- Context matters for understanding

### The 88.7% Benchmark is Misleading

The automated benchmark measures "fact presence" not "answer quality":
- A fact can be present but unusable (truncated, buried, wrong context)
- String matching can't distinguish "100 connections" from "2000 connections"
- No penalty for noise around the relevant fact

**Recommendation**: Use LLM-based evaluation for more realistic benchmarking, or add quality weights to fact matching.

---

## Appendix: How to Fix the Automated Benchmark

### Option 1: Add Quality Weights

Instead of binary found/not-found, weight by:
- Position in chunk (beginning = better)
- Surrounding context relevance
- Chunk rank (top chunk = better)

### Option 2: LLM-Based Fact Verification

```python
def verify_fact(fact: str, chunks: list[str]) -> float:
    prompt = f"""
    Fact to verify: {fact}
    Retrieved text: {chunks}
    
    Score 0-1: Is this fact clearly and usably present?
    - 1.0: Fact is clearly stated and easy to find
    - 0.5: Fact is present but buried/truncated
    - 0.0: Fact is not present or wrong
    """
    return call_llm(prompt)
```

### Option 3: Use Recall@K with Context

Instead of just "is fact present?", check:
- Is fact in top-1 chunk? (most valuable)
- Is fact in top-3 chunks? (good)
- Is fact in top-5 chunks? (acceptable)

Weight accordingly: top-1 = 1.0, top-3 = 0.7, top-5 = 0.5
