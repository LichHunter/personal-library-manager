# Draft: LLM Query Rewrite Investigation

## Context

**Problem**: 59% failure rate (Hit@5 = 40.75%) on realistic user questions.

**Previous hypothesis** (INVALIDATED): Users use "wrong" terms → add term graph to map user vocabulary to docs vocabulary.

**New hypothesis**: The LLM rewrite layer already handles vocabulary transformation. The rewrite itself may be:
1. Losing original intent during transformation
2. Adding wrong/misleading terms
3. Over-generalizing specific user questions
4. Removing important context

## Research Findings

### Current Query Flow

```
User Query → LLM Rewrite (Claude Haiku) → Domain Expansion → BM25 + Semantic → RRF Fusion → Results
```

### LLM Rewrite Implementation

**File**: `poc/chunking_benchmark_v2/retrieval/query_rewrite.py`

**Prompt** (lines 19-35):
```
You are a technical documentation search expert. Your task is to rewrite user questions as direct documentation lookup queries.

Guidelines:
1. Convert problem descriptions to feature/capability questions
2. Expand abbreviations and acronyms to full terms
3. Replace casual language with technical terminology
4. Align with documentation vocabulary and structure
5. Keep the rewritten query concise (one line)

Examples:
- "Why can't I schedule workflows every 30 seconds?" → "workflow scheduling minimum interval frequency constraints"
- "Why does my token stop working after 3600 seconds?" → "token expiration TTL lifetime 3600 seconds"
- "What's the RPO and RTO?" → "recovery point objective recovery time objective disaster recovery"
```

**Model**: Claude Haiku (5s timeout)

**Behavior**: Falls back to original query on timeout/error

### Existing Logging (disabled by default)

The retrieval strategy HAS tracing but it's disabled:
- `enriched_hybrid_llm.py` line 99: `debug=False`
- When `debug=True`, logs:
  - `ORIGINAL_QUERY: {query}` (line 186)
  - `REWRITTEN_QUERY: {rewritten_query}` (line 197)
  - `EXPANDED_QUERY: {expanded_query}` (line 208)

### How to Enable Debug

```python
strategy = create_retrieval_strategy("enriched_hybrid_llm", debug=True)
```

The `**kwargs` in `create_retrieval_strategy()` pass through to the constructor.

### Existing Benchmark: needle_haystack

**File**: `benchmark_needle_haystack.py`

**Corpus**: 200 K8s docs (`corpus/kubernetes_sample_200/`)

**Questions available**:
- `needle_questions.json` - 20 baseline questions (90% pass rate)
- `needle_questions_adversarial.json` - 20 hard questions (65% pass rate)
  - Categories: VERSION (40%), COMPARISON (100%), NEGATION (60%), VOCABULARY (60%)

**Current benchmark flow** (lines 268-296):
1. Load questions
2. For each question: `strategy.retrieve(q["question"], k=5)`
3. Check if needle doc in results
4. Save to `results/needle_haystack_retrieval.json`

**What's missing**: No logging of rewritten query!

## Implementation Plan

### Simple Approach

1. **Modify `benchmark_needle_haystack.py`**:
   - Add `debug=True` to strategy creation
   - Capture rewritten query for each question
   - Add to results: `{original, rewritten, expanded}`

2. **Run on adversarial questions** (20 questions, 65% baseline):
   - These are the hardest questions
   - Include VOCABULARY category - most relevant for rewrite analysis

3. **Analyze output**:
   - Look at failed questions
   - Compare original vs rewritten
   - Determine if rewrite helped/hurt

### Changes Needed

In `benchmark_needle_haystack.py`:

```python
# Line 246: Enable debug
strategy = create_retrieval_strategy("enriched_hybrid_llm", debug=True)

# Line 270: Capture rewrite info
# Need to access strategy internals or modify retrieve() to return metadata
```

**Problem**: The `retrieve()` method doesn't return rewrite info. Options:
1. Add a method to get last rewrite: `strategy.get_last_rewrite()`
2. Parse debug logs after each query
3. Call `rewrite_query()` separately before retrieve

**Simplest**: Call `rewrite_query()` directly in benchmark loop, log it, then call retrieve.

## Scope

- IN: Add rewrite logging to needle_haystack benchmark, run on adversarial questions
- OUT: Modifying the rewrite prompt, running on full 200 questions (yet)
