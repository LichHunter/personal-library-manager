# Draft: Benchmark LLM Grading Enhancement

## Requirements (confirmed)

### Grading Context
- Sonnet receives: Question + Retrieved Chunks + Expected Answer
- Grades on scale 1-10: "Would these chunks help solve the user's problem?"

### Metrics to Track
1. **LLM Grade** (1-10): Sonnet's assessment of retrieval usefulness
2. **Hit@1**: Is needle doc at position 1?
3. **Hit@5**: Is needle doc in top 5?
4. **MRR**: Mean Reciprocal Rank (1/position of first correct result)

### Benchmark Behavior
- Always run LLM grading (no flag needed)
- Grade all questions in the benchmark

## Technical Decisions

### LLM Integration
- Use existing `call_llm()` from `utils/llm_provider.py`
- Model: `"claude-sonnet"` → `claude-sonnet-4-20250514`
- Timeout: 30s per grading call

### Grading Prompt Structure
```
You are evaluating retrieval quality for a documentation search system.

**User Question:** {question}

**Expected Answer:** {expected_answer}

**Retrieved Chunks:**
{chunks formatted with content}

**Task:** Grade how well these retrieved chunks would help the user solve their problem.

Scale:
- 10: Perfect - chunks contain exact answer
- 7-9: Good - chunks contain relevant info to solve problem
- 4-6: Partial - some useful info but incomplete
- 1-3: Poor - chunks not helpful for this question

Respond with ONLY a JSON object:
{"grade": <1-10>, "reasoning": "<brief explanation>"}
```

### MRR Calculation
```python
def calculate_mrr(results):
    reciprocal_ranks = []
    for r in results:
        if r["rank"]:  # Found in top-k
            reciprocal_ranks.append(1.0 / r["rank"])
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

### Hit@k Calculation
```python
hit_at_1 = sum(1 for r in results if r["rank"] == 1) / len(results)
hit_at_5 = sum(1 for r in results if r["rank"] and r["rank"] <= 5) / len(results)
```

## Research Findings

### Existing LLM Pattern (from query_rewriter.py)
```python
from ..utils.llm_provider import call_llm

result = call_llm(prompt, model="claude-sonnet", timeout=30)
```

### Current Benchmark Flow
1. Load questions (with expected_answer field)
2. Load documents, chunk them
3. Index chunks
4. For each question:
   - Retrieve top-5 chunks
   - Check if needle_doc_id in results (binary)
5. Calculate accuracy

### New Flow
1. Load questions (with expected_answer field)
2. Load documents, chunk them
3. Index chunks
4. For each question:
   - Retrieve top-5 chunks
   - Calculate rank of needle doc (for Hit@1, Hit@5, MRR)
   - Call Sonnet to grade retrieval quality (1-10)
5. Calculate metrics:
   - LLM Grade (average)
   - Hit@1, Hit@5
   - MRR
   - Legacy accuracy (for comparison)

## Open Questions
- None - all requirements clarified

## Total Score Formula (NEW)
Composite score combining LLM quality grade with ranking position:

```
total_score = llm_grade × position_weight

Position weights:
- Rank 1:     1.0  (full credit - best possible)
- Rank 2-5:   0.8  (80% credit - slightly worse)
- Not found:  0.5  (50% credit - penalize but credit good content)
```

Examples:
- LLM=10, Rank=1 → 10.0 (perfect)
- LLM=8, Rank=3 → 6.4 (good but not top)
- LLM=8, Not found → 4.0 (good chunks, wrong doc)
- LLM=3, Rank=1 → 3.0 (right doc, bad chunks)

Aggregate: `avg_total_score` = average of all total_scores

## Scope Boundaries
- INCLUDE: LLM grading, Hit@1, Hit@5, MRR metrics, Total Score
- INCLUDE: Update results JSON structure
- EXCLUDE: Caching LLM grades (each run is fresh)
- EXCLUDE: Changing retrieval logic (only evaluation changes)
