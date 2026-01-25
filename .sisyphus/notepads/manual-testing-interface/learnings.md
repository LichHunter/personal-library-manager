# Manual Testing Interface - Learnings

## 2026-01-25: Implementation Complete

### Overview
Built a CLI tool for manual validation of RAG retrieval results. The tool generates human-like questions using Claude Haiku, executes retrieval with enriched_hybrid_llm strategy, grades results, and produces markdown reports.

### Key Implementation Decisions

#### 1. LLM Model Selection
**Initial Plan**: Use Claude Sonnet for grading (higher quality)
**Reality**: OAuth credentials only support Claude Haiku models
**Solution**: Use Claude Haiku for both question generation AND grading
**Outcome**: Works well, Haiku is capable enough for both tasks

#### 2. Grading Rubric
- 10-point scale (1=Failed, 10=Perfect)
- Explicit criteria for each score level
- Verdict mapping: PASS (≥7), PARTIAL (4-6), FAIL (≤3)
- Validation thresholds: VALIDATED (≥7.5), INCONCLUSIVE (5.5-7.4), INVALIDATED (<5.5)

#### 3. Question Generation Strategy
- Agent reads ALL 5 corpus documents (~17K words, ~23K tokens)
- Haiku's 200K context window handles this easily
- Auto-decides question count: 2-3 per document, max 15
- Follows existing query patterns from ground_truth_realistic.json
- Verifies answers exist in corpus before including questions

#### 4. Report Structure
- Summary: Average score, benchmark comparison, validation verdict
- Results Table: Question | Score | Verdict | Source
- Detailed Findings: Full question, expected answer, explanation, retrieved chunks (truncated to 200 chars)
- Conclusion: Validation verdict with reasoning

### Technical Challenges

#### Challenge 1: OAuth Credential Restrictions
**Problem**: OpenCode OAuth tokens only work with Claude Code, not direct API calls
**Error**: `400 {"type":"invalid_request_error","message":"This credential is only authorized for use with Claude Code"}`
**Root Cause**: OAuth token scope restricted to specific models
**Solution**: The `enrichment/provider.py` already handles this correctly with OAuth refresh logic
**Lesson**: Always check which models are available with current credentials

#### Challenge 2: Nix Environment Library Paths
**Problem**: `ImportError: libstdc++.so.6: cannot open shared object file`
**Root Cause**: Python C extensions (torch, numpy) need Nix-provided system libraries
**Solution**: Already fixed in `flake.nix` with proper LD_LIBRARY_PATH
**Verification**: Must run in `nix develop` shell, not bare venv
**Lesson**: Document environment requirements clearly

#### Challenge 3: Model Availability
**Problem**: `claude-sonnet` returned API 400 errors
**Investigation**: Tested both models:
  - `claude-haiku` → Works ✓
  - `claude-sonnet` → Fails with credential error ✗
**Solution**: Changed `grade_result()` to use `claude-haiku` instead
**Lesson**: Test actual API access before assuming model availability

### Code Patterns Established

#### Pattern 1: LLM Integration
```python
from enrichment.provider import call_llm

response = call_llm(
    prompt="Your prompt here",
    model="claude-haiku",  # or "claude-sonnet" if available
    timeout=90
)
```
- Returns empty string on failure (no exception)
- Handles OAuth token refresh automatically
- Logs all requests/responses for debugging

#### Pattern 2: Retrieval Execution
```python
# High-precision timing
start = time.perf_counter()
retrieved = strategy.retrieve(query, k=k)
latency_ms = (time.perf_counter() - start) * 1000

# Extract chunk data
for chunk in retrieved:
    chunk_data = {
        "content": chunk.content,
        "score": getattr(chunk, "score", None),  # Graceful fallback
        "doc_id": chunk.doc_id,
    }
```

#### Pattern 3: JSON Parsing from LLM
```python
# Robust extraction
response_text = response.strip()
if response_text.startswith("["):
    data = json.loads(response_text)
else:
    # Find JSON in response
    start_idx = response_text.find("[")
    end_idx = response_text.rfind("]") + 1
    if start_idx >= 0 and end_idx > start_idx:
        data = json.loads(response_text[start_idx:end_idx])
```

### Test Results

#### Run 1: 6 Questions (2026-01-25 17:12)
- Average Score: 5.3/10
- Verdict: INVALIDATED (below 5.5 threshold)
- Findings:
  - Question 1 (workflow timeout): 5/10 - answer not directly stated
  - Question 2 (max steps): 3/10 - no relevant content
  - Question 3 (auth methods): 7/10 - partial match, truncated
  - Question 4 (connection pool): 6/10 - tangentially related
  - Question 5 (retry policy): 6/10 - hints but no specifics
  - Question 6 (JWT validity): 5/10 - related but not direct

**Interpretation**: The enriched_hybrid_llm strategy often retrieves tangentially related content rather than direct answers. The 88.7% benchmark claim may be overstated.

### Files Created

1. **`poc/chunking_benchmark_v2/manual_test.py`** (234 lines)
   - CLI with argparse (--questions, --output)
   - Document loading and strategy initialization
   - Question generation (Claude Haiku)
   - Retrieval execution with timing
   - Grading logic (Claude Haiku, 1-10 scale)
   - Markdown report generation

2. **Sample Reports**:
   - `results/manual_test_20260125_171125.md` (6 questions, 5.3/10 avg)
   - Multiple test runs for validation

### Commits

| Commit | Description |
|--------|-------------|
| `a19345b` | Tasks 1-2: Grading rubric + script scaffold |
| `8cbfaa7` | Tasks 3-4: Question generation + retrieval |
| `8fc4c3b` | Tasks 5-6: Grading logic + report generation |
| `d214c2a` | Task 7: Haiku fix + end-to-end validation |

### Usage

```bash
cd poc/chunking_benchmark_v2
source .venv/bin/activate

# Run with default (agent decides question count)
python manual_test.py

# Run with specific question count
python manual_test.py --questions 10

# Custom output path
python manual_test.py --output my_report.md
```

**Important**: Must run in `nix develop` shell for proper library paths.

### Future Improvements (Not Implemented)

1. **Manual Grading Mode**: Remove automatic LLM grading, let humans grade for best accuracy
2. **Full Chunk Display**: Show complete chunk content instead of truncated (200 chars)
3. **Multi-Strategy Comparison**: Test multiple strategies side-by-side
4. **Interactive Mode**: REPL for exploring retrieval results
5. **Export to CSV**: Machine-readable format for analysis

### Lessons Learned

1. **OAuth vs API Keys**: OpenCode OAuth tokens have model restrictions - always verify access
2. **Haiku is Capable**: Don't assume you need Sonnet - Haiku handles both generation and grading well
3. **Environment Matters**: Nix shell required for C extension libraries
4. **LLM Error Handling**: `call_llm()` returns empty string on failure, not exception
5. **Grading Consistency**: Temperature=0 helps but LLMs still have variance
6. **Benchmark Validation**: Manual testing revealed the 88.7% claim may be optimistic

### Next Steps (User Decision)

1. Run more comprehensive tests (10-20 questions)
2. Manually review and re-grade results for accuracy
3. Investigate why certain questions fail (vocabulary mismatch? chunking issues?)
4. Compare against other strategies (semantic, hybrid, etc.)
5. Consider implementing proposition chunking or other advanced techniques
