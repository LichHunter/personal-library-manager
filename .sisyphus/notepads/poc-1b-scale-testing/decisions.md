# Decisions - POC-1b Scale Testing

## 2026-02-08: run_scale_pipeline.py Architecture

### Decision
Implemented `run_scale_pipeline.py` that reuses functions from `test_dplus_v3_sweep.py` via imports rather than copy/paste.

### Rationale
- Avoids code duplication and drift between files
- F25_ULTIMATE filter, scoring functions, and prompts are maintained in one place
- CLI interface matches expected usage pattern from task specification

### Key Imports Used
```python
from test_dplus_v3_sweep import (
    enhanced_noise_filter,      # F25_ULTIMATE filter
    many_to_many_score,         # m2m_v3 scoring methodology
    v3_match,                   # Matching function with prefix/suffix support
    normalize_term,             # Term normalization
    verify_span,                # Span grounding verification
    is_structural_term,         # Structural term detection
    smart_dedup,                # Smart deduplication
    parse_terms_response,       # Parse LLM extraction output
    parse_approval_response,    # Parse Sonnet review output
    EXHAUSTIVE_PROMPT,          # For Sonnet/Haiku extraction
    SIMPLE_PROMPT,              # For Haiku simple extraction
    V_BASELINE,                 # For Sonnet review
)
```

### Output Format
```json
{
  "metadata": { "gt_file": "...", "filter_version": "F25_ULTIMATE", ... },
  "aggregate": { "m2m_v3": { "precision": X, "recall": Y, "hallucination": Z, "f1": W } },
  "per_chunk": [ { "chunk_id": "...", "extracted_terms": [...], "scores": {...} } ]
}
```
