# Draft: POC-1b Scale Testing & Documentation

## Requirements (confirmed)

### User Request
- Document current results and strategy in POC folder
- Test enhanced filter on 30, 50, and 100 chunks (not just 15)
- Ensure detailed logging for later analysis
- Ensure new GT has very high quality (double/triple check)
- Create comprehensive todo and plan

### Quality Decisions
- **GT Quality Approach**: Opus + exhaustive span verification (strict - reject any term not literally in text)
- **Chunk Selection**: Random sampling from available pool
- **Logging Detail**: Full audit trail (every term tracked through entire pipeline)

## Current State

### What We Have
- Enhanced noise filter (F25_ULTIMATE) integrated into `test_dplus_v3_sweep.py`
- Current GT: 15 chunks, 277 terms (v2)
- Current results on 15 chunks:
  - Precision: 98.2% (target: 95%) ✅
  - Recall: 94.6% (target: 95%) - 0.4% short
  - Hallucination: 1.8% (target: <5%) ✅
  - F1: 0.963

### Files Created
- `test_filter_upgrades.py` - Filter testing harness
- `artifacts/filter_upgrade_results.json` - Individual filter test results
- `artifacts/enhanced_filter_results.json` - Final integrated results

## Technical Decisions

### GT Expansion Strategy
1. Use Opus for exhaustive extraction (high recall)
2. Apply strict span grounding (every term must appear verbatim in text)
3. Tier classification (1=core, 2=important, 3=contextual)
4. No hallucinations by construction (span verification)

### Logging Requirements
For each term, track:
- Extraction source(s): sonnet_exhaustive, haiku_exhaustive, haiku_simple
- Vote count (1, 2, or 3)
- Grounding result (grounded/ungrounded, match type)
- Structural filter decision
- Vote routing (auto_keep/sonnet_review)
- Sonnet discrimination decision + reasoning (if applicable)
- Enhanced noise filter decision + reason
- Final status (KEPT/REJECTED + reason)

### Test Matrix
| Test Set | Chunks | GT Terms (est.) | Purpose |
|----------|--------|-----------------|---------|
| Small (current) | 15 | 277 | Baseline, already done |
| Medium | 30 | ~550 | 2x scale validation |
| Large | 50 | ~920 | Diversity test |
| Full | 100 | ~1850 | Production-scale test |

## Open Questions
- Where are the source chunks located? Need to verify availability of 100+ chunks
- What's the GT expansion script? Likely `expand_ground_truth.py`
- Should we re-validate existing 15-chunk GT with the new strict approach?

## Scope Boundaries

### INCLUDE
- Documentation of results and strategy
- GT expansion to 30, 50, 100 chunks with quality verification
- Running extraction + filter pipeline on all test sets
- Detailed logging with full audit trail
- Results analysis and comparison

### EXCLUDE
- Modifying the extraction prompts (Phase 1)
- Modifying the Sonnet discrimination prompts (Phase 4)
- Changing the enhanced noise filter (already finalized)
- GT audit/correction of existing 15 chunks (unless quality issues found)
