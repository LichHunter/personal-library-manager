# Sentence-Level Extraction Test - RUNNING

## Status
**Test started:** 2026-02-05 17:39:40  
**Process ID:** 15644  
**Status:** RUNNING ✓

## Progress
- **Chunks completed:** 1 / 10
- **Estimated completion:** ~30-40 minutes (based on 18s/chunk + Sonnet filtering)

## Enhanced Logging
The test now includes **detailed Sonnet filtering decisions** showing:
- ✓ KEEP decisions with reasoning and score
- ✗ REMOVE decisions with specific reasons

### Example from Chunk 1:
```
✗ REMOVE 'Documentation': Generic term, not K8s-specific
✗ REMOVE 'sitemap': Generic web/documentation structure term
✗ REMOVE 'priority': Context shows it's about documentation priority, not K8s PriorityClass
✓ KEEP 'Kubernetes' (score=10): Core platform name
```

## Monitoring
```bash
# Check progress
bash monitor_test.sh

# Watch live
tail -f test_run.log

# Check detailed log
tail -f artifacts/logs/sentence_level_20260205_173940.log
```

## What We're Testing
1. **Sentence-level extraction**: Each sentence processed individually
2. **Quote-verify at sentence level**: Terms must exist in sentence with exact quotes
3. **Reasoning requirement**: Haiku must explain WHY each term is K8s-specific
4. **Sonnet filtering with detailed logging**: Now shows WHY terms are kept/removed
5. **Full 10-chunk test**: Complete statistical picture

## Expected Results
Based on preliminary 5-chunk results:
- **Precision:** 90-95% (better than baseline)
- **Recall:** 55-65% (worse than baseline) ⚠️
- **Hallucination:** 5-10% (better than baseline)

The key question: **Will the detailed logging reveal why Sonnet is over-filtering?**

## Next Steps After Completion
1. Analyze Sonnet's removal reasoning patterns
2. Identify if it's rejecting valid K8s terms
3. Decide: Fix Sonnet prompt OR abandon sentence-level approach
