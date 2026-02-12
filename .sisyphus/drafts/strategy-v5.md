# Draft: Strategy v5 - Cost-Effective NER Pipeline

## Requirements (confirmed)

### Core Objective
Replace expensive 2-Sonnet extraction with 3-Haiku + NER, while maintaining quality through smart validation routing.

### Architecture Decisions

**Step 1: Extraction (3 Haiku + NER)**
- Haiku-1: Few-shot with retrieved examples (like current retrieval_sonnet)
- Haiku-2: Full taxonomy prompt (like current exhaustive_sonnet)
- Haiku-3: Simplified prompt (current haiku_simple)
- NER model: Adds +1 vote (NOT auto-keep)

**Step 2-3: Grounding + Filtering**
- Keep existing logic unchanged
- Verify spans exist in text
- Apply stop words, gerunds, adjectives, negatives filtering

**Step 4: Confidence Scoring**
- HIGH: vote_count >= 3 OR entity_ratio >= 0.8 OR structural_pattern
- MEDIUM: vote_count == 2 OR entity_ratio 0.5-0.8
- LOW: vote_count == 1 AND entity_ratio < 0.5 AND no structural markers

**Step 5: Validation Routing**
- HIGH → KEEP (no validation needed)
- MEDIUM → Sonnet validation with contrastive examples → KEEP or REJECT
- LOW → REJECT + LOG STATISTICS (no Opus for now)

**Step 6: Statistics Tracking (NEW)**
- Log all LOW confidence rejected terms
- Track: term, doc_id, vote_count, entity_ratio, sources, was_in_GT
- Output: low_confidence_stats.jsonl
- Purpose: Data-driven decision on whether Opus is needed later

## Technical Decisions

### NER Model Choice
- User did not specify exact model
- Options: BERTOverflow (79% F1), GLiNER (zero-shot), spaCy (not recommended)
- Default: Need to ask user OR research what's available

### Haiku Prompt Strategy
- Haiku-1: Reuse RETRIEVAL_PROMPT_TEMPLATE but call Haiku instead of Sonnet
- Haiku-2: Reuse EXHAUSTIVE_PROMPT but call Haiku (may need simplification)
- Haiku-3: Keep current HAIKU_SIMPLE_PROMPT

### Validation Optimization
- Skip Sonnet validation if entity_ratio >= 0.7 (saves ~40% Sonnet calls)
- Keep term_index for calibration

### Statistics File Format
```jsonl
{"term": "...", "doc_id": "...", "vote_count": 1, "entity_ratio": 0.32, "sources": ["haiku_2"], "in_gt": true}
```

## Research Findings (from Oracle + Librarian)

### Haiku Recall Risk
- Expected 2-5% worse recall than Sonnet
- Concentrated in: contextual disambiguation, implicit references, compound terms
- Mitigation: Run A/B test on 50 docs before full commit

### NER Limitations
- 40-60% false negative rate on unseen entities
- Over-extracts common words ("node", "pod" as generic nouns)
- Should be validation signal, not auto-keep

### Cost Savings
- Current: ~$0.06/doc (2 Sonnet + 1 Haiku + validation)
- Proposed: ~$0.026/doc (3 Haiku + NER + validation)
- Savings: ~57%

## Scope Boundaries

### INCLUDE
- New strategy_v5 preset in hybrid_ner.py
- 3 Haiku extraction functions (or reuse with model param)
- NER integration as +1 vote source
- Confidence tier routing logic
- LOW confidence statistics logging
- Benchmark comparison on 10 docs

### EXCLUDE (for now)
- Opus integration (collecting data first)
- NER model training/fine-tuning
- Batch processing infrastructure
- UI/dashboard for statistics

## Open Questions

1. **NER Model**: Which model to use? BERTOverflow requires setup. GLiNER is zero-shot but needs threshold tuning. Start with simpler option?

2. **Haiku Prompt Complexity**: Should we test Haiku with full exhaustive prompt first, or simplify upfront?

3. **Statistics Storage**: JSONL file per benchmark run? Or append to single file?

4. **Test Strategy**: TDD for new functions? Or tests after implementation?

## Success Criteria

- [ ] strategy_v5 achieves P >= 90%, R >= 88% on 10-doc benchmark
- [ ] Cost per doc reduced by >= 50%
- [ ] LOW confidence statistics are captured correctly
- [ ] No regression in HIGH confidence term accuracy
