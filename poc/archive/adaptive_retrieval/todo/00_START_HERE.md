# Adaptive Retrieval POC - Execution Order

## How to Use This

1. Execute TODO plans in numerical order
2. Each plan is self-contained with preparation, measurement, and conclusion
3. Do not skip plans - baseline must be done first
4. After each plan, follow the "Proceed to Next" instructions

## Execution Order

### Phase 0: Preparation
- `01_prepare_test_set.md` - Create labeled test queries (REQUIRED)

### Phase 1: Baseline
- `02_baseline.md` - Measure current PLM performance (REQUIRED)

### Phase 2: P0 Approaches (Highest Priority)
- `03_reranking.md`
- `04_parent_child.md`
- `05_auto_merging.md`
- `06_p0_review.md` - Review P0 results, decide if sufficient

### Phase 3: P1 Approaches (If P0 Insufficient)
- `07_adaptive_classifier.md`
- `08_iterative_expansion.md`
- `09_recursive_retriever.md`
- `10_sentence_window.md`
- `11_p1_review.md`

### Phase 4: P2 Approaches (If P1 Insufficient)
- `12_multi_scale.md`
- `13_crag.md`
- `14_adaptive_k.md`
- `15_multi_query.md`
- `16_p2_review.md`

### Phase 5: P3 Approaches (Last Resort)
- `17_remaining_approaches.md`

### Phase 6: Final
- `18_final_report.md`

## Stopping Criteria

You may stop early if:
- Answer Success Rate improves ≥10% over baseline
- Further approaches unlikely to provide additional value
- Document decision to stop in phase review

## Start

Begin with: `01_prepare_test_set.md`
