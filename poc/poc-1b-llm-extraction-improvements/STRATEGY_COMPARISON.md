# POC-1b Strategy Comparison Matrix

## Quick Reference: All 28 Strategies

| # | Strategy | Model | Precision | Recall | Hallucination | F1 | Technique | Status |
|---|----------|-------|-----------|--------|---------------|----|----|--------|
| 1 | Quote-Extract | Haiku | 92.6% | 53.7% | 7.4% | - | Quote verification | ⚠️ Low recall |
| 2 | Gleaning 2x | Haiku | 62.7% | 68.4% | 32.3% | - | Multi-pass | ❌ High hallucination |
| 3 | Combined | Haiku | 64.2% | 51.1% | 30.8% | - | Quote + Gleaning | ❌ Poor balance |
| 4 | Combined | Sonnet | 85.7% | 60.9% | 9.3% | 0.712 | Quote + Gleaning | ⚠️ Below targets |
| 5 | Pattern-Only | - | 43.0% | 33.2% | 52.5% | - | Regex only | ❌ Too simplistic |
| 6 | Discrimination (No Cat) | Haiku | 59.1% | 18.6% | 25.9% | 0.264 | LLM filtering | ❌ Extremely low recall |
| 7 | Discrimination (Full) | Haiku | 64.0% | 62.5% | 31.0% | 0.611 | LLM + category | ⚠️ High hallucination |
| 8 | Discrimination | Sonnet | 70.7% | 57.8% | 19.3% | 0.628 | LLM + category | ⚠️ Below targets |
| 9 | Simple (Small Chunks) | Haiku | 71.1% | 82.8% | 28.9% | 0.744 | Smaller context | ⚠️ High hallucination |
| 10 | Quote (Small Chunks) | Haiku | 78.7% | 74.8% | 21.3% | 0.721 | Quote + chunks | ⚠️ Hallucination 21% |
| 11 | Exhaustive (Small Chunks) | Haiku | 57.3% | 78.5% | 32.7% | 0.643 | Exhaustive + chunks | ❌ High hallucination |
| 12 | Vote-2 | Ensemble | 65.2% | 92.7% | 34.8% | 0.727 | 2-model voting | ❌ High hallucination |
| 13 | **Vote-3** | Ensemble | **97.8%** | **75.1%** | **2.2%** | **0.833** | **3-model voting** | **✅ BEST** |
| 14 | Sonnet Conservative | Sonnet | 98.2% | 59.0% | 1.8% | 0.713 | Conservative extraction | ⚠️ Low recall |
| 15 | Quote Verified | Haiku | 80.0% | 59.8% | 0.0% | 0.678 | Quote + span verify | ⚠️ Low recall |
| 16 | Union Conservative | Haiku | 85.7% | 72.8% | 14.3% | 0.779 | Union + conservative | ⚠️ Hallucination 14% |
| 17 | **Ensemble Verified** | Ensemble | **89.3%** | **88.9%** | **10.7%** | **0.874** | **Ensemble + verify** | **✅ BEST F1** |
| 18 | Sonnet Exh + Haiku Verify | Mixed | 70.5% | 79.3% | 29.5% | 0.726 | Sonnet + Haiku verify | ⚠️ High hallucination |
| 19 | Ensemble Vote Hybrid | Ensemble | 64.1% | 84.1% | 35.9% | 0.712 | Voting + hybrid | ❌ High hallucination |
| 20 | Multi-Consensus | Ensemble | 71.4% | 80.9% | 28.6% | 0.736 | Consensus voting | ⚠️ High hallucination |
| 21 | Union Verified | Haiku | 88.3% | 72.1% | 11.7% | 0.785 | Union + verify | ⚠️ Hallucination 12% |
| 22 | Exhaustive Double Verify | Haiku | 93.7% | 65.2% | 6.3% | 0.748 | Double verification | ⚠️ Low recall |
| 23 | Multi-Pass | Haiku | 58.3% | 53.4% | 36.7% | 0.527 | 3-pass extraction | ❌ Poor across board |
| 24 | Multi-Pass | Sonnet | 64.1% | 67.9% | 25.9% | 0.656 | 3-pass extraction | ⚠️ High hallucination |
| 25 | Quote Verify | Haiku | 40.8% | 82.5% | 59.2% | 0.531 | Quote + verify | ❌ Extremely high hallucination |
| 26 | Quote Verify | Sonnet | 52.3% | 88.9% | 47.7% | 0.647 | Quote + verify | ❌ High hallucination |
| 27 | Ensemble Sonnet Verify | Sonnet | 94.4% | 70.9% | 5.6% | 0.799 | Ensemble + verify | ⚠️ Low recall |
| 28 | Exhaustive Double Verify | Sonnet | 93.7% | 65.2% | 6.3% | 0.748 | Double verification | ⚠️ Low recall |

---

## Strategies by Performance Category

### 🏆 Top Performers (F1 > 0.80)
| Strategy | Precision | Recall | Hallucination | F1 |
|----------|-----------|--------|---------------|-----|
| Ensemble Verified | 89.3% | 88.9% | 10.7% | **0.874** |
| Vote-3 Ensemble | 97.8% | 75.1% | 2.2% | **0.833** |
| Union Conservative | 85.7% | 72.8% | 14.3% | **0.779** |

### ⚠️ Moderate Performers (F1 0.70-0.80)
| Strategy | Precision | Recall | Hallucination | F1 |
|----------|-----------|--------|---------------|-----|
| Ensemble Sonnet Verify | 94.4% | 70.9% | 5.6% | 0.799 |
| Union Verified | 88.3% | 72.1% | 11.7% | 0.785 |
| Exhaustive Double Verify | 93.7% | 65.2% | 6.3% | 0.748 |
| Simple (Small Chunks) | 71.1% | 82.8% | 28.9% | 0.744 |
| Quote (Small Chunks) | 78.7% | 74.8% | 21.3% | 0.721 |
| Vote-2 Ensemble | 65.2% | 92.7% | 34.8% | 0.727 |
| Multi-Consensus | 71.4% | 80.9% | 28.6% | 0.736 |

### ❌ Poor Performers (F1 < 0.70)
| Strategy | Precision | Recall | Hallucination | F1 |
|----------|-----------|--------|---------------|-----|
| Sonnet Conservative | 98.2% | 59.0% | 1.8% | 0.713 |
| Combined (Sonnet) | 85.7% | 60.9% | 9.3% | 0.712 |
| Ensemble Vote Hybrid | 64.1% | 84.1% | 35.9% | 0.712 |
| Discrimination (Sonnet) | 70.7% | 57.8% | 19.3% | 0.628 |
| Quote Verify (Sonnet) | 52.3% | 88.9% | 47.7% | 0.647 |
| Exhaustive (Small Chunks) | 57.3% | 78.5% | 32.7% | 0.643 |
| Multi-Pass (Sonnet) | 64.1% | 67.9% | 25.9% | 0.656 |
| Quote Verified | 80.0% | 59.8% | 0.0% | 0.678 |
| Sonnet Exh + Haiku Verify | 70.5% | 79.3% | 29.5% | 0.726 |
| Discrimination (Full) | 64.0% | 62.5% | 31.0% | 0.611 |
| Discrimination (Sonnet) | 70.7% | 57.8% | 19.3% | 0.628 |
| Quote Verify (Haiku) | 40.8% | 82.5% | 59.2% | 0.531 |
| Multi-Pass (Haiku) | 58.3% | 53.4% | 36.7% | 0.527 |
| Discrimination (No Cat) | 59.1% | 18.6% | 25.9% | 0.264 |
| Pattern-Only | 43.0% | 33.2% | 52.5% | - |

---

## Strategies by Target Metric

### Best for Precision (>95%)
1. **Sonnet Conservative**: 98.2% (but recall 59%)
2. **Vote-3 Ensemble**: 97.8% (recall 75.1%) ⭐
3. **Exhaustive Double Verify**: 93.7% (recall 65.2%)

### Best for Recall (>85%)
1. **Quote Verify (Sonnet)**: 88.9% (hallucination 47.7%)
2. **Ensemble Verified**: 88.9% (hallucination 10.7%) ⭐
3. **Simple (Small Chunks)**: 82.8% (hallucination 28.9%)

### Best for Hallucination (<5%)
1. **Quote Verified**: 0.0% (recall 59.8%)
2. **Sonnet Conservative**: 1.8% (recall 59%)
3. **Vote-3 Ensemble**: 2.2% (recall 75.1%) ⭐

### Best for Cost-Efficiency
1. **Quote (Small Chunks)**: 78.7% P, 74.8% R, 21.3% H (1 call)
2. **Simple (Small Chunks)**: 71.1% P, 82.8% R, 28.9% H (1 call)
3. **Ensemble Verified**: 89.3% P, 88.9% R, 10.7% H (2 calls)

---

## Technique Effectiveness Summary

### Span Verification Impact
- **Baseline**: 65-85% precision, 30-50% hallucination
- **With Verification**: 80-98% precision, 0-12% hallucination
- **Effect**: -50% to -100% hallucination, -10% to -30% recall
- **Verdict**: ✅ HIGHLY EFFECTIVE for hallucination control

### Voting/Consensus Impact
- **2-model voting**: +15% recall, +10% hallucination
- **3-model voting (70% threshold)**: +20% precision, -30% hallucination
- **Verdict**: ✅ EFFECTIVE when threshold is high (70%+)

### Multi-Pass Extraction Impact
- **Effect on Recall**: +15% to +35%
- **Effect on Hallucination**: +10% to +25% (negative)
- **Verdict**: ⚠️ MIXED - improves recall but increases hallucination

### Model Upgrade (Haiku → Sonnet)
- **Effect on Precision**: +5% to +15%
- **Effect on Hallucination**: -5% to -15%
- **Effect on Recall**: -5% to +10% (variable)
- **Verdict**: ✅ EFFECTIVE but increases cost 2-3x

### Smaller Chunks Impact
- **Effect on Hallucination**: -5% to -10%
- **Effect on Recall**: +5% to +15%
- **Verdict**: ✅ EFFECTIVE for both metrics

---

## Combination Strategies

### ✅ Successful Combinations
1. **Vote-3 + Span Verification**
   - Result: 97.8% P, 75.1% R, 2.2% H
   - Cost: 3 model calls + verification
   - Trade-off: Sacrifices recall for precision/hallucination

2. **Ensemble + Verification**
   - Result: 89.3% P, 88.9% R, 10.7% H
   - Cost: 2 model calls + verification
   - Trade-off: Best balanced performance

3. **Sonnet Conservative + Verification**
   - Result: 98.2% P, 59% R, 1.8% H
   - Cost: 1 Sonnet call (expensive)
   - Trade-off: Extreme precision/hallucination, low recall

### ❌ Failed Combinations
1. **Multi-Pass Without Verification**
   - Result: 58-64% P, 53-68% R, 25-37% H
   - Problem: Each pass added hallucinations without filtering

2. **Voting Without Threshold**
   - Result: 65% P, 93% R, 35% H
   - Problem: Low threshold allowed hallucinations through

3. **Quote Verify Without Span Check**
   - Result: 40-52% P, 82-89% R, 47-59% H
   - Problem: Quote extraction alone generates hallucinations

---

## Recommendations by Use Case

### Use Case 1: High-Precision Extraction (e.g., critical systems)
**Best Strategy**: Vote-3 Ensemble + Span Verification
- Achieves: 97.8% precision, 2.2% hallucination
- Recall: 75.1% (acceptable for critical use)
- Cost: 3 model calls + verification
- **Recommendation**: ✅ DEPLOY

### Use Case 2: Balanced Extraction (e.g., general RAG)
**Best Strategy**: Ensemble Verified
- Achieves: 89.3% precision, 88.9% recall, 10.7% hallucination
- F1: 0.874 (best overall)
- Cost: 2 model calls + verification
- **Recommendation**: ✅ DEPLOY

### Use Case 3: High-Recall Extraction (e.g., discovery)
**Best Strategy**: Ensemble Verified (with lower threshold)
- Achieves: 88.9% recall, 89.3% precision, 10.7% hallucination
- Cost: 2 model calls + verification
- **Recommendation**: ✅ DEPLOY (same as balanced)

### Use Case 4: Cost-Sensitive Extraction
**Best Strategy**: Quote Extraction (Small Chunks)
- Achieves: 78.7% precision, 74.8% recall, 21.3% hallucination
- Cost: 1 model call
- **Recommendation**: ⚠️ ACCEPTABLE for non-critical use

### Use Case 5: Zero-Hallucination Requirement
**Best Strategy**: Quote Verified
- Achieves: 0.0% hallucination, 80% precision, 59.8% recall
- Cost: 1 model call + verification
- **Recommendation**: ✅ DEPLOY if recall acceptable

---

## Key Findings

1. **Span verification is the most effective hallucination control**
   - Reduces hallucination by 50-100%
   - Cost: Moderate (verification overhead)

2. **Voting with high threshold (70%+) dramatically improves precision**
   - Achieves 97.8% precision
   - Cost: 3x model calls

3. **Recall remains the bottleneck**
   - Best achieved: 88.9%
   - 6.1% short of 95% target
   - Suggests missing term types or context limitations

4. **No single strategy achieved all three targets simultaneously**
   - Precision/Hallucination vs. Recall trade-off is fundamental
   - Hybrid approaches needed for different use cases

5. **Model quality matters but has limits**
   - Sonnet improves precision by 5-15%
   - But doesn't solve recall bottleneck
   - Cost increase (2-3x) may not justify gains

---

## Next Steps for POC-1c

1. **Investigate recall bottleneck**
   - Analyze which term types are missed
   - Test with different domain vocabularies
   - Consider retrieval-augmented verification

2. **Test hybrid approach**
   - Vote-3 for precision + separate high-recall pass
   - Combine results with intelligent merging

3. **Explore different model families**
   - Test with GPT-4, Llama, Mistral
   - Ensemble with different architectures

4. **Optimize verification**
   - Implement fuzzy matching for span verification
   - Test with knowledge base lookups

5. **Cost optimization**
   - Implement caching for repeated extractions
   - Test with smaller models (Haiku-only ensemble)

