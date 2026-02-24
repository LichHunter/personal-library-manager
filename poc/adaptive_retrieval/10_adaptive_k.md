# 10. Adaptive-k Retrieval

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P2 |
| **Complexity** | MEDIUM |
| **Expected Improvement** | Optimal document count per query |
| **PLM Changes Required** | Dynamic k selection logic |
| **External Dependencies** | None or small classifier |

## Description

Instead of fixed top-k retrieval, dynamically select how many documents to retrieve based on query characteristics and retrieval confidence.

## How It Works

```
1. Analyze query characteristics
2. Predict optimal k (number of documents)
3. Retrieve top-k with predicted k
4. Optionally: adjust based on retrieval scores
```

## Why It Works

- Simple queries may need k=3 (focused)
- Complex queries may need k=15 (broad coverage)
- Fixed k either over-retrieves or under-retrieves
- Adaptive k optimizes context window usage

## Methods

### Method A: Query-Based Prediction

```python
def predict_k(query: str) -> int:
    # Simple heuristics
    words = query.split()
    
    # Short specific query → fewer docs
    if len(words) < 5:
        return 5
    
    # Comparison/multi-aspect → more docs
    if "compare" in query.lower() or "vs" in query.lower():
        return 15
    
    # How-to/procedural → medium
    if query.lower().startswith(("how", "steps", "guide")):
        return 10
    
    return 8  # Default
```

### Method B: Score-Based Cutoff

```python
def adaptive_retrieve(query: str, max_k: int = 20, score_threshold: float = 0.7):
    # Retrieve more than needed
    results = retrieve(query, k=max_k)
    
    # Cut off at score drop
    filtered = []
    for i, result in enumerate(results):
        if i == 0:
            filtered.append(result)
        elif result.score >= score_threshold * results[0].score:
            filtered.append(result)
        else:
            break  # Score dropped below threshold
    
    return filtered
```

### Method C: Trained Classifier

```python
class KPredictor:
    def __init__(self):
        self.model = load_model("k-predictor")
    
    def predict(self, query: str) -> int:
        features = extract_features(query)
        k = self.model.predict(features)
        return max(3, min(20, int(k)))  # Clamp to reasonable range
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `min_k` | Minimum documents | 3 |
| `max_k` | Maximum documents | 20 |
| `score_threshold` | Relative score cutoff | 0.7 |
| `method` | query, score, or classifier | score (simple start) |

## Expected Behavior

| Query Type | Predicted k | Reason |
|------------|-------------|--------|
| "What is X?" | 3-5 | Single concept |
| "How do I X?" | 8-10 | Procedure, multiple steps |
| "Compare X and Y" | 12-15 | Multiple aspects |
| "Explain X architecture" | 10-15 | Comprehensive coverage |

## Tradeoffs

| Pros | Cons |
|------|------|
| Optimizes context usage | Prediction can be wrong |
| Reduces noise for simple queries | Adds decision complexity |
| Better coverage for complex queries | May miss relevant docs if k too low |

## Implementation Steps

1. Implement score-based cutoff (simplest)
2. Analyze query distribution to set thresholds
3. (Optional) Add query-based heuristics
4. (Optional) Train k predictor on labeled data
5. Benchmark: fixed k vs adaptive k

## Metrics

| Metric | Description |
|--------|-------------|
| Average k | How many docs retrieved on average |
| Context efficiency | Relevant tokens / total tokens |
| Recall impact | Did lower k hurt recall? |

## References

- [Adaptive-k Paper](https://arxiv.org/abs/2506.08479)
- [Efficient Context Selection for Long-Context QA](https://arxiv.org/abs/2506.08479)
