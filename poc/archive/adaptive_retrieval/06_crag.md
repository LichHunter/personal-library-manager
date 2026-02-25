# 06. CRAG (Corrective Retrieval-Augmented Generation)

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P2 |
| **Complexity** | MEDIUM |
| **Expected Improvement** | Reduces bad retrievals |
| **PLM Changes Required** | Add retrieval evaluator + correction logic |
| **External Dependencies** | Evaluator model (small LLM or classifier) |

## Description

Add a lightweight evaluator that assesses retrieval quality and triggers corrective actions when retrieval is poor or ambiguous.

## How It Works

```
1. Retrieve documents via standard pipeline
2. Evaluator scores retrieval quality: CORRECT / INCORRECT / AMBIGUOUS
3. Take action based on score:
   - CORRECT → Use retrieved context as-is
   - INCORRECT → Discard, try alternative strategy (web search, different query)
   - AMBIGUOUS → Decompose query, retrieve more context
```

## Why It Works

- Standard RAG blindly trusts retrieval results
- Bad retrieval → bad generation (garbage in, garbage out)
- Evaluator catches failures before they reach LLM
- Enables graceful degradation and recovery

## Evaluator Options

### Option A: Small Classifier

```python
class RetrievalEvaluator:
    def __init__(self):
        self.model = load_classifier("retrieval-quality-classifier")
    
    def evaluate(self, query: str, documents: list[str]) -> str:
        # Classify as CORRECT / INCORRECT / AMBIGUOUS
        features = self.extract_features(query, documents)
        return self.model.predict(features)
    
    def extract_features(self, query, docs):
        # Features: keyword overlap, embedding similarity, doc count, etc.
        pass
```

### Option B: LLM-Based Evaluator

```python
def evaluate_retrieval(query: str, documents: list[str]) -> str:
    doc_text = "\n---\n".join(documents[:3])  # Top 3
    
    prompt = f"""Evaluate if these documents can answer the query.

Query: {query}

Documents:
{doc_text}

Rate as:
- CORRECT: Documents directly answer the query
- AMBIGUOUS: Documents partially relevant, need more info
- INCORRECT: Documents not relevant to query

Reply with ONE word: CORRECT, AMBIGUOUS, or INCORRECT"""
    
    response = llm.generate(prompt, max_tokens=10)
    return parse_response(response)
```

## Algorithm

```python
def crag_retrieve(query: str, k: int = 10):
    # 1. Initial retrieval
    documents = retrieve(query, k=k)
    
    # 2. Evaluate
    quality = evaluate_retrieval(query, documents)
    
    if quality == "CORRECT":
        return documents
    
    elif quality == "AMBIGUOUS":
        # Expand context
        expanded = expand_to_headings(documents)
        return expanded
    
    elif quality == "INCORRECT":
        # Try alternative strategies
        # Option 1: Reformulate query
        new_query = reformulate_query(query)
        documents = retrieve(new_query, k=k)
        
        # Option 2: Web search fallback (if available)
        # documents = web_search(query)
        
        return documents
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `evaluator_type` | classifier or llm | classifier (faster) |
| `ambiguous_action` | What to do for AMBIGUOUS | expand |
| `incorrect_action` | What to do for INCORRECT | reformulate |
| `max_retries` | Max correction attempts | 2 |

## Tradeoffs

| Pros | Cons |
|------|------|
| Catches retrieval failures | Adds evaluator latency |
| Enables recovery strategies | Evaluator can be wrong too |
| Improves overall reliability | More complex pipeline |
| Explicit quality signal | Needs evaluator training/tuning |

## Implementation Steps

1. Implement feature-based retrieval quality heuristics
2. Create labeled dataset of (query, docs, quality) tuples
3. Train lightweight classifier or use LLM evaluator
4. Implement correction strategies (expand, reformulate)
5. Benchmark: standard RAG vs CRAG

## Metrics

| Metric | Description |
|--------|-------------|
| Evaluator Accuracy | Does evaluator correctly classify quality? |
| Recovery Rate | % of INCORRECT retrievals successfully corrected |
| False Positive Rate | CORRECT classified as INCORRECT (causes unnecessary work) |

## References

- [CRAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph Adaptive RAG Tutorial](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)
