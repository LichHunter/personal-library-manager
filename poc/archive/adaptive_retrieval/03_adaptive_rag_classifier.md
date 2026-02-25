# 03. Adaptive-RAG Query Classifier

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P1 |
| **Complexity** | MEDIUM |
| **Expected Improvement** | Routes to optimal granularity |
| **PLM Changes Required** | Add classifier + routing logic |
| **External Dependencies** | Small classifier model (optional) |

## Description

Train or build a classifier that predicts query complexity from the query text alone, then route to the appropriate retrieval strategy.

## How It Works

```
1. Query → Classifier → Complexity Level
2. Route based on level:
   - SIMPLE → Chunk retrieval (no expansion)
   - MODERATE → Heading-level retrieval
   - COMPLEX → Document-level or iterative retrieval
```

## Why It Works

Research shows query complexity is **predictable from query text alone**:
- Short, specific queries → usually need less context
- "How to" queries → usually need procedures (more context)
- "Why" / "Explain" queries → need comprehensive context

## Classifier Options

### Option A: Rule-Based Heuristics (No Training)

```python
def classify_query(query: str) -> str:
    query_lower = query.lower()
    words = query.split()
    
    # Length heuristic
    if len(words) < 5:
        return "chunk"
    
    # Question type heuristics
    if query_lower.startswith(("what is", "what's the", "define")):
        return "chunk"
    if query_lower.startswith(("how do i", "how to", "steps to", "guide")):
        return "heading"
    if query_lower.startswith(("why", "explain", "describe how")):
        return "heading"
    if "vs" in query_lower or "compare" in query_lower or "difference" in query_lower:
        return "multi_heading"
    
    return "heading"  # Default
```

### Option B: Small BERT Classifier (Requires Training)

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3  # chunk, heading, document
)

def classify_query(query: str) -> str:
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class = outputs.logits.argmax().item()
    return ["chunk", "heading", "document"][predicted_class]
```

### Option C: Zero-Shot LLM (No Training, Higher Latency)

```python
def classify_query(query: str) -> str:
    prompt = f"""Classify this query's complexity:
Query: {query}

Reply with ONE word: SIMPLE, MODERATE, or COMPLEX"""
    
    response = llm.generate(prompt, max_tokens=10)
    # Parse response...
```

## Training Data Generation

For Option B, generate labels from oracle experiments:
1. Run retrieval at all granularities (chunk, heading, document)
2. Label each query with smallest granularity that produces correct answer
3. Train classifier on (query, label) pairs

## Algorithm

```python
def adaptive_retrieve(query: str, k: int = 10):
    complexity = classify_query(query)
    
    if complexity == "chunk":
        return retrieve_chunks(query, k=k)
    elif complexity == "heading":
        return retrieve_headings(query, k=k)  # Use heading embeddings
    elif complexity == "document":
        return retrieve_documents(query, k=k)  # Use document embeddings
    else:  # multi_heading
        chunks = retrieve_chunks(query, k=k*2)
        return expand_to_headings(chunks)
```

## Parameters

| Parameter | Description | Options |
|-----------|-------------|---------|
| `classifier_type` | Which classifier to use | rules, bert, llm |
| `confidence_threshold` | Min confidence to trust prediction | 0.7 |
| `fallback_granularity` | Default if uncertain | heading |

## Tradeoffs

| Option | Latency | Accuracy | Training Needed |
|--------|---------|----------|-----------------|
| Rules | ~0ms | Medium | No |
| BERT | ~10ms | High | Yes |
| LLM | ~200ms | High | No |

## Implementation Steps

1. Implement rule-based classifier (baseline)
2. Create labeled dataset from benchmark queries
3. (Optional) Train BERT classifier
4. Add routing logic to retriever
5. Benchmark each classifier option

## References

- [Adaptive-RAG Paper (NAACL 2024)](https://arxiv.org/abs/2403.14403)
- [GitHub: starsuzi/Adaptive-RAG](https://github.com/starsuzi/Adaptive-RAG)
