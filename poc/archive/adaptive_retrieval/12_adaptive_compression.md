# 12. Adaptive Context Compression

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P3 |
| **Complexity** | HIGH |
| **Expected Improvement** | Efficient context usage |
| **PLM Changes Required** | Compression model + logic |
| **External Dependencies** | Compression model (LLM or trained) |

## Description

Instead of adapting retrieval granularity, retrieve broadly then compress context to fit query needs. Variable compression rate based on query complexity.

## How It Works

```
1. Retrieve fixed set of documents
2. Analyze query complexity
3. Apply compression:
   - Simple query → Heavy compression (extract key facts)
   - Complex query → Light compression (preserve detail)
4. Return compressed context
```

## Why It Works

- LLM context windows are limited and expensive
- Not all retrieved content is equally relevant
- Compression removes noise, keeps signal
- Adaptive rate matches query needs

## Compression Methods

### Method A: Extractive (Select Sentences)

```python
def extract_compress(query: str, documents: list[str], ratio: float = 0.3):
    all_sentences = []
    for doc in documents:
        all_sentences.extend(split_sentences(doc))
    
    # Score sentences by query relevance
    query_emb = embed(query)
    scored = []
    for sent in all_sentences:
        sent_emb = embed(sent)
        score = cosine_similarity(query_emb, sent_emb)
        scored.append((sent, score))
    
    # Select top by ratio
    scored.sort(key=lambda x: x[1], reverse=True)
    n_select = int(len(scored) * ratio)
    selected = [s[0] for s in scored[:n_select]]
    
    return " ".join(selected)
```

### Method B: Abstractive (LLM Summarization)

```python
def abstractive_compress(query: str, documents: list[str], target_tokens: int = 500):
    context = "\n\n".join(documents)
    
    prompt = f"""Compress the following context to answer this query.
Keep ONLY information relevant to: {query}
Target length: ~{target_tokens} tokens

Context:
{context}

Compressed context:"""
    
    return llm.generate(prompt, max_tokens=target_tokens)
```

### Method C: Learned Compression (LLMLingua Style)

```python
# Uses trained model to score token importance
def learned_compress(query: str, context: str, ratio: float = 0.5):
    # Score each token's importance
    token_scores = compression_model.score_tokens(query, context)
    
    # Keep top tokens by score
    tokens = tokenize(context)
    keep_n = int(len(tokens) * ratio)
    
    top_indices = token_scores.argsort()[-keep_n:]
    top_indices.sort()  # Maintain order
    
    compressed = [tokens[i] for i in top_indices]
    return detokenize(compressed)
```

## Adaptive Rate Selection

```python
def get_compression_rate(query: str) -> float:
    # Simple heuristics
    words = query.split()
    
    if len(words) < 5:  # Simple query
        return 0.2  # Heavy compression
    elif "explain" in query.lower() or "how" in query.lower():
        return 0.6  # Light compression
    else:
        return 0.4  # Medium
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `method` | extractive, abstractive, learned | extractive (start) |
| `base_rate` | Default compression ratio | 0.4 |
| `min_tokens` | Minimum output size | 200 |
| `max_tokens` | Maximum output size | 1000 |

## Tradeoffs

| Pros | Cons |
|------|------|
| Works with any retrieval | Compression can lose info |
| Optimizes context window | Adds processing latency |
| Reduces LLM costs | Extractive may be choppy |
| Adaptive to query | Abstractive may hallucinate |

## PLM Consideration

Lower priority because:
- Adds complexity and latency
- Risk of losing important information
- Retrieval-side adaptation (granularity) may be cleaner

Consider if:
- Context window is severely limited
- LLM token costs are major concern
- After retrieval optimization exhausted

## Implementation Steps

1. Implement extractive compression (simplest)
2. Add adaptive rate selection
3. Benchmark: uncompressed vs compressed
4. Measure information loss (answer quality)
5. (Optional) Add abstractive compression

## References

- [Adaptive Context Compression Paper](https://arxiv.org/abs/2507.22931)
- [LLMLingua](https://github.com/microsoft/LLMLingua)
- [RECOMP](https://arxiv.org/abs/2310.04408)
