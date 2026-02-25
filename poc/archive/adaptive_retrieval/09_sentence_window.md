# 09. Sentence Window Retrieval

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P1 |
| **Complexity** | LOW |
| **Expected Improvement** | Better context for precise matches |
| **PLM Changes Required** | Store sentence-level + window metadata |
| **External Dependencies** | None |

## Description

Index at sentence level for precise matching, but return a window of surrounding sentences for context. Each sentence knows its neighbors.

## How It Works

```
1. Index individual sentences (very precise embeddings)
2. Store window metadata: surrounding N sentences
3. At retrieval: match sentences, return windows
```

## Why It Works

- Sentence embeddings are highly precise (single concept)
- But single sentence rarely enough to answer
- Window provides context without full document
- Good middle ground between chunk and section

## Algorithm

### Indexing

```python
def index_with_windows(document: str, window_size: int = 5):
    sentences = split_sentences(document)
    
    for i, sentence in enumerate(sentences):
        # Calculate window bounds
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)
        
        # Store sentence with window metadata
        store(
            content=sentence,
            embedding=embed(sentence),
            metadata={
                "sentence_idx": i,
                "window_start": start,
                "window_end": end,
                "doc_id": doc_id
            }
        )
```

### Retrieval

```python
def retrieve_with_windows(query: str, k: int = 5):
    # 1. Search sentences
    sentences = search(query, k=k)
    
    # 2. Expand to windows
    results = []
    for sent in sentences:
        window_sentences = get_sentences(
            doc_id=sent.doc_id,
            start=sent.window_start,
            end=sent.window_end
        )
        window_text = " ".join(window_sentences)
        results.append(window_text)
    
    return results
```

## Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `window_size` | Sentences on each side | 3-5 |
| `overlap_handling` | How to handle overlapping windows | merge |
| `sentence_splitter` | How to split sentences | spacy or nltk |

## Window Size Tradeoffs

| Window Size | Pros | Cons |
|-------------|------|------|
| 1-2 | Very precise | May lack context |
| 3-5 | Good balance | Default choice |
| 7-10 | More context | Approaches chunk size |

## Comparison to Chunks

| Aspect | Sentence Window | Chunks |
|--------|-----------------|--------|
| Indexing granularity | Sentence | 512 tokens |
| Embedding precision | Very high | Medium |
| Context returned | Window | Fixed chunk |
| Adaptability | Window size tunable | Fixed |
| Storage | Higher (more embeddings) | Lower |

## PLM Consideration

PLM currently indexes at ~512 token chunks. Switching to sentence-level would require:
1. Re-chunking all documents to sentences
2. Re-embedding (many more embeddings)
3. Significant storage increase

**May be better as enhancement, not replacement:**
- Keep chunk index for broad retrieval
- Add sentence index for precision (hybrid)

## Implementation Steps

1. Implement sentence splitter
2. Create sentence-level index with window metadata
3. Implement window expansion at retrieval
4. Benchmark: chunks vs sentence-windows

## Storage Analysis

For PLM corpus:
- Current: ~20K chunks
- Sentence level: ~200K-500K sentences (10-25x more)
- Embedding storage: 10-25x increase

## References

- [LlamaIndex SentenceWindowNodeParser](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo/)
- [Sentence Window Retrieval Blog](https://www.llamaindex.ai/blog/advanced-rag-patterns)
