# 07. Self-RAG (Self-Reflective RAG)

## Overview

| Attribute | Value |
|-----------|-------|
| **Priority** | P3 |
| **Complexity** | HIGH |
| **Expected Improvement** | Better retrieval decisions |
| **PLM Changes Required** | Fine-tuned LLM with special tokens |
| **External Dependencies** | Fine-tuned model |

## Description

Fine-tune an LLM to generate special "reflection" tokens that decide whether to retrieve, assess relevance of retrieved content, and critique its own outputs.

## How It Works

```
1. LLM generates with special tokens inline:
   - [Retrieve]: yes/no/continue - Should I retrieve?
   - [IsRel]: relevant/irrelevant - Is this passage relevant?
   - [IsSup]: supported/partial/none - Does evidence support response?
   - [IsUse]: useful/not useful - Is response useful?

2. System acts on tokens:
   - If [Retrieve]=yes → Fetch documents
   - If [IsRel]=irrelevant → Skip passage
   - If [IsSup]=none → Retrieve more or regenerate
```

## Why It Works

- LLM learns WHEN retrieval helps (not always)
- Self-critique catches hallucinations
- Adaptive: different behavior for different queries
- End-to-end optimization (retrieval + generation)

## Special Tokens

| Token | Values | Purpose |
|-------|--------|---------|
| `[Retrieve]` | yes, no, continue | Should retrieve now? |
| `[IsRel]` | relevant, irrelevant | Passage relevance |
| `[IsSup]` | fully, partially, no | Evidence support level |
| `[IsUse]` | 1-5 | Response utility score |

## Algorithm

```python
def self_rag_generate(query: str, corpus: Index):
    context = []
    
    while True:
        # Generate with reflection tokens
        output = llm.generate(
            query=query,
            context=context,
            enable_reflection_tokens=True
        )
        
        # Parse tokens from output
        if "[Retrieve]=yes" in output:
            new_docs = corpus.search(query)
            # Filter by relevance
            for doc in new_docs:
                relevance = llm.generate(f"[IsRel] for: {doc}")
                if relevance == "relevant":
                    context.append(doc)
        
        if "[IsSup]=fully" in output:
            # Confident answer
            return extract_answer(output)
        
        if "[IsSup]=no" in output:
            # Need more evidence
            continue
```

## Training Requirements

1. **Base model**: 7B+ parameter LLM
2. **Training data**: Examples with reflection tokens labeled
3. **Fine-tuning**: Supervised fine-tuning on labeled examples
4. **Compute**: Significant GPU resources

## Parameters

| Parameter | Description | Notes |
|-----------|-------------|-------|
| `base_model` | Starting LLM | Llama-2-7B, Mistral-7B |
| `reflection_tokens` | Which tokens to use | All 4 or subset |
| `retrieval_threshold` | When to act on [Retrieve] | 0.5 confidence |

## Tradeoffs

| Pros | Cons |
|------|------|
| End-to-end optimization | Requires fine-tuning |
| Self-correcting | High compute cost |
| Learns when to retrieve | Complex training pipeline |
| State-of-the-art results | Model-specific (not plug-and-play) |

## PLM Applicability

**Low priority for PLM because:**
- Requires fine-tuning (we use API models)
- High complexity
- Simpler approaches (reranking, auto-merge) likely sufficient
- Consider only if other approaches fail

## Implementation Steps (If Pursued)

1. Collect training data with reflection labels
2. Fine-tune base model with special tokens
3. Implement token parsing in generation loop
4. Integrate with PLM retrieval
5. Extensive evaluation

## References

- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [GitHub: AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)
- [HuggingFace Models](https://huggingface.co/selfrag)
