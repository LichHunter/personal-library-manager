# Chunking Strategy Benchmark

## What

Benchmarked 6 document chunking strategies for RAG retrieval accuracy on a synthetic 16-document corpus with 30 ground-truth queries.

## Why

The Personal Knowledge Assistant needs to chunk 1000+ documents for semantic search. The chunking strategy directly impacts retrieval quality and ultimately answer accuracy. We needed data to choose between fixed-size, heading-based, paragraph-based, and hierarchical approaches.

## Hypothesis

Expected heading-based or hierarchical strategies to win due to preserving semantic boundaries. Fixed-size was expected to perform worst.

## Results

| Strategy | Recall@5 | MRR | Chunks | Avg Tokens | Index Time |
|----------|----------|-----|--------|------------|------------|
| **paragraphs_50_256** | **96.4%** | **0.940** | 103 | 55 | 76ms |
| heading_based_h3 | 91.4% | 0.912 | 220 | 25 | 114ms |
| heading_limited_512 | 91.4% | 0.874 | 188 | 29 | 98ms |
| hierarchical_h4 | 89.4% | 0.929 | 257 | 22 | 111ms |
| fixed_size_512 | 88.6% | 0.908 | 33 | 196 | 264ms |
| heading_paragraph_h3 | 86.1% | 0.894 | 511 | 22 | 193ms |

## Key Findings

### 1. Paragraph-based chunking wins decisively

The `paragraphs_50_256` strategy achieved 96.4% Recall@5, outperforming all others by 5+ percentage points. This was unexpected - we hypothesized that heading-aware strategies would preserve context better.

**Why paragraphs work well:**
- Natural semantic boundaries (authors write in coherent paragraphs)
- Consistent chunk sizes (50-256 token range prevents outliers)
- Merging small paragraphs prevents fragmentation
- Each chunk is self-contained and readable

### 2. Chunk size sweet spot: 50-100 tokens

| Avg Tokens | Best Recall@5 |
|------------|---------------|
| 22-29 | 86-91% |
| 55 | **96.4%** |
| 196 | 88.6% |

Very small chunks (heading-based, ~25 tokens) lose context. Very large chunks (fixed-size, ~200 tokens) dilute relevance with off-topic content. The 50-100 token range balances specificity with context.

### 3. More chunks ≠ better retrieval

| Strategy | Chunks | Recall@5 |
|----------|--------|----------|
| heading_paragraph_h3 | 511 | 86.1% (worst) |
| paragraphs_50_256 | 103 | 96.4% (best) |

The heading_paragraph strategy created 5x more chunks but performed worst. Too many tiny chunks means:
- Query matches spread across many low-relevance chunks
- Top-K results contain duplicates/near-duplicates
- Each chunk lacks sufficient context

### 4. Fixed-size is a reasonable baseline

Despite cutting mid-sentence, fixed_size_512 achieved 88.6% Recall@5. The sentence-boundary heuristic helped. For simple use cases, this is viable.

### 5. Hierarchical complexity didn't pay off

The hierarchical strategy (parent-child relationships) added implementation complexity but performed mid-pack (89.4%). The benefit of multi-granularity retrieval didn't materialize in this benchmark.

## Recommendations

### Primary: Use paragraph-based chunking

```python
ParagraphStrategy(min_tokens=50, max_tokens=256)
```

- Split on `\n\n` (paragraph boundaries)
- Merge consecutive paragraphs under 50 tokens
- Split paragraphs over 256 tokens at sentence boundaries
- Prepend section heading to each chunk for context

### Secondary considerations

1. **Heading context**: Prepend the current section heading to each paragraph chunk. This preserves structural context without the complexity of hierarchical storage.

2. **Code blocks**: Treat code blocks as atomic units - don't split them.

3. **Tables/lists**: Keep together when possible, split at row/item boundaries if too large.

## Limitations

1. **Small corpus**: 16 documents, 30 queries. Results may differ at scale.
2. **Synthetic data**: Real documents have more variety in structure.
3. **Single embedding model**: Tested only all-MiniLM-L6-v2.
4. **No LLM evaluation**: Measured retrieval, not final answer quality.

## Files

```
poc/chunking_benchmark/
├── README.md                 # This file
├── DESIGN.md                 # Original experiment design
├── run_benchmark.py          # Main benchmark script
├── pyproject.toml            # Dependencies
│
├── strategies/               # Chunking implementations
│   ├── base.py               # Document, Chunk, ChunkingStrategy
│   ├── fixed_size.py         # 512 tokens + overlap
│   ├── heading_based.py      # Split by H1-H3
│   ├── heading_limited.py    # Heading + size limit
│   ├── hierarchical.py       # Parent-child tree
│   ├── paragraphs.py         # Paragraph-based (WINNER)
│   └── heading_paragraph.py  # Hybrid
│
├── corpus/
│   ├── generate_corpus.py    # Generates 16 CloudFlow docs
│   ├── documents/            # Generated markdown files
│   ├── corpus_metadata.json  # Document index
│   └── ground_truth.json     # 30 test queries
│
├── evaluation/               # Benchmark harness
│   ├── embedder.py           # Sentence-transformers wrapper
│   ├── retriever.py          # Cosine similarity search
│   ├── metrics.py            # Recall@K, MRR calculations
│   └── benchmark.py          # Full benchmark runner
│
└── results/
    ├── benchmark_summary.json
    └── report.md
```

## Running the Benchmark

```bash
cd /home/fujin/Code/personal-library-manager
source .venv/bin/activate
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"  # For CUDA on NixOS

cd poc/chunking_benchmark
python run_benchmark.py
```

Requires: `sentence-transformers`, `numpy`, `torch` (with CUDA for speed)
