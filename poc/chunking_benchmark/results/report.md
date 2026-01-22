# Chunking Benchmark Results

**Corpus:** 16 documents, 30 test queries
**Device:** cuda

## Summary

| Strategy | Chunks | Avg Tokens | Recall@5 | MRR | Index Time |
|----------|--------|------------|----------|-----|------------|
| paragraphs_50_256 | 103 | 55 | 96.4% | 0.940 | 76ms |
| heading_based_h3 | 220 | 25 | 91.4% | 0.912 | 114ms |
| heading_limited_512 | 188 | 29 | 91.4% | 0.874 | 98ms |
| hierarchical_h4 | 257 | 22 | 89.4% | 0.929 | 111ms |
| fixed_size_512 | 33 | 196 | 88.6% | 0.908 | 264ms |
| heading_paragraph_h3 | 511 | 22 | 86.1% | 0.894 | 193ms |

## Recommendation

**Best strategy: paragraphs_50_256**

- Recall@5: 96.4%
- MRR: 0.940
- 103 chunks averaging 55 tokens

## Per-Strategy Details

### paragraphs_50_256
- Chunks: 103 (range: 11-96 tokens)
- Recall@5: 96.4%, MRR: 0.940
- Timing: 1ms chunking, 74ms embedding

### heading_based_h3
- Chunks: 220 (range: 2-171 tokens)
- Recall@5: 91.4%, MRR: 0.912
- Timing: 1ms chunking, 113ms embedding

### heading_limited_512
- Chunks: 188 (range: 3-171 tokens)
- Recall@5: 91.4%, MRR: 0.874
- Timing: 1ms chunking, 97ms embedding

### hierarchical_h4
- Chunks: 257 (range: 2-107 tokens)
- Recall@5: 89.4%, MRR: 0.929
- Timing: 1ms chunking, 110ms embedding

### fixed_size_512
- Chunks: 33 (range: 48-497 tokens)
- Recall@5: 88.6%, MRR: 0.908
- Timing: 0ms chunking, 263ms embedding

### heading_paragraph_h3
- Chunks: 511 (range: 3-171 tokens)
- Recall@5: 86.1%, MRR: 0.894
- Timing: 2ms chunking, 191ms embedding
