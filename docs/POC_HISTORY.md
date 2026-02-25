# POC History & Research Findings

This document captures the results from proof-of-concept experiments. All POCs are archived in `poc/archive/`.

## POC-1: LLM Term Extraction (Complete)

**Status**: Complete (POC-1, POC-1b, POC-1c)  
**Original Target**: 95% precision / 95% recall / 5% hallucination

The POC-1 series investigated LLM-based technical term extraction, testing vocabulary-assisted and vocabulary-free approaches.

### Final Results

| Approach | Precision | Recall | Hallucination | F1 | Vocabulary |
|----------|-----------|--------|---------------|-----|------------|
| V6 (best with vocab) | 90.7% | 95.8% | 9.3% | 0.932 | 176 terms |
| V6 @ 50 docs | 84.2% | 92.8% | 15.8% | 0.883 | 176 terms |
| Retrieval few-shot | 81.6% | 80.6% | 18.4% | 0.811 | **0 terms** |
| SLIMER zero-shot | 84.9% | 66.0% | 15.1% | 0.743 | **0 terms** |

### Key Findings

1. **95/95/5 NOT achievable** — benchmark ceiling is ~P=94%, R=96%, H=6% due to ground truth annotation gaps
2. **Vocabulary-free maximum: ~80/90/20** — retrieval few-shot eliminates maintenance with ~10% F1 drop
3. **GLiNER rejected** — produces garbage results for software entities, no usable signal
4. **Heuristic patterns work** — CamelCase, backticks, ALL_CAPS, dot.paths viable for fast extraction

### Recommendation

Use **retrieval few-shot** for production (zero vocabulary maintenance). For RAG, recall matters more than matching benchmark conventions. The precision/hallucination tradeoff is acceptable.

### Implementation

The production package at `src/plm/extraction/` implements:
- **Fast system**: Heuristic-based extraction (regex patterns, confidence scoring)
- **Slow system**: V6 LLM pipeline (5 stages: Extract → Ground → Filter → Validate → Postprocess)

### Detailed Reports

- [POC-1c Results](../poc/archive/poc-1c-scalable-ner/RESULTS.md)
- [V6 Analysis](../poc/archive/poc-1c-scalable-ner/docs/V6_RESULTS.md)
- [POC-1b Results](../poc/archive/poc-1b-llm-extraction-improvements/RESEARCH_RESULTS.md)

---

## POC-2: Confidence Scoring

**Location**: `poc/archive/poc-2-confidence-scoring/`

Investigated confidence calibration for extraction outputs.

---

## Retrieval Benchmarks

### Chunking Benchmark V2

**Location**: `poc/archive/chunking_benchmark_v2/`

Compared chunking strategies for retrieval quality.

### SPLADE Benchmark

**Location**: `poc/archive/splade_benchmark/`

Evaluated SPLADE sparse retrieval against BM25.

### Convex Fusion Benchmark

**Location**: `poc/archive/convex_fusion_benchmark/`

Tested convex combination fusion strategies for hybrid search.

### Adaptive Retrieval

**Location**: `poc/archive/adaptive_retrieval/`

Explored adaptive retrieval strategies.

---

## Other POCs

| POC | Location | Status |
|-----|----------|--------|
| Modular Retrieval Pipeline | `poc/archive/modular_retrieval_pipeline/` | Archived |
| PLM vs RAG Benchmark | `poc/archive/plm_vs_rag_benchmark/` | Archived |
| RAPTOR Test | `poc/archive/raptor_test/` | Archived |
| Retrieval Benchmark | `poc/archive/retrieval_benchmark/` | Archived |

---

## Running Archived POCs

Each POC has its own `pyproject.toml` and isolated environment:

```bash
cd poc/archive/<poc-name>
uv sync
source .venv/bin/activate
# Follow POC-specific README
```

---

*Archived: 2026-02-25*
