# PLM vs RAG Benchmark

## What

Compare PLM's sophisticated hybrid retrieval (BM25 + semantic + RRF + enrichment + expansion) against naive out-of-the-box LangChain RAG using 3 variants to isolate feature contributions.

## Why

Prove that PLM's advanced features deliver measurable improvement over baseline RAG, and quantify how much each feature (enrichment vs RRF/BM25/expansion) contributes to the improvement.

## Hypothesis

- PLM Full should outperform both baselines
- Enrichment alone should provide ~40% of the improvement
- RRF + BM25 + expansion should provide ~60% of the improvement

## Three Comparison Variants

| Variant | Description | What It Tests |
|---------|-------------|---------------|
| **PLM Full** | Complete system: BM25 + semantic + RRF + enrichment + expansion | "Does full PLM beat naive RAG?" |
| **Baseline + Enriched** | LangChain + FAISS + PLM's enriched chunks | "How much does enrichment alone help?" |
| **Baseline Naive** | LangChain + FAISS + raw chunks | "What's the naive RAG baseline?" |

## Setup

```bash
cd /home/susano/Code/personal-library-manager
direnv allow  # or: nix develop

cd poc/plm_vs_rag_benchmark
uv sync
```

## Usage

```bash
# Quick validation (20 needle questions, no LLM grading)
python benchmark_runner.py --questions needle --no-llm-grade

# Full benchmark (400 realistic questions, with LLM grading)
python benchmark_runner.py --questions realistic --llm-grade

# Run specific variant only
python benchmark_runner.py --questions needle --variant plm
```

## File Structure

```
poc/plm_vs_rag_benchmark/
├── pyproject.toml          # Dependencies
├── README.md               # This file
├── chunk_extractor.py      # Extract chunks from PLM SQLite
├── baseline_langchain.py   # LangChain RAG (raw + enriched modes)
├── benchmark_runner.py     # Unified benchmark for all 3 variants
├── corpus/                 # Symlink → ../chunking_benchmark_v2/corpus
└── results/
    ├── needle_benchmark.json
    ├── realistic_benchmark.json
    └── BENCHMARK_REPORT.md
```

## Results

See `results/BENCHMARK_REPORT.md` for full analysis after running benchmarks.

## Key Metrics

- **MRR** (Mean Reciprocal Rank): How high is the correct document ranked?
- **Hit@k**: Is the correct document in top k results?
- **Recall@k**: What fraction of relevant docs are retrieved?
- **LLM Grade**: Does the retrieved content enable a good answer? (1-10)
- **Bootstrap 95% CI**: Statistical confidence on MRR differences

## Attribution Analysis

After running benchmarks, the report shows:
- Total PLM improvement over naive baseline
- Enrichment contribution (Baseline-Enriched vs Baseline-Naive)
- RRF/BM25/expansion contribution (PLM vs Baseline-Enriched)
