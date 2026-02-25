# RAPTOR Test POC

## What

A proof-of-concept to benchmark RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) with local models. Tests indexing performance and retrieval accuracy.

## Why

Before building our own hierarchical retrieval system, we need to understand:
1. How long does RAPTOR indexing take?
2. Where is the bottleneck?
3. What accuracy can we expect?
4. Is RAPTOR worth adopting or should we simplify?

## Hypothesis

- RAPTOR will provide better retrieval than flat embedding search
- Summarization will be the main bottleneck (LLM calls are slow)
- Local models (Ollama + SBERT) will be good enough for testing

## What This Tests

1. **Indexing Performance**: Time to build the hierarchical tree
   - Chunking speed
   - Embedding generation (local SentenceTransformers)
   - Clustering (UMAP + GMM)
   - Summarization (local Ollama LLM)

2. **Retrieval Accuracy**: Can we find the right information?
   - Tests if relevant chunks are retrieved for questions

3. **QA Accuracy**: Are answers correct and grounded?
   - Tests against 10 ground-truth Q&A pairs

## Setup

### 1. Enter the Nix development shell

```bash
cd /home/fujin/Code/personal-library-manager
direnv allow  # or: nix develop
```

### 2. Install Python dependencies

```bash
cd poc/raptor_test
uv sync
```

### 3. Start Ollama and pull a model

```bash
# Start Ollama server (in another terminal)
ollama serve

# Pull a model (in this terminal)
ollama pull llama3.2:3b   # Fast, good quality
# or
ollama pull llama3.2:1b   # Faster, lower quality
# or
ollama pull mistral:7b    # Better quality, slower
```

## Running the Benchmark

### Quick Test

```bash
# Test with default settings (llama3.2:3b, built-in test document)
python benchmark.py
```

### Custom Options

```bash
# Use a different model
python benchmark.py --model mistral:7b

# Use a different chunk size
python benchmark.py --chunk-size 150

# Use your own document
python benchmark.py --document /path/to/your/document.txt

# Save results to file
python benchmark.py --output results.json
```

## Expected Output

```
RAPTOR Benchmark

Initializing models...

Building RAPTOR tree...

┌─────────────────────────────────┐
│        Tree Statistics          │
├─────────────────┬───────────────┤
│ Metric          │ Value         │
├─────────────────┼───────────────┤
│ Chunks          │ 25            │
│ Total Nodes     │ 35            │
│ Layers          │ 2             │
│ Summaries       │ 10            │
└─────────────────┴───────────────┘

┌────────────────────────────────────────────────┐
│              Timing Breakdown                   │
├──────────────────┬───────────┬─────────────────┤
│ Phase            │ Time (s)  │ % of Total      │
├──────────────────┼───────────┼─────────────────┤
│ Chunking         │ 0.01      │ 0.1%            │
│ Embedding        │ 2.50      │ 15.0%           │
│ Clustering       │ 0.50      │ 3.0%            │
│ Summarization    │ 13.50     │ 81.9%           │
│ Total            │ 16.51     │ 100%            │
└──────────────────┴───────────┴─────────────────┘

Testing retrieval and QA accuracy...

Q1: Who created Python?
  ✓ OK - Found 2/2 keywords

...

┌─────────────────────────────────────────────────┐
│              Accuracy Results                    │
├─────────────────────────────┬───────────────────┤
│ Metric                      │ Score             │
├─────────────────────────────┼───────────────────┤
│ Retrieval Accuracy          │ 90.0% (9/10)      │
│ QA Accuracy                 │ 80.0% (8/10)      │
└─────────────────────────────┴───────────────────┘
```

## File Structure

```
poc/raptor_test/
├── pyproject.toml      # Python dependencies
├── local_models.py     # Local model implementations (SBERT, Ollama)
├── raptor_lite.py      # Simplified RAPTOR implementation
├── benchmark.py        # Main benchmark script
└── README.md           # This file
```

## Key Findings to Collect

After running, note:

1. **Indexing Time**: How long for different document sizes?
2. **Bottleneck**: What takes the most time? (Usually summarization)
3. **Retrieval Accuracy**: Can the system find relevant chunks?
4. **QA Accuracy**: Are answers correct?
5. **Model Comparison**: How do different Ollama models compare?

## Troubleshooting

### "Ollama not available"
```bash
# Make sure Ollama is running
ollama serve

# Check if model is pulled
ollama list
```

### UMAP import error
UMAP will automatically fall back to PCA if not available. For full functionality:
```bash
uv pip install umap-learn
```

### Out of memory
- Use a smaller model: `--model llama3.2:1b`
- Reduce chunk size: `--chunk-size 50`

---

## Results

### Actual Benchmark (results.json)

| Metric | Value |
|--------|-------|
| Chunks | 21 |
| Total Nodes | 26 |
| Layers | 1 |
| Summaries Generated | 5 |

### Timing Breakdown

| Phase | Time (s) | % of Total |
|-------|----------|------------|
| Chunking | 0.001 | 0.003% |
| Embedding | 0.17 | 0.4% |
| Clustering | 0.34 | 0.8% |
| **Summarization** | **40.28** | **98.6%** |
| **Total** | **40.86** | 100% |

### Accuracy

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | 80% (8/10 questions found relevant chunks) |
| QA Accuracy | 70% (7/10 answers correct) |

### Per-Question Results

| Question | Retrieval | QA | Notes |
|----------|-----------|-----|-------|
| Who created Python? | Hit | Correct | Found "Guido van Rossum" |
| When was Python first released? | Hit | Correct | Found "1991" |
| What inspired the name Python? | Hit | Correct | Found "Monty Python" |
| When did Python 2 reach end of life? | Hit | Correct | Found "January 1, 2020" |
| What is PEP 20? | **Miss** | **Wrong** | Couldn't find "Zen of Python" |
| How many packages are on PyPI? | Hit | Correct | Found "400,000" |
| What web framework does Instagram use? | Hit | Correct | Found "Django" |
| Who manages Python after Guido stepped down? | Hit | **Wrong** | Said "PSF" instead of "steering council" |
| What is Python's design philosophy? | Hit | Correct | Found "readability and simplicity" |
| What is PyPy? | **Miss** | **Wrong** | Information not in test document |

---

## Conclusion

### Key Findings

1. **Summarization is 98.6% of indexing time**
   - This is the critical bottleneck for RAPTOR
   - For 1K documents, indexing would take days
   - Need to either: batch/parallelize summarization, or skip it

2. **70-80% accuracy is achievable with local models**
   - Retrieval: 80% (good)
   - QA: 70% (acceptable for local 3B model)
   - Failures were either missing info or subtle errors

3. **RAPTOR's complexity may not be worth it**
   - The hierarchical tree adds significant indexing cost
   - For our use case (explicit document structure), we can use headings directly
   - Skip clustering, use document structure as the hierarchy

### Design Decisions Informed

| Decision | Rationale |
|----------|-----------|
| Use document headings instead of RAPTOR clustering | We have explicit structure, no need to infer it |
| Skip recursive summarization during indexing | Too slow; summarize on-demand or at query time |
| Use local SBERT for embeddings | Fast (0.17s), good quality |
| Use tree traversal search | doc summary → section → content (10x fewer comparisons) |

### What to Adopt from RAPTOR

- Hierarchical tree concept
- Multi-level retrieval (broad queries match summaries, specific match details)
- Clean extension points (swap embedding/summarization models)

### What to Change

- Use explicit document structure (headings) instead of clustering
- No recursive summarization during indexing
- LLM only at query time for final answer synthesis
