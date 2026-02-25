# PLM Benchmark

End-to-end retrieval benchmark for evaluating search service quality against ground-truth question-answer pairs.

## Overview

The benchmark sends queries to the search service HTTP API and measures retrieval accuracy by checking if the expected document appears in the top-k results. It supports three datasets with different characteristics and can capture detailed service traces for debugging.

## Quick Start

```bash
# Start search service (must have indexed data)
docker compose -f docker/docker-compose.search.yml up -d

# Run benchmark on all datasets
python -m plm.benchmark.cli

# Run on specific dataset with trace logging
python -m plm.benchmark.cli --datasets needle --trace-log /path/to/search_trace.log
```

## Datasets

| Dataset | Questions | Description |
|---------|-----------|-------------|
| `needle` | 20 | Specific technical questions with single correct document |
| `realistic` | 400 | Broad questions simulating real user queries |
| `informed` | 25 | Expert-level questions using domain terminology |

Dataset files are stored in `corpus/` as JSON with structure:
```json
{
  "per_question": [
    {
      "id": "q_001",
      "question": "How do I scale a deployment?",
      "target_doc_id": "concepts_workloads_controllers_deployment"
    }
  ]
}
```

## CLI Options

```
python -m plm.benchmark.cli [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--datasets` | all | Datasets to evaluate: `needle`, `realistic`, `informed` |
| `--service-url` | `http://localhost:8000` | Search service URL |
| `--k` | 10 | Number of results to retrieve |
| `--use-rewrite` | false | Enable LLM query rewriting |
| `--use-rerank` | false | Enable cross-encoder reranking |
| `--trace-log` | auto-detect | Path to service trace log |
| `--logs-dir` | `./logs` | Output directory for reports |
| `--limit` | none | Limit questions (for testing) |
| `--verbose`, `-v` | false | Enable DEBUG level logging |

## Output

Each run creates a timestamped directory in `logs/` containing:

```
logs/20260226_120000/
├── benchmark_info.log    # Run info and progress
├── benchmark_trace.log   # Detailed trace log
├── report.txt            # Human-readable report
└── results.json          # Machine-readable results
```

### Metrics

| Metric | Description |
|--------|-------------|
| Hit@k | Percentage of queries where target doc appears in top-k |
| MRR | Mean Reciprocal Rank (1/rank averaged across queries) |
| Latency | Mean, P50, P95 response times |

### Sample Report

```
SUMMARY STATISTICS
------------------
Total queries: 445

Hit Rates:
  Hit@1:  75.3%
  Hit@5:  89.2%
  Hit@10: 92.1%

MRR: 0.8234

Latency:
  Mean: 245.3ms
  P50:  198.2ms
  P95:  512.4ms
```

## Service Trace Integration

The benchmark can capture and display service-side traces for each query, showing the full retrieval pipeline:

```
WORST PERFORMING QUESTIONS
--------------------------
1. [needle] NOT FOUND
   ID: q_001
   Q: My pod keeps getting rejected...
   Target: tasks_administer-cluster_topology-manager
   Trace [9a33e68c...]:
      [receive] query='...' k=10 rewrite=False
      [expand] added=[]
      [sparse] found=100 top5=[18233, 7646, ...]
      [semantic] computed=20801 top5=[18232, ...]
      [rrf] fused=184 top_k=[18233, 6620, ...]
      [complete] results=10
```

### Trace Discovery

1. **Explicit path**: `--trace-log /path/to/search_trace.log`
2. **Docker auto-detect**: Inspects `plm-search-service` container for `/data/logs` mount
3. **Disabled**: If neither works, benchmark runs without traces

### Enabling Traces

Start the search service with `PLM_LOG_LEVEL=TRACE`:

```bash
docker run -d --name plm-search-service \
  -v /tmp/logs:/data/logs \
  -e PLM_LOG_LEVEL=TRACE \
  plm-search-service:0.2.0

# Run benchmark (auto-detects trace log)
python -m plm.benchmark.cli --datasets needle
```

## Examples

### Quick validation (5 questions)
```bash
python -m plm.benchmark.cli --datasets needle --limit 5
```

### Full benchmark with reranking
```bash
python -m plm.benchmark.cli --use-rerank --k 20
```

### Compare configurations
```bash
# Baseline
python -m plm.benchmark.cli --logs-dir logs/baseline

# With query rewriting
python -m plm.benchmark.cli --use-rewrite --logs-dir logs/rewrite

# Compare results.json files
```

### Remote service
```bash
python -m plm.benchmark.cli --service-url http://prod-server:8000
```

## Programmatic Usage

```python
from plm.benchmark.loader import load_questions
from plm.benchmark.runner import BenchmarkRunner, RunnerConfig
from plm.benchmark.metrics import calculate_metrics

# Load questions
questions = load_questions(Path("corpus"), datasets=["needle"])

# Run benchmark
config = RunnerConfig(service_url="http://localhost:8000", k=10)
runner = BenchmarkRunner(config)
results = runner.run_all(questions)
runner.close()

# Calculate metrics
metrics = calculate_metrics(results, k=10)
print(f"Hit@10: {metrics.hit_at_10:.1%}")
print(f"MRR: {metrics.mrr:.4f}")
```

## Adding New Datasets

Create a JSON file in `corpus/` with naming convention `{name}_benchmark.json`:

```json
{
  "per_question": [
    {
      "id": "unique_id",
      "question": "The query text",
      "target_doc_id": "expected_doc_id_prefix"
    }
  ]
}
```

Note: `target_doc_id` uses prefix matching - the benchmark checks if any retrieved doc_id starts with this value.
