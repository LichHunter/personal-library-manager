# PLM Benchmark Framework

## Overview
Externally-validated benchmark framework using StackOverflow data.

## Architecture
[Pipeline diagram: SO Data → Signals → Generation → Verification → Assembly → Evaluation]

## CLI Commands

### Data Extraction
plm-benchmark extract-so --db <path> --output <path>

### Signal Extraction
plm-benchmark extract-signals --so-data <path> --mappings <dir> --corpus-db <path> --output <dir>

### Generation
plm-benchmark generate --signals <path> --output <dir>

### Verification
plm-benchmark verify --generated <dir> --corpus-db <path> --output <dir>

### Assembly
plm-benchmark assemble --verified <path> --regenerated <path> --signals <path> --output <dir>

### Evaluation
plm-benchmark evaluate --dataset gold.json --k 10 --url http://localhost:8000

### Integration Analysis
plm-benchmark analyze-integration --dataset gold.json --analysis all

## Configuration

### Environment Variables
- PLM_LLM_MODEL: LLM model for generation

### Tier Thresholds
- GOLD: fragment match OR quote >= 30 chars
- SILVER: URL + high trust signals
- BRONZE: URL + score >= 5

## Interpreting Results

### Metrics
- Hit@k: Proportion with relevant chunk in top k
- MRR: Mean reciprocal rank of first relevant
- NDCG@k: Normalized discounted cumulative gain

### Quality Targets
- Quarantine rate < 5%
- Recovery rate > 50%
