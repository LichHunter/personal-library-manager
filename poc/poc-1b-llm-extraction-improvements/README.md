# POC-1b: LLM Term Extraction Improvements

## What

Test advanced LLM-only techniques to achieve **95%+ precision, 95%+ recall, <1% hallucination** for Kubernetes term extraction, building on POC-1 findings.

## Why

POC-1 achieved 81% precision and 7.4% hallucination with guardrails, but missed the <5% hallucination target. This POC tests research-backed techniques to close that gap using LLM-only approaches (no NER hybrids).

## Hypothesis

LLM extraction with Instructor structured output + span verification + self-consistency voting will achieve **95%+ precision, 85%+ recall, <1% hallucination**.

## Strategies Tested

| Strategy | Description | Expected Impact |
|----------|-------------|-----------------|
| **E** | Instructor structured output (baseline) | Guaranteed JSON format |
| **F** | Instructor + span verification validators | Hallucination <1% |
| **G** | Self-consistency voting (N=10, 70% threshold) | +25% precision |
| **H** | Multi-pass extraction ("what did I miss?") | +15% recall |
| **I** | Combined (F + G + H) | Target: 95% P, 85% R, <1% H |

## Setup

```bash
# Enter project Nix shell
cd /home/fujin/Code/personal-library-manager
direnv allow

# Navigate to POC
cd poc/poc-1b-llm-extraction-improvements

# Install dependencies
uv sync
source .venv/bin/activate

# Verify POC-1 ground truth exists
ls ../poc-1-llm-extraction-guardrails/artifacts/phase-2-ground-truth.json
```

## Usage

```bash
# Run full experiment (1350 conditions, ~3-4 hours)
python run_experiment.py

# Results saved to artifacts/
```

## Results

See [RESULTS.md](./RESULTS.md) after execution.

## POC-1 Baseline (for comparison)

| Metric | Baseline A | Best D (Haiku) | Best D (Sonnet) |
|--------|------------|----------------|-----------------|
| Precision | 69.4% | 79.3% | 81.0% |
| Recall | 63.2% | 45.4% | 63.7% |
| Hallucination | 24.7% | 7.4% | 16.8% |

## Key Research Findings Used

1. **Instructor library**: Structured output with Pydantic validation + automatic retries
2. **Span verification**: Field validators check term exists in source text
3. **Self-consistency voting**: N=10 samples, 70% agreement threshold
4. **Multi-pass extraction**: 3 passes with increasing specificity
5. **Temperature 0.8**: Required for self-consistency diversity

## Files

| File | Purpose |
|------|---------|
| `SPEC.md` | Full specification |
| `run_experiment.py` | Main experiment runner |
| `utils/llm_provider.py` | Anthropic OAuth provider (from POC-1) |
| `artifacts/` | Results and checkpoints |

## Related

- [POC-1 Results](../poc-1-llm-extraction-guardrails/RESULTS.md) - Baseline findings
- [RAG Pipeline Architecture](../../RAG_PIPELINE_ARCHITECTURE.md) - System design
