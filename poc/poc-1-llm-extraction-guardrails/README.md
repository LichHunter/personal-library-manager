# POC-1: LLM Term Extraction Guardrails

## What

Test whether LLMs can reliably extract domain-specific terms from Kubernetes documentation with guardrail prompts, achieving >80% precision and <5% hallucination rate.

## Why

The RAG pipeline's slow system relies on LLM extraction for novel terms. This POC validates whether LLM extraction is reliable enough to use, and which guardrail strategy works best.

## Hypothesis

LLM extraction with full guardrails (evidence citation + output constraints) will achieve >80% precision and <5% hallucination rate.

## Setup

```bash
# Enter project Nix shell
cd /home/fujin/Code/personal-library-manager
direnv allow  # or: nix develop

# Navigate to POC
cd poc/poc-1-llm-extraction-guardrails

# Install dependencies
uv sync
source .venv/bin/activate

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Install Ollama and models (for local model testing)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3:8b
ollama pull mistral:7b
```

## Usage

```bash
# Phase 1: Verify environment
python verify_setup.py

# Phase 2: Generate ground truth
python generate_ground_truth.py

# Phase 3: Run evaluation harness tests
python test_harness.py

# Phase 4: Run main experiment
python run_experiment.py

# Phase 5: Generate analysis
python analyze_results.py
```

## Results

See [RESULTS.md](./RESULTS.md) after execution (copy from RESULTS_TEMPLATE.md).

## Conclusion

**POC-1 Status: PARTIAL PASS**

| Hypothesis | Verdict |
|------------|---------|
| H1: Full guardrails >80% precision, <5% hallucination | REJECTED - Met precision (81%), but hallucination at 16.8% |
| H2: Evidence citation reduces hallucination >50% | SUPPORTED - 55% reduction (27% -> 12%) |
| H3: Sonnet > Haiku by 10% | REJECTED - Models performed comparably |
| H4: Local models >70% precision | NOT TESTED - Ollama unavailable |

**Key Finding**: LLM extraction with guardrails is viable, but requires post-processing span verification to meet the <5% hallucination target reliably. Claude Haiku with Variant D (full guardrails) offers the best cost-efficiency balance at 79.3% precision and 7.4% hallucination.

## Files

| File | Purpose |
|------|---------|
| `SPEC.md` | Full POC specification with requirements and success criteria |
| `RESULTS_TEMPLATE.md` | Template for documenting results |
| `RESULTS.md` | Filled results (created during execution) |
| `artifacts/` | Checkpoint artifacts for each phase |
| `verify_setup.py` | Phase 1: Environment verification |
| `generate_ground_truth.py` | Phase 2: Create Opus-annotated ground truth |
| `test_harness.py` | Phase 3: Validate evaluation code |
| `run_experiment.py` | Phase 4: Execute 2400 extraction conditions |
| `analyze_results.py` | Phase 5: Compute metrics and generate report |

## Related

- [POC Specification](./SPEC.md) - Detailed requirements
- [RAG Pipeline Architecture](../../RAG_PIPELINE_ARCHITECTURE.md) - System design
- [POC Template](../POC_SPECIFICATION_TEMPLATE.md) - Template this POC follows
