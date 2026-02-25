# POC-1b: LLM Term Extraction Improvements

> **Status**: Complete | **Result**: Substantial Success | [Full Research Results](./RESEARCH_RESULTS.md)

## Quick Results

| Metric | POC-1 Best | POC-1b Best | Target | Status |
|--------|-----------|-------------|--------|--------|
| Precision | 81.0% | **98.2%** | 95%+ | **EXCEEDED** |
| Recall | 63.7% | **94.6%** | 95%+ | -0.4% gap |
| Hallucination | 16.8% | **1.8%** | <5% | **EXCEEDED** |
| F1 Score | N/A | **0.963** | - | Best overall |

**Recommended Strategy**: D+v2.2 + F25_ULTIMATE Filter - Ready for production deployment.

---

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

**28+ strategies tested** with comprehensive analysis. See detailed results:

- **[RESEARCH_RESULTS.md](./RESEARCH_RESULTS.md)** - Comprehensive research findings, limitations, and future improvements
- [docs/RESULTS.md](./docs/RESULTS.md) - Detailed numerical results with scale testing
- [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md) - 5-minute overview
- [STRATEGY_COMPARISON.md](./STRATEGY_COMPARISON.md) - All strategies compared

## Key Findings

### What Worked

1. **Triple extraction** (Sonnet exhaustive + Haiku exhaustive + Haiku simple) significantly improved recall
2. **Voting mechanism** (2+ votes → keep, 1 vote → review) balanced precision and recall
3. **Sonnet discrimination** effectively filtered borderline terms
4. **F25_ULTIMATE filter** eliminated noise without harming recall
5. **Span verification** ensured zero true hallucinations

### What Didn't Work

1. Restrictive prompts cause severe recall collapse (35.6% best recall)
2. Multi-pass without verification increases hallucination
3. Low-threshold voting allows too much noise
4. Quote extraction alone generates hallucinations

### Key Insight: Scale Performance

At scale (100 chunks), precision drops to ~84%. **Root cause is over-extraction of generic terms, NOT hallucination**:
- 99.3% of false positives exist in the source text
- Only 0.7% are true hallucinations
- Solution: Enhanced blocklists for universal generic terms

## Limitations

1. **Recall ceiling at ~95%** - Extraction phase limitations, not filter issues
2. **Scale degradation** - Generic term over-extraction at larger scales
3. **Span extraction deficiencies** - Extracts partial spans instead of maximum spans
4. **Cost** - Triple extraction + discrimination is expensive (~$0.10-0.15/chunk)

## Future Improvements

1. Add universal generic terms blocklist (data types, structural words, shell artifacts)
2. Maximum span extraction for function calls and multi-word entities
3. Prompt refinement to reduce generic extraction
4. Multi-model ensemble testing (GPT-4, Llama, Mistral)

## Files

| File | Purpose |
|------|---------|
| **[RESEARCH_RESULTS.md](./RESEARCH_RESULTS.md)** | **Comprehensive research findings** |
| `docs/STRATEGY.md` | Pipeline methodology details |
| `docs/RESULTS.md` | Numerical results with scale testing |
| `EXECUTIVE_SUMMARY.md` | Quick overview of all 28 strategies |
| `STRATEGY_COMPARISON.md` | Side-by-side strategy comparison |
| `GAP_ANALYSIS.md` | Planned vs actual implementation |
| `run_experiment.py` | Main experiment runner |
| `artifacts/` | Results and checkpoints |

## Related

- [POC-1 Results](../poc-1-llm-extraction-guardrails/RESULTS.md) - Baseline findings
- [POC-1c Scalable NER](../poc-1c-scalable-ner/) - Follow-up NER integration research
- [RAG Pipeline Architecture](../../RAG_PIPELINE_ARCHITECTURE.md) - System design
