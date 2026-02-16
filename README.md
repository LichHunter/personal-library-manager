# Personal Knowledge Assistant

A local-first, NotebookLM-like system for querying a personal document corpus with grounded, cited answers.

## Vision

Build a trustworthy personal knowledge assistant that:
- Handles 1000+ documents with complex queries
- Provides synthesized answers with proper citations
- **No hallucinations** - critical requirement
- Runs locally on consumer hardware (32GB RAM, 8GB VRAM)

## Project Status

**Current Phase**: Research & Proof of Concept

| Phase | Status | Notes |
|-------|--------|-------|
| Requirements | Complete | See [REQUIREMENTS.md](./REQUIREMENTS.md) |
| Architecture Design | In Progress | See [DESIGN_PLAN.md](./DESIGN_PLAN.md) |
| RAG Pipeline Research | In Progress | See [RAG_PIPELINE_ARCHITECTURE.md](./RAG_PIPELINE_ARCHITECTURE.md) |
| Term Extraction Research | **Complete** | POC-1 series done. Target 95/95/5 not achieved; max ~80/90/20. See [POC-1 Results](#poc-1-llm-term-extraction) |

---

## Key Research Findings

### POC-1: LLM Term Extraction (Complete)

**Status**: Complete (POC-1, POC-1b, POC-1c) | **Original Target**: 95% precision / 95% recall / 5% hallucination

The POC-1 series investigated LLM-based technical term extraction, testing vocabulary-assisted and vocabulary-free approaches.

#### Final Results

| Approach | Precision | Recall | Hallucination | F1 | Vocabulary |
|----------|-----------|--------|---------------|-----|------------|
| V6 (best with vocab) | 90.7% | 95.8% | 9.3% | 0.932 | 176 terms |
| V6 @ 50 docs | 84.2% | 92.8% | 15.8% | 0.883 | 176 terms |
| Retrieval few-shot | 81.6% | 80.6% | 18.4% | 0.811 | **0 terms** |
| SLIMER zero-shot | 84.9% | 66.0% | 15.1% | 0.743 | **0 terms** |

#### Key Findings

1. **95/95/5 NOT achievable** — benchmark ceiling is ~P=94%, R=96%, H=6% due to ground truth annotation gaps
2. **Vocabulary-free maximum: ~80/90/20** — retrieval few-shot eliminates maintenance with ~10% F1 drop
3. **GLiNER rejected** — produces garbage results for software entities, no usable signal
4. **Heuristic patterns work** — CamelCase, backticks, ALL_CAPS, dot.paths viable for fast extraction

#### Recommendation

Use **retrieval few-shot** for production (zero vocabulary maintenance). For RAG, recall matters more than matching benchmark conventions. The precision/hallucination tradeoff is acceptable.

#### Production Package

The `src/plm/extraction/` package implements:
- **Fast system**: Heuristic-based extraction (regex patterns, confidence scoring)
- **Slow system**: V6 LLM pipeline (5 stages: Extract → Ground → Filter → Validate → Postprocess)

**Full Details**: 
- [POC-1c Results](./poc/poc-1c-scalable-ner/RESULTS.md)
- [V6 Analysis](./poc/poc-1c-scalable-ner/docs/V6_RESULTS.md)
- [POC-1b Results](./poc/poc-1b-llm-extraction-improvements/RESEARCH_RESULTS.md)

---

## Project Structure

```
personal-library-manager/
├── REQUIREMENTS.md           # System requirements
├── DESIGN_PLAN.md            # Design phase tracker
├── RAG_PIPELINE_ARCHITECTURE.md  # RAG system design
├── RESEARCH.md               # Research notes
│
├── src/plm/                  # Production extraction package
│   ├── extraction/           # Fast (heuristic) + Slow (V6 LLM) systems
│   └── shared/llm/           # LLM providers (Anthropic, Gemini)
│
├── poc/                      # Proof of Concept experiments
│   ├── poc-1-llm-extraction-guardrails/   # Initial extraction research
│   ├── poc-1b-llm-extraction-improvements/ # Advanced extraction (Complete)
│   ├── poc-1c-scalable-ner/              # Scalable NER (Complete)
│   ├── modular_retrieval_pipeline/       # Retrieval experiments
│   ├── retrieval_benchmark/              # Retrieval strategy comparison
│   └── chunking_benchmark_v2/            # Chunking strategy comparison
│
└── docs/                     # Additional documentation
    └── research/             # Research documentation
```

## Quick Links

### Research Results

| POC | Status | Key Finding | Location |
|-----|--------|-------------|----------|
| POC-1 (1, 1b, 1c) | **Complete** | Target 95/95/5 not achieved; max ~80/90/20 vocab-free | [poc/poc-1c-scalable-ner/](./poc/poc-1c-scalable-ner/) |
| Retrieval | In Progress | Strategy comparison | [poc/retrieval_benchmark/](./poc/retrieval_benchmark/) |

### Design Documents

- [Requirements](./REQUIREMENTS.md) - System requirements and success criteria
- [Design Plan](./DESIGN_PLAN.md) - Development phase tracker
- [RAG Architecture](./RAG_PIPELINE_ARCHITECTURE.md) - Detailed pipeline design
- [Research Notes](./RESEARCH.md) - Research findings and decisions

---

## Development

### Prerequisites

- Nix (with flakes enabled)
- Python 3.11+
- Anthropic API access (for LLM testing)

### Setup

```bash
# Enter development shell
cd personal-library-manager
direnv allow  # or: nix develop

# Navigate to specific POC
cd poc/poc-1b-llm-extraction-improvements
uv sync
source .venv/bin/activate
```

### Running POCs

Each POC has its own README with setup and execution instructions. See [poc/README.md](./poc/README.md) for guidelines.

---

## Core Principles

1. **No Hallucinations** - Critical requirement, especially no fake citations
2. **Trustworthy Answers** - Must be reliable enough to use without verifying everything
3. **Local-First** - Designed for consumer hardware
4. **Grounded Responses** - Every claim backed by source citations

---

*Last Updated: 2026-02-16*
