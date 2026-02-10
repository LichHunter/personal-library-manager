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
| Term Extraction Research | **Complete** | See [POC-1b Results](#poc-1b-llm-term-extraction) |

---

## Key Research Findings

### POC-1b: LLM Term Extraction

**Status**: Complete | **Recommendation**: Ready for Production

POC-1b investigated LLM-based technical term extraction for Kubernetes documentation, achieving significant improvements in precision and hallucination control.

#### Results Summary

| Metric | Baseline (POC-1) | POC-1b Best | Improvement |
|--------|-----------------|-------------|-------------|
| Precision | 81.0% | **98.2%** | +17.2% |
| Recall | 63.7% | **94.6%** | +30.9% |
| Hallucination | 16.8% | **1.8%** | -15.0% |

#### Key Contributions

1. **D+v2.2 Pipeline**: Multi-phase extraction with triple extraction + voting + discrimination
2. **F25_ULTIMATE Filter**: Enhanced noise removal achieving 98.2% precision with zero recall loss
3. **Scale Testing**: Revealed over-extraction of generic terms as primary challenge (not true hallucination)
4. **Prompt Variant Analysis**: Confirmed filtering is more effective than prompt restrictions

#### Limitations Identified

- **Recall ceiling** at ~95% due to extraction phase limitations
- **Scale degradation** - precision drops with larger document sets
- **Span extraction** - pipeline extracts partial instead of maximum spans
- **Cost** - full pipeline is expensive (~$0.10-0.15/chunk)

#### Future Directions

- Universal generic terms blocklist
- Maximum span extraction improvements
- Multi-model ensemble testing
- Integration with NER systems (POC-1c)

**Full Details**: [poc/poc-1b-llm-extraction-improvements/RESEARCH_RESULTS.md](./poc/poc-1b-llm-extraction-improvements/RESEARCH_RESULTS.md)

---

## Project Structure

```
personal-library-manager/
├── REQUIREMENTS.md           # System requirements
├── DESIGN_PLAN.md            # Design phase tracker
├── RAG_PIPELINE_ARCHITECTURE.md  # RAG system design
├── RESEARCH.md               # Research notes
│
├── poc/                      # Proof of Concept experiments
│   ├── poc-1-llm-extraction-guardrails/   # Initial extraction research
│   ├── poc-1b-llm-extraction-improvements/ # Advanced extraction (Complete)
│   ├── poc-1c-scalable-ner/              # NER integration research
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
| POC-1b | **Complete** | 98.2% precision, 1.8% hallucination | [RESEARCH_RESULTS.md](./poc/poc-1b-llm-extraction-improvements/RESEARCH_RESULTS.md) |
| POC-1c | In Progress | NER-integrated extraction | [poc/poc-1c-scalable-ner/](./poc/poc-1c-scalable-ner/) |
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

*Last Updated: 2026-02-10*
