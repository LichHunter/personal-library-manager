# Test Data Generator POC

## What

A framework for generating test datasets and evaluating retrieval accuracy for RAG systems. Creates documents with ground truth Q&A pairs that can be used to benchmark different retrieval strategies.

## Why

To measure retrieval accuracy, we need:
1. **Documents** with known structure (sections, headings)
2. **Ground truth Q&A pairs** with known correct answers and source locations
3. **Evaluation metrics** (recall@k, evidence overlap)

This POC answers: "Can we automatically generate high-quality test data for retrieval evaluation?"

## Hypothesis

- Wikipedia articles provide good structured test documents (headings = sections)
- LLMs can generate reasonable Q&A pairs from documents
- Evidence needs to be extracted programmatically (LLMs are unreliable at verbatim quotes)

## Setup

```bash
cd /home/fujin/Code/personal-library-manager
direnv allow  # or: nix develop

cd poc/test_data
uv sync
source .venv/bin/activate

# Ensure Ollama is running with a model
ollama serve  # in another terminal
ollama pull llama3.2:3b
```

## Usage

### Generate Test Data

```bash
# Generate from Wikipedia articles
python -m test_data.cli generate \
  --source wikipedia \
  --titles "Python (programming language)" "Rust (programming language)" \
  --output ./output \
  --generate-gt \
  --questions-per-doc 5

# Generate from synthetic documents (Ollama creates them)
python -m test_data.cli generate \
  --source synthetic \
  --topics "machine learning" "database systems" \
  --output ./output \
  --generate-gt
```

### Output Structure

```
output/
├── manifest.json           # Index of all generated data
├── documents/
│   └── wiki_*.json         # Document with sections
└── ground_truth.json       # Q&A pairs with section IDs and evidence
```

### Evaluate a Retriever

```python
from test_data.models import GroundTruth, Retriever, SearchResult
from test_data.evaluator import RetrievalEvaluator
import json

# Load ground truth
with open("output/ground_truth.json") as f:
    gts = [GroundTruth.from_dict(d) for d in json.load(f)]

# Implement the Retriever protocol
class MyRetriever:
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # Your retrieval logic here
        return [SearchResult(document_id="...", section_id="...", content="...", score=0.9)]

# Evaluate
evaluator = RetrievalEvaluator(gts)
report = evaluator.evaluate(MyRetriever(), top_k=5)
evaluator.print_report(report)
```

## Results

### Ground Truth Generation Quality

| Metric | Value | Notes |
|--------|-------|-------|
| Q&A pairs generated | ~60-80% of requested | Some filtered due to missing evidence |
| Evidence accuracy | High | Post-processed to extract verbatim quotes |
| Section ID accuracy | Good | Corrected by evidence extraction |

### Key Findings

1. **LLMs are bad at verbatim quotes**: Initial approach asked LLM for exact quotes, but it returned vague references like "Python documentation". Solution: LLM generates Q&A, then we programmatically extract evidence.

2. **Evidence extraction works**: Keyword matching between answer and document sentences reliably finds supporting evidence.

3. **Section IDs from LLM are unreliable**: LLM sometimes hallucinates section IDs. Solution: Use the section where evidence was actually found.

4. **Wikipedia is good test data**: Articles have clear section structure, factual content, and consistent formatting.

## Conclusion

**What works:**
- Wikipedia → Document with sections (reliable)
- LLM → Question + Answer (reliable)
- Keyword matching → Evidence extraction (reliable)

**What doesn't work:**
- LLM → Verbatim quotes (unreliable)
- LLM → Section ID accuracy (unreliable)

**Design decision:** Use two-stage generation:
1. LLM generates Q&A pairs
2. Post-process to extract real evidence and correct section IDs

## Files

| File | Purpose |
|------|---------|
| `models.py` | Data models: Document, Section, GroundTruth, SearchResult, EvaluationReport |
| `sources/wikipedia.py` | Fetch Wikipedia articles with section parsing |
| `sources/synthetic.py` | Generate synthetic documents with Ollama |
| `generator.py` | Orchestrates document generation |
| `ground_truth.py` | Generates Q&A pairs with evidence extraction |
| `evaluator.py` | Evaluates retriever against ground truth |
| `cli.py` | Command-line interface |

## Data Models

### Document
```python
Document(
    id="wiki_abc123",
    title="Python (programming language)",
    source="wikipedia",
    summary="Python is a high-level...",
    sections=[Section(...), ...]
)
```

### GroundTruth
```python
GroundTruth(
    id="gt_wiki_abc123_1",
    question="Who created Python?",
    answer="Guido van Rossum created Python...",
    document_id="wiki_abc123",
    section_ids=["sec_1"],
    evidence=["Python was conceived in the late 1980s by Guido van Rossum..."],
    difficulty="easy",
    query_type="factual"
)
```

### Retriever Protocol
```python
class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]: ...
```

## Next Steps

1. Generate larger dataset (50+ Wikipedia articles)
2. Implement embedding-based retriever
3. Benchmark flat search vs tree traversal search
4. Measure if 90-95% recall@5 is achievable
