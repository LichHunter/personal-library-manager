# Quick Reference: Test File Patterns

## File Structure (Copy This Order)
```
1. Shebang + Docstring
2. Imports (stdlib → third-party → local)
3. Path setup + print header
4. load_ground_truth()
5. Parsing utilities (parse_terms, strict_span_verify)
6. Prompts (CONSTANTS)
7. Extraction functions
8. Metrics functions (normalize_term, match_terms, calculate_metrics)
9. Strategy functions
10. run_experiment() main loop
11. if __name__ == "__main__": run_experiment()
```

## Essential Functions (Copy These)

### 1. Load Ground Truth
```python
def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)["chunks"]
```

### 2. Parse Terms from LLM Response
```python
def parse_terms(response: str, require_quotes: bool = False) -> list[str]:
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            terms = data.get("terms", [])
            if require_quotes:
                return [t.get("term", "") for t in terms if isinstance(t, dict)]
            elif isinstance(terms, list):
                if terms and isinstance(terms[0], dict):
                    return [t.get("term", "") for t in terms]
                return [str(t) for t in terms]
    except:
        pass
    return []
```

### 3. Verify Terms Exist in Content
```python
def strict_span_verify(term: str, content: str) -> bool:
    if not term or len(term) < 2:
        return False
    content_lower = content.lower()
    term_lower = term.lower().strip()
    if term_lower in content_lower:
        return True
    normalized = term_lower.replace("_", " ").replace("-", " ")
    if normalized in content_lower.replace("_", " ").replace("-", " "):
        return True
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True
    return False
```

### 4. Calculate Metrics
```python
def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")

def match_terms(extracted: str, ground_truth: str) -> bool:
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)
    if ext_norm == gt_norm:
        return True
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True
    return False

def calculate_metrics(extracted: list[str], ground_truth: list[dict]) -> dict:
    gt_terms = [t.get("term", "") for t in ground_truth]
    matched_gt = set()
    matched_ext = set()
    tp = 0
    for i, ext in enumerate(extracted):
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if match_terms(ext, gt):
                matched_gt.add(j)
                matched_ext.add(i)
                tp += 1
                break
    fp = len(extracted) - tp
    fn = len(gt_terms) - tp
    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extracted_count": len(extracted),
        "gt_count": len(gt_terms),
    }
```

## Common Prompts

### High Recall (Exhaustive)
```python
EXHAUSTIVE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
Be EXHAUSTIVE. Include: resources, components, concepts, feature gates, lifecycle stages, CLI flags, API terms.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""
```

### High Precision (Conservative)
```python
CONSERVATIVE_PROMPT = """Extract ONLY the most important Kubernetes technical terms.
Be CONSERVATIVE. Only core resources, key components, essential concepts.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""
```

### Verification
```python
VERIFY_PROMPT = """Filter this list of extracted terms from Kubernetes docs.
Keep ONLY Kubernetes-specific technical terms. Remove generic English words.
CHUNK: {content}
TERMS: {terms}
Output JSON: {{"terms": ["term1", ...]}}"""
```

## Strategy Patterns

### Single Extraction
```python
def extract_simple(content: str, model: str = "claude-haiku") -> list[str]:
    prompt = SIMPLE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]
```

### Ensemble (Multiple Extractions + Verification)
```python
def strategy_ensemble_verified(content: str) -> list[str]:
    all_terms = set()
    all_terms.update(extract_simple(content))
    all_terms.update(extract_exhaustive(content))
    if not all_terms:
        return []
    prompt = VERIFY_PROMPT.format(content=content[:2000], terms=json.dumps(sorted(all_terms)))
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=1000)
    verified = parse_terms(response)
    return [t for t in verified if strict_span_verify(t, content)]
```

### Voting (Multiple Strategies)
```python
def strategy_intersection_vote(content: str, min_votes: int = 2) -> list[str]:
    simple = set(extract_simple(content))
    exhaustive = set(extract_exhaustive(content))
    conservative = set(extract_conservative(content))
    term_votes = {}
    for term_set in [simple, exhaustive, conservative]:
        for term in term_set:
            key = term.lower()
            if key not in term_votes:
                term_votes[key] = {"canonical": term, "votes": 0}
            term_votes[key]["votes"] += 1
    return [info["canonical"] for info in term_votes.values() if info["votes"] >= min_votes]
```

## Key Imports
```python
import json
import re
import sys
import time
from pathlib import Path
from rapidfuzz import fuzz
sys.path.insert(0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"))
from utils.llm_provider import call_llm
```

## Key Constants
```python
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"
```

## LLM Call Pattern
```python
response = call_llm(
    prompt=PROMPT.format(content=content[:2500]),
    model="claude-haiku",  # or "claude-sonnet"
    temperature=0,         # Always 0 for deterministic extraction
    max_tokens=1000        # Adjust based on expected output size
)
```

## Metrics Interpretation
- **Precision**: % of extracted terms that are correct (TP / (TP + FP))
- **Recall**: % of ground truth terms that were found (TP / (TP + FN))
- **Hallucination**: % of extracted terms that are false positives (FP / extracted)
- **F1**: Harmonic mean of precision and recall

## Target Thresholds
- ✓ Excellent: Recall ≥ 95%, Hallucination < 10%
- ~ Good: Recall ≥ 85%, Hallucination < 20%
- Space: Needs improvement

## Files to Check
- Ground truth: `artifacts/small_chunk_ground_truth.json`
- Results: `artifacts/test_results.json` (created after run)
- Shared utils: `../poc-1-llm-extraction-guardrails/utils/llm_provider.py`
