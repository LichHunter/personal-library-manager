# Test File Patterns Analysis - POC-1b

## Overview

This document extracts common patterns from existing test files (`test_fast_combined.py` and `test_hybrid_final.py`) to provide a template for creating new test files that match the codebase style.

---

## 1. Test Structure Patterns

### File Header & Imports
```python
#!/usr/bin/env python3
"""[One-line description of what this test does]

[2-3 sentence explanation of strategy/approach]
"""

import json
import re
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz

# Add paths for shared utilities
sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm
```

**Key Points:**
- Shebang for direct execution
- Docstring explains the test's purpose and strategy
- Imports are organized: stdlib → third-party → local
- Path manipulation to access shared utilities from POC-1
- Always import `call_llm` from `utils.llm_provider`

### Initialization & Paths
```python
print("POC-1b: [Test Name]", flush=True)
print("=" * 70, flush=True)

# Paths
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"
```

**Key Points:**
- Print header with test name (flush=True for immediate output)
- Use Path objects for file operations
- Define constants for artifact and ground truth paths
- All print statements use `flush=True` for real-time output

---

## 2. Ground Truth Loading & Validation

### Loading Pattern
```python
def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)["chunks"]
```

**Ground Truth Format:**
```json
{
  "created_at": "2026-02-04 10:49:09",
  "total_chunks": 15,
  "total_terms": 163,
  "chunks": [
    {
      "chunk_id": "concepts__index_chunk_0",
      "doc_id": "concepts__index",
      "heading": "Content",
      "source_file": "concepts__index.md",
      "content": "...",
      "terms": [
        {"term": "Kubernetes", "tier": 1},
        {"term": "cluster", "tier": 1}
      ],
      "term_count": 2
    }
  ]
}
```

**Key Points:**
- Ground truth is a JSON file with `chunks` array
- Each chunk has: `chunk_id`, `doc_id`, `heading`, `source_file`, `content`, `terms`, `term_count`
- Terms are objects with `term` and `tier` fields
- Use type hints in function signatures

---

## 3. Shared Utilities

### LLM Provider (`utils.llm_provider.call_llm`)
```python
def call_llm(
    prompt: str,
    model: str = "claude-haiku-4-5",
    timeout: int = 90,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> str:
    """Call Claude API with automatic token refresh and rate limiting."""
```

**Usage Pattern:**
```python
response = call_llm(
    prompt=PROMPT_TEMPLATE.format(content=content[:2500]),
    model="claude-haiku",
    temperature=0,
    max_tokens=1000
)
```

**Key Points:**
- Always use `temperature=0` for deterministic extraction
- Truncate content to 2500 chars to fit token limits
- Use model aliases: `"claude-haiku"`, `"claude-sonnet"`, `"claude-opus"`
- Handles OAuth token refresh automatically
- Includes exponential backoff for rate limiting

### Parsing Utilities (Custom in Each Test)
```python
def parse_terms(response: str, require_quotes: bool = False) -> list[str]:
    """Extract terms from LLM JSON response."""
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

**Key Points:**
- Handles markdown code blocks (`` ```json ... ``` ``)
- Extracts JSON from response text
- Handles both list and dict term formats
- Returns empty list on parse failure (graceful degradation)
- Each test file defines its own parsing logic

---

## 4. Span Verification Pattern

### Strict Span Verification
```python
def strict_span_verify(term: str, content: str) -> bool:
    """Verify term exists in content with normalization."""
    if not term or len(term) < 2:
        return False
    content_lower = content.lower()
    term_lower = term.lower().strip()
    
    # Exact match
    if term_lower in content_lower:
        return True
    
    # Normalized match (underscores/hyphens)
    normalized = term_lower.replace("_", " ").replace("-", " ")
    if normalized in content_lower.replace("_", " ").replace("-", " "):
        return True
    
    # CamelCase expansion
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True
    
    return False
```

**Key Points:**
- Filters out hallucinations by verifying terms exist in source
- Handles multiple normalization strategies:
  - Exact match (case-insensitive)
  - Underscore/hyphen normalization
  - CamelCase expansion
- Minimum term length of 2 characters
- Used to filter extracted terms before metrics calculation

---

## 5. Metrics Calculation Pattern

### Term Matching
```python
def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")

def match_terms(extracted: str, ground_truth: str) -> bool:
    """Check if extracted term matches ground truth term."""
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)
    
    # Exact match
    if ext_norm == gt_norm:
        return True
    
    # Fuzzy match (85% similarity)
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True
    
    # Token overlap (80% of GT tokens present)
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True
    
    return False
```

### Metrics Calculation
```python
def calculate_metrics(extracted: list[str], ground_truth: list[dict]) -> dict:
    """Calculate precision, recall, hallucination, F1."""
    gt_terms = [t.get("term", "") for t in ground_truth]
    matched_gt = set()
    matched_ext = set()
    tp = 0
    
    # Bipartite matching: each extracted term matches at most one GT term
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
    f1 = (
        2 * precision * recall / (precision + recall) 
        if (precision + recall) > 0 else 0
    )
    
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

**Key Points:**
- **Bipartite matching**: Each extracted term matches at most one GT term
- **Precision**: TP / (TP + FP) = correct / extracted
- **Recall**: TP / (TP + FN) = correct / ground_truth
- **Hallucination**: FP / extracted = false positives / extracted
- **F1**: Harmonic mean of precision and recall
- Returns dict with all metrics plus TP/FP/FN counts

---

## 6. Prompt Patterns

### Extraction Prompts
```python
SIMPLE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""

EXHAUSTIVE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
Be EXHAUSTIVE. Include: resources, components, concepts, feature gates, lifecycle stages, CLI flags, API terms.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""

QUOTE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
For EACH term, provide the exact quote where it appears.
CHUNK:
{content}
Output JSON: {{"terms": [{{"quote": "exact text", "term": "Term"}}]}}"""

CONSERVATIVE_PROMPT = """Extract ONLY the most important Kubernetes technical terms.
Be CONSERVATIVE. Only core resources, key components, essential concepts.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""
```

### Verification Prompts
```python
VERIFY_PROMPT = """Filter this list of extracted terms from Kubernetes docs.
Keep ONLY Kubernetes-specific technical terms. Remove generic English words.
CHUNK: {content}
TERMS: {terms}
Output JSON: {{"terms": ["term1", ...]}}"""

STRICT_VERIFY_PROMPT = """You are a Kubernetes documentation expert filtering extracted terms.

DOCUMENTATION CHUNK:
{content}

CANDIDATE TERMS:
{terms}

Your task: Keep ONLY terms that meet ALL criteria:
1. Is a Kubernetes-specific technical term (not generic English)
2. Would be valuable in a documentation search index
3. Represents a specific K8s concept, resource, component, or feature

STRICTLY REMOVE:
- Generic words (memory, network, process, information, section)
- YAML/JSON structural keywords (title, stages, value, content)
- Version numbers or dates
- File paths, URLs
- Common verbs/adjectives used in tech writing

Be CONSERVATIVE. When in doubt, REMOVE the term.

Output ONLY the filtered terms as JSON:
{{"terms": ["term1", "term2", ...]}}"""
```

**Key Points:**
- Prompts use `{content}` and `{terms}` placeholders
- Always specify JSON output format explicitly
- Use "ALL" and "EXHAUSTIVE" for high-recall extraction
- Use "CONSERVATIVE" for high-precision extraction
- Verification prompts are more detailed with explicit criteria

---

## 7. Strategy Implementation Pattern

### Single Extraction Strategy
```python
def extract_simple(content: str, model: str = "claude-haiku") -> list[str]:
    prompt = SIMPLE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]
```

### Combined Strategy (Ensemble)
```python
def strategy_ensemble_verified(content: str) -> list[str]:
    """High recall ensemble + LLM verification."""
    # Step 1: Collect terms from multiple extraction methods
    all_terms = set()
    all_terms.update(extract_simple(content))
    all_terms.update(extract_quote(content))
    all_terms.update(extract_exhaustive(content))
    
    if not all_terms:
        return []
    
    # Step 2: Verify with LLM
    prompt = VERIFY_PROMPT.format(
        content=content[:2000], terms=json.dumps(sorted(all_terms))
    )
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=1000)
    verified = parse_terms(response)
    
    # Step 3: Final span verification
    return [t for t in verified if strict_span_verify(t, content)]
```

### Voting Strategy
```python
def strategy_intersection_vote(content: str, min_votes: int = 2) -> list[str]:
    """Keep terms with multiple strategy votes."""
    simple = set(extract_simple(content))
    quote = set(extract_quote(content))
    exhaustive = set(extract_exhaustive(content))
    conservative = set(extract_conservative(content))
    
    term_votes = {}
    for term_set in [simple, quote, exhaustive, conservative]:
        for term in term_set:
            key = term.lower()
            if key not in term_votes:
                term_votes[key] = {"canonical": term, "votes": 0}
            term_votes[key]["votes"] += 1
    
    return [
        info["canonical"] for info in term_votes.values() 
        if info["votes"] >= min_votes
    ]
```

**Key Points:**
- Strategies are functions that take `content: str` and return `list[str]`
- Use `set()` for deduplication across multiple extractions
- Always apply `strict_span_verify` as final filter
- Voting strategies track canonical form (original casing)
- Strategies can be parameterized (e.g., `min_votes`)

---

## 8. Main Experiment Loop Pattern

### Experiment Runner
```python
def run_experiment(num_chunks: int = 10):
    ground_truth = load_ground_truth()
    test_chunks = ground_truth[:num_chunks]
    
    print(f"\nTesting on {len(test_chunks)} chunks", flush=True)
    
    # Define strategies
    strategies = {
        "simple_haiku": lambda c: extract_simple(c),
        "quote_haiku": lambda c: extract_quote(c),
        "ensemble_verified": strategy_ensemble_verified,
        "vote_2": lambda c: strategy_intersection_vote(c, 2),
        "vote_3": lambda c: strategy_intersection_vote(c, 3),
    }
    
    results = {name: [] for name in strategies}
    
    print(f"\n{'=' * 70}", flush=True)
    
    # Per-chunk evaluation
    for i, chunk in enumerate(test_chunks):
        print(
            f"\n[{i + 1}/{len(test_chunks)}] {chunk['chunk_id']} (GT: {chunk['term_count']} terms)",
            flush=True,
        )
        
        for name, extractor in strategies.items():
            try:
                start = time.time()
                extracted = extractor(chunk["content"])
                elapsed = time.time() - start
                
                metrics = calculate_metrics(extracted, chunk["terms"])
                metrics["elapsed"] = elapsed
                results[name].append(metrics)
                
                # Per-chunk output
                r_mark = (
                    "✓"
                    if metrics["recall"] >= 0.95
                    else ("~" if metrics["recall"] >= 0.85 else " ")
                )
                h_mark = (
                    "✓"
                    if metrics["hallucination"] < 0.10
                    else ("~" if metrics["hallucination"] < 0.20 else " ")
                )
                
                print(
                    f"  {name:<20}: R={metrics['recall']:>5.0%}{r_mark} H={metrics['hallucination']:>5.0%}{h_mark} ({metrics['extracted_count']:>2} ext)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {name:<20}: ERROR - {e}", flush=True)
    
    # Aggregate results
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(
        f"{'Strategy':<22} {'Precision':>10} {'Recall':>10} {'Halluc':>10} {'F1':>8}",
        flush=True,
    )
    print("-" * 55, flush=True)
    
    summary = {}
    for name, metrics_list in results.items():
        if not metrics_list:
            continue
        avg_p = sum(m["precision"] for m in metrics_list) / len(metrics_list)
        avg_r = sum(m["recall"] for m in metrics_list) / len(metrics_list)
        avg_h = sum(m["hallucination"] for m in metrics_list) / len(metrics_list)
        avg_f1 = sum(m["f1"] for m in metrics_list) / len(metrics_list)
        
        r_mark = "✓" if avg_r >= 0.95 else ("~" if avg_r >= 0.85 else "  ")
        h_mark = "✓" if avg_h < 0.10 else ("~" if avg_h < 0.20 else "  ")
        
        print(
            f"{name:<22} {avg_p:>10.1%} {avg_r:>8.1%} {r_mark} {avg_h:>8.1%} {h_mark} {avg_f1:>7.1%}",
            flush=True,
        )
        summary[name] = {
            "precision": avg_p,
            "recall": avg_r,
            "hallucination": avg_h,
            "f1": avg_f1,
        }
    
    # Target check
    print(f"\n{'=' * 70}", flush=True)
    print("TARGET CHECK: 95%+ recall AND <10% hallucination", flush=True)
    print("=" * 70, flush=True)
    
    meeting_both = [
        (n, m)
        for n, m in summary.items()
        if m["recall"] >= 0.95 and m["hallucination"] < 0.10
    ]
    
    if meeting_both:
        print("✅ STRATEGIES MEETING BOTH TARGETS:")
        for name, m in sorted(meeting_both, key=lambda x: -x[1]["recall"]):
            print(
                f"   {name}: R={m['recall']:.1%}, H={m['hallucination']:.1%}, F1={m['f1']:.1%}"
            )
    else:
        print("❌ No strategy met both targets.")
    
    # Save results
    results_path = ARTIFACTS_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {results_path}", flush=True)
    
    return summary


if __name__ == "__main__":
    run_experiment(num_chunks=5)
```

**Key Points:**
- `run_experiment()` is the main entry point
- Strategies are defined as dict of name → function
- Results are accumulated per-chunk, then aggregated
- Per-chunk output shows recall/hallucination with visual markers:
  - `✓` = excellent (R≥95%, H<10%)
  - `~` = good (R≥85%, H<20%)
  - ` ` = needs improvement
- Aggregate output shows averages across all chunks
- Target check identifies strategies meeting both goals
- Results saved to JSON in artifacts directory

---

## 9. Output & Logging Format

### Console Output Pattern
```
POC-1b: Fast Combined Strategy Test
======================================================================

Testing on 5 chunks

======================================================================

[1/5] _index_chunk_0 (GT: 1 terms)
  simple_haiku        : R= 100%✓ H=  0%✓ ( 1 ext)
  quote_haiku         : R= 100%✓ H=  0%✓ ( 1 ext)
  ensemble_verified   : R= 100%✓ H=  0%✓ ( 1 ext)

[2/5] concepts__index_chunk_0 (GT: 2 terms)
  simple_haiku        : R=  50% ~ H= 50%   ( 2 ext)
  quote_haiku         : R= 100%✓ H=  0%✓ ( 2 ext)
  ensemble_verified   : R= 100%✓ H=  0%✓ ( 2 ext)

======================================================================
AGGREGATE RESULTS
======================================================================
Strategy                 Precision     Recall     Halluc       F1
-------------------------------------------------------
simple_haiku                 75.0%      75.0%       25.0%     75.0%
quote_haiku                  90.0%      90.0%       10.0%     90.0%
ensemble_verified            95.0%      95.0%        5.0%     95.0%

======================================================================
TARGET CHECK: 95%+ recall AND <10% hallucination
======================================================================
✅ STRATEGIES MEETING BOTH TARGETS:
   ensemble_verified: R=95.0%, H=5.0%, F1=95.0%

Saved to: /path/to/artifacts/test_results.json
```

**Key Points:**
- Header with test name and separator lines
- Per-chunk output with chunk ID and GT term count
- Per-strategy output with recall/hallucination marks
- Aggregate table with averages
- Target check section with visual indicators
- Final save confirmation

---

## 10. Results File Format

### JSON Results Structure
```json
{
  "simple_haiku": {
    "precision": 0.75,
    "recall": 0.75,
    "hallucination": 0.25,
    "f1": 0.75
  },
  "quote_haiku": {
    "precision": 0.90,
    "recall": 0.90,
    "hallucination": 0.10,
    "f1": 0.90
  },
  "ensemble_verified": {
    "precision": 0.95,
    "recall": 0.95,
    "hallucination": 0.05,
    "f1": 0.95
  }
}
```

**Key Points:**
- Top-level keys are strategy names
- Each strategy has: precision, recall, hallucination, f1
- All values are floats (0.0-1.0)
- Saved with `indent=2` for readability

---

## Template for New Test File

```python
#!/usr/bin/env python3
"""[Test name and description].

[2-3 sentence explanation of what this test validates]
"""

import json
import re
import sys
import time
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

print("POC-1b: [Test Name]", flush=True)
print("=" * 70, flush=True)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"


def load_ground_truth() -> list[dict]:
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)["chunks"]


def strict_span_verify(term: str, content: str) -> bool:
    """Verify term exists in content."""
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


def parse_terms(response: str, require_quotes: bool = False) -> list[str]:
    """Parse terms from LLM JSON response."""
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


# ============================================================================
# PROMPTS
# ============================================================================

EXTRACTION_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def extract_terms(content: str, model: str = "claude-haiku") -> list[str]:
    """Extract terms using [strategy name]."""
    prompt = EXTRACTION_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_terms(response)
    return [t for t in terms if strict_span_verify(t, content)]


# ============================================================================
# METRICS
# ============================================================================


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
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

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


# ============================================================================
# MAIN
# ============================================================================


def run_experiment(num_chunks: int = 5):
    ground_truth = load_ground_truth()
    test_chunks = ground_truth[:num_chunks]

    print(f"\nTesting on {len(test_chunks)} chunks", flush=True)

    strategies = {
        "strategy_name": extract_terms,
    }

    results = {name: [] for name in strategies}

    print(f"\n{'=' * 70}", flush=True)

    for i, chunk in enumerate(test_chunks):
        print(
            f"\n[{i + 1}/{len(test_chunks)}] {chunk['chunk_id']} (GT: {chunk['term_count']} terms)",
            flush=True,
        )

        for name, extractor in strategies.items():
            try:
                start = time.time()
                extracted = extractor(chunk["content"])
                elapsed = time.time() - start

                metrics = calculate_metrics(extracted, chunk["terms"])
                metrics["elapsed"] = elapsed
                results[name].append(metrics)

                r_mark = (
                    "✓"
                    if metrics["recall"] >= 0.95
                    else ("~" if metrics["recall"] >= 0.85 else " ")
                )
                h_mark = (
                    "✓"
                    if metrics["hallucination"] < 0.10
                    else ("~" if metrics["hallucination"] < 0.20 else " ")
                )

                print(
                    f"  {name:<20}: R={metrics['recall']:>5.0%}{r_mark} H={metrics['hallucination']:>5.0%}{h_mark} ({metrics['extracted_count']:>2} ext)",
                    flush=True,
                )
            except Exception as e:
                print(f"  {name:<20}: ERROR - {e}", flush=True)

    # Aggregate
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("=" * 70, flush=True)
    print(
        f"{'Strategy':<22} {'Precision':>10} {'Recall':>10} {'Halluc':>10} {'F1':>8}",
        flush=True,
    )
    print("-" * 55, flush=True)

    summary = {}
    for name, metrics_list in results.items():
        if not metrics_list:
            continue
        avg_p = sum(m["precision"] for m in metrics_list) / len(metrics_list)
        avg_r = sum(m["recall"] for m in metrics_list) / len(metrics_list)
        avg_h = sum(m["hallucination"] for m in metrics_list) / len(metrics_list)
        avg_f1 = sum(m["f1"] for m in metrics_list) / len(metrics_list)

        r_mark = "✓" if avg_r >= 0.95 else ("~" if avg_r >= 0.85 else "  ")
        h_mark = "✓" if avg_h < 0.10 else ("~" if avg_h < 0.20 else "  ")

        print(
            f"{name:<22} {avg_p:>10.1%} {avg_r:>8.1%} {r_mark} {avg_h:>8.1%} {h_mark} {avg_f1:>7.1%}",
            flush=True,
        )
        summary[name] = {
            "precision": avg_p,
            "recall": avg_r,
            "hallucination": avg_h,
            "f1": avg_f1,
        }

    # Target check
    print(f"\n{'=' * 70}", flush=True)
    print("TARGET CHECK: 95%+ recall AND <10% hallucination", flush=True)
    print("=" * 70, flush=True)

    meeting_both = [
        (n, m)
        for n, m in summary.items()
        if m["recall"] >= 0.95 and m["hallucination"] < 0.10
    ]

    if meeting_both:
        print("✅ STRATEGIES MEETING BOTH TARGETS:")
        for name, m in sorted(meeting_both, key=lambda x: -x[1]["recall"]):
            print(
                f"   {name}: R={m['recall']:.1%}, H={m['hallucination']:.1%}, F1={m['f1']:.1%}"
            )
    else:
        print("❌ No strategy met both targets.")

    # Save
    results_path = ARTIFACTS_DIR / "test_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to: {results_path}", flush=True)

    return summary


if __name__ == "__main__":
    run_experiment(num_chunks=5)
```

---

## Summary of Key Patterns

| Aspect | Pattern |
|--------|---------|
| **File Structure** | Shebang → Docstring → Imports → Constants → Functions → Main |
| **Imports** | Always import `call_llm` from `utils.llm_provider` |
| **Ground Truth** | Load from `small_chunk_ground_truth.json`, extract `["chunks"]` |
| **Parsing** | Custom `parse_terms()` that handles markdown code blocks and JSON |
| **Verification** | `strict_span_verify()` filters hallucinations by checking term exists in content |
| **Metrics** | Bipartite matching for TP/FP/FN, then precision/recall/hallucination/F1 |
| **Prompts** | Use `{content}` and `{terms}` placeholders, specify JSON output format |
| **Strategies** | Functions taking `content: str` → `list[str]`, apply span verify as final filter |
| **Experiment Loop** | Per-chunk evaluation → aggregate → target check → save JSON |
| **Output** | Console with visual markers (✓/~/space), JSON results file |
| **Logging** | All prints use `flush=True` for real-time output |

---

## Next Steps for New Test File

1. **Copy template** above and customize:
   - Change test name in header and print statement
   - Define extraction prompts for your strategy
   - Implement extraction functions
   - Define strategies dict with your approaches

2. **Run experiment**:
   ```bash
   python test_sentence_extraction.py
   ```

3. **Check results**:
   - Console output shows per-chunk and aggregate metrics
   - JSON file saved to `artifacts/test_results.json`
   - Compare against targets: 95%+ recall, <10% hallucination

4. **Iterate**:
   - Adjust prompts based on results
   - Try different strategy combinations
   - Document findings in POC README
