#!/usr/bin/env python3
"""Analyze false positives to understand if ground truth is too conservative.

The key question: Are the "hallucinations" actually valid technical terms
that Opus missed in the ground truth?
"""

import json
import re
import sys
from pathlib import Path

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"


def load_ground_truth():
    with open(GROUND_TRUTH_PATH) as f:
        return json.load(f)["chunks"]


def strict_span_verify(term, content):
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


def parse_terms(response):
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            terms = data.get("terms", [])
            if isinstance(terms, list):
                if terms and isinstance(terms[0], dict):
                    return [t.get("term", "") for t in terms]
                return [str(t) for t in terms]
    except:
        pass
    return []


def normalize_term(term):
    return term.lower().strip().replace("-", " ").replace("_", " ")


def match_terms(extracted, ground_truth):
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


EXHAUSTIVE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.
Be EXHAUSTIVE. Include: resources, components, concepts, feature gates, lifecycle stages, CLI flags, API terms.
CHUNK:
{content}
Output JSON: {{"terms": ["term1", "term2", ...]}}"""


VALIDATE_PROMPT = """You are a Kubernetes expert. Given a term extracted from documentation, determine if it's a valid technical term worth indexing.

TERM: {term}
CONTEXT: {context}

Answer ONLY "YES" or "NO":
- YES if this is a valid Kubernetes/container/cloud technical term
- NO if this is generic English, structural text, or not worth indexing

Answer:"""


def analyze_chunk(chunk):
    """Analyze false positives for a single chunk."""
    content = chunk["content"]
    gt_terms = [t.get("term", "") for t in chunk["terms"]]

    # Extract with exhaustive Sonnet (highest recall)
    prompt = EXHAUSTIVE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model="claude-sonnet", temperature=0, max_tokens=1000)
    extracted = parse_terms(response)
    extracted = [t for t in extracted if strict_span_verify(t, content)]

    # Find false positives (extracted but not in GT)
    false_positives = []
    for ext in extracted:
        is_match = any(match_terms(ext, gt) for gt in gt_terms)
        if not is_match:
            false_positives.append(ext)

    return {
        "chunk_id": chunk["chunk_id"],
        "gt_terms": gt_terms,
        "extracted": extracted,
        "false_positives": false_positives,
        "content": content[:500] + "..." if len(content) > 500 else content,
    }


def validate_term(term, context):
    """Ask LLM if a term is a valid technical term."""
    prompt = VALIDATE_PROMPT.format(term=term, context=context[:500])
    response = call_llm(prompt, model="claude-haiku", temperature=0, max_tokens=10)
    return "YES" in response.upper()


def main():
    print("Analyzing False Positives in Term Extraction")
    print("=" * 70)

    ground_truth = load_ground_truth()

    # Analyze first 5 chunks
    all_fps = []
    for i, chunk in enumerate(ground_truth[:5]):
        print(f"\n[{i + 1}/5] Analyzing {chunk['chunk_id']}...")
        result = analyze_chunk(chunk)

        print(f"  Ground Truth: {len(result['gt_terms'])} terms")
        print(f"  Extracted: {len(result['extracted'])} terms")
        print(f"  False Positives: {len(result['false_positives'])} terms")

        if result["false_positives"]:
            print(f"\n  False positive terms:")
            for fp in result["false_positives"][:10]:
                all_fps.append(
                    {
                        "term": fp,
                        "chunk_id": chunk["chunk_id"],
                        "context": chunk["content"][:300],
                    }
                )
                print(f"    - {fp}")

    # Validate a sample of false positives
    print(f"\n{'=' * 70}")
    print("VALIDATING SAMPLE FALSE POSITIVES")
    print("=" * 70)

    sample = all_fps[:15]
    valid_count = 0

    for fp in sample:
        is_valid = validate_term(fp["term"], fp["context"])
        status = "✓ VALID" if is_valid else "✗ NOISE"
        print(f"  {fp['term']:<30} -> {status}")
        if is_valid:
            valid_count += 1

    print(f"\n{'=' * 70}")
    print(
        f"SUMMARY: {valid_count}/{len(sample)} 'false positives' are actually VALID terms"
    )
    print(
        f"This suggests ground truth is missing ~{valid_count / len(sample) * 100:.0f}% of valid terms"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
