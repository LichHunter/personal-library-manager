#!/usr/bin/env python3
"""Analyze WHY recall dropped so dramatically in prompt variants."""

import json
from pathlib import Path
from utils.llm_provider import AnthropicProvider

provider = AnthropicProvider()

# Load GT
gt_file = Path("artifacts/gt_100_chunks.json")
with open(gt_file) as f:
    gt_data = json.load(f)

# Get first 3 chunks for detailed analysis
chunks_to_analyze = gt_data["completed_chunks"][:3]

# The prompts we tested
PROMPTS = {
    "V0_BASELINE": """Extract ALL technical terms from this documentation chunk.

TEXT:
{text}

Return JSON: {{"terms": ["term1", "term2", ...]}}""",

    "V1_DOMAIN_SPECIFIC": """Extract ONLY domain-specific technical terms from this documentation.

EXCLUDE these categories:
- Generic data types: string, int, boolean, byte, object, array, map, list
- Structural words: name, kind, spec, status, field, key, value, items, type
- Action verbs: create, delete, get, set, run, update, list, watch, apply
- Shell/CLI artifacts: cat, rm, bash, shell, EOF, tty, null, vi, echo
- Data formats: yaml, json, xml, binary, base64

INCLUDE: Named APIs, specific tools, protocols, frameworks, domain concepts

TEXT:
{text}

Return JSON: {{"terms": ["term1", "term2", ...]}}"""
}

def extract_terms(text: str, prompt_template: str) -> list[str]:
    """Call LLM and extract terms."""
    prompt = prompt_template.format(text=text)
    response = provider.generate(prompt, model="claude-haiku-4-5", max_tokens=2000, temperature=0.0)
    
    import re
    match = re.search(r'\{[^{}]*"terms"\s*:\s*\[[^\]]*\][^{}]*\}', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return [str(t).lower() for t in data.get("terms", []) if t]
        except:
            pass
    return []

print("=" * 80)
print("DETAILED FAILURE ANALYSIS")
print("=" * 80)

for chunk in chunks_to_analyze:
    chunk_id = chunk["chunk_id"]
    text = chunk["content"]
    gt_terms = [t["term"].lower() for t in chunk["terms"]]
    
    print(f"\n{'='*80}")
    print(f"CHUNK: {chunk_id}")
    print(f"GT TERMS ({len(gt_terms)}): {gt_terms}")
    print(f"{'='*80}")
    
    for variant_name, prompt_template in PROMPTS.items():
        extracted = extract_terms(text, prompt_template)
        
        gt_set = set(gt_terms)
        extracted_set = set(extracted)
        
        tp = gt_set & extracted_set
        fp = extracted_set - gt_set
        fn = gt_set - extracted_set
        
        print(f"\n--- {variant_name} ---")
        print(f"Extracted ({len(extracted)}): {sorted(extracted)}")
        print(f"TRUE POSITIVES ({len(tp)}): {sorted(tp)}")
        print(f"FALSE POSITIVES ({len(fp)}): {sorted(fp)}")
        print(f"FALSE NEGATIVES ({len(fn)}): {sorted(fn)}")
        print(f"Precision: {len(tp)/len(extracted)*100:.1f}%" if extracted else "N/A")
        print(f"Recall: {len(tp)/len(gt_terms)*100:.1f}%")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
