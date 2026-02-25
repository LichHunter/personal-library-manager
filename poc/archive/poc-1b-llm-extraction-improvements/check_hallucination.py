#!/usr/bin/env python3
"""Check REAL hallucination - terms not in source text."""

import json
from pathlib import Path
from utils.llm_provider import AnthropicProvider

provider = AnthropicProvider()

# Load GT
gt_file = Path("artifacts/gt_100_chunks.json")
with open(gt_file) as f:
    gt_data = json.load(f)

chunks_to_analyze = gt_data["completed_chunks"][:5]

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

def term_in_text(term: str, text: str) -> bool:
    """Check if term exists in text (case-insensitive)."""
    return term.lower() in text.lower()

print("=" * 80)
print("HALLUCINATION ANALYSIS - Terms NOT in source text")
print("=" * 80)

total_hallucinations = {"V0_BASELINE": 0, "V1_DOMAIN_SPECIFIC": 0}
total_extracted = {"V0_BASELINE": 0, "V1_DOMAIN_SPECIFIC": 0}

for chunk in chunks_to_analyze:
    chunk_id = chunk["chunk_id"]
    text = chunk["content"]
    
    print(f"\n{'='*80}")
    print(f"CHUNK: {chunk_id[:60]}...")
    print(f"TEXT LENGTH: {len(text)} chars")
    print(f"{'='*80}")
    
    for variant_name, prompt_template in PROMPTS.items():
        extracted = extract_terms(text, prompt_template)
        total_extracted[variant_name] += len(extracted)
        
        hallucinations = []
        for term in extracted:
            if not term_in_text(term, text):
                hallucinations.append(term)
                total_hallucinations[variant_name] += 1
        
        print(f"\n--- {variant_name} ---")
        print(f"Extracted: {len(extracted)} terms")
        if hallucinations:
            print(f"HALLUCINATIONS ({len(hallucinations)}): {hallucinations}")
        else:
            print(f"HALLUCINATIONS: 0 (all terms exist in text)")

print("\n" + "=" * 80)
print("SUMMARY - TRUE HALLUCINATION RATE")
print("=" * 80)
for variant in PROMPTS:
    rate = total_hallucinations[variant] / total_extracted[variant] * 100 if total_extracted[variant] else 0
    print(f"{variant}: {total_hallucinations[variant]}/{total_extracted[variant]} = {rate:.1f}% hallucination")
