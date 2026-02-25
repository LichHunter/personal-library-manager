#!/usr/bin/env python3
"""Analyze WHERE GT terms appear and WHY extraction missed them."""

import json
import re
from pathlib import Path
from utils.llm_provider import AnthropicProvider

provider = AnthropicProvider()

# Load GT
gt_file = Path("artifacts/gt_100_chunks.json")
with open(gt_file) as f:
    gt_data = json.load(f)

def find_term_location(term: str, text: str) -> str:
    """Determine where in the text a term appears."""
    term_lower = term.lower()
    text_lower = text.lower()
    
    if term_lower not in text_lower:
        return "NOT_IN_TEXT"
    
    # Check different sections
    lines = text.split('\n')
    
    # YAML frontmatter (between --- markers)
    in_frontmatter = False
    frontmatter_lines = []
    content_lines = []
    dash_count = 0
    
    for line in lines:
        if line.strip() == '---':
            dash_count += 1
            if dash_count == 2:
                in_frontmatter = False
            else:
                in_frontmatter = True
            continue
        if in_frontmatter or dash_count < 2:
            frontmatter_lines.append(line)
        else:
            content_lines.append(line)
    
    frontmatter_text = '\n'.join(frontmatter_lines).lower()
    content_text = '\n'.join(content_lines).lower()
    
    locations = []
    
    if term_lower in frontmatter_text:
        locations.append("YAML_FRONTMATTER")
    
    # Check if in URL/path
    url_pattern = r'\[.*?\]\((.*?)\)'
    urls = re.findall(url_pattern, text)
    for url in urls:
        if term_lower in url.lower():
            locations.append("URL_PATH")
            break
    
    # Check if in code block
    code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
    for block in code_blocks:
        if term_lower in block.lower():
            locations.append("CODE_BLOCK")
            break
    
    # Check if in inline code
    inline_code = re.findall(r'`([^`]+)`', text)
    for code in inline_code:
        if term_lower in code.lower():
            locations.append("INLINE_CODE")
            break
    
    # Check if in prose (not in special locations)
    prose_text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
    prose_text = re.sub(r'\[.*?\]\(.*?\)', '', prose_text)  # Remove links
    prose_text = re.sub(r'`[^`]+`', '', prose_text)  # Remove inline code
    prose_text = re.sub(r'---.*?---', '', prose_text, flags=re.DOTALL)  # Remove frontmatter
    
    if term_lower in prose_text.lower():
        locations.append("PROSE")
    
    # Check link text (not URL)
    link_texts = re.findall(r'\[([^\]]+)\]', text)
    for link_text in link_texts:
        if term_lower in link_text.lower():
            locations.append("LINK_TEXT")
            break
    
    return ", ".join(locations) if locations else "UNKNOWN"

def extract_terms(text: str) -> list[str]:
    """Extract terms using baseline prompt."""
    prompt = f"""Extract ALL technical terms from this documentation chunk.

TEXT:
{text}

Return JSON: {{"terms": ["term1", "term2", ...]}}"""
    
    response = provider.generate(prompt, model="claude-haiku-4-5", max_tokens=2000, temperature=0.0)
    
    match = re.search(r'\{[^{}]*"terms"\s*:\s*\[[^\]]*\][^{}]*\}', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return [str(t).lower() for t in data.get("terms", []) if t]
        except:
            pass
    return []

# Analyze first 5 chunks
chunks = gt_data["completed_chunks"][:5]

print("=" * 100)
print("EXTRACTION FAILURE ANALYSIS - WHERE DO MISSED TERMS APPEAR?")
print("=" * 100)

location_stats = {}
missed_by_location = {}

for chunk in chunks:
    chunk_id = chunk["chunk_id"]
    text = chunk["content"]
    gt_terms = [t["term"].lower() for t in chunk["terms"]]
    
    extracted = extract_terms(text)
    extracted_set = set(extracted)
    
    print(f"\n{'='*100}")
    print(f"CHUNK: {chunk_id}")
    print(f"GT: {len(gt_terms)} terms | Extracted: {len(extracted)} terms")
    print(f"{'='*100}")
    
    print("\n| GT Term | Location | Extracted? |")
    print("|---------|----------|------------|")
    
    for gt_term in gt_terms:
        location = find_term_location(gt_term, text)
        was_extracted = "YES" if gt_term in extracted_set else "NO"
        
        print(f"| {gt_term[:30]:30} | {location[:25]:25} | {was_extracted:10} |")
        
        # Stats
        if location not in location_stats:
            location_stats[location] = {"total": 0, "extracted": 0}
        location_stats[location]["total"] += 1
        if was_extracted == "YES":
            location_stats[location]["extracted"] += 1
        else:
            if location not in missed_by_location:
                missed_by_location[location] = []
            missed_by_location[location].append(gt_term)

print("\n" + "=" * 100)
print("SUMMARY: EXTRACTION RATE BY TERM LOCATION")
print("=" * 100)
print("\n| Location | Total | Extracted | Rate |")
print("|----------|-------|-----------|------|")
for loc, stats in sorted(location_stats.items(), key=lambda x: x[1]["total"], reverse=True):
    rate = stats["extracted"] / stats["total"] * 100 if stats["total"] else 0
    print(f"| {loc:25} | {stats['total']:5} | {stats['extracted']:9} | {rate:5.1f}% |")

print("\n" + "=" * 100)
print("MISSED TERMS BY LOCATION")
print("=" * 100)
for loc, terms in missed_by_location.items():
    print(f"\n{loc}:")
    for t in terms[:10]:  # Show first 10
        print(f"  - {t}")
    if len(terms) > 10:
        print(f"  ... and {len(terms) - 10} more")
