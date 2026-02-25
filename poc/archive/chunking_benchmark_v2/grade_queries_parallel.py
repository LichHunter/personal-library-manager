#!/usr/bin/env python3
"""
Grade all queries in parallel using separate Sonnet agents.
Each query gets its own agent for grading.
"""

import json
import re
from pathlib import Path

# Read the results file
results_file = Path("results/gems_adaptive_hybrid_2026-01-26-184607.md")
content = results_file.read_text()

# Split by query sections
query_sections = re.split(r"^## Query: ", content, flags=re.MULTILINE)[
    1:
]  # Skip header

# Extract query data
queries = []
for section in query_sections:
    lines = section.split("\n")
    query_id = lines[0].strip()

    # Extract query text
    query_match = re.search(r"\*\*Query\*\*: (.+)", section)
    query_text = query_match.group(1) if query_match else ""

    # Extract expected answer
    expected_match = re.search(r"\*\*Expected Answer\*\*: (.+)", section)
    expected_answer = expected_match.group(1) if expected_match else ""

    # Extract retrieved chunks (everything between "**Retrieved Chunks**:" and "**Baseline Score**:")
    chunks_match = re.search(
        r"\*\*Retrieved Chunks\*\*:\n(.+?)\n\*\*Baseline Score\*\*:", section, re.DOTALL
    )
    retrieved_chunks = chunks_match.group(1).strip() if chunks_match else ""

    # Extract baseline score
    baseline_match = re.search(r"\*\*Baseline Score\*\*: (\d+)/10", section)
    baseline_score = int(baseline_match.group(1)) if baseline_match else 0

    queries.append(
        {
            "id": query_id,
            "query": query_text,
            "expected_answer": expected_answer,
            "retrieved_chunks": retrieved_chunks,
            "baseline_score": baseline_score,
            "section_text": section,
        }
    )

# Save queries for grading
output = {"total_queries": len(queries), "queries": queries}

output_file = Path("results/queries_to_grade.json")
output_file.write_text(json.dumps(output, indent=2))

print(f"Extracted {len(queries)} queries")
print(f"Saved to: {output_file}")
print("\nQuery IDs:")
for q in queries:
    print(f"  - {q['id']}: {q['query'][:60]}...")
