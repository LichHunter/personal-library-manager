#!/usr/bin/env python3
"""
Consolidate existing test queries and categorize them for adaptive retrieval evaluation.

This script:
1. Loads queries from existing POC datasets
2. Maps existing query types to the 5 adaptive retrieval categories
3. Identifies gaps (categories that need more queries)
4. Outputs consolidated queries and gap analysis
"""

import json
from pathlib import Path
from collections import defaultdict
from typing import TypedDict, Literal

# Type definitions
QueryType = Literal["factoid", "procedural", "explanatory", "comparison", "troubleshooting"]
Granularity = Literal["chunk", "heading", "document"]

class ConsolidatedQuery(TypedDict, total=False):
    id: str
    query: str
    query_type: QueryType
    expected_answer: str
    optimal_granularity: Granularity
    difficulty: str
    source: str
    source_file: str
    original_type: str
    doc_id: str


# Mapping from existing query types to new categories
TYPE_MAPPING: dict[str, QueryType] = {
    # From needle_questions.json
    "fact-lookup": "factoid",
    "how-to": "procedural",
    "conceptual": "explanatory",  # Default, but some may be comparison
    "problem-based": "troubleshooting",
    
    # From adversarial questions
    "VERSION": "factoid",
    "COMPARISON": "comparison",
    "NEGATION": "explanatory",  # Negation questions often explain limitations
    "VOCABULARY": "factoid",  # Vocabulary mismatch but still fact lookup
}

# Expected optimal granularity by query type
DEFAULT_GRANULARITY: dict[QueryType, Granularity] = {
    "factoid": "chunk",
    "procedural": "heading",
    "explanatory": "heading",
    "comparison": "heading",
    "troubleshooting": "heading",
}

# Minimum required queries per type
MINIMUM_COUNTS: dict[QueryType, int] = {
    "factoid": 50,
    "procedural": 50,
    "explanatory": 50,
    "comparison": 30,
    "troubleshooting": 30,
}


def load_needle_questions(path: Path) -> list[ConsolidatedQuery]:
    """Load and convert needle questions."""
    queries: list[ConsolidatedQuery] = []
    
    with open(path) as f:
        data = json.load(f)
    
    for q in data.get("questions", []):
        original_type = q.get("type", "unknown")
        query_type = TYPE_MAPPING.get(original_type, "explanatory")
        
        # Check if conceptual question is actually a comparison
        question_lower = q.get("question", "").lower()
        if original_type == "conceptual" and any(word in question_lower for word in ["difference", "vs", "versus", "compare", "differ"]):
            query_type = "comparison"
        
        queries.append({
            "id": f"needle_{q.get('id', '')}",
            "query": q.get("question", ""),
            "query_type": query_type,
            "expected_answer": q.get("expected_answer", ""),
            "optimal_granularity": DEFAULT_GRANULARITY[query_type],
            "difficulty": q.get("difficulty", "medium"),
            "source": "existing",
            "source_file": str(path.name),
            "original_type": original_type,
            "doc_id": data.get("needle_doc_id", ""),
        })
    
    return queries


def load_adversarial_questions(path: Path) -> list[ConsolidatedQuery]:
    """Load and convert adversarial questions."""
    queries: list[ConsolidatedQuery] = []
    
    with open(path) as f:
        data = json.load(f)
    
    for q in data.get("questions", []):
        category = q.get("category", "unknown")
        original_type = q.get("type", category)
        query_type = TYPE_MAPPING.get(category, "explanatory")
        
        queries.append({
            "id": f"adv_{q.get('id', '')}",
            "query": q.get("question", ""),
            "query_type": query_type,
            "expected_answer": q.get("expected_answer", ""),
            "optimal_granularity": DEFAULT_GRANULARITY[query_type],
            "difficulty": q.get("difficulty", "medium"),
            "source": "existing",
            "source_file": str(path.name),
            "original_type": f"{category}/{original_type}",
            "doc_id": data.get("needle_doc_id", ""),
        })
    
    return queries


def load_informed_questions(path: Path) -> list[ConsolidatedQuery]:
    """Load and convert informed questions from kubefix."""
    queries: list[ConsolidatedQuery] = []
    
    with open(path) as f:
        data = json.load(f)
    
    for q in data.get("questions", []):
        question = q.get("question", "")
        question_lower = question.lower()
        
        # Classify based on question patterns
        if any(word in question_lower for word in ["what is", "what are", "what does", "which", "when did", "what protocol", "what format", "what tools"]):
            if "difference" in question_lower or "types" in question_lower:
                query_type: QueryType = "comparison"
            else:
                query_type = "factoid"
        elif any(word in question_lower for word in ["how do", "how can", "how to", "steps to"]):
            query_type = "procedural"
        elif any(word in question_lower for word in ["why", "purpose", "role", "how does"]):
            query_type = "explanatory"
        elif any(word in question_lower for word in ["difference", "vs", "compare"]):
            query_type = "comparison"
        else:
            query_type = "explanatory"  # Default
        
        # Skip duplicate variants (only take q1 variant)
        if q.get("variant", "").endswith("_q2"):
            continue
        
        queries.append({
            "id": f"informed_{q.get('id', '')}",
            "query": question,
            "query_type": query_type,
            "expected_answer": q.get("expected_answer", ""),
            "optimal_granularity": DEFAULT_GRANULARITY[query_type],
            "difficulty": "medium",
            "source": "existing",
            "source_file": str(path.name),
            "original_type": "kubefix",
            "doc_id": q.get("doc_id", ""),
        })
    
    return queries


def analyze_gaps(queries: list[ConsolidatedQuery]) -> dict[QueryType, int]:
    """Analyze how many more queries are needed per type."""
    counts: dict[QueryType, int] = defaultdict(int)
    for q in queries:
        counts[q["query_type"]] += 1
    
    gaps: dict[QueryType, int] = {}
    for qtype, minimum in MINIMUM_COUNTS.items():
        current = counts.get(qtype, 0)
        gaps[qtype] = max(0, minimum - current)
    
    return gaps


def main():
    poc_root = Path(__file__).parent.parent.parent
    
    # Source files
    sources = {
        "needle": poc_root / "plm_vs_rag_benchmark/corpus/needle_questions.json",
        "adversarial": poc_root / "plm_vs_rag_benchmark/corpus/needle_questions_adversarial.json",
        "informed": poc_root / "modular_retrieval_pipeline/corpus/informed_questions.json",
    }
    
    # Load all queries
    all_queries: list[ConsolidatedQuery] = []
    
    if sources["needle"].exists():
        all_queries.extend(load_needle_questions(sources["needle"]))
        print(f"Loaded {len(load_needle_questions(sources['needle']))} queries from needle_questions.json")
    
    if sources["adversarial"].exists():
        all_queries.extend(load_adversarial_questions(sources["adversarial"]))
        print(f"Loaded {len(load_adversarial_questions(sources['adversarial']))} queries from adversarial")
    
    if sources["informed"].exists():
        informed = load_informed_questions(sources["informed"])
        all_queries.extend(informed)
        print(f"Loaded {len(informed)} queries from informed_questions.json")
    
    # Count by type
    counts: dict[str, int] = defaultdict(int)
    for q in all_queries:
        counts[q["query_type"]] += 1
    
    print("\n=== Current Query Counts ===")
    for qtype in MINIMUM_COUNTS.keys():
        current = counts.get(qtype, 0)
        minimum = MINIMUM_COUNTS[qtype]
        status = "OK" if current >= minimum else f"NEED {minimum - current} MORE"
        print(f"  {qtype:15}: {current:3} / {minimum:3} ({status})")
    
    # Gap analysis
    gaps = analyze_gaps(all_queries)
    total_needed = sum(gaps.values())
    
    print(f"\n=== Gap Analysis ===")
    print(f"Total existing queries: {len(all_queries)}")
    print(f"Total queries needed: {sum(MINIMUM_COUNTS.values())}")
    print(f"Queries to generate: {total_needed}")
    
    for qtype, gap in gaps.items():
        if gap > 0:
            print(f"  {qtype}: need {gap} more")
    
    # Save consolidated queries
    output_path = poc_root / "adaptive_retrieval/benchmarks/datasets/consolidated_existing.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        "metadata": {
            "total_queries": len(all_queries),
            "query_counts": dict(counts),
            "gaps": gaps,
            "sources": list(sources.keys()),
        },
        "queries": all_queries,
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved consolidated queries to: {output_path}")
    
    # Print sample queries by type
    print("\n=== Sample Queries by Type ===")
    for qtype in MINIMUM_COUNTS.keys():
        type_queries = [q for q in all_queries if q["query_type"] == qtype]
        if type_queries:
            print(f"\n{qtype.upper()} ({len(type_queries)} queries):")
            for q in type_queries[:2]:
                print(f"  - {q['query'][:80]}...")


if __name__ == "__main__":
    main()
