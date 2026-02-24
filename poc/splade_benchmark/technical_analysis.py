#!/usr/bin/env python3
"""Technical term analysis for SPLADE benchmark.

Analyzes how SPLADE handles technical terminology queries.

Usage:
    cd poc/splade_benchmark
    .venv/bin/python technical_analysis.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from splade_encoder import SPLADEEncoder
from splade_index import SPLADEIndex
from baseline_bm25 import load_informed_queries


POC_DIR = Path(__file__).parent
ARTIFACTS_DIR = POC_DIR / "artifacts"
SPLADE_INDEX_PATH = ARTIFACTS_DIR / "splade_index"


TECHNICAL_TERM_QUERIES = [
    {"id": "tech_camel_1", "query": "SubjectAccessReview", "category": "CamelCase"},
    {"id": "tech_camel_2", "query": "PodSecurityPolicy", "category": "CamelCase"},
    {"id": "tech_camel_3", "query": "ClusterRoleBinding", "category": "CamelCase"},
    {"id": "tech_abbrev_1", "query": "RBAC permissions", "category": "Abbreviation"},
    {"id": "tech_abbrev_2", "query": "HPA autoscaling", "category": "Abbreviation"},
    {"id": "tech_abbrev_3", "query": "k8s cluster", "category": "Abbreviation"},
    {"id": "tech_multi_1", "query": "webhook token authenticator", "category": "Multi-word"},
    {"id": "tech_multi_2", "query": "admission controller webhook", "category": "Multi-word"},
    {"id": "tech_multi_3", "query": "persistent volume claim", "category": "Multi-word"},
    {"id": "tech_hyphen_1", "query": "kube-apiserver", "category": "Hyphenated"},
    {"id": "tech_hyphen_2", "query": "cluster-admin role", "category": "Hyphenated"},
    {"id": "tech_hyphen_3", "query": "kube-controller-manager", "category": "Hyphenated"},
    {"id": "tech_dotted_1", "query": "kubernetes.io/hostname", "category": "Dotted"},
    {"id": "tech_dotted_2", "query": "node.kubernetes.io/instance-type", "category": "Dotted"},
]


@dataclass
class TermAnalysis:
    """Analysis of a single technical term query."""
    query_id: str
    query: str
    category: str
    top_expansion_terms: list[tuple[str, float]]
    num_nonzero_terms: int
    splade_rank: Optional[int]
    bm25_rank: Optional[int]
    rank_improvement: Optional[int]


class TechnicalTermAnalyzer:
    """Analyzer for technical terminology handling in SPLADE."""
    
    def __init__(
        self,
        splade_index_path: Path = SPLADE_INDEX_PATH,
    ):
        print("[TechnicalAnalyzer] Loading SPLADE encoder...")
        self.encoder = SPLADEEncoder()
        
        if splade_index_path.exists():
            print("[TechnicalAnalyzer] Loading SPLADE index...")
            self.index = SPLADEIndex.load(str(splade_index_path), encoder=self.encoder)
        else:
            print("[TechnicalAnalyzer] SPLADE index not found - expansion analysis only")
            self.index = None
    
    def analyze_expansion(self, query: str, top_k: int = 20) -> tuple[list[tuple[str, float]], int]:
        """Analyze expansion terms for a query.
        
        Returns:
            (top_k_terms, total_nonzero_terms)
        """
        sparse_vec = self.encoder.encode(query, return_tokens=False)
        assert isinstance(sparse_vec, dict)
        
        top_terms = self.encoder.get_top_terms(sparse_vec, k=top_k)
        
        return top_terms, len(sparse_vec)
    
    def analyze_query(
        self,
        query_id: str,
        query: str,
        category: str,
        ground_truth_doc_id: Optional[str] = None,
        bm25_rank: Optional[int] = None,
    ) -> TermAnalysis:
        """Analyze a single technical term query."""
        top_terms, num_terms = self.analyze_expansion(query)
        
        splade_rank = None
        if self.index is not None and ground_truth_doc_id:
            results = self.index.search(query, k=10)
            for i, r in enumerate(results):
                if ground_truth_doc_id in r["doc_id"]:
                    splade_rank = i + 1
                    break
        
        rank_improvement = None
        if splade_rank is not None and bm25_rank is not None:
            rank_improvement = bm25_rank - splade_rank
        
        return TermAnalysis(
            query_id=query_id,
            query=query,
            category=category,
            top_expansion_terms=top_terms,
            num_nonzero_terms=num_terms,
            splade_rank=splade_rank,
            bm25_rank=bm25_rank,
            rank_improvement=rank_improvement,
        )
    
    def run_analysis(self, queries: list[dict]) -> list[TermAnalysis]:
        """Run analysis on a list of queries."""
        results = []
        
        for q in tqdm(queries, desc="Analyzing"):
            result = self.analyze_query(
                q["id"],
                q["query"],
                q.get("category", "unknown"),
                q.get("ground_truth_doc_id"),
                q.get("bm25_rank"),
            )
            results.append(result)
        
        return results


def load_bm25_ranks() -> dict[str, int]:
    """Load BM25 ranks from baseline results."""
    baseline_path = ARTIFACTS_DIR / "baseline_bm25.json"
    if not baseline_path.exists():
        return {}
    
    with open(baseline_path) as f:
        data = json.load(f)
    
    ranks = {}
    for benchmark in ["informed", "needle", "realistic"]:
        if benchmark in data:
            for q in data[benchmark]["per_query"]:
                if q.get("rank"):
                    ranks[q["query_id"]] = q["rank"]
    
    return ranks


def format_expansion_terms(terms: list[tuple[str, float]], max_display: int = 10) -> str:
    """Format expansion terms for display."""
    formatted = []
    for term, weight in terms[:max_display]:
        formatted.append(f"{term}({weight:.2f})")
    return ", ".join(formatted)


def main():
    """Run technical term analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Technical Term Analysis")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/technical_term_analysis.json",
    )
    parser.add_argument(
        "--include-informed",
        action="store_true",
        help="Also analyze informed benchmark queries",
    )
    
    args = parser.parse_args()
    
    analyzer = TechnicalTermAnalyzer()
    
    print("\n" + "="*60)
    print("Analyzing Technical Term Queries")
    print("="*60)
    
    tech_results = analyzer.run_analysis(TECHNICAL_TERM_QUERIES)
    
    informed_results = []
    if args.include_informed:
        print("\n" + "="*60)
        print("Analyzing Informed Benchmark Queries")
        print("="*60)
        
        bm25_ranks = load_bm25_ranks()
        informed = load_informed_queries()
        
        informed_with_ranks = [
            {
                **q,
                "category": "informed",
                "bm25_rank": bm25_ranks.get(q["id"]),
            }
            for q in informed
        ]
        
        informed_results = analyzer.run_analysis(informed_with_ranks)
    
    print("\n" + "="*60)
    print("Technical Term Expansion Analysis")
    print("="*60)
    
    by_category: dict[str, list[TermAnalysis]] = {}
    for r in tech_results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)
    
    for category, results in by_category.items():
        print(f"\n{category.upper()}:")
        for r in results:
            print(f"  Query: '{r.query}'")
            print(f"    Non-zero terms: {r.num_nonzero_terms}")
            print(f"    Top terms: {format_expansion_terms(r.top_expansion_terms, 8)}")
    
    if informed_results:
        print("\n" + "="*60)
        print("Informed Query Rank Comparison")
        print("="*60)
        
        improved = [r for r in informed_results if r.rank_improvement and r.rank_improvement > 0]
        regressed = [r for r in informed_results if r.rank_improvement and r.rank_improvement < 0]
        unchanged = [r for r in informed_results if r.rank_improvement == 0]
        
        print(f"\nImproved: {len(improved)}")
        for r in improved[:5]:
            print(f"  {r.query[:50]}... BM25={r.bm25_rank} -> SPLADE={r.splade_rank} (+{r.rank_improvement})")
        
        print(f"\nRegressed: {len(regressed)}")
        for r in regressed[:5]:
            print(f"  {r.query[:50]}... BM25={r.bm25_rank} -> SPLADE={r.splade_rank} ({r.rank_improvement})")
        
        print(f"\nUnchanged: {len(unchanged)}")
    
    output_path = POC_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def analysis_to_dict(a: TermAnalysis) -> dict:
        d = asdict(a)
        d["top_expansion_terms"] = [(t, round(w, 4)) for t, w in a.top_expansion_terms]
        return d
    
    output_data = {
        "technical_terms": [analysis_to_dict(r) for r in tech_results],
        "informed": [analysis_to_dict(r) for r in informed_results] if informed_results else [],
        "summary": {
            "categories": list(by_category.keys()),
            "avg_nonzero_terms": sum(r.num_nonzero_terms for r in tech_results) / len(tech_results),
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
