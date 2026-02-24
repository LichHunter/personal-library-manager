#!/usr/bin/env python3
"""Unified benchmark runner comparing PLM vs baseline RAG variants.

This script runs identical queries through 3 retrieval systems:
1. PLM Full - Complete hybrid system (BM25 + semantic + RRF + enrichment + expansion)
2. Baseline-Enriched - LangChain + FAISS with enriched content
3. Baseline-Naive - LangChain + FAISS with raw content

Usage:
    # Quick validation (20 needle questions, no LLM grading)
    python benchmark_runner.py --questions needle --no-llm-grade
    
    # Full benchmark (400 realistic questions, with LLM grading)
    python benchmark_runner.py --questions realistic --llm-grade
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add src to path for PLM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from baseline_langchain import BaselineLangChainRAG, load_chunks


# PLM database paths
PLM_DB_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data/index.db"
PLM_BM25_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data"

# Question file paths
NEEDLE_QUESTIONS_PATH = Path(__file__).parent / "corpus" / "needle_questions.json"
REALISTIC_QUESTIONS_PATH = Path(__file__).parent / "corpus" / "kubernetes" / "realistic_questions.json"
INFORMED_QUESTIONS_PATH = Path(__file__).parent / "corpus" / "kubernetes" / "informed_questions.json"

# Chunks file
CHUNKS_PATH = Path(__file__).parent / "chunks.json"


def load_questions(questions_type: str) -> tuple[list[dict], Optional[str]]:
    """Load questions based on type.
    
    Args:
        questions_type: 'needle', 'realistic', or 'informed'
        
    Returns:
        (questions, global_needle_doc_id) - global_needle_doc_id is None for realistic/informed
    """
    if questions_type == "needle":
        with open(NEEDLE_QUESTIONS_PATH) as f:
            data = json.load(f)
        return data["questions"], data.get("needle_doc_id")
    
    elif questions_type == "realistic":
        with open(REALISTIC_QUESTIONS_PATH) as f:
            data = json.load(f)
        # Expand q1/q2 variants
        expanded = []
        for i, q in enumerate(data.get("questions", [])):
            doc_id = q.get("doc_id", "")
            if q.get("realistic_q1"):
                expanded.append({
                    "id": f"q_{i:03d}_q1",
                    "question": q["realistic_q1"],
                    "doc_id": doc_id,
                    "expected_answer": q.get("original_instruction", ""),
                })
            if q.get("realistic_q2"):
                expanded.append({
                    "id": f"q_{i:03d}_q2",
                    "question": q["realistic_q2"],
                    "doc_id": doc_id,
                    "expected_answer": q.get("original_instruction", ""),
                })
        return expanded, None
    
    elif questions_type == "informed":
        with open(INFORMED_QUESTIONS_PATH) as f:
            data = json.load(f)
        questions = []
        for i, q in enumerate(data.get("questions", [])):
            questions.append({
                "id": f"informed_{i:03d}",
                "question": q["original_instruction"],
                "doc_id": q["doc_id"],
            })
        return questions, None
    
    else:
        raise ValueError(f"Unknown questions type: {questions_type}")


def calculate_mrr(ranks: list[Optional[int]]) -> float:
    """Calculate Mean Reciprocal Rank.
    
    Args:
        ranks: List of ranks (1-indexed) or None if not found
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    reciprocals = []
    for rank in ranks:
        if rank is not None:
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    return sum(reciprocals) / len(reciprocals) if reciprocals else 0.0


def calculate_hit_at_k(ranks: list[Optional[int]], k: int) -> float:
    """Calculate Hit@k (percentage of queries where target found in top k).
    
    Args:
        ranks: List of ranks (1-indexed) or None if not found
        k: Number of top results to check
        
    Returns:
        Hit@k as percentage (0-100)
    """
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return (hits / len(ranks) * 100) if ranks else 0.0


def bootstrap_ci(values: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> tuple[float, float]:
    """Calculate bootstrap confidence interval.
    
    Args:
        values: List of values to bootstrap
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        (lower_bound, upper_bound) of confidence interval
    """
    if not values:
        return (0.0, 0.0)
    
    values_arr = np.array(values)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(values_arr, size=len(values_arr), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrap_means, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 + ci) / 2 * 100)
    
    return (float(lower), float(upper))


class PLMRetriever:
    """Wrapper for PLM's HybridRetriever."""
    
    def __init__(self, db_path: str = PLM_DB_PATH, bm25_path: str = PLM_BM25_PATH,
                 use_rerank: bool = False, candidates_k: int = 50):
        from plm.search.retriever import HybridRetriever
        self.retriever = HybridRetriever(db_path, bm25_path)
        self.use_rerank = use_rerank
        self.candidates_k = candidates_k
        
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        return self.retriever.retrieve(
            query, k=k, use_rewrite=False,
            use_rerank=self.use_rerank, candidates_k=self.candidates_k,
        )
    
    def get_stats(self) -> dict:
        chunks = self.retriever.storage.get_all_chunks()
        return {
            "total_chunks": len(chunks) if chunks else 0,
            "type": "PLM HybridRetriever",
            "use_rerank": self.use_rerank,
            "candidates_k": self.candidates_k if self.use_rerank else None,
        }


def run_benchmark(
    questions: list[dict],
    plm: PLMRetriever,
    baseline_enriched: BaselineLangChainRAG,
    baseline_naive: BaselineLangChainRAG,
    global_needle_doc_id: Optional[str] = None,
    k: int = 5,
) -> dict:
    """Run benchmark on all 3 variants.
    
    Args:
        questions: List of question dicts with 'question', 'doc_id', etc.
        plm: PLM retriever instance
        baseline_enriched: Baseline with enriched content
        baseline_naive: Baseline with raw content
        global_needle_doc_id: Optional global target doc ID (for needle questions)
        k: Number of results to retrieve
        
    Returns:
        Results dictionary with metrics for all variants
    """
    results = {
        "plm": {"ranks": [], "latencies": []},
        "baseline_enriched": {"ranks": [], "latencies": []},
        "baseline_naive": {"ranks": [], "latencies": []},
    }
    
    per_question_results = []
    
    for q in tqdm(questions, desc="Running benchmark"):
        question_text = q["question"]
        target_doc_id = q.get("doc_id") or global_needle_doc_id
        
        q_result = {
            "id": q.get("id", ""),
            "question": question_text,
            "target_doc_id": target_doc_id,
        }
        
        # Run PLM
        start = time.perf_counter()
        plm_results = plm.retrieve(question_text, k=k)
        plm_latency = (time.perf_counter() - start) * 1000
        plm_rank = None
        for i, r in enumerate(plm_results, 1):
            if target_doc_id and target_doc_id in r.get("doc_id", ""):
                plm_rank = i
                break
        results["plm"]["ranks"].append(plm_rank)
        results["plm"]["latencies"].append(plm_latency)
        q_result["plm_rank"] = plm_rank
        q_result["plm_latency_ms"] = plm_latency
        
        # Run Baseline-Enriched
        start = time.perf_counter()
        enriched_results = baseline_enriched.retrieve(question_text, k=k)
        enriched_latency = (time.perf_counter() - start) * 1000
        enriched_rank = None
        for i, r in enumerate(enriched_results, 1):
            if target_doc_id and target_doc_id in r.get("doc_id", ""):
                enriched_rank = i
                break
        results["baseline_enriched"]["ranks"].append(enriched_rank)
        results["baseline_enriched"]["latencies"].append(enriched_latency)
        q_result["baseline_enriched_rank"] = enriched_rank
        q_result["baseline_enriched_latency_ms"] = enriched_latency
        
        # Run Baseline-Naive
        start = time.perf_counter()
        naive_results = baseline_naive.retrieve(question_text, k=k)
        naive_latency = (time.perf_counter() - start) * 1000
        naive_rank = None
        for i, r in enumerate(naive_results, 1):
            if target_doc_id and target_doc_id in r.get("doc_id", ""):
                naive_rank = i
                break
        results["baseline_naive"]["ranks"].append(naive_rank)
        results["baseline_naive"]["latencies"].append(naive_latency)
        q_result["baseline_naive_rank"] = naive_rank
        q_result["baseline_naive_latency_ms"] = naive_latency
        
        per_question_results.append(q_result)
    
    # Calculate aggregate metrics for each variant
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(questions),
            "k": k,
        },
        "per_question": per_question_results,
    }
    
    for variant in ["plm", "baseline_enriched", "baseline_naive"]:
        ranks = results[variant]["ranks"]
        latencies = results[variant]["latencies"]
        
        # Calculate reciprocal ranks for bootstrap CI
        reciprocal_ranks = [1.0/r if r else 0.0 for r in ranks]
        
        output[variant] = {
            "total_queries": len(ranks),
            "mrr": calculate_mrr(ranks),
            "hit_at_1": calculate_hit_at_k(ranks, 1),
            "hit_at_3": calculate_hit_at_k(ranks, 3),
            "hit_at_5": calculate_hit_at_k(ranks, 5),
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "found_count": sum(1 for r in ranks if r is not None),
            "not_found_count": sum(1 for r in ranks if r is None),
        }
    
    # Calculate bootstrap confidence intervals for MRR differences
    plm_rr = [1.0/r if r else 0.0 for r in results["plm"]["ranks"]]
    enriched_rr = [1.0/r if r else 0.0 for r in results["baseline_enriched"]["ranks"]]
    naive_rr = [1.0/r if r else 0.0 for r in results["baseline_naive"]["ranks"]]
    
    # Pairwise differences
    plm_vs_naive_diff = [p - n for p, n in zip(plm_rr, naive_rr)]
    plm_vs_enriched_diff = [p - e for p, e in zip(plm_rr, enriched_rr)]
    enriched_vs_naive_diff = [e - n for e, n in zip(enriched_rr, naive_rr)]
    
    output["statistics"] = {
        "mrr_ci_95": {
            "plm_vs_naive": bootstrap_ci(plm_vs_naive_diff),
            "plm_vs_enriched": bootstrap_ci(plm_vs_enriched_diff),
            "enriched_vs_naive": bootstrap_ci(enriched_vs_naive_diff),
        },
        "attribution": {
            "total_improvement": output["plm"]["mrr"] - output["baseline_naive"]["mrr"],
            "enrichment_contribution": output["baseline_enriched"]["mrr"] - output["baseline_naive"]["mrr"],
            "rrf_bm25_expansion_contribution": output["plm"]["mrr"] - output["baseline_enriched"]["mrr"],
        }
    }
    
    return output


def print_results(results: dict) -> None:
    """Print results summary to console."""
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total queries: {results['metadata']['total_queries']}")
    print(f"k: {results['metadata']['k']}")
    print()
    
    for variant in ["plm", "baseline_enriched", "baseline_naive"]:
        v = results[variant]
        print(f"{variant.upper()}:")
        print(f"  MRR:     {v['mrr']:.4f}")
        print(f"  Hit@1:   {v['hit_at_1']:.1f}%")
        print(f"  Hit@5:   {v['hit_at_5']:.1f}%")
        print(f"  Found:   {v['found_count']}/{v['total_queries']}")
        print(f"  Avg lat: {v['avg_latency_ms']:.1f}ms")
        print()
    
    print("ATTRIBUTION ANALYSIS:")
    attr = results["statistics"]["attribution"]
    print(f"  Total PLM improvement:      {attr['total_improvement']:+.4f} MRR")
    print(f"  Enrichment contribution:    {attr['enrichment_contribution']:+.4f} MRR")
    print(f"  RRF/BM25/expansion:         {attr['rrf_bm25_expansion_contribution']:+.4f} MRR")
    print()
    
    print("95% CONFIDENCE INTERVALS (MRR difference):")
    ci = results["statistics"]["mrr_ci_95"]
    print(f"  PLM vs Naive:     [{ci['plm_vs_naive'][0]:+.4f}, {ci['plm_vs_naive'][1]:+.4f}]")
    print(f"  PLM vs Enriched:  [{ci['plm_vs_enriched'][0]:+.4f}, {ci['plm_vs_enriched'][1]:+.4f}]")
    print(f"  Enriched vs Naive:[{ci['enriched_vs_naive'][0]:+.4f}, {ci['enriched_vs_naive'][1]:+.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Run PLM vs baseline RAG benchmark"
    )
    parser.add_argument(
        "--questions",
        type=str,
        choices=["needle", "realistic", "informed"],
        default="needle",
        help="Question set to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: results/<questions>_benchmark.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for quick testing)",
    )
    parser.add_argument(
        "--llm-grade",
        action="store_true",
        help="Run LLM grading (not implemented yet)",
    )
    parser.add_argument(
        "--no-llm-grade",
        action="store_true",
        help="Skip LLM grading",
    )
    parser.add_argument(
        "--plm-db",
        type=str,
        default=None,
        help="Custom PLM database path (default: production path)",
    )
    parser.add_argument(
        "--plm-bm25",
        type=str,
        default=None,
        help="Custom PLM BM25 index path (default: production path)",
    )
    parser.add_argument(
        "--use-rerank",
        action="store_true",
        help="Enable cross-encoder reranking in PLM retriever",
    )
    parser.add_argument(
        "--candidates-k",
        type=int,
        default=50,
        help="Number of candidates for reranking (default: 50)",
    )
    
    args = parser.parse_args()
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        suffix = "_rerank" if args.use_rerank else ""
        output_path = Path(__file__).parent / "results" / f"{args.questions}{suffix}_benchmark.json"
    
    # Load questions
    print(f"Loading {args.questions} questions...")
    questions, global_needle_doc_id = load_questions(args.questions)
    
    if args.limit:
        questions = questions[:args.limit]
    
    print(f"Loaded {len(questions)} questions")
    if global_needle_doc_id:
        print(f"Global needle doc ID: {global_needle_doc_id}")
    
    # Load chunks for baseline
    print(f"\nLoading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(str(CHUNKS_PATH))
    print(f"Loaded {len(chunks)} chunks")
    
    # Initialize PLM
    plm_db = args.plm_db or PLM_DB_PATH
    plm_bm25 = args.plm_bm25 or PLM_BM25_PATH
    print(f"\nInitializing PLM from {plm_db}...")
    plm = PLMRetriever(
        db_path=plm_db, bm25_path=plm_bm25,
        use_rerank=args.use_rerank, candidates_k=args.candidates_k,
    )
    print(f"PLM stats: {plm.get_stats()}")
    
    # Initialize baselines
    print("\nInitializing baseline variants...")
    
    print("  Baseline-Enriched (embedding enriched content)...")
    baseline_enriched = BaselineLangChainRAG()
    baseline_enriched.ingest_chunks(chunks, use_enriched=True)
    
    print("  Baseline-Naive (embedding raw content)...")
    baseline_naive = BaselineLangChainRAG()
    baseline_naive.ingest_chunks(chunks, use_enriched=False)
    
    # Run benchmark
    print(f"\nRunning benchmark with k={args.k}...")
    results = run_benchmark(
        questions=questions,
        plm=plm,
        baseline_enriched=baseline_enriched,
        baseline_naive=baseline_naive,
        global_needle_doc_id=global_needle_doc_id,
        k=args.k,
    )
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print_results(results)
    
    return 0


if __name__ == "__main__":
    exit(main())
