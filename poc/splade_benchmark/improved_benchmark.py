#!/usr/bin/env python3
"""
Improved SPLADE Benchmark with Corrected Methodology

Addresses Oracle's concerns:
1. Statistical Validity: Uses all 200 informed queries (was 25, TREC minimum is 50)
2. Needle Benchmark: Groups by target document for multi-document analysis
3. Realistic Benchmark: Uses all 400 queries (200 questions × 2 variants)
4. Confidence Intervals: Bootstrap sampling for statistical significance
5. Per-Document Analysis: Breaks down performance by document to catch bias
"""

import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plm.search.components.sparse import SPLADERetriever
from plm.search.components.sparse.bm25_retriever import BM25Retriever

POC_DIR = Path(__file__).parent
TEST_DB_PATH = POC_DIR / "test_db"
CORPUS_DIR = POC_DIR / "corpus"
ARTIFACTS_DIR = POC_DIR / "artifacts"
SPLADE_INDEX_PATH = ARTIFACTS_DIR / "splade_index"


def load_doc_mapping():
    """Load chunk index to doc_id mapping."""
    conn = sqlite3.connect(str(TEST_DB_PATH / "index.db"))
    rows = conn.execute("SELECT doc_id FROM chunks ORDER BY rowid").fetchall()
    conn.close()
    return {i: row[0] for i, row in enumerate(rows)}


def load_all_informed_queries():
    """Load ALL 200 informed queries (original_instruction from realistic_questions.json)."""
    path = CORPUS_DIR / "kubernetes" / "realistic_questions.json"
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", [])
    return [
        {
            "id": f"informed_{i:03d}",
            "query": q["original_instruction"],
            "expected_doc_id": q["doc_id"],
        }
        for i, q in enumerate(questions)
    ]


def load_all_realistic_queries():
    """Load ALL 400 realistic queries (200 questions × 2 variants)."""
    path = CORPUS_DIR / "kubernetes" / "realistic_questions.json"
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", [])
    results = []
    for i, q in enumerate(questions):
        results.append(
            {
                "id": f"realistic_{i:03d}_q1",
                "query": q["realistic_q1"],
                "expected_doc_id": q["doc_id"],
            }
        )
        results.append(
            {
                "id": f"realistic_{i:03d}_q2",
                "query": q["realistic_q2"],
                "expected_doc_id": q["doc_id"],
            }
        )
    return results


def evaluate_queries(retriever, queries, idx_to_doc, k=10, desc="Evaluating"):
    """Run evaluation and return detailed results."""
    results = []

    for q in tqdm(queries, desc=desc, leave=False):
        start = time.time()
        search_results = retriever.search(q["query"], k=k)
        latency = (time.time() - start) * 1000

        rank = None
        matched_doc_id = None
        for j, r in enumerate(search_results):
            doc_id = idx_to_doc.get(r["index"], "")
            if q["expected_doc_id"] in doc_id:
                rank = j + 1
                matched_doc_id = doc_id
                break

        results.append(
            {
                "query_id": q["id"],
                "query": q["query"],
                "expected_doc_id": q["expected_doc_id"],
                "rank": rank,
                "latency_ms": latency,
                "matched_doc_id": matched_doc_id,
            }
        )

    return results


def compute_metrics(results):
    """Compute MRR and other metrics from results."""
    ranks = [r["rank"] for r in results]
    found = [r for r in ranks if r is not None]

    mrr = sum(1.0 / r for r in found) / len(ranks) if ranks else 0
    hit_at_1 = sum(1 for r in found if r <= 1) / len(ranks) * 100 if ranks else 0
    hit_at_5 = sum(1 for r in found if r <= 5) / len(ranks) * 100 if ranks else 0
    hit_at_10 = sum(1 for r in found if r <= 10) / len(ranks) * 100 if ranks else 0
    recall = len(found) / len(ranks) * 100 if ranks else 0

    latencies = [r["latency_ms"] for r in results]

    return {
        "mrr": mrr,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "recall_at_k": recall,
        "found": len(found),
        "total": len(ranks),
        "avg_latency_ms": np.mean(latencies),
        "median_latency_ms": np.median(latencies),
    }


def bootstrap_ci(results, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for MRR."""
    ranks = [r["rank"] for r in results]
    n = len(ranks)

    mrr_samples = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        sample_indices = np.random.choice(n, size=n, replace=True)
        sample_ranks = [ranks[i] for i in sample_indices]
        found = [r for r in sample_ranks if r is not None]
        mrr = sum(1.0 / r for r in found) / len(sample_ranks) if sample_ranks else 0
        mrr_samples.append(mrr)

    lower = np.percentile(mrr_samples, (1 - ci) / 2 * 100)
    upper = np.percentile(mrr_samples, (1 + ci) / 2 * 100)
    return lower, upper


def analyze_by_document(results):
    """Analyze results grouped by target document."""
    by_doc = defaultdict(list)
    for r in results:
        by_doc[r["expected_doc_id"]].append(r)

    doc_metrics = []
    for doc_id, doc_results in by_doc.items():
        ranks = [r["rank"] for r in doc_results]
        found = [r for r in ranks if r is not None]
        mrr = sum(1.0 / r for r in found) / len(ranks) if ranks else 0
        doc_metrics.append(
            {
                "doc_id": doc_id,
                "query_count": len(doc_results),
                "found": len(found),
                "mrr": mrr,
            }
        )

    # Sort by query count descending
    doc_metrics.sort(key=lambda x: x["query_count"], reverse=True)
    return doc_metrics


def main():
    print("=" * 80)
    print("IMPROVED SPLADE BENCHMARK - Corrected Methodology")
    print("=" * 80)

    # Load data
    idx_to_doc = load_doc_mapping()
    print(f"\nLoaded doc mapping: {len(idx_to_doc)} chunks")

    informed = load_all_informed_queries()
    realistic = load_all_realistic_queries()

    print(f"Loaded queries:")
    print(f"  - Informed: {len(informed)} (was 25, now all 200)")
    print(f"  - Realistic: {len(realistic)} (was 100, now all 400)")

    # Count unique documents in informed queries
    unique_docs = len(set(q["expected_doc_id"] for q in informed))
    print(f"  - Unique target documents: {unique_docs}")

    # Load retrievers
    print("\nLoading retrievers...")
    splade = SPLADERetriever.load(str(SPLADE_INDEX_PATH))
    print(f"  SPLADE: {splade.document_count} documents")

    bm25 = BM25Retriever.load(str(TEST_DB_PATH))
    print(f"  BM25: {bm25.document_count} documents")

    # Run benchmarks
    print("\n" + "-" * 80)
    print("Running benchmarks...")
    print("-" * 80)

    results = {}

    # Informed benchmark
    print("\n[1/4] Informed queries (BM25)...")
    bm25_informed = evaluate_queries(bm25, informed, idx_to_doc, desc="BM25 Informed")
    results["bm25_informed"] = compute_metrics(bm25_informed)
    results["bm25_informed"]["ci_95"] = bootstrap_ci(bm25_informed)

    print("[2/4] Informed queries (SPLADE)...")
    splade_informed = evaluate_queries(splade, informed, idx_to_doc, desc="SPLADE Informed")
    results["splade_informed"] = compute_metrics(splade_informed)
    results["splade_informed"]["ci_95"] = bootstrap_ci(splade_informed)

    # Realistic benchmark
    print("[3/4] Realistic queries (BM25)...")
    bm25_realistic = evaluate_queries(bm25, realistic, idx_to_doc, desc="BM25 Realistic")
    results["bm25_realistic"] = compute_metrics(bm25_realistic)
    results["bm25_realistic"]["ci_95"] = bootstrap_ci(bm25_realistic)

    print("[4/4] Realistic queries (SPLADE)...")
    splade_realistic = evaluate_queries(splade, realistic, idx_to_doc, desc="SPLADE Realistic")
    results["splade_realistic"] = compute_metrics(splade_realistic)
    results["splade_realistic"]["ci_95"] = bootstrap_ci(splade_realistic)

    # Per-document analysis (multi-document needle test)
    print("\nAnalyzing per-document performance...")
    bm25_by_doc = analyze_by_document(bm25_informed)
    splade_by_doc = analyze_by_document(splade_informed)

    # Results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    print("\n{:<20} {:>10} {:>10} {:>12} {:>18}".format(
        "Benchmark", "BM25 MRR", "SPLADE", "Improvement", "95% CI (SPLADE)"
    ))
    print("-" * 80)

    for name in ["informed", "realistic"]:
        bm25_mrr = results[f"bm25_{name}"]["mrr"]
        splade_mrr = results[f"splade_{name}"]["mrr"]
        improvement = (splade_mrr - bm25_mrr) / bm25_mrr * 100 if bm25_mrr > 0 else 0
        ci = results[f"splade_{name}"]["ci_95"]

        print("{:<20} {:>10.4f} {:>10.4f} {:>+11.1f}% {:>8.4f} - {:.4f}".format(
            name.capitalize(), bm25_mrr, splade_mrr, improvement, ci[0], ci[1]
        ))

    # Query count summary
    print("\n{:<20} {:>10} {:>10}".format("", "Total", "Found/BM25→SPLADE"))
    print("-" * 80)
    for name in ["informed", "realistic"]:
        total = results[f"bm25_{name}"]["total"]
        bm25_found = results[f"bm25_{name}"]["found"]
        splade_found = results[f"splade_{name}"]["found"]
        print("{:<20} {:>10} {:>10}".format(
            name.capitalize(), total, f"{bm25_found}→{splade_found}"
        ))

    # Per-document analysis (top 10)
    print("\n" + "-" * 80)
    print("PER-DOCUMENT ANALYSIS (Top 10 by query count)")
    print("-" * 80)
    print("{:<50} {:>8} {:>10} {:>10}".format("Document", "Queries", "BM25 MRR", "SPLADE"))
    print("-" * 80)

    # Create lookup for SPLADE results
    splade_doc_lookup = {d["doc_id"]: d for d in splade_by_doc}

    for doc in bm25_by_doc[:10]:
        doc_id = doc["doc_id"]
        splade_doc = splade_doc_lookup.get(doc_id, {"mrr": 0})
        print("{:<50} {:>8} {:>10.3f} {:>10.3f}".format(
            doc_id[:50], doc["query_count"], doc["mrr"], splade_doc["mrr"]
        ))

    # Statistical significance check
    print("\n" + "-" * 80)
    print("STATISTICAL SIGNIFICANCE")
    print("-" * 80)

    for name in ["informed", "realistic"]:
        bm25_mrr = results[f"bm25_{name}"]["mrr"]
        splade_ci = results[f"splade_{name}"]["ci_95"]

        if splade_ci[0] > bm25_mrr:
            sig = "YES - SPLADE CI lower bound > BM25 mean"
        elif splade_ci[1] < bm25_mrr:
            sig = "YES - BM25 significantly better (unexpected)"
        else:
            sig = "NO - Confidence intervals overlap"

        print(f"  {name.capitalize()}: {sig}")

    # Save detailed results
    output = {
        "metadata": {
            "methodology_version": "v2_improved",
            "informed_queries": len(informed),
            "realistic_queries": len(realistic),
            "unique_target_documents": unique_docs,
            "bootstrap_samples": 1000,
            "confidence_level": 0.95,
        },
        "results": results,
        "per_document_bm25": bm25_by_doc,
        "per_document_splade": splade_by_doc,
        "detailed_results": {
            "bm25_informed": bm25_informed,
            "splade_informed": splade_informed,
            "bm25_realistic": bm25_realistic,
            "splade_realistic": splade_realistic,
        },
    }

    output_path = ARTIFACTS_DIR / "improved_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nDetailed results saved to {output_path}")

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    informed_improvement = (
        (results["splade_informed"]["mrr"] - results["bm25_informed"]["mrr"])
        / results["bm25_informed"]["mrr"]
        * 100
    )
    realistic_improvement = (
        (results["splade_realistic"]["mrr"] - results["bm25_realistic"]["mrr"])
        / results["bm25_realistic"]["mrr"]
        * 100
    )

    print(f"\n  Informed Benchmark ({len(informed)} queries):")
    print(f"    BM25 MRR:     {results['bm25_informed']['mrr']:.4f}")
    print(f"    SPLADE MRR:   {results['splade_informed']['mrr']:.4f}")
    print(f"    Improvement:  {informed_improvement:+.1f}%")
    print(f"    95% CI:       [{results['splade_informed']['ci_95'][0]:.4f}, {results['splade_informed']['ci_95'][1]:.4f}]")

    print(f"\n  Realistic Benchmark ({len(realistic)} queries):")
    print(f"    BM25 MRR:     {results['bm25_realistic']['mrr']:.4f}")
    print(f"    SPLADE MRR:   {results['splade_realistic']['mrr']:.4f}")
    print(f"    Improvement:  {realistic_improvement:+.1f}%")
    print(f"    95% CI:       [{results['splade_realistic']['ci_95'][0]:.4f}, {results['splade_realistic']['ci_95'][1]:.4f}]")

    # Overall assessment
    informed_sig = results["splade_informed"]["ci_95"][0] > results["bm25_informed"]["mrr"]
    realistic_sig = results["splade_realistic"]["ci_95"][0] > results["bm25_realistic"]["mrr"]

    print("\n  Assessment:")
    if informed_sig and realistic_sig:
        print("    ✓ SPLADE significantly outperforms BM25 on BOTH benchmarks")
    elif informed_sig:
        print("    ⚠ SPLADE significantly better on Informed only")
    elif realistic_sig:
        print("    ⚠ SPLADE significantly better on Realistic only")
    else:
        print("    ✗ No statistically significant improvement detected")


if __name__ == "__main__":
    main()
