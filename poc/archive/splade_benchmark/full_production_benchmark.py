#!/usr/bin/env python3
"""Full benchmark of production SPLADE integration against POC baselines."""

import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plm.search.components.sparse import SPLADERetriever

POC_DIR = Path(__file__).parent
TEST_DB_PATH = POC_DIR / "test_db"
CORPUS_DIR = POC_DIR / "corpus"
ARTIFACTS_DIR = POC_DIR / "artifacts"
SPLADE_INDEX_PATH = ARTIFACTS_DIR / "splade_index"


def load_doc_mapping():
    conn = sqlite3.connect(str(TEST_DB_PATH / "index.db"))
    rows = conn.execute("SELECT doc_id FROM chunks ORDER BY rowid").fetchall()
    conn.close()
    return {i: row[0] for i, row in enumerate(rows)}


def load_informed_queries():
    path = CORPUS_DIR / "kubernetes" / "informed_questions.json"
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", [])
    return [
        {"id": f"informed_{i}", "query": q["original_instruction"], "expected_doc_id": q["doc_id"]}
        for i, q in enumerate(questions)
    ]


def load_needle_queries():
    path = CORPUS_DIR / "needle_questions.json"
    with open(path) as f:
        data = json.load(f)
    needle_doc_id = data.get("needle_doc_id", "")
    questions = data.get("questions", [])
    return [
        {"id": q["id"], "query": q["question"], "expected_doc_id": needle_doc_id}
        for q in questions
    ]


def load_realistic_queries(limit=50):
    path = CORPUS_DIR / "kubernetes" / "realistic_questions.json"
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", [])[:limit]
    results = []
    for i, q in enumerate(questions):
        results.append({"id": f"realistic_{i}_q1", "query": q["realistic_q1"], "expected_doc_id": q["doc_id"]})
        results.append({"id": f"realistic_{i}_q2", "query": q["realistic_q2"], "expected_doc_id": q["doc_id"]})
    return results


def evaluate_queries(retriever, queries, idx_to_doc, k=10, desc="Evaluating"):
    ranks = []
    latencies = []
    
    for q in tqdm(queries, desc=desc):
        start = time.time()
        results = retriever.search(q["query"], k=k)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        
        rank = None
        for j, r in enumerate(results):
            doc_id = idx_to_doc.get(r["index"], "")
            if q["expected_doc_id"] in doc_id:
                rank = j + 1
                break
        ranks.append(rank)
    
    found = [r for r in ranks if r is not None]
    mrr = sum(1.0 / r for r in found) / len(ranks) if ranks else 0
    hit_at_1 = sum(1 for r in found if r <= 1) / len(ranks) * 100 if ranks else 0
    hit_at_5 = sum(1 for r in found if r <= 5) / len(ranks) * 100 if ranks else 0
    hit_at_10 = sum(1 for r in found if r <= 10) / len(ranks) * 100 if ranks else 0
    
    return {
        "mrr": mrr,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "hit_at_10": hit_at_10,
        "found": len(found),
        "total": len(ranks),
        "avg_latency_ms": np.mean(latencies),
    }


def main():
    print("=" * 70)
    print("PRODUCTION SPLADE INTEGRATION - FULL BENCHMARK")
    print("=" * 70)

    idx_to_doc = load_doc_mapping()
    print(f"\nLoaded doc mapping: {len(idx_to_doc)} chunks")

    print(f"\nLoading production SPLADERetriever...")
    retriever = SPLADERetriever.load(str(SPLADE_INDEX_PATH))
    print(f"Loaded: {retriever.document_count} documents")

    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)
    
    print(f"\nLoaded queries: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")

    print("\n" + "-" * 70)
    print("Running benchmarks with production SPLADERetriever...")
    print("-" * 70)

    informed_results = evaluate_queries(retriever, informed, idx_to_doc, desc="Informed")
    needle_results = evaluate_queries(retriever, needle, idx_to_doc, desc="Needle")
    realistic_results = evaluate_queries(retriever, realistic, idx_to_doc, desc="Realistic")

    with open(ARTIFACTS_DIR / "baseline_bm25.json") as f:
        baseline = json.load(f)

    with open(ARTIFACTS_DIR / "splade_only_results.json") as f:
        poc_splade = json.load(f)

    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print("\n{:<12} {:>12} {:>12} {:>12} {:>12}".format(
        "Benchmark", "BM25 MRR", "POC SPLADE", "Prod SPLADE", "Diff"
    ))
    print("-" * 70)

    for name, results in [("Informed", informed_results), ("Needle", needle_results), ("Realistic", realistic_results)]:
        bm25_mrr = baseline[name.lower()]["mrr"]
        poc_mrr = poc_splade[name.lower()]["mrr"]
        prod_mrr = results["mrr"]
        diff = (prod_mrr - poc_mrr) / poc_mrr * 100 if poc_mrr > 0 else 0
        
        print("{:<12} {:>12.4f} {:>12.4f} {:>12.4f} {:>11.1f}%".format(
            name, bm25_mrr, poc_mrr, prod_mrr, diff
        ))

    print("\n" + "-" * 70)
    print("LATENCY")
    print("-" * 70)
    print(f"  Informed:  {informed_results['avg_latency_ms']:.0f}ms")
    print(f"  Needle:    {needle_results['avg_latency_ms']:.0f}ms")
    print(f"  Realistic: {realistic_results['avg_latency_ms']:.0f}ms")

    print("\n" + "-" * 70)
    print("IMPROVEMENT vs BM25 BASELINE")
    print("-" * 70)

    for name, results in [("Informed", informed_results), ("Needle", needle_results), ("Realistic", realistic_results)]:
        bm25_mrr = baseline[name.lower()]["mrr"]
        improvement = (results["mrr"] - bm25_mrr) / bm25_mrr * 100
        print(f"  {name}: {improvement:+.1f}%")

    output = {
        "production_splade": {
            "informed": informed_results,
            "needle": needle_results,
            "realistic": realistic_results,
        },
        "comparison": {
            "informed_vs_bm25": (informed_results["mrr"] - baseline["informed"]["mrr"]) / baseline["informed"]["mrr"] * 100,
            "needle_vs_bm25": (needle_results["mrr"] - baseline["needle"]["mrr"]) / baseline["needle"]["mrr"] * 100,
            "realistic_vs_bm25": (realistic_results["mrr"] - baseline["realistic"]["mrr"]) / baseline["realistic"]["mrr"] * 100,
            "informed_vs_poc": (informed_results["mrr"] - poc_splade["informed"]["mrr"]) / poc_splade["informed"]["mrr"] * 100,
        }
    }

    output_path = ARTIFACTS_DIR / "production_splade_full_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    informed_pass = output["comparison"]["informed_vs_bm25"] > 20
    parity_ok = abs(output["comparison"]["informed_vs_poc"]) < 5

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Target: +20% MRR on Informed (vs BM25): {'PASS' if informed_pass else 'FAIL'} ({output['comparison']['informed_vs_bm25']:+.1f}%)")
    print(f"  Parity with POC (<5% diff): {'PASS' if parity_ok else 'FAIL'} ({output['comparison']['informed_vs_poc']:+.1f}%)")

    if informed_pass and parity_ok:
        print("\n  OVERALL: PASS - Production SPLADE integration verified")
    else:
        print("\n  OVERALL: NEEDS REVIEW")


if __name__ == "__main__":
    main()
