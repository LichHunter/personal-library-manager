#!/usr/bin/env python3
"""Benchmark production HybridRetriever with SPLADE integration.

Compares:
1. BM25-only (existing baseline)
2. Production HybridRetriever with BM25 + Semantic (current)
3. Production HybridRetriever with SPLADE-only (new)

Usage:
    cd poc/splade_benchmark
    .venv/bin/python production_benchmark.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plm.search.retriever import HybridRetriever
from plm.search.config import RetrievalConfig, SparseRetrieverType, SemanticSettings, SPLADESettings

POC_DIR = Path(__file__).parent
TEST_DB_PATH = POC_DIR / "test_db"
CORPUS_DIR = POC_DIR / "corpus"
ARTIFACTS_DIR = POC_DIR / "artifacts"


@dataclass
class QueryResult:
    query_id: str
    query: str
    ground_truth_doc_id: str
    rank: Optional[int]
    score: Optional[float]
    top_k_doc_ids: list[str]
    latency_ms: float


@dataclass
class BenchmarkResults:
    name: str
    total_queries: int
    k: int
    mrr: float
    recall_at_k: float
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    avg_latency_ms: float
    found_count: int
    not_found_count: int
    per_query: list[QueryResult]


def load_informed_queries() -> list[dict]:
    path = CORPUS_DIR / "kubernetes" / "informed_questions.json"
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    return [
        {"id": f"informed_{i}", "query": q["original_instruction"], "ground_truth_doc_id": q["doc_id"]}
        for i, q in enumerate(questions)
    ]


def load_needle_queries() -> list[dict]:
    path = CORPUS_DIR / "needle_questions.json"
    with open(path) as f:
        data = json.load(f)
    return [
        {"id": q["question_id"], "query": q["question"], "ground_truth_doc_id": q["source_doc"]}
        for q in data
    ]


def load_realistic_queries(limit: int = 50) -> list[dict]:
    path = CORPUS_DIR / "kubernetes" / "realistic_questions.json"
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    results = []
    for i, q in enumerate(questions[:limit]):
        results.append({"id": f"realistic_{i}_q1", "query": q["realistic_q1"], "ground_truth_doc_id": q["doc_id"]})
        results.append({"id": f"realistic_{i}_q2", "query": q["realistic_q2"], "ground_truth_doc_id": q["doc_id"]})
    return results


def evaluate_query(
    retriever: HybridRetriever,
    query_id: str,
    query: str,
    ground_truth_doc_id: str,
    k: int = 10,
) -> QueryResult:
    start = time.time()
    results = retriever.retrieve(query, k=k)
    latency_ms = (time.time() - start) * 1000

    rank = None
    score = None
    top_k_doc_ids = []

    for i, r in enumerate(results):
        doc_id = r["doc_id"]
        top_k_doc_ids.append(doc_id)
        if ground_truth_doc_id in doc_id:
            rank = i + 1
            score = r["score"]

    return QueryResult(
        query_id=query_id,
        query=query,
        ground_truth_doc_id=ground_truth_doc_id,
        rank=rank,
        score=score,
        top_k_doc_ids=top_k_doc_ids,
        latency_ms=latency_ms,
    )


def run_benchmark(
    retriever: HybridRetriever,
    name: str,
    queries: list[dict],
    k: int = 10,
) -> BenchmarkResults:
    results = []
    for q in tqdm(queries, desc=f"[{name}]"):
        result = evaluate_query(
            retriever,
            q["id"],
            q["query"],
            q["ground_truth_doc_id"],
            k=k,
        )
        results.append(result)

    ranks = [r.rank for r in results]
    found_ranks = [r for r in ranks if r is not None]

    mrr = sum(1.0 / r for r in found_ranks) / len(ranks) if ranks else 0
    recall_at_k = len(found_ranks) / len(ranks) if ranks else 0
    hit_at_1 = sum(1 for r in found_ranks if r <= 1) / len(ranks) if ranks else 0
    hit_at_5 = sum(1 for r in found_ranks if r <= 5) / len(ranks) if ranks else 0
    hit_at_10 = sum(1 for r in found_ranks if r <= 10) / len(ranks) if ranks else 0
    avg_latency = float(np.mean([r.latency_ms for r in results]))

    return BenchmarkResults(
        name=name,
        total_queries=len(queries),
        k=k,
        mrr=mrr,
        recall_at_k=recall_at_k,
        hit_at_1=hit_at_1 * 100,
        hit_at_5=hit_at_5 * 100,
        hit_at_10=hit_at_10 * 100,
        avg_latency_ms=avg_latency,
        found_count=len(found_ranks),
        not_found_count=len(ranks) - len(found_ranks),
        per_query=results,
    )


def load_baseline() -> dict:
    path = ARTIFACTS_DIR / "baseline_bm25.json"
    with open(path) as f:
        return json.load(f)


def print_comparison(name: str, baseline_mrr: float, new_mrr: float):
    improvement = new_mrr - baseline_mrr
    improvement_pct = (improvement / baseline_mrr * 100) if baseline_mrr > 0 else 0
    status = "REGRESSION" if new_mrr < baseline_mrr * 0.95 else "OK"
    print(f"\n{name.upper()}:")
    print(f"  Baseline MRR: {baseline_mrr:.4f}")
    print(f"  New MRR:      {new_mrr:.4f}")
    print(f"  Change:       {improvement:+.4f} ({improvement_pct:+.1f}%) [{status}]")


def main():
    print("=" * 60)
    print("Production HybridRetriever Benchmark with SPLADE")
    print("=" * 60)

    db_path = str(TEST_DB_PATH / "index.db")

    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)

    print(f"\nLoaded: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")

    baseline = load_baseline()

    print("\n" + "=" * 60)
    print("Mode 1: SPLADE-only (PLM_SPARSE_RETRIEVER=splade, PLM_SEMANTIC_ENABLED=false)")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        splade_index_path = os.path.join(tmpdir, "splade_index")

        splade_config = RetrievalConfig(
            sparse_retriever=SparseRetrieverType.SPLADE,
            splade=SPLADESettings(batch_size=16),
            semantic=SemanticSettings(enabled=False),
        )

        print("\n[1/2] Building SPLADE index from test_db chunks...")
        retriever = HybridRetriever(db_path, splade_index_path, config=splade_config)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        chunks_data = conn.execute(
            "SELECT doc_id, content, enriched_content FROM chunks ORDER BY rowid"
        ).fetchall()
        conn.close()

        print(f"[1/2] Loaded {len(chunks_data)} chunks from test_db")

        documents = [
            {
                "doc_id": row["doc_id"],
                "source_file": row["doc_id"],
                "chunks": [
                    {
                        "content": row["content"],
                        "keywords": [],
                        "entities": {},
                    }
                ],
            }
            for row in chunks_data
        ]

        print("[1/2] Batch ingesting into SPLADE retriever...")
        retriever.batch_ingest(documents, batch_size=16, show_progress=True)

        print("\n[2/2] Running benchmarks...")
        splade_informed = run_benchmark(retriever, "splade_informed", informed)
        splade_needle = run_benchmark(retriever, "splade_needle", needle)
        splade_realistic = run_benchmark(retriever, "splade_realistic", realistic)

    print("\n" + "=" * 60)
    print("Results: SPLADE-only vs BM25-only Baseline")
    print("=" * 60)

    print_comparison("Informed", baseline["informed"]["mrr"], splade_informed.mrr)
    print_comparison("Needle", baseline["needle"]["mrr"], splade_needle.mrr)
    print_comparison("Realistic", baseline["realistic"]["mrr"], splade_realistic.mrr)

    print("\n" + "=" * 60)
    print("Latency")
    print("=" * 60)
    print(f"  Informed avg: {splade_informed.avg_latency_ms:.1f}ms")
    print(f"  Needle avg:   {splade_needle.avg_latency_ms:.1f}ms")
    print(f"  Realistic avg: {splade_realistic.avg_latency_ms:.1f}ms")

    output = {
        "mode": "splade_only_production",
        "baseline": {
            "informed_mrr": baseline["informed"]["mrr"],
            "needle_mrr": baseline["needle"]["mrr"],
            "realistic_mrr": baseline["realistic"]["mrr"],
        },
        "splade_only": {
            "informed": {
                "mrr": splade_informed.mrr,
                "hit_at_1": splade_informed.hit_at_1,
                "hit_at_5": splade_informed.hit_at_5,
                "hit_at_10": splade_informed.hit_at_10,
                "avg_latency_ms": splade_informed.avg_latency_ms,
            },
            "needle": {
                "mrr": splade_needle.mrr,
                "hit_at_1": splade_needle.hit_at_1,
                "hit_at_5": splade_needle.hit_at_5,
                "hit_at_10": splade_needle.hit_at_10,
                "avg_latency_ms": splade_needle.avg_latency_ms,
            },
            "realistic": {
                "mrr": splade_realistic.mrr,
                "hit_at_1": splade_realistic.hit_at_1,
                "hit_at_5": splade_realistic.hit_at_5,
                "hit_at_10": splade_realistic.hit_at_10,
                "avg_latency_ms": splade_realistic.avg_latency_ms,
            },
        },
        "improvement": {
            "informed_pct": (splade_informed.mrr - baseline["informed"]["mrr"]) / baseline["informed"]["mrr"] * 100,
            "needle_pct": (splade_needle.mrr - baseline["needle"]["mrr"]) / baseline["needle"]["mrr"] * 100,
            "realistic_pct": (splade_realistic.mrr - baseline["realistic"]["mrr"]) / baseline["realistic"]["mrr"] * 100,
        },
    }

    output_path = ARTIFACTS_DIR / "production_splade_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    informed_pass = output["improvement"]["informed_pct"] > 10
    needle_ok = output["improvement"]["needle_pct"] > -5
    realistic_ok = output["improvement"]["realistic_pct"] > -5

    print("\n" + "=" * 60)
    print("Success Criteria")
    print("=" * 60)
    print(f"  Informed MRR >10% improvement: {'PASS' if informed_pass else 'FAIL'} ({output['improvement']['informed_pct']:+.1f}%)")
    print(f"  Needle MRR no major regression: {'PASS' if needle_ok else 'FAIL'} ({output['improvement']['needle_pct']:+.1f}%)")
    print(f"  Realistic MRR no major regression: {'PASS' if realistic_ok else 'FAIL'} ({output['improvement']['realistic_pct']:+.1f}%)")

    if informed_pass and needle_ok and realistic_ok:
        print("\nVERDICT: PASS - Production SPLADE integration verified")
    else:
        print("\nVERDICT: FAIL - Some criteria not met")


if __name__ == "__main__":
    main()
