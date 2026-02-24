#!/usr/bin/env python3
"""Quick validation of production SPLADE integration against POC results."""

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plm.search.components.sparse import SPLADERetriever, SPLADEConfig

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


def load_sample_queries():
    path = CORPUS_DIR / "kubernetes" / "informed_questions.json"
    with open(path) as f:
        data = json.load(f)
    questions = data.get("questions", [])[:5]
    return [
        {"query": q["original_instruction"], "expected_doc_id": q["doc_id"]}
        for q in questions
    ]


def main():
    print("=" * 60)
    print("Production SPLADE Integration Validation")
    print("=" * 60)

    idx_to_doc = load_doc_mapping()
    print(f"\nLoaded doc mapping: {len(idx_to_doc)} chunks")

    print(f"\nLoading production SPLADERetriever from {SPLADE_INDEX_PATH}...")
    retriever = SPLADERetriever.load(str(SPLADE_INDEX_PATH))
    print(f"Loaded: {retriever.document_count} documents, is_ready={retriever.is_ready}")

    queries = load_sample_queries()
    print(f"\nRunning {len(queries)} sample queries...\n")

    hits = 0
    total_latency = 0

    for i, q in enumerate(queries):
        start = time.time()
        results = retriever.search(q["query"], k=10)
        latency = (time.time() - start) * 1000
        total_latency += latency

        found = False
        rank = None
        for j, r in enumerate(results):
            doc_id = idx_to_doc.get(r["index"], "unknown")
            if q["expected_doc_id"] in doc_id:
                found = True
                rank = j + 1
                break

        status = f"FOUND@{rank}" if found else "NOT FOUND"
        if found:
            hits += 1

        print(f"[{i+1}] {status} ({latency:.0f}ms)")
        print(f"    Query: {q['query'][:60]}...")
        print(f"    Expected: {q['expected_doc_id']}")
        if results:
            top_doc = idx_to_doc.get(results[0]["index"], "unknown")
            print(f"    Top result: {top_doc} (score={results[0]['score']:.2f})")
        print()

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Hits: {hits}/{len(queries)} ({hits/len(queries)*100:.0f}%)")
    print(f"  Avg latency: {total_latency/len(queries):.0f}ms")

    if hits == len(queries):
        print("\nVALIDATION: PASS - All sample queries found expected documents")
    elif hits >= len(queries) * 0.6:
        print("\nVALIDATION: PARTIAL - Most queries found expected documents")
    else:
        print("\nVALIDATION: FAIL - Too many queries failed")


if __name__ == "__main__":
    main()
