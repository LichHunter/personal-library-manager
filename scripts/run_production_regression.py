"""Run production pipeline directly — no test wrappers."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "poc" / "chunking_benchmark_v2" / "corpus" / "kubernetes"
NEEDLE_QUESTIONS = PROJECT_ROOT / "poc" / "chunking_benchmark_v2" / "corpus" / "needle_questions.json"
INFORMED_QUESTIONS = PROJECT_ROOT / "poc" / "modular_retrieval_pipeline" / "corpus" / "informed_questions.json"
CACHE_DIR = PROJECT_ROOT / ".cache" / "regression_index"
DB_PATH = str(CACHE_DIR / "regression.db")
BM25_PATH = str(CACHE_DIR / "bm25_index")

from plm.extraction.fast.document_processor import process_document
from plm.search.adapters.gliner_adapter import document_result_to_chunks
from plm.search.retriever import HybridRetriever

BATCH_SIZE = 50


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever(DB_PATH, BM25_PATH)

    if not retriever.is_indexed():
        print("=== BUILDING INDEX (production pipeline) ===", flush=True)
        t0 = time.time()
        files = sorted(CORPUS_DIR.glob("*.md"))
        print(f"Processing {len(files)} files in batches of {BATCH_SIZE}...", flush=True)

        total_chunks = 0
        for batch_start in range(0, len(files), BATCH_SIZE):
            batch_files = files[batch_start:batch_start + BATCH_SIZE]
            batch_docs = []

            for filepath in batch_files:
                doc_result = process_document(filepath)
                chunks = document_result_to_chunks(doc_result)
                batch_docs.append({
                    "doc_id": filepath.stem,
                    "source_file": str(filepath),
                    "chunks": chunks,
                })
                total_chunks += len(chunks)

            retriever.batch_ingest(batch_docs, batch_size=256, show_progress=False, rebuild_index=False)
            batch_end = min(batch_start + BATCH_SIZE, len(files))
            print(f"  {batch_end}/{len(files)} done, {total_chunks} chunks ({time.time()-t0:.0f}s)", flush=True)

        print("Building BM25 index...", flush=True)
        retriever.rebuild_index()
        print(f"Index built in {time.time()-t0:.0f}s", flush=True)
    else:
        doc_count = retriever.storage.get_document_count()
        chunk_count = retriever.storage.get_chunk_count()
        print(f"=== LOADED CACHED INDEX: {doc_count} docs, {chunk_count} chunks ===", flush=True)

    with open(NEEDLE_QUESTIONS) as f:
        ndata = json.load(f)
    needle_doc_id = ndata["needle_doc_id"]
    needle_qs = [{"question": q["question"], "doc_id": needle_doc_id, "id": q["id"]}
                 for q in ndata["questions"]]

    with open(INFORMED_QUESTIONS) as f:
        informed_qs = json.load(f)["questions"]

    for name, questions, baseline in [
        ("Needle (20q)", needle_qs, 90.0),
        ("Informed (50q)", informed_qs, 80.0),
    ]:
        target = baseline - 5.0
        hits = 0
        for q in questions:
            results = retriever.retrieve(q["question"], k=5)
            found = any(r["doc_id"] == q["doc_id"] for r in results)
            if found:
                hits += 1
            else:
                got = [r["doc_id"] for r in results]
                print(f"  MISS [{q.get('id','?')}]: target={q['doc_id']}  got={got}", flush=True)

        accuracy = hits / len(questions) * 100
        status = "PASS" if accuracy >= target else "FAIL"
        print(f"[{status}] {name}: {hits}/{len(questions)} = {accuracy:.1f}% "
              f"(POC {baseline:.0f}%, target >= {target:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
