"""Regression accuracy test: production search vs POC baseline.

Uses the production HybridRetriever pipeline end-to-end.
Index is cached to disk — first run builds it (~5 min), subsequent runs load
instantly and only run retrieval (~10s).

Extraction must be run separately BEFORE this test (via fast extraction CLI).
This test reads the extraction JSON output and ingests it through the
production pipeline.

Baselines (from POC BENCHMARK_RESULTS.md):
  - Needle questions (20q): POC 90%, target >= 85%
  - Informed questions (50q): POC 80%, target >= 75%

Run:  python tests/search/test_regression_accuracy.py [--rebuild] [extraction_dir]
Clear cache:  python tests/search/test_regression_accuracy.py --rebuild
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from plm.search.adapters import load_extraction_directory
from plm.search.retriever import HybridRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NEEDLE_QUESTIONS = PROJECT_ROOT / "poc" / "chunking_benchmark_v2" / "corpus" / "needle_questions.json"
INFORMED_QUESTIONS = PROJECT_ROOT / "poc" / "modular_retrieval_pipeline" / "corpus" / "informed_questions.json"
DEFAULT_EXTRACTION_DIR = PROJECT_ROOT / ".cache" / "extraction_output"

CACHE_DIR = PROJECT_ROOT / ".cache" / "regression_index"
DB_PATH = str(CACHE_DIR / "regression.db")
BM25_PATH = str(CACHE_DIR / "bm25_index")


def load_needle_questions() -> list[dict]:
    with open(NEEDLE_QUESTIONS) as f:
        data = json.load(f)
    needle_doc_id = data["needle_doc_id"]
    return [{"question": q["question"], "doc_id": needle_doc_id, "id": q["id"]}
            for q in data["questions"]]


def load_informed_questions() -> list[dict]:
    with open(INFORMED_QUESTIONS) as f:
        return json.load(f)["questions"]


def build_index(retriever: HybridRetriever, extraction_dir: Path) -> None:
    t0 = time.time()

    print(f"[build] Loading extraction output from {extraction_dir}...", flush=True)
    documents = load_extraction_directory(extraction_dir)
    total_chunks = sum(len(d["chunks"]) for d in documents)
    print(f"[build] {len(documents)} documents, {total_chunks} chunks", flush=True)

    if not documents:
        print("[build] ERROR: No extraction JSONs found. Run fast extraction first.", flush=True)
        sys.exit(1)

    def progress(step: int, total: int, msg: str) -> None:
        print(f"[build]   Step {step}/{total}: {msg}", flush=True)

    retriever.batch_ingest(
        documents,
        batch_size=256,
        show_progress=True,
        on_progress=progress,
    )
    print(f"[build] Total build time: {time.time()-t0:.0f}s", flush=True)


def evaluate(
    name: str,
    questions: list[dict],
    retriever: HybridRetriever,
    poc_baseline: float,
    tolerance: float = 5.0,
) -> dict:
    target = poc_baseline - tolerance

    t0 = time.time()
    hits = 0
    miss_details = []

    for q in questions:
        results = retriever.retrieve(q["question"], k=5)
        found = any(r["doc_id"] == q["doc_id"] for r in results)
        if found:
            hits += 1
        else:
            miss_details.append((q.get("id", "?"), q["doc_id"],
                                 [r["doc_id"] for r in results]))

    elapsed = time.time() - t0
    accuracy = hits / len(questions) * 100
    passed = accuracy >= target

    print(f"\n--- {name} ---", flush=True)
    print(f"  POC baseline: {poc_baseline:.0f}%  |  Target: >= {target:.0f}%", flush=True)
    for qid, tgt, got in miss_details:
        print(f"  MISS [{qid}]: target={tgt}  got={got}", flush=True)

    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {hits}/{len(questions)} = {accuracy:.1f}% ({elapsed:.1f}s)", flush=True)

    return {"name": name, "accuracy": accuracy, "hits": hits, "total": len(questions),
            "poc_baseline": poc_baseline, "target": target, "passed": passed}


def main() -> bool:
    rebuild = "--rebuild" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    extraction_dir = Path(args[0]) if args else DEFAULT_EXTRACTION_DIR

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("REGRESSION ACCURACY TEST", flush=True)
    print("=" * 70, flush=True)

    retriever = HybridRetriever(DB_PATH, BM25_PATH)

    if rebuild or not retriever.is_indexed():
        if rebuild:
            for p in CACHE_DIR.iterdir():
                if p.is_file():
                    p.unlink()
            retriever = HybridRetriever(DB_PATH, BM25_PATH)
        print(f"[cache] Index not found at {CACHE_DIR}, building...", flush=True)
        build_index(retriever, extraction_dir)
    else:
        chunk_count = retriever.storage.get_chunk_count()
        doc_count = retriever.storage.get_document_count()
        print(f"[cache] Loaded existing index: {doc_count} docs, {chunk_count} chunks", flush=True)

    r_needle = evaluate("Needle (20q)", load_needle_questions(), retriever, poc_baseline=90.0)
    r_informed = evaluate("Informed (50q)", load_informed_questions(), retriever, poc_baseline=80.0)

    print(f"\n{'=' * 70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 70}", flush=True)
    all_passed = True
    for r in [r_needle, r_informed]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}: {r['accuracy']:.1f}% "
              f"(POC {r['poc_baseline']:.0f}%, target >= {r['target']:.0f}%)", flush=True)
        if not r["passed"]:
            all_passed = False

    print(f"\n  {'*** ALL PASSED ***' if all_passed else '*** SOME FAILED ***'}", flush=True)
    return all_passed


if __name__ == "__main__":
    passed = main()
    sys.exit(0 if passed else 1)
