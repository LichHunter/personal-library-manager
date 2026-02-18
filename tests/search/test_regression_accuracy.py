"""Regression accuracy test: production search vs POC baseline.

Uses the production HybridRetriever pipeline end-to-end.
Index is cached to disk — first run builds it (~20 min), subsequent runs load
instantly and only run retrieval (~10s).

Baselines (from POC BENCHMARK_RESULTS.md):
  - Needle questions (20q): POC 90%, target >= 85%
  - Informed questions (50q): POC 80%, target >= 75%

Run:  python tests/search/test_regression_accuracy.py
Clear cache:  python tests/search/test_regression_accuracy.py --rebuild
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "poc" / "chunking_benchmark_v2"))

import yake
from strategies import Document, MarkdownSemanticStrategy

from plm.search.retriever import HybridRetriever

CORPUS_DIR = PROJECT_ROOT / "poc" / "chunking_benchmark_v2" / "corpus" / "kubernetes"
NEEDLE_QUESTIONS = PROJECT_ROOT / "poc" / "chunking_benchmark_v2" / "corpus" / "needle_questions.json"
INFORMED_QUESTIONS = PROJECT_ROOT / "poc" / "modular_retrieval_pipeline" / "corpus" / "informed_questions.json"

CACHE_DIR = PROJECT_ROOT / ".cache" / "regression_index"
DB_PATH = str(CACHE_DIR / "regression.db")
BM25_PATH = str(CACHE_DIR / "bm25_index")

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
ENTITY_TYPES = {"ORG", "PRODUCT", "GPE", "PERSON", "WORK_OF_ART", "LAW", "EVENT", "FAC", "NORP"}

_yake_ext: yake.KeywordExtractor | None = None
_spacy_nlp = None


def _get_yake() -> yake.KeywordExtractor:
    global _yake_ext
    if _yake_ext is None:
        _yake_ext = yake.KeywordExtractor(
            lan="en", n=2, top=10, dedupLim=0.9,
            dedupFunc="seqm", windowsSize=1,
        )
    return _yake_ext


def _get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def extract_keywords(text: str) -> list[str]:
    if not text or len(text.strip()) < 50:
        return []
    return [kw for kw, _ in _get_yake().extract_keywords(text)][:10]


def extract_entities(text: str) -> dict[str, list[str]]:
    if not text or len(text.strip()) < 50:
        return {}
    nlp = _get_spacy()
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_RE.finditer(text))
    code_ratio = code_chars / len(text) if text else 0.0
    clean = CODE_BLOCK_RE.sub(" ", text) if code_ratio > 0.3 else text
    doc = nlp(clean[:5000])
    entities: dict[str, list[str]] = {}
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            bucket = entities.setdefault(ent.label_, [])
            if ent.text not in bucket:
                bucket.append(ent.text)
    for label in entities:
        entities[label] = entities[label][:5]
    return entities


# ---------------------------------------------------------------------------
# Corpus loading — same logic as POC benchmark.py
# ---------------------------------------------------------------------------

def load_documents() -> list[Document]:
    docs = []
    for md_path in sorted(CORPUS_DIR.glob("*.md")):
        doc_id = md_path.stem
        content = md_path.read_text(encoding="utf-8")
        title = doc_id
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("title:"):
                title = line.replace("title:", "").strip().strip("\"'")
                break
            elif line.startswith("# "):
                title = line[2:].strip()
                break
        docs.append(Document(id=doc_id, title=title, content=content, path=str(md_path)))
    return docs


def chunk_documents(documents: list[Document]):
    strategy = MarkdownSemanticStrategy(
        max_heading_level=4, target_chunk_size=400,
        min_chunk_size=50, max_chunk_size=800, overlap_sentences=1,
    )
    all_chunks = []
    for doc in documents:
        all_chunks.extend(strategy.chunk(doc))
    return all_chunks


def load_needle_questions() -> list[dict]:
    with open(NEEDLE_QUESTIONS) as f:
        data = json.load(f)
    needle_doc_id = data["needle_doc_id"]
    return [{"question": q["question"], "doc_id": needle_doc_id, "id": q["id"]}
            for q in data["questions"]]


def load_informed_questions() -> list[dict]:
    with open(INFORMED_QUESTIONS) as f:
        return json.load(f)["questions"]


# ---------------------------------------------------------------------------
# Build index through production pipeline
# ---------------------------------------------------------------------------

def build_index(retriever: HybridRetriever) -> None:
    t0 = time.time()

    print("[build] Loading corpus...", flush=True)
    documents = load_documents()
    print(f"[build] {len(documents)} documents loaded", flush=True)

    print("[build] Chunking...", flush=True)
    poc_chunks = chunk_documents(documents)
    print(f"[build] {len(poc_chunks)} chunks created", flush=True)

    print("[build] Extracting keywords + entities...", flush=True)
    doc_map: dict[str, dict] = {}
    for i, c in enumerate(poc_chunks):
        if (i + 1) % 1000 == 0:
            print(f"[build]   {i+1}/{len(poc_chunks)} enriched ({time.time()-t0:.0f}s)", flush=True)

        entry = doc_map.setdefault(c.doc_id, {
            "doc_id": c.doc_id,
            "source_file": f"{c.doc_id}.md",
            "chunks": [],
        })
        entry["chunks"].append({
            "content": c.content,
            "keywords": extract_keywords(c.content),
            "entities": extract_entities(c.content),
        })

    print(f"[build] Enrichment done in {time.time()-t0:.0f}s", flush=True)
    print(f"[build] Batch ingesting {len(doc_map)} documents...", flush=True)

    def progress(step: int, total: int, msg: str) -> None:
        print(f"[build]   Step {step}/{total}: {msg}", flush=True)

    retriever.batch_ingest(
        list(doc_map.values()),
        batch_size=256,
        show_progress=True,
        on_progress=progress,
    )
    print(f"[build] Total build time: {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> bool:
    rebuild = "--rebuild" in sys.argv
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
        build_index(retriever)
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
