#!/usr/bin/env python3
"""Benchmark individual extractors against GT to measure per-extractor recall/precision.

Usage:
    python benchmark_extractors.py --n-docs 3
    python benchmark_extractors.py --n-docs 10 --extractors haiku_simple sonnet_taxonomy
    python benchmark_extractors.py --n-docs 3 --doc-ids Q39079773 Q45734089
"""

import argparse
import json
import time
from pathlib import Path

from parse_so_ner import select_documents
from scoring import many_to_many_score


def load_json(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_extractor(
    name: str,
    doc: dict,
    train_docs: list[dict],
    index,
    embed_model,
) -> tuple[list[str], float]:
    t0 = time.time()

    if name == "haiku_fewshot":
        from hybrid_ner import _extract_haiku_fewshot
        terms, _ = _extract_haiku_fewshot(doc, train_docs, index, embed_model)
    elif name == "sonnet_taxonomy":
        from hybrid_ner import _extract_haiku_taxonomy
        terms, _ = _extract_haiku_taxonomy(doc, llm_model="sonnet")
    elif name == "haiku_simple":
        from hybrid_ner import _extract_haiku_simple
        terms, _ = _extract_haiku_simple(doc)
    elif name == "heuristic":
        from hybrid_ner import _extract_heuristic
        terms = _extract_heuristic(doc)
    elif name == "seeds":
        from hybrid_ner import _extract_seeds
        vocab_path = Path(__file__).parent / "artifacts" / "auto_vocab.json"
        with open(vocab_path) as f:
            auto_vocab = json.load(f)
        seeds_list = list(auto_vocab.get("seeds", []))
        terms = _extract_seeds(doc, seeds_list)
    else:
        raise ValueError(f"Unknown extractor: {name}")

    elapsed = time.time() - t0
    return terms, elapsed


ALL_EXTRACTORS = [
    "haiku_fewshot",
    "sonnet_taxonomy",
    "haiku_simple",
    "heuristic",
    "seeds",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark individual extractors")
    parser.add_argument("--n-docs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--extractors",
        nargs="+",
        default=ALL_EXTRACTORS,
        choices=ALL_EXTRACTORS,
    )
    parser.add_argument("--doc-ids", nargs="+", default=None)
    args = parser.parse_args()

    artifacts = Path(__file__).parent / "artifacts"
    train_docs = load_json(artifacts / "train_documents.json")
    test_docs = load_json(artifacts / "test_documents.json")
    print(f"Loaded {len(train_docs)} train, {len(test_docs)} test docs")

    if args.doc_ids:
        selected = [d for d in test_docs if d["doc_id"] in args.doc_ids]
        if not selected:
            print(f"No docs found with IDs: {args.doc_ids}")
            return
    else:
        selected = select_documents(test_docs, args.n_docs, seed=args.seed)

    retrieval_index = None
    retrieval_model = None
    if any(e in args.extractors for e in ("haiku_fewshot",)):
        from retrieval_ner import build_retrieval_index
        retrieval_index, _, retrieval_model = build_retrieval_index(train_docs)

    print(f"\nBenchmarking {len(args.extractors)} extractors on {len(selected)} docs\n")

    per_extractor: dict[str, dict] = {
        name: {"tp": 0, "fp": 0, "fn": 0, "total_time": 0.0, "docs": []}
        for name in args.extractors
    }

    for doc_idx, doc in enumerate(selected):
        doc_id = doc["doc_id"]
        gt = doc["gt_terms"]
        print(f"{'='*70}")
        print(f"Doc [{doc_idx+1}/{len(selected)}]: {doc_id} (GT: {len(gt)} terms)")
        print(f"{'='*70}")

        for name in args.extractors:
            terms, elapsed = run_extractor(
                name, doc, train_docs, retrieval_index, retrieval_model,
            )
            scores = many_to_many_score(terms, gt)

            per_extractor[name]["tp"] += scores["tp"]
            per_extractor[name]["fp"] += scores["fp"]
            per_extractor[name]["fn"] += scores["fn"]
            per_extractor[name]["total_time"] += elapsed
            per_extractor[name]["docs"].append({
                "doc_id": doc_id,
                "extracted_count": len(terms),
                "tp": scores["tp"],
                "fp": scores["fp"],
                "fn": scores["fn"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "fp_terms": scores["fp_terms"],
                "fn_terms": scores["fn_terms"],
            })

            print(
                f"  {name:20s}: P={scores['precision']*100:5.1f}%  "
                f"R={scores['recall']*100:5.1f}%  "
                f"F1={scores['f1']:.3f}  "
                f"({len(terms)} extracted, {elapsed:.1f}s)"
            )
            if scores["fn_terms"]:
                print(f"    FN: {scores['fn_terms'][:10]}")
            if scores["fp_terms"]:
                print(f"    FP: {scores['fp_terms'][:10]}")

        print()

    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}\n")
    print(f"{'Extractor':20s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Time':>7s}")
    print("-" * 70)

    for name in args.extractors:
        data = per_extractor[name]
        tp, fp, fn = data["tp"], data["fp"], data["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(
            f"{name:20s} {p*100:6.1f}% {r*100:6.1f}% {f1:6.3f} {tp:5d} {fp:5d} {fn:5d} {data['total_time']:6.1f}s"
        )

    all_fn: dict[str, list[str]] = {}
    for name in args.extractors:
        for doc_data in per_extractor[name]["docs"]:
            for fn_term in doc_data["fn_terms"]:
                if fn_term not in all_fn:
                    all_fn[fn_term] = []
                all_fn[fn_term].append(name)

    missed_by_all = [
        term for term, extractors in all_fn.items()
        if len(extractors) == len(args.extractors)
    ]
    if missed_by_all:
        print(f"\nTerms missed by ALL extractors ({len(missed_by_all)}):")
        for term in sorted(missed_by_all):
            print(f"  - {term}")

    out_path = artifacts / "results" / "extractor_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.extractors:
        data = per_extractor[name]
        tp, fp, fn = data["tp"], data["fp"], data["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "total_time": round(data["total_time"], 1),
            "docs": data["docs"],
        }

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
