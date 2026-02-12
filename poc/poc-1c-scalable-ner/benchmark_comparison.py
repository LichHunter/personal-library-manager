#!/usr/bin/env python3
"""Head-to-head benchmark comparing NER approaches.

Usage:
    python benchmark_comparison.py --approach retrieval --n-docs 10
    python benchmark_comparison.py --approach slimer --n-docs 10
    python benchmark_comparison.py --approach hybrid --n-docs 10
    python benchmark_comparison.py --approach all --n-docs 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

from parse_so_ner import extract_gt_terms, select_documents
from scoring import many_to_many_score


def load_json(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_benchmark(
    approach: str,
    test_docs: list[dict],
    train_docs: list[dict] | None = None,
    n_docs: int = 100,
    seed: int = 42,
    strategy_name: str | None = None,
) -> dict:
    selected = select_documents(test_docs, n_docs, seed=seed)

    label = f"{approach}" + (f"/{strategy_name}" if strategy_name else "")
    results: dict = {
        "approach": approach,
        "strategy": strategy_name,
        "n_docs": len(selected),
        "seed": seed,
        "documents": [],
        "totals": {"tp": 0, "fp": 0, "fn": 0},
    }

    retrieval_index = None
    retrieval_model = None
    auto_vocab = None

    if approach in ("retrieval", "hybrid", "hybrid_v5"):
        from retrieval_ner import build_retrieval_index
        assert train_docs is not None, f"{approach} approach requires train_docs"
        retrieval_index, _embeddings, retrieval_model = build_retrieval_index(train_docs)

    term_index = None
    strategy_config = None

    if approach in ("hybrid", "hybrid_v5"):
        vocab_path = Path(__file__).parent / "artifacts" / "auto_vocab.json"
        with open(vocab_path) as f:
            auto_vocab = json.load(f)
        print(f"  Loaded auto_vocab: {auto_vocab['stats']}")

        from hybrid_ner import build_term_index, get_strategy_config
        assert train_docs is not None
        term_index = build_term_index(train_docs)
        print(f"  Built term_index: {len(term_index)} terms")

        if strategy_name:
            strategy_config = get_strategy_config(strategy_name)
            print(f"  Strategy: {strategy_config.name}")

    total_start = time.time()

    for i, doc in enumerate(selected):
        doc_start = time.time()
        gt_terms = doc["gt_terms"]
        doc_id = doc["doc_id"]

        if approach == "retrieval":
            from retrieval_ner import extract_with_retrieval
            assert train_docs is not None
            extracted = extract_with_retrieval(doc, train_docs, retrieval_index, retrieval_model, k=5)
        elif approach == "slimer":
            from slimer_ner import extract_with_slimer
            extracted = extract_with_slimer(doc)
        elif approach == "hybrid":
            from hybrid_ner import extract_hybrid
            assert train_docs is not None and auto_vocab is not None
            extracted = extract_hybrid(
                doc, train_docs, retrieval_index, retrieval_model,
                auto_vocab, term_index=term_index, strategy=strategy_config,
            )
        elif approach == "hybrid_v5":
            from hybrid_ner import extract_hybrid_v5
            assert train_docs is not None and auto_vocab is not None
            extracted = extract_hybrid_v5(
                doc, train_docs, retrieval_index, retrieval_model,
                auto_vocab, term_index=term_index, strategy=strategy_config,
            )
        else:
            raise ValueError(f"Unknown approach: {approach}")

        scores = many_to_many_score(extracted, gt_terms)

        doc_elapsed = time.time() - doc_start

        print(
            f"  [{i+1}/{len(selected)}] {doc_id}: "
            f"P={scores['precision']*100:.1f}% R={scores['recall']*100:.1f}% "
            f"H={scores['hallucination']*100:.1f}% F1={scores['f1']:.3f} "
            f"({doc_elapsed:.1f}s)"
        )

        if scores["fp_terms"]:
            print(f"    FP: {scores['fp_terms']}")
        if scores["fn_terms"]:
            print(f"    FN: {scores['fn_terms']}")

        results["documents"].append({
            "doc_id": doc_id,
            "extracted": extracted,
            "gt_terms": gt_terms,
            "tp": scores["tp"],
            "fp": scores["fp"],
            "fn": scores["fn"],
            "fp_terms": scores["fp_terms"],
            "fn_terms": scores["fn_terms"],
            "precision": scores["precision"],
            "recall": scores["recall"],
            "elapsed": round(doc_elapsed, 2),
        })
        results["totals"]["tp"] += scores["tp"]
        results["totals"]["fp"] += scores["fp"]
        results["totals"]["fn"] += scores["fn"]

    total_elapsed = time.time() - total_start

    tp = results["totals"]["tp"]
    fp = results["totals"]["fp"]
    fn = results["totals"]["fn"]
    results["metrics"] = {
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "hallucination": fp / (tp + fp) if (tp + fp) > 0 else 0,
        "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
    }
    results["total_elapsed_seconds"] = round(total_elapsed, 1)

    return results


def print_summary(results: dict) -> None:
    m = results["metrics"]
    print(f"\n{'='*60}")
    print(f"RESULTS: {results['approach']} ({results['n_docs']} docs)")
    print(f"{'='*60}")
    print(f"  Precision:     {m['precision']*100:.1f}%")
    print(f"  Recall:        {m['recall']*100:.1f}%")
    print(f"  Hallucination: {m['hallucination']*100:.1f}%")
    print(f"  F1:            {m['f1']:.3f}")
    t = results["totals"]
    print(f"  TP={t['tp']} FP={t['fp']} FN={t['fn']}")
    print(f"  Time: {results.get('total_elapsed_seconds', '?')}s")

    all_fp: dict[str, int] = {}
    all_fn: dict[str, int] = {}
    for doc in results["documents"]:
        for term in doc.get("fp_terms", []):
            all_fp[term] = all_fp.get(term, 0) + 1
        for term in doc.get("fn_terms", []):
            all_fn[term] = all_fn.get(term, 0) + 1

    top_fp = sorted(all_fp.items(), key=lambda x: -x[1])[:15]
    top_fn = sorted(all_fn.items(), key=lambda x: -x[1])[:15]

    if top_fp:
        print(f"\n  Top FPs: {top_fp}")
    if top_fn:
        print(f"  Top FNs: {top_fn}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NER approaches")
    parser.add_argument(
        "--approach",
        choices=["retrieval", "slimer", "hybrid", "hybrid_v5", "all"],
        required=True,
    )
    parser.add_argument("--n-docs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        default=None,
        help="Strategy preset name (baseline, strategy_a..e, strategy_v5). Only for hybrid/hybrid_v5.",
    )
    parser.add_argument(
        "--clear-stats",
        action="store_true",
        help="Clear low_confidence_stats.jsonl before running.",
    )
    args = parser.parse_args()

    if args.clear_stats:
        from hybrid_ner import clear_low_confidence_stats
        clear_low_confidence_stats()
        print("Cleared low_confidence_stats.jsonl")

    artifacts = Path(__file__).parent / "artifacts"

    train_docs = load_json(artifacts / "train_documents.json")
    test_docs = load_json(artifacts / "test_documents.json")
    print(f"Loaded {len(train_docs)} train docs, {len(test_docs)} test docs")

    approaches = ["retrieval", "slimer", "hybrid"] if args.approach == "all" else [args.approach]

    for approach in approaches:
        strategy_name = args.strategy if approach in ("hybrid", "hybrid_v5") else None
        label = f"{approach}" + (f"/{strategy_name}" if strategy_name else "")
        print(f"\n{'='*60}")
        print(f"Running {label} on {args.n_docs} documents")
        print(f"{'='*60}\n")

        results = run_benchmark(
            approach, test_docs, train_docs, args.n_docs, args.seed,
            strategy_name=strategy_name,
        )

        suffix = f"_{strategy_name}" if strategy_name else ""
        out_path = artifacts / "results" / f"{approach}{suffix}_results.json"
        save_json(results, out_path)
        print(f"\nSaved to {out_path}")

        print_summary(results)


if __name__ == "__main__":
    main()
