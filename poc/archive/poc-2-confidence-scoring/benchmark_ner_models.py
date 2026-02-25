#!/usr/bin/env python3
"""
Benchmark runner for NER model evaluation.

Usage:
    # Run all models on SO NER data (100 docs)
    python benchmark_ner_models.py

    # Run specific models
    python benchmark_ner_models.py --models heuristic bert-base-ner gliner

    # Run on OOD data too (must generate first)
    python benchmark_ner_models.py --include-ood

    # Quick test (10 docs)
    python benchmark_ner_models.py --n-docs 10 --models heuristic bert-base-ner

    # List available models
    python benchmark_ner_models.py --list-models
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from eval_framework import (
    evaluate_model,
    load_so_ner_test,
    load_ood_dataset,
    report_to_dict,
    print_report,
)
from ner_models import get_all_models


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def run_benchmark(
    model_names: list[str],
    n_docs: int = 100,
    include_ood: bool = False,
) -> dict:
    registry = get_all_models()

    for name in model_names:
        if name not in registry:
            print(f"ERROR: Unknown model '{name}'. Available: {list(registry.keys())}")
            sys.exit(1)

    print(f"Loading SO NER test data ({n_docs} docs)...")
    so_docs = load_so_ner_test(n_docs=n_docs)
    print(f"  Loaded {len(so_docs)} documents\n")

    ood_docs = None
    if include_ood:
        try:
            ood_docs = load_ood_dataset()
            print(f"Loaded OOD dataset ({len(ood_docs)} docs)\n")
        except FileNotFoundError as e:
            print(f"WARNING: {e}")
            print("Run `python generate_ood_dataset.py` first. Skipping OOD.\n")

    all_reports = []

    for model_name in model_names:
        print(f"\n{'#' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'#' * 60}")

        try:
            print(f"  Loading model...")
            model = registry[model_name]()
            print(f"  Model loaded: {model.name}")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            all_reports.append({
                "model_name": model_name,
                "error": str(e),
                "dataset_name": "so-ner",
            })
            continue

        print(f"\n  Evaluating on SO NER ({len(so_docs)} docs)...")
        try:
            so_report = evaluate_model(model, so_docs, dataset_name="so-ner")
            print_report(so_report)
            all_reports.append(report_to_dict(so_report))
        except Exception as e:
            print(f"  FAILED on SO NER: {e}")
            all_reports.append({
                "model_name": model_name,
                "error": str(e),
                "dataset_name": "so-ner",
            })

        if ood_docs:
            print(f"\n  Evaluating on OOD ({len(ood_docs)} docs)...")
            try:
                ood_report = evaluate_model(model, ood_docs, dataset_name="ood")
                print_report(ood_report)
                all_reports.append(report_to_dict(ood_report))
            except Exception as e:
                print(f"  FAILED on OOD: {e}")
                all_reports.append({
                    "model_name": model_name,
                    "error": str(e),
                    "dataset_name": "ood",
                })

    return {
        "timestamp": datetime.now().isoformat(),
        "n_docs_so": len(so_docs),
        "n_docs_ood": len(ood_docs) if ood_docs else 0,
        "models_tested": model_names,
        "reports": all_reports,
    }


def print_comparison_table(results: dict) -> None:
    reports = [r for r in results["reports"] if "error" not in r]
    if not reports:
        print("\nNo successful reports to compare.")
        return

    so_reports = [r for r in reports if r["dataset_name"] == "so-ner"]
    ood_reports = [r for r in reports if r["dataset_name"] == "ood"]

    print(f"\n{'=' * 90}")
    print("  COMPARISON TABLE — SO NER")
    print(f"{'=' * 90}")
    print(f"  {'Model':<25} {'P':>6} {'R':>6} {'F1':>6} {'Hall':>6} {'Conf':>6} {'Sep':>6} {'ms':>8}")
    print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
    for r in sorted(so_reports, key=lambda x: x["avg_f1"], reverse=True):
        print(
            f"  {r['model_name']:<25}"
            f" {r['avg_precision']:6.3f}"
            f" {r['avg_recall']:6.3f}"
            f" {r['avg_f1']:6.3f}"
            f" {r['avg_hallucination']:6.3f}"
            f" {r['avg_confidence']:6.3f}"
            f" {r['confidence_separation']:+6.3f}"
            f" {r['avg_time_ms']:8.1f}"
        )

    if ood_reports:
        print(f"\n{'=' * 90}")
        print("  COMPARISON TABLE — OOD (Renamed Entities)")
        print(f"{'=' * 90}")
        print(f"  {'Model':<25} {'P':>6} {'R':>6} {'F1':>6} {'Hall':>6} {'Conf':>6} {'Sep':>6} {'ms':>8}")
        print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*8}")
        for r in sorted(ood_reports, key=lambda x: x["avg_f1"], reverse=True):
            print(
                f"  {r['model_name']:<25}"
                f" {r['avg_precision']:6.3f}"
                f" {r['avg_recall']:6.3f}"
                f" {r['avg_f1']:6.3f}"
                f" {r['avg_hallucination']:6.3f}"
                f" {r['avg_confidence']:6.3f}"
                f" {r['confidence_separation']:+6.3f}"
                f" {r['avg_time_ms']:8.1f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Benchmark NER models for extraction quality")
    parser.add_argument("--models", nargs="+", help="Model names to test (default: all)")
    parser.add_argument("--n-docs", type=int, default=100, help="Number of SO NER docs (default: 100)")
    parser.add_argument("--include-ood", action="store_true", help="Also test on OOD dataset")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    args = parser.parse_args()

    if args.list_models:
        registry = get_all_models()
        print("Available models:")
        for name in sorted(registry.keys()):
            print(f"  - {name}")
        return

    model_names = args.models or list(get_all_models().keys())
    print(f"Models to test: {model_names}")
    print(f"SO NER docs: {args.n_docs}")
    print(f"Include OOD: {args.include_ood}\n")

    results = run_benchmark(model_names, n_docs=args.n_docs, include_ood=args.include_ood)

    print_comparison_table(results)

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    output_path = ARTIFACTS_DIR / "ner_model_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    errors = [r for r in results["reports"] if "error" in r]
    if errors:
        print(f"\n{len(errors)} model(s) failed:")
        for e in errors:
            print(f"  - {e['model_name']}: {e['error']}")


if __name__ == "__main__":
    main()
