#!/usr/bin/env python3
"""
Phase 1: Prepare Evaluation Dataset

Loads 100 random SO NER documents, runs fast extraction, calculates metrics and signals,
saves structured dataset for grading and analysis.
"""

import json
import random
import sys
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "poc-1c-scalable-ner"))

from plm.extraction.fast.heuristic import extract_all_heuristic
from scoring import many_to_many_score
from signals import (
    known_term_ratio,
    coverage_score,
    entity_density,
    section_type_mismatch,
)


def load_test_documents(n_docs: int = 100, seed: int = 42) -> list[dict[str, Any]]:
    """Load n random documents from SO NER test set."""
    test_path = Path(__file__).parent.parent / "poc-1c-scalable-ner" / "artifacts" / "test_documents.json"
    
    with open(test_path) as f:
        all_docs = json.load(f)
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Sample n documents
    if len(all_docs) < n_docs:
        raise ValueError(f"Only {len(all_docs)} documents available, requested {n_docs}")
    
    return random.sample(all_docs, n_docs)


def load_vocabulary() -> set[str]:
    """Load term vocabulary from term_index.json."""
    vocab_path = Path(__file__).parent.parent.parent / "data" / "vocabularies" / "term_index.json"
    
    with open(vocab_path) as f:
        term_index = json.load(f)
    
    # Extract all term keys (lowercase for matching)
    vocab = {term.lower() for term in term_index.keys()}
    return vocab


def process_document(doc: dict[str, Any], vocab: set[str]) -> dict[str, Any]:
    """Process a single document: extract terms, calculate metrics and signals."""
    doc_id = doc["doc_id"]
    text = doc["text"]
    gt_terms = doc["gt_terms"]
    
    # Run fast extraction
    extracted_terms = extract_all_heuristic(text)
    
    # Calculate metrics using many_to_many_score
    score_result = many_to_many_score(extracted_terms, gt_terms)
    metrics = {
        "precision": score_result["precision"],
        "recall": score_result["recall"],
        "f1": score_result["f1"],
        "hallucination": score_result["hallucination"],
    }
    
    # Calculate confidence signals
    signals = {
        "known_term_ratio": known_term_ratio(extracted_terms, vocab),
        "coverage": coverage_score(extracted_terms, text),
        "entity_density": entity_density(extracted_terms, text),
        "section_type_mismatch": section_type_mismatch(extracted_terms, None),
    }
    
    return {
        "doc_id": doc_id,
        "text": text,
        "gt_terms": gt_terms,
        "extracted_terms": extracted_terms,
        "metrics": metrics,
        "signals": signals,
    }


def calculate_statistics(dataset: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate summary statistics for the dataset."""
    n = len(dataset)
    
    # Average metrics
    avg_precision = sum(d["metrics"]["precision"] for d in dataset) / n
    avg_recall = sum(d["metrics"]["recall"] for d in dataset) / n
    avg_f1 = sum(d["metrics"]["f1"] for d in dataset) / n
    avg_hallucination = sum(d["metrics"]["hallucination"] for d in dataset) / n
    
    # Signal distributions
    def signal_stats(signal_name: str) -> dict[str, float]:
        values = [d["signals"][signal_name] for d in dataset]
        values_sorted = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": values_sorted[len(values_sorted) // 2],
        }
    
    # Extraction statistics
    avg_extracted = sum(len(d["extracted_terms"]) for d in dataset) / n
    avg_gt = sum(len(d["gt_terms"]) for d in dataset) / n
    
    return {
        "total_documents": n,
        "avg_metrics": {
            "precision": avg_precision,
            "recall": avg_recall,
            "f1": avg_f1,
            "hallucination": avg_hallucination,
        },
        "signal_distributions": {
            "known_term_ratio": signal_stats("known_term_ratio"),
            "coverage": signal_stats("coverage"),
            "entity_density": signal_stats("entity_density"),
            "section_type_mismatch": signal_stats("section_type_mismatch"),
        },
        "extraction_stats": {
            "avg_extracted_terms": avg_extracted,
            "avg_gt_terms": avg_gt,
        },
    }


def write_summary(stats: dict[str, Any], output_path: Path) -> None:
    """Write summary statistics to markdown file."""
    with open(output_path, "w") as f:
        f.write("# Phase 1 Dataset Summary\n\n")
        
        f.write(f"## Overview\n\n")
        f.write(f"- **Total Documents**: {stats['total_documents']}\n")
        f.write(f"- **Average Extracted Terms**: {stats['extraction_stats']['avg_extracted_terms']:.2f}\n")
        f.write(f"- **Average Ground Truth Terms**: {stats['extraction_stats']['avg_gt_terms']:.2f}\n\n")
        
        f.write("## Average Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        for metric, value in stats["avg_metrics"].items():
            f.write(f"| {metric.capitalize()} | {value:.4f} |\n")
        f.write("\n")
        
        f.write("## Signal Distributions\n\n")
        for signal_name, signal_stats in stats["signal_distributions"].items():
            f.write(f"### {signal_name}\n\n")
            f.write("| Statistic | Value |\n")
            f.write("|-----------|-------|\n")
            for stat_name, stat_value in signal_stats.items():
                f.write(f"| {stat_name.capitalize()} | {stat_value:.4f} |\n")
            f.write("\n")


def main():
    """Main execution function."""
    print("Phase 1: Preparing Evaluation Dataset")
    print("=" * 50)
    
    # Create artifacts directory
    artifacts_dir = Path(__file__).parent / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\n1. Loading test documents...")
    documents = load_test_documents(n_docs=100, seed=42)
    print(f"   Loaded {len(documents)} documents")
    
    print("\n2. Loading vocabulary...")
    vocab = load_vocabulary()
    print(f"   Loaded {len(vocab)} terms")
    
    # Process documents
    print("\n3. Processing documents...")
    dataset = []
    for i, doc in enumerate(documents, 1):
        if i % 10 == 0:
            print(f"   Processed {i}/{len(documents)} documents")
        result = process_document(doc, vocab)
        dataset.append(result)
    
    print(f"   Processed all {len(dataset)} documents")
    
    # Save dataset
    output_path = artifacts_dir / "phase-1-dataset.json"
    print(f"\n4. Saving dataset to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"   Saved {len(dataset)} entries")
    
    # Calculate and save statistics
    print("\n5. Calculating statistics...")
    stats = calculate_statistics(dataset)
    
    summary_path = artifacts_dir / "phase-1-summary.md"
    print(f"   Writing summary to {summary_path}...")
    write_summary(stats, summary_path)
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Documents processed: {stats['total_documents']}")
    print(f"Average Precision: {stats['avg_metrics']['precision']:.4f}")
    print(f"Average Recall: {stats['avg_metrics']['recall']:.4f}")
    print(f"Average F1: {stats['avg_metrics']['f1']:.4f}")
    print(f"Average Hallucination: {stats['avg_metrics']['hallucination']:.4f}")
    print(f"\nAverage Extracted Terms: {stats['extraction_stats']['avg_extracted_terms']:.2f}")
    print(f"Average GT Terms: {stats['extraction_stats']['avg_gt_terms']:.2f}")
    print("\n✓ Phase 1 complete!")


if __name__ == "__main__":
    main()
