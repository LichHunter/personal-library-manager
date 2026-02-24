#!/usr/bin/env python3
"""Compare alternative SPLADE models on a small sample.

This script tests different SPLADE model variants for:
1. Encoding speed (latency per query)
2. Term expansion quality (number and type of expanded terms)
3. Technical term handling (CamelCase, acronyms)

Run this while the main index is being built to evaluate alternatives.

Usage:
    python compare_models.py [--sample-size 100]
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np


# Models to compare
MODELS = {
    "ensembledistil": "naver/splade-cocondenser-ensembledistil",
    "selfdistil": "naver/splade-cocondenser-selfdistil",
    # "splade-v3": "naver/splade-v3",  # Requires HF login
    "splade-tiny": "rasyosef/splade-tiny",
    "splade-mini": "rasyosef/splade-mini",
    "splade-small": "rasyosef/splade-small",
}

# Technical test queries (from informed questions)
TEST_QUERIES = [
    "SubjectAccessReview webhook",
    "RBAC authorization policies",
    "kube-apiserver configuration flags",
    "Pod affinity and anti-affinity",
    "NetworkPolicy ingress egress rules",
    "StatefulSet persistent volume claims",
    "DaemonSet node selector tolerations",
    "ConfigMap environment variables",
    "ServiceAccount token volume projection",
    "ResourceQuota limit ranges",
]


@dataclass
class ModelStats:
    """Statistics for a single model."""
    model_name: str
    model_id: str
    vocab_size: int
    load_time_s: float
    avg_encode_time_ms: float
    min_encode_time_ms: float
    max_encode_time_ms: float
    avg_terms_per_query: float
    min_terms: int
    max_terms: int
    # Term analysis
    sample_expansions: dict[str, list[str]]  # query -> top expanded terms


def load_model(model_id: str, device: str = "cpu") -> tuple:
    """Load SPLADE model and tokenizer."""
    print(f"  Loading {model_id}...")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    
    load_time = time.time() - start
    return tokenizer, model, load_time


def encode_splade(
    text: str,
    tokenizer,
    model,
    device: str = "cpu",
    max_length: int = 256,
) -> tuple[dict[int, float], float]:
    """Encode text with SPLADE, return sparse vector and time."""
    start = time.time()
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True,
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # SPLADE aggregation
    relu_log = torch.log1p(torch.relu(logits))
    sparse_vec, _ = torch.max(relu_log, dim=1)
    sparse_vec = sparse_vec.squeeze(0).cpu().numpy()
    
    # Extract non-zero
    nonzero_indices = np.nonzero(sparse_vec)[0]
    id_weights = {
        int(idx): float(sparse_vec[idx])
        for idx in nonzero_indices
    }
    
    encode_time = (time.time() - start) * 1000  # ms
    return id_weights, encode_time


def get_top_terms(
    sparse_vec: dict[int, float],
    tokenizer,
    k: int = 15,
) -> list[tuple[str, float]]:
    """Get top-k terms by weight."""
    id_to_token = {v: k for k, v in tokenizer.vocab.items()}
    sorted_terms = sorted(sparse_vec.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(id_to_token.get(idx, f"[UNK:{idx}]"), weight) for idx, weight in sorted_terms]


def evaluate_model(
    model_name: str,
    model_id: str,
    queries: list[str],
    device: str = "cpu",
) -> ModelStats:
    """Evaluate a single model on test queries."""
    print(f"\n[{model_name}] {model_id}")
    
    # Load model
    tokenizer, model, load_time = load_model(model_id, device)
    vocab_size = tokenizer.vocab_size
    print(f"  Loaded in {load_time:.2f}s, vocab: {vocab_size}")
    
    # Encode all queries
    encode_times = []
    term_counts = []
    sample_expansions = {}
    
    for query in queries:
        sparse_vec, encode_time = encode_splade(query, tokenizer, model, device)
        encode_times.append(encode_time)
        term_counts.append(len(sparse_vec))
        
        # Store top terms for first 3 queries
        if len(sample_expansions) < 3:
            top_terms = get_top_terms(sparse_vec, tokenizer, k=15)
            sample_expansions[query] = [t[0] for t in top_terms]
    
    # Calculate stats
    stats = ModelStats(
        model_name=model_name,
        model_id=model_id,
        vocab_size=vocab_size,
        load_time_s=load_time,
        avg_encode_time_ms=np.mean(encode_times),
        min_encode_time_ms=np.min(encode_times),
        max_encode_time_ms=np.max(encode_times),
        avg_terms_per_query=np.mean(term_counts),
        min_terms=min(term_counts),
        max_terms=max(term_counts),
        sample_expansions=sample_expansions,
    )
    
    print(f"  Encode: {stats.avg_encode_time_ms:.1f}ms avg ({stats.min_encode_time_ms:.1f}-{stats.max_encode_time_ms:.1f}ms)")
    print(f"  Terms: {stats.avg_terms_per_query:.0f} avg ({stats.min_terms}-{stats.max_terms})")
    
    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return stats


def load_sample_chunks(db_path: Path, limit: int = 100) -> list[str]:
    """Load sample chunks from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT content FROM chunks LIMIT ?",
        (limit,)
    )
    chunks = [row[0] for row in cursor.fetchall()]
    conn.close()
    return chunks


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare SPLADE models")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of chunks to test (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cuda/cpu, default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/model_comparison.json",
        help="Output JSON file",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("SPLADE Model Comparison")
    print("=" * 60)
    
    # Test with technical queries
    print(f"\nTesting {len(TEST_QUERIES)} technical queries...")
    
    results = {}
    
    for model_name, model_id in MODELS.items():
        try:
            stats = evaluate_model(model_name, model_id, TEST_QUERIES, args.device)
            results[model_name] = asdict(stats)
        except Exception as e:
            print(f"  ERROR: {e}")
            results[model_name] = {"error": str(e)}
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    # Print comparison table
    print(f"\n{'Model':<20} {'Encode (ms)':<15} {'Terms':<10} {'Load (s)':<10}")
    print("-" * 55)
    
    for model_name, data in results.items():
        if "error" in data:
            print(f"{model_name:<20} ERROR: {data['error'][:30]}")
        else:
            print(
                f"{model_name:<20} "
                f"{data['avg_encode_time_ms']:<15.1f} "
                f"{data['avg_terms_per_query']:<10.0f} "
                f"{data['load_time_s']:<10.2f}"
            )
    
    print(f"\nResults saved to: {output_path}")
    
    # Show sample expansions from best model
    print("\n" + "=" * 60)
    print("SAMPLE TERM EXPANSIONS (ensembledistil)")
    print("=" * 60)
    
    if "ensembledistil" in results and "sample_expansions" in results["ensembledistil"]:
        for query, terms in results["ensembledistil"]["sample_expansions"].items():
            print(f"\nQuery: {query}")
            print(f"  Top terms: {', '.join(terms[:10])}")


if __name__ == "__main__":
    main()
