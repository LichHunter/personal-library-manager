#!/usr/bin/env python3
"""Analyze which facts consistently fail across retrieval strategies."""

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))

from strategies import Document, FixedSizeStrategy
from retrieval import create_retrieval_strategy


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def exact_match(fact: str, text: str) -> bool:
    return fact.lower() in text.lower()


def fuzzy_match(fact: str, text: str) -> bool:
    if exact_match(fact, text):
        return True
    words = fact.lower().split()
    if len(words) >= 3:
        return all(w in text.lower() for w in words)
    return False


@dataclass
class FactResult:
    query_id: str
    query: str
    fact: str
    expected_docs: list[str]
    found_by: list[str] = field(default_factory=list)
    retrieved_docs: dict[str, list[str]] = field(default_factory=dict)


def main():
    base_dir = Path(__file__).parent
    device = "cuda" if torch.cuda.is_available() else "cpu"

    docs_dir = base_dir / "corpus" / "expanded_documents"
    gt_path = base_dir / "corpus" / "ground_truth_expanded.json"

    log("Loading corpus...")
    documents = []
    for f in sorted(docs_dir.glob("*.md")):
        documents.append(
            Document(
                id=f.stem,
                title=f.stem,
                content=f.read_text(),
                metadata={"source": f.name},
            )
        )
    log(f"Loaded {len(documents)} documents")

    with open(gt_path) as f:
        ground_truth = json.load(f)
    queries = ground_truth["queries"]
    log(f"Loaded {len(queries)} queries with {ground_truth['total_key_facts']} facts")

    log("Chunking with fixed_512...")
    chunker = FixedSizeStrategy(chunk_size=512, overlap=0)
    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))
    log(f"Created {len(chunks)} chunks")

    log("Loading embedder...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    strategies_config = [
        ("semantic", {"type": "semantic"}),
        ("hybrid", {"type": "hybrid", "rrf_k": 60, "candidate_multiplier": 10}),
    ]

    fact_results: dict[tuple[str, str], FactResult] = {}

    for strategy_name, config in strategies_config:
        log(f"\nTesting {strategy_name}...")

        strategy = create_retrieval_strategy(
            strategy_type=config["type"],
        )
        strategy.set_embedder(embedder)
        strategy.index(chunks)

        for q in queries:
            query_id = q["id"]
            query_text = q["query"]
            expected_docs = q.get("expected_docs", [])
            key_facts = q.get("key_facts", [])

            retrieved = strategy.retrieve(query_text, k=10)
            retrieved_text = " ".join(c.content for c in retrieved)
            retrieved_doc_ids = list(set(c.doc_id for c in retrieved))

            for fact in key_facts:
                key = (query_id, fact)

                if key not in fact_results:
                    fact_results[key] = FactResult(
                        query_id=query_id,
                        query=query_text,
                        fact=fact,
                        expected_docs=expected_docs,
                    )

                fact_results[key].retrieved_docs[strategy_name] = retrieved_doc_ids

                if fuzzy_match(fact, retrieved_text):
                    fact_results[key].found_by.append(strategy_name)

    log("\n" + "=" * 80)
    log("ANALYSIS: Facts that FAILED across ALL strategies")
    log("=" * 80)

    always_failed = []
    sometimes_failed = []
    always_found = []

    for key, result in fact_results.items():
        n_found = len(result.found_by)
        n_strategies = len(strategies_config)

        if n_found == 0:
            always_failed.append(result)
        elif n_found < n_strategies:
            sometimes_failed.append(result)
        else:
            always_found.append(result)

    log(f"\nSummary:")
    log(f"  Always found:     {len(always_found)} facts")
    log(f"  Sometimes failed: {len(sometimes_failed)} facts")
    log(f"  Always failed:    {len(always_failed)} facts")

    log(f"\n{'=' * 80}")
    log("ALWAYS FAILED FACTS (not found by any strategy at k=10)")
    log("=" * 80)

    failed_by_query = defaultdict(list)
    for result in always_failed:
        failed_by_query[result.query_id].append(result)

    for query_id, results in sorted(failed_by_query.items()):
        r = results[0]
        log(f"\n[{query_id}] {r.query}")
        log(f"  Expected docs: {r.expected_docs}")
        log(f"  Retrieved docs (semantic): {r.retrieved_docs.get('semantic', [])}")
        for res in results:
            log(f'  MISSING FACT: "{res.fact}"')

    log(f"\n{'=' * 80}")
    log("ROOT CAUSE ANALYSIS")
    log("=" * 80)

    doc_contents = {doc.id: doc.content for doc in documents}

    facts_not_in_docs = []
    facts_in_docs_but_missed = []

    for result in always_failed:
        fact = result.fact
        expected_docs = result.expected_docs

        fact_found_in_doc = False
        for doc_id in expected_docs:
            if doc_id in doc_contents:
                if fuzzy_match(fact, doc_contents[doc_id]):
                    fact_found_in_doc = True
                    break

        if fact_found_in_doc:
            facts_in_docs_but_missed.append(result)
        else:
            facts_not_in_docs.append(result)

    log(f"\nOf {len(always_failed)} always-failed facts:")
    log(f"  Fact NOT in expected docs (ground truth issue): {len(facts_not_in_docs)}")
    log(
        f"  Fact IS in docs but not retrieved (retrieval issue): {len(facts_in_docs_but_missed)}"
    )

    if facts_not_in_docs:
        log(f"\n--- GROUND TRUTH ISSUES (fact not in expected docs) ---")
        for r in facts_not_in_docs:
            log(f'  [{r.query_id}] Fact "{r.fact}" not found in {r.expected_docs}')

    if facts_in_docs_but_missed:
        log(f"\n--- RETRIEVAL ISSUES (fact in docs but not retrieved) ---")
        for r in facts_in_docs_but_missed:
            log(
                f'  [{r.query_id}] Fact "{r.fact}" is in {r.expected_docs} but chunks not retrieved'
            )
            log(f"    Retrieved instead: {r.retrieved_docs.get('semantic', [])[:5]}")

    output = {
        "summary": {
            "total_facts": len(fact_results),
            "always_found": len(always_found),
            "sometimes_failed": len(sometimes_failed),
            "always_failed": len(always_failed),
            "ground_truth_issues": len(facts_not_in_docs),
            "retrieval_issues": len(facts_in_docs_but_missed),
        },
        "always_failed_facts": [
            {
                "query_id": r.query_id,
                "query": r.query,
                "fact": r.fact,
                "expected_docs": r.expected_docs,
                "retrieved_docs": r.retrieved_docs.get("semantic", []),
                "issue_type": "ground_truth" if r in facts_not_in_docs else "retrieval",
            }
            for r in always_failed
        ],
    }

    output_path = base_dir / "results" / "failed_facts_analysis.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    log(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
