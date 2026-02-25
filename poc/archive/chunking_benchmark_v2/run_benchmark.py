#!/usr/bin/env python3
"""Comprehensive Retrieval Benchmark Runner.

Loads configuration from YAML and executes the full benchmark matrix:
- Multiple chunking strategies
- Multiple embedding models
- Multiple retrieval strategies
- Multiple rerankers and LLMs

Usage:
    python run_benchmark.py                    # Use default config.yaml
    python run_benchmark.py --config custom.yaml
    python run_benchmark.py --config config.yaml --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from strategies import (
    Document,
    Chunk,
    ChunkingStrategy,
    FixedSizeStrategy,
    RecursiveSplitterStrategy,
    ClusterSemanticStrategy,
    ParagraphHeadingStrategy,
    HeadingBasedStrategy,
    HeadingLimitedStrategy,
    HierarchicalStrategy,
    HeadingParagraphStrategy,
    ParagraphStrategy,
)
from retrieval import (
    create_retrieval_strategy,
    parse_document_sections,
    StructuredDocument,
    RetrievalStrategy,
)
from logger import BenchmarkLogger, set_logger


# =============================================================================
# LOGGING (module-level logger, set in main)
# =============================================================================

logger: BenchmarkLogger | None = None


def log(msg: str, level: str = "INFO"):
    if logger:
        getattr(logger, level.lower(), logger.info)(msg)
    else:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {msg}", flush=True)


def log_section(title: str):
    if logger:
        logger.section(title)
    else:
        print(f"\n{'=' * 80}", flush=True)
        log(title)
        print("=" * 80, flush=True)


# =============================================================================
# CHUNKING STRATEGY FACTORY
# =============================================================================

CHUNKING_STRATEGIES = {
    "fixed_size": FixedSizeStrategy,
    "recursive_splitter": RecursiveSplitterStrategy,
    "cluster_semantic": ClusterSemanticStrategy,
    "paragraph_heading": ParagraphHeadingStrategy,
    "heading_based": HeadingBasedStrategy,
    "heading_limited": HeadingLimitedStrategy,
    "hierarchical": HierarchicalStrategy,
    "heading_paragraph": HeadingParagraphStrategy,
    "paragraphs": ParagraphStrategy,
}


def create_chunking_strategy(config: dict) -> ChunkingStrategy:
    """Create a chunking strategy from config."""
    strategy_type = config["type"]
    params = config.get("params", {})

    if strategy_type not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy_type}")

    return CHUNKING_STRATEGIES[strategy_type](**params)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_corpus(
    config: dict, base_dir: Path
) -> tuple[list[Document], list[StructuredDocument]]:
    """Load corpus documents."""
    docs_dir = base_dir / config["corpus"]["documents_dir"]
    metadata_path = base_dir / config["corpus"]["metadata_file"]

    with open(metadata_path) as f:
        metadata = json.load(f)

    flat_docs = []
    structured_docs = []

    for doc_meta in metadata:
        doc_path = docs_dir / doc_meta["filename"]
        if doc_path.exists():
            doc = Document(
                id=doc_meta["id"],
                title=doc_meta["title"],
                content=doc_path.read_text(),
                path=str(doc_path),
            )
            flat_docs.append(doc)
            structured_docs.append(parse_document_sections(doc))

    return flat_docs, structured_docs


def load_queries(config: dict, base_dir: Path) -> list[dict]:
    """Load ground truth queries."""
    gt_path = base_dir / config["corpus"]["ground_truth_file"]
    with open(gt_path) as f:
        data = json.load(f)
    return data.get("queries", data)


# =============================================================================
# EVALUATION
# =============================================================================


def exact_match(fact: str, text: str) -> bool:
    """Exact substring match."""
    return fact.lower() in text.lower()


def fuzzy_match(fact: str, text: str) -> bool:
    """Word-level fuzzy match."""
    text_lower = text.lower()
    fact_lower = fact.lower()

    if fact_lower in text_lower:
        return True

    words = fact_lower.split()
    if len(words) >= 2 and all(w in text_lower for w in words):
        return True

    return False


MATCH_FUNCTIONS = {
    "exact_match": exact_match,
    "fuzzy_match": fuzzy_match,
}


DIMENSIONS = ["original", "synonym", "problem", "casual", "contextual", "negation"]


def get_query_text(q: dict) -> str:
    return q.get("query") or q.get("original_query", "")


def evaluate_single_query(
    strategy: RetrievalStrategy,
    query_text: str,
    key_facts: list[str],
    expected_docs: list[str],
    k: int,
    match_fn,
) -> dict:
    start = time.perf_counter()
    retrieved = strategy.retrieve(query_text, k=k)
    latency_ms = (time.perf_counter() - start) * 1000

    combined_text = " ".join(c.content for c in retrieved)

    found_facts = [f for f in key_facts if match_fn(f, combined_text)]
    missed_facts = [f for f in key_facts if not match_fn(f, combined_text)]

    retrieved_chunks = []
    for chunk in retrieved:
        retrieved_chunks.append(
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.id,
                "content": chunk.content,
                "score": getattr(chunk, "score", None),
            }
        )

    if logger and logger.min_level <= logger.LEVELS.get("TRACE", -1):
        logger.trace(f"=== FACT COVERAGE ANALYSIS ===")
        logger.trace(f"Query: {query_text}")
        logger.trace(f"Total key facts: {len(key_facts)}")
        logger.trace(f"Found facts: {len(found_facts)}/{len(key_facts)}")

        for i, fact in enumerate(found_facts, 1):
            logger.trace(f"  FOUND[{i}] {fact}")

        if missed_facts:
            logger.trace(f"Missed facts: {len(missed_facts)}/{len(key_facts)}")
            for i, fact in enumerate(missed_facts, 1):
                logger.trace(f"  MISSED[{i}] {fact}")

                if hasattr(strategy, "chunks") and strategy.chunks:  # type: ignore
                    for chunk in strategy.chunks:  # type: ignore
                        if match_fn(fact, chunk.content):
                            logger.trace(
                                f"    -> Found in chunk_id={chunk.id} (rank not in top-{k})"
                            )
                            break

        logger.trace(f"=== END FACT COVERAGE ===")

    return {
        "key_facts": key_facts,
        "found_facts": found_facts,
        "missed_facts": missed_facts,
        "coverage": len(found_facts) / len(key_facts) if key_facts else 0,
        "expected_docs": expected_docs,
        "retrieved_chunks": retrieved_chunks,
        "latency_ms": latency_ms,
    }


def evaluate_strategy(
    strategy: RetrievalStrategy,
    queries: list[dict],
    k: int,
    metric_name: str,
) -> dict:
    match_fn = MATCH_FUNCTIONS.get(metric_name, exact_match)

    per_query_results = []

    for q in queries:
        query_id = q.get("id", "unknown")
        key_facts = q.get("key_facts", [])
        expected_docs = q.get("expected_docs", [])
        original_query = get_query_text(q)
        human_queries = q.get("human_queries", [])

        original_result = evaluate_single_query(
            strategy, original_query, key_facts, expected_docs, k, match_fn
        )
        original_result["query_id"] = query_id
        original_result["dimension"] = "original"
        original_result["query"] = original_query
        per_query_results.append(original_result)

        for hq in human_queries:
            hq_result = evaluate_single_query(
                strategy, hq["query"], key_facts, expected_docs, k, match_fn
            )
            hq_result["query_id"] = query_id
            hq_result["dimension"] = hq.get("dimension", "unknown")
            hq_result["query"] = hq["query"]
            per_query_results.append(hq_result)

    dimension_stats = {}
    for dim in DIMENSIONS:
        dim_results = [r for r in per_query_results if r["dimension"] == dim]
        if dim_results:
            total_facts = sum(len(r["key_facts"]) for r in dim_results)
            found_facts = sum(len(r["found_facts"]) for r in dim_results)
            latencies = [r["latency_ms"] for r in dim_results]
            dimension_stats[dim] = {
                "coverage": found_facts / total_facts if total_facts else 0,
                "found": found_facts,
                "total": total_facts,
                "avg_latency_ms": np.mean(latencies),
                "p95_latency_ms": np.percentile(latencies, 95) if latencies else 0,
            }

    return {
        "aggregate": dimension_stats,
        "per_query": per_query_results,
    }


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================


class ModelCache:
    """Cache for loaded models to avoid reloading."""

    def __init__(self, device: str):
        self.device = device
        self.embedders: dict[str, SentenceTransformer] = {}
        self.rerankers: dict[str, CrossEncoder] = {}

    def get_embedder(self, name: str) -> SentenceTransformer:
        if name not in self.embedders:
            log(f"Loading embedder: {name}")
            self.embedders[name] = SentenceTransformer(name, device=self.device)
        return self.embedders[name]

    def get_reranker(self, name: str) -> CrossEncoder:
        if name not in self.rerankers:
            log(f"Loading reranker: {name}")
            self.rerankers[name] = CrossEncoder(name, device=self.device)
        return self.rerankers[name]

    def clear(self):
        """Clear cache to free memory."""
        self.embedders.clear()
        self.rerankers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================


def check_ollama_available() -> bool:
    """Check if Ollama is available."""
    import subprocess

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def check_claude_available() -> bool:
    """Check if Claude is available via OpenCode auth."""
    import os
    from pathlib import Path

    auth_path = Path(os.path.expanduser("~/.local/share/opencode/auth.json"))
    if not auth_path.exists():
        return False

    try:
        import json

        with open(auth_path) as f:
            data = json.load(f)
        return "anthropic" in data
    except Exception:
        return False


def run_benchmark(
    config: dict,
    base_dir: Path,
    dry_run: bool = False,
) -> dict:
    log_section("BENCHMARK CONFIGURATION")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ollama_available = check_ollama_available()
    claude_available = check_claude_available()
    llm_available = ollama_available or claude_available

    if logger:
        logger.metric("device", device)
        logger.metric("ollama_available", ollama_available)
        logger.metric("claude_available", claude_available)
        logger.metric("llm_available", llm_available)

    log(f"Device: {device}")
    log(f"Ollama available: {ollama_available}")
    log(f"Claude available: {claude_available}")
    log(f"LLM available: {llm_available}")

    log("Loading corpus...")
    flat_docs, structured_docs = load_corpus(config, base_dir)
    queries = load_queries(config, base_dir)
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    has_human_queries = any(q.get("human_queries") for q in queries)

    if logger:
        logger.metric("documents", len(flat_docs))
        logger.metric("queries", len(queries))
        logger.metric("total_facts", total_facts)
        logger.metric("has_human_queries", has_human_queries)

    log(f"Loaded {len(flat_docs)} documents")
    log(f"Loaded {len(queries)} queries")
    log(f"Total key facts: {total_facts}")
    log(f"Human query variations: {has_human_queries}")

    enabled_embedders = [
        e for e in config["embedding_models"] if e.get("enabled", True)
    ]
    enabled_rerankers = [r for r in config["reranker_models"] if r.get("enabled", True)]
    enabled_llms = [l for l in config["llm_models"] if l.get("enabled", True)]
    enabled_chunking = [
        c for c in config["chunking_strategies"] if c.get("enabled", True)
    ]
    enabled_retrieval = [
        r for r in config["retrieval_strategies"] if r.get("enabled", True)
    ]

    if logger:
        logger.info(f"Enabled embedders: {len(enabled_embedders)}")
        logger.info(f"Enabled rerankers: {len(enabled_rerankers)}")
        logger.info(f"Enabled LLMs: {len(enabled_llms)}")
        logger.info(f"Enabled chunking strategies: {len(enabled_chunking)}")
        logger.info(f"Enabled retrieval strategies: {len(enabled_retrieval)}")

    k_values = config["evaluation"]["k_values"]
    metrics = config["evaluation"]["metrics"]

    total_combinations = 0
    for retrieval_cfg in enabled_retrieval:
        n_embedders = len(enabled_embedders)
        n_rerankers = (
            len(enabled_rerankers) if retrieval_cfg.get("requires_reranker") else 1
        )
        n_llms = len(enabled_llms) if retrieval_cfg.get("requires_llm") else 1
        n_chunking = (
            1
            if retrieval_cfg.get("requires_structured_docs")
            else len(enabled_chunking)
        )
        total_combinations += n_embedders * n_rerankers * n_llms * n_chunking

    total_evaluations = total_combinations * len(k_values) * len(metrics)

    if logger:
        logger.metric("total_combinations", total_combinations)
        logger.metric("total_evaluations", total_evaluations)

    log(f"Total strategy combinations: {total_combinations}")
    log(f"Total evaluations (with k and metrics): {total_evaluations}")

    if dry_run:
        log("DRY RUN - not executing benchmark")
        return {}

    model_cache = ModelCache(device)

    all_results: list[dict] = []
    combination_idx = 0

    for retrieval_cfg in enabled_retrieval:
        retrieval_type = retrieval_cfg["type"]
        retrieval_name = retrieval_cfg["name"]
        requires_reranker = retrieval_cfg.get("requires_reranker", False)
        requires_llm = retrieval_cfg.get("requires_llm", False)
        requires_structured = retrieval_cfg.get("requires_structured_docs", False)

        log_section(f"RETRIEVAL STRATEGY: {retrieval_name}")

        if requires_structured:
            chunking_configs = [{"name": "structured", "type": None}]
        else:
            chunking_configs = enabled_chunking

        reranker_configs = enabled_rerankers if requires_reranker else [None]
        llm_configs = enabled_llms if requires_llm else [None]

        for embedder_cfg in enabled_embedders:
            embedder_name = embedder_cfg["name"]
            use_prefix = embedder_cfg.get("use_prefix", False)
            embedder = model_cache.get_embedder(embedder_name)

            for reranker_cfg in reranker_configs:
                reranker_name = reranker_cfg["name"] if reranker_cfg else None
                reranker = (
                    model_cache.get_reranker(reranker_name) if reranker_name else None
                )

                for llm_cfg in llm_configs:
                    llm_name = llm_cfg["name"] if llm_cfg else None

                    if llm_name and not llm_available:
                        log(
                            f"Skipping {retrieval_name} with {llm_name} (No LLM available)"
                        )
                        continue

                    for chunking_cfg in chunking_configs:
                        chunking_name = chunking_cfg["name"]
                        combination_idx += 1

                        parts = [retrieval_name, embedder_name.split("/")[-1]]
                        if reranker_name:
                            parts.append(reranker_name.split("/")[-1])
                        if llm_name:
                            parts.append(llm_name.replace(":", "_"))
                        if chunking_name != "structured":
                            parts.append(chunking_name)
                        full_name = "_".join(parts)

                        if logger:
                            logger.progress(
                                combination_idx, total_combinations, full_name
                            )
                        else:
                            log(f"[{combination_idx}/{total_combinations}] {full_name}")

                        try:
                            if chunking_name == "structured":
                                chunks = None
                            else:
                                chunker = create_chunking_strategy(chunking_cfg)
                                if hasattr(chunker, "set_embedder"):
                                    chunker.set_embedder(embedder)
                                chunks = chunker.chunk_many(flat_docs)
                                log(f"  Created {len(chunks)} chunks")

                            strategy = create_retrieval_strategy(
                                retrieval_type,
                                name=full_name,
                                **retrieval_cfg.get("params", {}),
                            )
                            strategy.set_embedder(embedder, use_prefix)

                            if hasattr(strategy, "set_reranker") and reranker:
                                strategy.set_reranker(reranker)
                            strategy_llm = retrieval_cfg.get("params", {}).get(
                                "llm_model"
                            )
                            effective_llm = (
                                strategy_llm or llm_name
                            )  # prefer strategy-specific over global
                            if hasattr(strategy, "set_llm_model") and effective_llm:
                                strategy.set_llm_model(effective_llm)

                            index_start = time.perf_counter()
                            strategy.index(
                                chunks=chunks,
                                documents=flat_docs,
                                structured_docs=structured_docs
                                if requires_structured
                                else None,
                            )
                            index_time = time.perf_counter() - index_start
                            log(f"  Indexed in {index_time:.2f}s")

                            stats = strategy.get_index_stats()
                            num_chunks = stats.get(
                                "num_chunks", len(chunks) if chunks else 0
                            )

                            for k in k_values:
                                for metric in metrics:
                                    eval_result = evaluate_strategy(
                                        strategy, queries, k, metric
                                    )

                                    orig_stats = eval_result["aggregate"].get(
                                        "original", {}
                                    )
                                    log(
                                        f"  k={k} {metric}: {orig_stats.get('coverage', 0):.1%} "
                                        f"({orig_stats.get('found', 0)}/{orig_stats.get('total', 0)})"
                                    )

                                    if has_human_queries:
                                        for dim in DIMENSIONS[1:]:
                                            dim_stats = eval_result["aggregate"].get(
                                                dim, {}
                                            )
                                            if dim_stats:
                                                log(
                                                    f"    {dim}: {dim_stats.get('coverage', 0):.1%}"
                                                )

                                    all_results.append(
                                        {
                                            "strategy": retrieval_name,
                                            "chunking": chunking_name,
                                            "embedder": embedder_name,
                                            "reranker": reranker_name,
                                            "llm": llm_name,
                                            "k": k,
                                            "metric": metric,
                                            "num_chunks": num_chunks,
                                            "index_time_s": index_time,
                                            "aggregate": eval_result["aggregate"],
                                            "per_query": eval_result["per_query"],
                                        }
                                    )

                        except Exception as e:
                            log(f"  ERROR: {e}", level="ERROR")
                            import traceback

                            traceback.print_exc()
                            continue

        model_cache.clear()

    return {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d_%H%M%S"),
            "num_documents": len(flat_docs),
            "num_queries": len(queries),
            "total_facts": total_facts,
            "has_human_queries": has_human_queries,
        },
        "evaluations": all_results,
    }


def save_results(benchmark_results: dict, config: dict, base_dir: Path) -> Path:
    results_base = base_dir / config["output"]["results_dir"]
    timestamp = benchmark_results["metadata"]["timestamp"]
    run_dir = results_base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    log(f"Saved full results to {results_path}")

    summary = generate_summary(benchmark_results)
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"Saved summary to {summary_path}")

    return run_dir


def generate_summary(benchmark_results: dict) -> dict:
    evaluations = benchmark_results.get("evaluations", [])
    if not evaluations:
        return {}

    summary = {
        "metadata": benchmark_results.get("metadata", {}),
        "dimension_comparison": {},
        "best_configurations": {},
        "degradation_analysis": [],
    }

    all_dimensions = set()
    for ev in evaluations:
        all_dimensions.update(ev.get("aggregate", {}).keys())

    for dim in sorted(all_dimensions):
        dim_coverages = []
        for ev in evaluations:
            agg = ev.get("aggregate", {}).get(dim, {})
            if agg:
                dim_coverages.append(agg.get("coverage", 0))
        if dim_coverages:
            summary["dimension_comparison"][dim] = {
                "avg_coverage": np.mean(dim_coverages),
                "min_coverage": min(dim_coverages),
                "max_coverage": max(dim_coverages),
            }

    if "original" in summary["dimension_comparison"]:
        orig_cov = summary["dimension_comparison"]["original"]["avg_coverage"]
        for dim, stats in summary["dimension_comparison"].items():
            if dim != "original":
                delta = stats["avg_coverage"] - orig_cov
                summary["degradation_analysis"].append(
                    {
                        "dimension": dim,
                        "avg_coverage": stats["avg_coverage"],
                        "delta_from_original": delta,
                        "percent_degradation": (delta / orig_cov * 100)
                        if orig_cov
                        else 0,
                    }
                )
        summary["degradation_analysis"].sort(key=lambda x: x["delta_from_original"])

    for ev in evaluations:
        key = f"{ev['strategy']}_{ev['chunking']}_{ev['embedder']}"
        orig_agg = ev.get("aggregate", {}).get("original", {})
        if orig_agg:
            summary["best_configurations"][key] = {
                "coverage": orig_agg.get("coverage", 0),
                "k": ev.get("k"),
                "metric": ev.get("metric"),
            }

    return summary


# =============================================================================
# MAIN
# =============================================================================


def main():
    global logger

    parser = argparse.ArgumentParser(
        description="Run comprehensive retrieval benchmark"
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Enable DEBUG level logging for detailed provider output",
    )
    parser.add_argument(
        "--trace",
        "-t",
        action="store_true",
        help="Enable TRACE level logging (shows full input/output text)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    config_path = base_dir / args.config

    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    results_dir = base_dir / "results" / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    log_level = "TRACE" if args.trace else ("DEBUG" if args.debug else "INFO")
    logger = BenchmarkLogger(
        log_dir=results_dir,
        log_file="benchmark.log",
        console=True,
        min_level=log_level,
    )
    set_logger(logger)

    if not config_path.exists():
        log(f"Config file not found: {config_path}", level="ERROR")
        logger.close()
        sys.exit(1)

    log(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    log_section("COMPREHENSIVE RETRIEVAL BENCHMARK")

    with logger.timer("total_benchmark"):
        benchmark_results = run_benchmark(config, base_dir, dry_run=args.dry_run)

    if benchmark_results and benchmark_results.get("evaluations"):
        benchmark_results["metadata"]["timestamp"] = timestamp

        log_section("SAVING RESULTS")
        save_results(benchmark_results, config, base_dir)

        log_section("BENCHMARK COMPLETE")
        log(f"Total evaluations: {len(benchmark_results['evaluations'])}")

        summary = generate_summary(benchmark_results)
        if summary.get("dimension_comparison"):
            rows = []
            orig_cov = (
                summary["dimension_comparison"]
                .get("original", {})
                .get("avg_coverage", 0)
            )
            for dim, stats in sorted(summary["dimension_comparison"].items()):
                delta = stats["avg_coverage"] - orig_cov if dim != "original" else 0
                rows.append(
                    [
                        dim,
                        f"{stats['avg_coverage']:.1%}",
                        f"{delta:+.1%}" if dim != "original" else "-",
                    ]
                )
            logger.table(
                ["Dimension", "Coverage", "Delta"], rows, "Coverage by Query Dimension"
            )

    logger.summary()
    logger.close()


if __name__ == "__main__":
    main()
