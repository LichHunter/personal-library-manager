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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from strategies import (
    Document, Chunk, ChunkingStrategy,
    FixedSizeStrategy, RecursiveSplitterStrategy, ClusterSemanticStrategy,
    ParagraphHeadingStrategy, HeadingBasedStrategy, HeadingLimitedStrategy,
    HierarchicalStrategy, HeadingParagraphStrategy, ParagraphStrategy,
)
from retrieval import (
    create_retrieval_strategy, parse_document_sections,
    StructuredDocument, RetrievalStrategy,
)


# =============================================================================
# LOGGING
# =============================================================================

def log(msg: str, level: str = "INFO"):
    """Simple timestamped logging."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)


def log_section(title: str):
    """Log a section header."""
    print(f"\n{'='*80}", flush=True)
    log(title)
    print('='*80, flush=True)


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

def load_corpus(config: dict, base_dir: Path) -> tuple[list[Document], list[StructuredDocument]]:
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


@dataclass
class EvaluationResult:
    """Result of evaluating a strategy."""
    strategy_name: str
    chunking_name: str
    embedder_name: str
    reranker_name: Optional[str]
    llm_name: Optional[str]
    k: int
    metric: str
    coverage: float
    found: int
    total: int
    avg_latency_ms: float
    p95_latency_ms: float
    num_chunks: int
    index_time_s: float


def evaluate_strategy(
    strategy: RetrievalStrategy,
    queries: list[dict],
    k: int,
    metric_name: str,
) -> dict:
    """Evaluate a retrieval strategy on queries."""
    match_fn = MATCH_FUNCTIONS.get(metric_name, exact_match)
    
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    found = 0
    latencies = []
    
    for q in queries:
        start = time.perf_counter()
        retrieved = strategy.retrieve(q["query"], k=k)
        latencies.append(time.perf_counter() - start)
        
        text = " ".join(c.content for c in retrieved)
        for fact in q.get("key_facts", []):
            if match_fn(fact, text):
                found += 1
    
    return {
        "coverage": found / total_facts if total_facts else 0,
        "found": found,
        "total": total_facts,
        "avg_latency_ms": np.mean(latencies) * 1000,
        "p95_latency_ms": np.percentile(latencies, 95) * 1000 if latencies else 0,
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


def run_benchmark(config: dict, base_dir: Path, dry_run: bool = False) -> list[EvaluationResult]:
    """Run the full benchmark matrix."""
    
    log_section("BENCHMARK CONFIGURATION")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    
    ollama_available = check_ollama_available()
    log(f"Ollama available: {ollama_available}")
    
    # Load data
    log("Loading corpus...")
    flat_docs, structured_docs = load_corpus(config, base_dir)
    log(f"Loaded {len(flat_docs)} documents")
    
    queries = load_queries(config, base_dir)
    log(f"Loaded {len(queries)} queries")
    
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    log(f"Total key facts: {total_facts}")
    
    # Get enabled items from config
    enabled_embedders = [e for e in config["embedding_models"] if e.get("enabled", True)]
    enabled_rerankers = [r for r in config["reranker_models"] if r.get("enabled", True)]
    enabled_llms = [l for l in config["llm_models"] if l.get("enabled", True)]
    enabled_chunking = [c for c in config["chunking_strategies"] if c.get("enabled", True)]
    enabled_retrieval = [r for r in config["retrieval_strategies"] if r.get("enabled", True)]
    
    log(f"Enabled embedders: {len(enabled_embedders)}")
    log(f"Enabled rerankers: {len(enabled_rerankers)}")
    log(f"Enabled LLMs: {len(enabled_llms)}")
    log(f"Enabled chunking strategies: {len(enabled_chunking)}")
    log(f"Enabled retrieval strategies: {len(enabled_retrieval)}")
    
    k_values = config["evaluation"]["k_values"]
    metrics = config["evaluation"]["metrics"]
    
    # Calculate total combinations
    total_combinations = 0
    for retrieval_cfg in enabled_retrieval:
        n_embedders = len(enabled_embedders)
        n_rerankers = len(enabled_rerankers) if retrieval_cfg.get("requires_reranker") else 1
        n_llms = len(enabled_llms) if retrieval_cfg.get("requires_llm") else 1
        n_chunking = 1 if retrieval_cfg.get("requires_structured_docs") else len(enabled_chunking)
        total_combinations += n_embedders * n_rerankers * n_llms * n_chunking
    
    total_evaluations = total_combinations * len(k_values) * len(metrics)
    log(f"Total strategy combinations: {total_combinations}")
    log(f"Total evaluations (with k and metrics): {total_evaluations}")
    
    if dry_run:
        log("DRY RUN - not executing benchmark")
        return []
    
    # Initialize model cache
    model_cache = ModelCache(device)
    
    results: list[EvaluationResult] = []
    combination_idx = 0
    
    # Main benchmark loop
    for retrieval_cfg in enabled_retrieval:
        retrieval_type = retrieval_cfg["type"]
        retrieval_name = retrieval_cfg["name"]
        requires_reranker = retrieval_cfg.get("requires_reranker", False)
        requires_llm = retrieval_cfg.get("requires_llm", False)
        requires_structured = retrieval_cfg.get("requires_structured_docs", False)
        
        log_section(f"RETRIEVAL STRATEGY: {retrieval_name}")
        
        # Determine which chunking strategies to use
        if requires_structured:
            # Hierarchical strategies do their own chunking
            chunking_configs = [{"name": "structured", "type": None}]
        else:
            chunking_configs = enabled_chunking
        
        # Determine rerankers to use
        reranker_configs = enabled_rerankers if requires_reranker else [None]
        
        # Determine LLMs to use
        llm_configs = enabled_llms if requires_llm else [None]
        
        for embedder_cfg in enabled_embedders:
            embedder_name = embedder_cfg["name"]
            use_prefix = embedder_cfg.get("use_prefix", False)
            
            embedder = model_cache.get_embedder(embedder_name)
            
            for reranker_cfg in reranker_configs:
                reranker_name = reranker_cfg["name"] if reranker_cfg else None
                reranker = model_cache.get_reranker(reranker_name) if reranker_name else None
                
                for llm_cfg in llm_configs:
                    llm_name = llm_cfg["name"] if llm_cfg else None
                    
                    # Skip LLM strategies if Ollama not available
                    if llm_name and not ollama_available:
                        log(f"Skipping {retrieval_name} with {llm_name} (Ollama not available)")
                        continue
                    
                    for chunking_cfg in chunking_configs:
                        chunking_name = chunking_cfg["name"]
                        combination_idx += 1
                        
                        # Build strategy name
                        parts = [retrieval_name, embedder_name.split("/")[-1]]
                        if reranker_name:
                            parts.append(reranker_name.split("/")[-1])
                        if llm_name:
                            parts.append(llm_name.replace(":", "_"))
                        if chunking_name != "structured":
                            parts.append(chunking_name)
                        
                        full_name = "_".join(parts)
                        
                        log(f"[{combination_idx}/{total_combinations}] {full_name}")
                        
                        try:
                            # Create chunks
                            if chunking_name == "structured":
                                chunks = None
                            else:
                                chunker = create_chunking_strategy(chunking_cfg)
                                # Some chunkers (like ClusterSemanticStrategy) need embedder
                                if hasattr(chunker, "set_embedder"):
                                    chunker.set_embedder(embedder)
                                chunks = chunker.chunk_many(flat_docs)
                                log(f"  Created {len(chunks)} chunks")
                            
                            # Create retrieval strategy
                            strategy = create_retrieval_strategy(
                                retrieval_type,
                                name=full_name,
                                **retrieval_cfg.get("params", {})
                            )
                            
                            # Configure strategy
                            strategy.set_embedder(embedder, use_prefix)
                            
                            if hasattr(strategy, "set_reranker") and reranker:
                                strategy.set_reranker(reranker)
                            
                            if hasattr(strategy, "set_llm_model") and llm_name:
                                strategy.set_llm_model(llm_name)
                            
                            # Index
                            index_start = time.perf_counter()
                            strategy.index(
                                chunks=chunks,
                                documents=flat_docs,
                                structured_docs=structured_docs if requires_structured else None,
                            )
                            index_time = time.perf_counter() - index_start
                            log(f"  Indexed in {index_time:.2f}s")
                            
                            # Get chunk count
                            stats = strategy.get_index_stats()
                            num_chunks = stats.get("num_chunks", len(chunks) if chunks else 0)
                            
                            # Evaluate for each k and metric
                            for k in k_values:
                                for metric in metrics:
                                    eval_result = evaluate_strategy(strategy, queries, k, metric)
                                    
                                    results.append(EvaluationResult(
                                        strategy_name=retrieval_name,
                                        chunking_name=chunking_name,
                                        embedder_name=embedder_name,
                                        reranker_name=reranker_name,
                                        llm_name=llm_name,
                                        k=k,
                                        metric=metric,
                                        coverage=eval_result["coverage"],
                                        found=eval_result["found"],
                                        total=eval_result["total"],
                                        avg_latency_ms=eval_result["avg_latency_ms"],
                                        p95_latency_ms=eval_result["p95_latency_ms"],
                                        num_chunks=num_chunks,
                                        index_time_s=index_time,
                                    ))
                                    
                                    log(f"  k={k} {metric}: {eval_result['coverage']:.1%} ({eval_result['found']}/{eval_result['total']})")
                        
                        except Exception as e:
                            log(f"  ERROR: {e}", level="ERROR")
                            continue
        
        # Clear cache between retrieval strategies to manage memory
        model_cache.clear()
    
    return results


def save_results(results: list[EvaluationResult], config: dict, base_dir: Path):
    """Save benchmark results."""
    results_dir = base_dir / config["output"]["results_dir"]
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    
    # Convert to dict for JSON
    results_data = [
        {
            "strategy": r.strategy_name,
            "chunking": r.chunking_name,
            "embedder": r.embedder_name,
            "reranker": r.reranker_name,
            "llm": r.llm_name,
            "k": r.k,
            "metric": r.metric,
            "coverage": r.coverage,
            "found": r.found,
            "total": r.total,
            "avg_latency_ms": r.avg_latency_ms,
            "p95_latency_ms": r.p95_latency_ms,
            "num_chunks": r.num_chunks,
            "index_time_s": r.index_time_s,
        }
        for r in results
    ]
    
    # Save detailed results
    if config["output"].get("save_detailed_results", True):
        detailed_path = results_dir / f"{timestamp}_detailed.json"
        with open(detailed_path, "w") as f:
            json.dump(results_data, f, indent=2)
        log(f"Saved detailed results to {detailed_path}")
    
    # Save summary
    if config["output"].get("save_summary", True):
        summary = generate_summary(results)
        summary_path = results_dir / f"{timestamp}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log(f"Saved summary to {summary_path}")
    
    return results_dir / f"{timestamp}_detailed.json"


def generate_summary(results: list[EvaluationResult]) -> dict:
    """Generate a summary of benchmark results."""
    if not results:
        return {}
    
    # Group by k and metric
    summary = {
        "best_by_k": {},
        "best_by_strategy_type": {},
        "best_by_embedder": {},
    }
    
    # Find best for each k value (using exact_match)
    k_values = set(r.k for r in results)
    for k in k_values:
        k_results = [r for r in results if r.k == k and r.metric == "exact_match"]
        if k_results:
            best = max(k_results, key=lambda r: r.coverage)
            summary["best_by_k"][f"k={k}"] = {
                "strategy": best.strategy_name,
                "chunking": best.chunking_name,
                "embedder": best.embedder_name,
                "coverage": best.coverage,
                "found": best.found,
                "total": best.total,
            }
    
    # Best by strategy type
    strategy_types = set(r.strategy_name for r in results)
    for st in strategy_types:
        st_results = [r for r in results if r.strategy_name == st and r.metric == "exact_match" and r.k == 5]
        if st_results:
            best = max(st_results, key=lambda r: r.coverage)
            summary["best_by_strategy_type"][st] = {
                "chunking": best.chunking_name,
                "embedder": best.embedder_name,
                "coverage": best.coverage,
            }
    
    # Best by embedder
    embedders = set(r.embedder_name for r in results)
    for emb in embedders:
        emb_results = [r for r in results if r.embedder_name == emb and r.metric == "exact_match" and r.k == 5]
        if emb_results:
            best = max(emb_results, key=lambda r: r.coverage)
            summary["best_by_embedder"][emb] = {
                "strategy": best.strategy_name,
                "chunking": best.chunking_name,
                "coverage": best.coverage,
            }
    
    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive retrieval benchmark")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Show what would be run without executing"
    )
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    config_path = base_dir / args.config
    
    if not config_path.exists():
        log(f"Config file not found: {config_path}", level="ERROR")
        sys.exit(1)
    
    log(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    log_section("COMPREHENSIVE RETRIEVAL BENCHMARK")
    
    results = run_benchmark(config, base_dir, dry_run=args.dry_run)
    
    if results:
        log_section("SAVING RESULTS")
        output_path = save_results(results, config, base_dir)
        
        log_section("BENCHMARK COMPLETE")
        log(f"Total evaluations: {len(results)}")
        
        # Print top results
        exact_k5 = [r for r in results if r.metric == "exact_match" and r.k == 5]
        if exact_k5:
            top5 = sorted(exact_k5, key=lambda r: r.coverage, reverse=True)[:5]
            log("\nTop 5 strategies (exact match, k=5):")
            for i, r in enumerate(top5, 1):
                name = f"{r.strategy_name}+{r.chunking_name}+{r.embedder_name.split('/')[-1]}"
                log(f"  {i}. {name}: {r.coverage:.1%}")


if __name__ == "__main__":
    main()
