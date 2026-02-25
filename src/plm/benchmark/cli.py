from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from plm.benchmark.loader import load_questions
from plm.benchmark.metrics import calculate_metrics
from plm.benchmark.report import generate_report
from plm.benchmark.runner import BenchmarkRunner, RunnerConfig
from plm.shared.logger import PipelineLogger, set_logger


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation against search service"
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(__file__).parent / "corpus",
        help="Directory containing benchmark question files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["needle", "realistic", "informed"],
        default=None,
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--service-url",
        default="http://localhost:8000",
        help="Search service URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--use-rewrite",
        action="store_true",
        help="Enable query rewriting",
    )
    parser.add_argument(
        "--use-rerank",
        action="store_true",
        help="Enable cross-encoder reranking",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).parent / "logs",
        help="Base directory for all output (default: src/plm/benchmark/logs)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose console output (DEBUG level)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to run (for testing)",
    )
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.logs_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log = PipelineLogger(
        log_file=run_dir / "benchmark_info.log",
        trace_file=run_dir / "benchmark_trace.log",
        console=True,
        min_level="DEBUG" if args.verbose else "INFO",
    )
    set_logger(log)
    
    log.section("BENCHMARK STARTING")
    log.info(f"Run directory: {run_dir}")
    
    log.info(f"Loading questions from: {args.corpus_dir}")
    questions = load_questions(args.corpus_dir, args.datasets)
    
    if not questions:
        log.error("No questions loaded. Exiting.")
        sys.exit(1)
    
    if args.limit:
        questions = questions[:args.limit]
        log.info(f"Limited to {len(questions)} questions")
    
    config = RunnerConfig(
        service_url=args.service_url,
        k=args.k,
        use_rewrite=args.use_rewrite,
        use_rerank=args.use_rerank,
    )
    
    runner = BenchmarkRunner(config, log)
    try:
        results = runner.run_all(questions)
    finally:
        runner.close()
    
    log.info("Calculating metrics...")
    metrics = calculate_metrics(results, k=args.k)
    
    log.info("Generating report...")
    config_dict = {
        "service_url": args.service_url,
        "k": args.k,
        "use_rewrite": args.use_rewrite,
        "use_rerank": args.use_rerank,
        "datasets": args.datasets or ["needle", "realistic", "informed"],
    }
    report = generate_report(results, metrics, config_dict, run_dir)
    
    print("\n" + report)
    
    log.section("BENCHMARK COMPLETE")
    log.summary()
    log.close()


if __name__ == "__main__":
    main()
