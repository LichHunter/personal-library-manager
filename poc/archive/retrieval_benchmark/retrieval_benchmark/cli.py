"""Command-line interface for the retrieval benchmark."""

import logging
import sys
from pathlib import Path

import click

from .config.schema import load_config
from .benchmark.runner import BenchmarkRunner


@click.group()
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Enable verbose logging",
)
def cli(verbose: bool) -> None:
    """Retrieval Benchmark - Compare document retrieval strategies."""
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


@cli.command()
@click.option(
    "-c", "--config",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to experiment configuration YAML file",
)
def run(config: Path) -> None:
    """Run a benchmark experiment from a configuration file."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading configuration from {config}")
    experiment_config = load_config(config)
    
    logger.info(f"Starting experiment: {experiment_config.id}")
    logger.info(f"Description: {experiment_config.description}")
    
    runner = BenchmarkRunner(experiment_config)
    runner.run()
    
    logger.info("Done!")


@cli.command()
@click.option(
    "-d", "--documents",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to documents directory",
)
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output path for ground truth JSON",
)
@click.option(
    "-q", "--questions-per-doc",
    default=3,
    type=int,
    help="Number of questions to generate per document",
)
def prepare(documents: Path, output: Path, questions_per_doc: int) -> None:
    """Generate ground truth questions from documents.
    
    This command requires the test_data POC to be available.
    Use it if ground truth doesn't exist yet.
    """
    logger = logging.getLogger(__name__)
    logger.error(
        "Ground truth generation is not implemented in this module. "
        "Use poc/test_data/batch_generate.py with --questions-per-doc flag instead."
    )
    sys.exit(1)


@cli.command()
@click.option(
    "-r", "--results",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to results directory",
)
def analyze(results: Path) -> None:
    """Analyze benchmark results and print summary.
    
    Reads summary.csv and prints key metrics in a readable format.
    """
    import csv
    
    logger = logging.getLogger(__name__)
    summary_path = results / "summary.csv"
    
    if not summary_path.exists():
        logger.error(f"summary.csv not found in {results}")
        sys.exit(1)
    
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        logger.error("No results found in summary.csv")
        sys.exit(1)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    
    # Header
    print(f"\n{'Strategy':<15} {'Backend':<10} {'Model':<20} "
          f"{'Doc@5':<8} {'Sec@5':<8} {'Avg(ms)':<10}")
    print("-" * 80)
    
    # Sort by doc_recall_at_5 descending
    rows_sorted = sorted(
        rows, 
        key=lambda r: float(r.get("doc_recall_at_5", 0)), 
        reverse=True
    )
    
    for row in rows_sorted:
        strategy = row.get("strategy", "")
        backend = row.get("backend", "")
        model = row.get("embedding_model", "")
        if "/" in model:
            model = model.split("/")[-1]  # Just model name
        
        doc_recall = float(row.get("doc_recall_at_5", 0))
        sec_recall = float(row.get("section_recall_at_5", 0))
        avg_time = float(row.get("avg_search_time_ms", 0))
        
        print(f"{strategy:<15} {backend:<10} {model:<20} "
              f"{doc_recall:>6.1%}  {sec_recall:>6.1%}  {avg_time:>8.1f}")
    
    print("-" * 80)
    print(f"Total configurations: {len(rows)}")
    print()


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
