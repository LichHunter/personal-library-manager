"""Integration analysis CLI for retrieval component evaluation.

Provides analyze-integration command to run complementarity, cascade, and ablation
analyses on benchmark results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plm.benchmark.integration.ablation import AblationResult, run_ablation
from plm.benchmark.integration.cascade import CascadeResult, analyze_cascade
from plm.benchmark.integration.complementarity import (
    ComplementarityResult,
    analyze_complementarity,
)
from plm.benchmark.runner import BenchmarkResults, load_benchmark_cases

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class IntegrationReport:
    """Complete integration analysis report.

    Attributes:
        run_timestamp: ISO 8601 timestamp of analysis run.
        dataset_used: Path to benchmark dataset.
        complementarity: Overlap analysis between retrievers.
        cascade: Stage contribution analysis.
        ablation: Component ablation results.
        recommendations: Generated recommendations based on analysis.
    """

    run_timestamp: str
    dataset_used: str
    complementarity: ComplementarityResult | None
    cascade: CascadeResult | None
    ablation: list[AblationResult]
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        if self.complementarity:
            result["complementarity"] = self.complementarity.to_dict()
        if self.cascade:
            result["cascade"] = self.cascade.to_dict()
        result["ablation"] = [a.to_dict() for a in self.ablation]
        return result


def generate_recommendations(
    complementarity: ComplementarityResult | None,
    cascade: CascadeResult | None,
    ablation: list[AblationResult],
) -> list[str]:
    """Generate actionable recommendations based on analysis results."""
    recommendations: list[str] = []

    if complementarity:
        if complementarity.error_correlation > 0.4:
            recommendations.append(
                f"High error correlation ({complementarity.error_correlation:.2f}): "
                "Consider adding a different retriever type (e.g., keyword, knowledge graph)"
            )
        if complementarity.overlap_at_10 > 0.7:
            recommendations.append(
                f"High overlap at top-10 ({complementarity.overlap_at_10:.2f}): "
                "Retrievers are too similar - consider diversifying retrieval methods"
            )
        if complementarity.fusion_potential < 0.3:
            recommendations.append(
                f"Low fusion potential ({complementarity.fusion_potential:.2f}): "
                "Hybrid fusion provides limited benefit - consider single retriever"
            )

    if cascade:
        if cascade.stage_contributions.get("rrf_to_rerank", 0) < 0.02:
            recommendations.append(
                "Reranker contributes less than 2% MRR improvement: "
                "Consider removing reranker to reduce latency"
            )
        if cascade.bm25_recall_at_100 < 0.5:
            recommendations.append(
                f"Low BM25 recall ({cascade.bm25_recall_at_100:.2f}): "
                "Consider increasing BM25 candidate pool or tuning tokenization"
            )

    no_rerank_result = next((a for a in ablation if a.config_name == "no_rerank"), None)
    if no_rerank_result and no_rerank_result.delta_vs_full > -0.02:
        recommendations.append(
            f"Reranker delta ({no_rerank_result.delta_vs_full:.3f}): "
            "Reranker provides minimal value - consider disabling"
        )

    no_rewrite_result = next((a for a in ablation if a.config_name == "no_rewrite"), None)
    if no_rewrite_result and no_rewrite_result.delta_vs_full > -0.02:
        recommendations.append(
            f"Query rewrite delta ({no_rewrite_result.delta_vs_full:.3f}): "
            "Query rewriting provides minimal value - consider disabling"
        )

    if not recommendations:
        recommendations.append("All components contributing effectively - no changes recommended")

    return recommendations


def run_analysis(
    benchmark_results_path: Path | None,
    dataset_path: Path,
    service_url: str,
    analysis_types: list[str],
    concurrency: int,
    timeout: int,
) -> IntegrationReport:
    """Run integration analysis on benchmark results."""
    complementarity_result: ComplementarityResult | None = None
    cascade_result: CascadeResult | None = None
    ablation_results: list[AblationResult] = []

    if benchmark_results_path and benchmark_results_path.exists():
        with benchmark_results_path.open() as f:
            benchmark_data = json.load(f)
        per_query_results = benchmark_data.get("per_query_results", [])
    else:
        per_query_results = []

    if "complementarity" in analysis_types or "all" in analysis_types:
        log.info("Running complementarity analysis...")
        if per_query_results:
            complementarity_result = analyze_complementarity(per_query_results)
            log.info(
                f"Complementarity: overlap@10={complementarity_result.overlap_at_10:.3f}, "
                f"error_corr={complementarity_result.error_correlation:.3f}"
            )
        else:
            log.warning("No benchmark results provided - skipping complementarity analysis")

    if "cascade" in analysis_types or "all" in analysis_types:
        log.info("Running cascade analysis...")
        if per_query_results:
            cascade_result = analyze_cascade(per_query_results)
            log.info(
                f"Cascade: BM25_R@100={cascade_result.bm25_recall_at_100:.3f}, "
                f"RRF_MRR={cascade_result.rrf_mrr:.3f}"
            )
        else:
            log.warning("No benchmark results provided - skipping cascade analysis")

    if "ablation" in analysis_types or "all" in analysis_types:
        log.info("Running ablation analysis...")
        cases = load_benchmark_cases(dataset_path)
        if cases:
            ablation_results = run_ablation(
                cases, service_url, k=10, concurrency=concurrency, timeout=timeout
            )
            for r in ablation_results:
                log.info(f"Ablation {r.config_name}: MRR={r.mrr:.3f}, delta={r.delta_vs_full:+.3f}")
        else:
            log.warning("No benchmark cases loaded - skipping ablation analysis")

    recommendations = generate_recommendations(
        complementarity_result, cascade_result, ablation_results
    )

    return IntegrationReport(
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_used=str(dataset_path),
        complementarity=complementarity_result,
        cascade=cascade_result,
        ablation=ablation_results,
        recommendations=recommendations,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Integration analysis for retrieval component evaluation"
    )
    parser.add_argument(
        "command",
        choices=["analyze-integration"],
        help="Command to run",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to benchmark dataset JSON file",
    )
    parser.add_argument(
        "--benchmark-results",
        type=Path,
        default=None,
        help="Path to existing benchmark results JSON (for complementarity/cascade)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Search service URL (for ablation)",
    )
    parser.add_argument(
        "--analysis",
        type=str,
        default="all",
        choices=["all", "complementarity", "cascade", "ablation"],
        help="Type of analysis to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of parallel requests",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds",
    )

    args = parser.parse_args()

    if args.command == "analyze-integration":
        if not args.dataset.exists():
            log.error(f"Dataset file not found: {args.dataset}")
            return 1

        analysis_types = [args.analysis] if args.analysis != "all" else ["all"]

        report = run_analysis(
            benchmark_results_path=args.benchmark_results,
            dataset_path=args.dataset,
            service_url=args.url,
            analysis_types=analysis_types,
            concurrency=args.concurrency,
            timeout=args.timeout,
        )

        output_json = json.dumps(report.to_dict(), indent=2)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_json)
            log.info(f"Report written to {args.output}")
        else:
            print(output_json)

        log.info("Recommendations:")
        for rec in report.recommendations:
            log.info(f"  - {rec}")

        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
