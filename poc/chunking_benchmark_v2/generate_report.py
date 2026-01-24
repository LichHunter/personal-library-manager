#!/usr/bin/env python3
"""Generate markdown reports from benchmark results.

Usage:
    python generate_report.py                           # Latest results
    python generate_report.py --input results/file.json # Specific file
    python generate_report.py --all                     # All results in directory
"""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class BenchmarkResult:
    strategy: str
    chunking: str
    embedder: str
    reranker: str | None
    llm: str | None
    k: int
    metric: str
    coverage: float
    found: int
    total: int
    avg_latency_ms: float
    p95_latency_ms: float
    num_chunks: int
    index_time_s: float

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResult":
        return cls(
            strategy=d["strategy"],
            chunking=d["chunking"],
            embedder=d["embedder"],
            reranker=d.get("reranker"),
            llm=d.get("llm"),
            k=d["k"],
            metric=d["metric"],
            coverage=d["coverage"],
            found=d["found"],
            total=d["total"],
            avg_latency_ms=d.get("avg_latency_ms", 0),
            p95_latency_ms=d.get("p95_latency_ms", 0),
            num_chunks=d.get("num_chunks", 0),
            index_time_s=d.get("index_time_s", 0),
        )

    @property
    def full_name(self) -> str:
        parts = [self.strategy, self.chunking, self.embedder.split("/")[-1]]
        if self.reranker:
            parts.append(self.reranker.split("/")[-1])
        if self.llm:
            parts.append(self.llm.replace(":", "_"))
        return " + ".join(parts)

    @property
    def short_name(self) -> str:
        return f"{self.strategy}+{self.chunking}"


class ReportGenerator:
    def __init__(self, results: list[BenchmarkResult], source_file: str = ""):
        self.results = results
        self.source_file = source_file
        self.generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def generate_markdown(self) -> str:
        sections = [
            self._header(),
            self._executive_summary(),
            self._top_strategies(),
            self._results_by_k(),
            self._results_by_strategy_type(),
            self._results_by_chunking(),
            self._results_by_embedder(),
            self._timing_analysis(),
            self._full_results_table(),
        ]
        return "\n\n".join(sections)

    def _header(self) -> str:
        lines = [
            "# Benchmark Report",
            "",
            f"**Generated**: {self.generated_at}",
            f"**Source**: `{self.source_file}`" if self.source_file else "",
            f"**Total Evaluations**: {len(self.results)}",
        ]
        return "\n".join(lines)

    def _executive_summary(self) -> str:
        lines = ["## Executive Summary"]

        exact_k5 = [r for r in self.results if r.metric == "exact_match" and r.k == 5]
        if not exact_k5:
            exact_k5 = [r for r in self.results if r.metric == "exact_match"]

        if not exact_k5:
            lines.append("\nNo exact_match results found.")
            return "\n".join(lines)

        best = max(exact_k5, key=lambda r: r.coverage)
        worst = min(exact_k5, key=lambda r: r.coverage)
        avg_coverage = sum(r.coverage for r in exact_k5) / len(exact_k5)

        lines.extend(
            [
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Best Coverage | **{best.coverage:.1%}** ({best.found}/{best.total}) |",
                f"| Best Strategy | {best.full_name} |",
                f"| Worst Coverage | {worst.coverage:.1%} ({worst.found}/{worst.total}) |",
                f"| Average Coverage | {avg_coverage:.1%} |",
                f"| Total Strategies Tested | {len(set(r.short_name for r in exact_k5))} |",
            ]
        )

        return "\n".join(lines)

    def _top_strategies(self) -> str:
        lines = ["## Top 10 Strategies"]

        exact_k5 = [r for r in self.results if r.metric == "exact_match" and r.k == 5]
        if not exact_k5:
            exact_k5 = [r for r in self.results if r.metric == "exact_match"]

        if not exact_k5:
            return "\n".join(lines + ["\nNo results available."])

        sorted_results = sorted(exact_k5, key=lambda r: r.coverage, reverse=True)[:10]

        lines.extend(
            [
                "",
                "| Rank | Strategy | Chunking | Embedder | Coverage | Found/Total | Latency |",
                "|------|----------|----------|----------|----------|-------------|---------|",
            ]
        )

        for i, r in enumerate(sorted_results, 1):
            embedder_short = r.embedder.split("/")[-1]
            lines.append(
                f"| {i} | {r.strategy} | {r.chunking} | {embedder_short} | "
                f"**{r.coverage:.1%}** | {r.found}/{r.total} | {r.avg_latency_ms:.1f}ms |"
            )

        return "\n".join(lines)

    def _results_by_k(self) -> str:
        lines = ["## Results by K Value"]

        k_values = sorted(set(r.k for r in self.results))
        if not k_values:
            return "\n".join(lines + ["\nNo results available."])

        lines.extend(
            [
                "",
                "Best strategy for each k value (exact_match):",
                "",
                "| K | Best Strategy | Coverage | Avg Latency |",
                "|---|---------------|----------|-------------|",
            ]
        )

        for k in k_values:
            k_results = [
                r for r in self.results if r.k == k and r.metric == "exact_match"
            ]
            if k_results:
                best = max(k_results, key=lambda r: r.coverage)
                lines.append(
                    f"| {k} | {best.short_name} | **{best.coverage:.1%}** | {best.avg_latency_ms:.1f}ms |"
                )

        return "\n".join(lines)

    def _results_by_strategy_type(self) -> str:
        lines = ["## Results by Retrieval Strategy"]

        exact_k5 = [r for r in self.results if r.metric == "exact_match" and r.k == 5]
        if not exact_k5:
            exact_k5 = [r for r in self.results if r.metric == "exact_match"]

        if not exact_k5:
            return "\n".join(lines + ["\nNo results available."])

        strategy_types = sorted(set(r.strategy for r in exact_k5))

        lines.extend(
            [
                "",
                "| Strategy | Best Chunking | Best Coverage | Avg Coverage | Count |",
                "|----------|---------------|---------------|--------------|-------|",
            ]
        )

        for st in strategy_types:
            st_results = [r for r in exact_k5 if r.strategy == st]
            if st_results:
                best = max(st_results, key=lambda r: r.coverage)
                avg = sum(r.coverage for r in st_results) / len(st_results)
                lines.append(
                    f"| {st} | {best.chunking} | **{best.coverage:.1%}** | {avg:.1%} | {len(st_results)} |"
                )

        return "\n".join(lines)

    def _results_by_chunking(self) -> str:
        lines = ["## Results by Chunking Strategy"]

        exact_k5 = [r for r in self.results if r.metric == "exact_match" and r.k == 5]
        if not exact_k5:
            exact_k5 = [r for r in self.results if r.metric == "exact_match"]

        if not exact_k5:
            return "\n".join(lines + ["\nNo results available."])

        chunking_types = sorted(set(r.chunking for r in exact_k5))

        lines.extend(
            [
                "",
                "| Chunking | Best Retrieval | Best Coverage | Avg Coverage | Chunks |",
                "|----------|----------------|---------------|--------------|--------|",
            ]
        )

        for ch in chunking_types:
            ch_results = [r for r in exact_k5 if r.chunking == ch]
            if ch_results:
                best = max(ch_results, key=lambda r: r.coverage)
                avg = sum(r.coverage for r in ch_results) / len(ch_results)
                chunks = ch_results[0].num_chunks if ch_results else 0
                lines.append(
                    f"| {ch} | {best.strategy} | **{best.coverage:.1%}** | {avg:.1%} | {chunks} |"
                )

        return "\n".join(lines)

    def _results_by_embedder(self) -> str:
        lines = ["## Results by Embedding Model"]

        exact_k5 = [r for r in self.results if r.metric == "exact_match" and r.k == 5]
        if not exact_k5:
            exact_k5 = [r for r in self.results if r.metric == "exact_match"]

        if not exact_k5:
            return "\n".join(lines + ["\nNo results available."])

        embedders = sorted(set(r.embedder for r in exact_k5))

        lines.extend(
            [
                "",
                "| Embedder | Best Strategy | Best Coverage | Avg Coverage |",
                "|----------|---------------|---------------|--------------|",
            ]
        )

        for emb in embedders:
            emb_results = [r for r in exact_k5 if r.embedder == emb]
            if emb_results:
                best = max(emb_results, key=lambda r: r.coverage)
                avg = sum(r.coverage for r in emb_results) / len(emb_results)
                emb_short = emb.split("/")[-1]
                lines.append(
                    f"| {emb_short} | {best.short_name} | **{best.coverage:.1%}** | {avg:.1%} |"
                )

        return "\n".join(lines)

    def _timing_analysis(self) -> str:
        lines = ["## Timing Analysis"]

        if not self.results:
            return "\n".join(lines + ["\nNo results available."])

        unique_strategies = {}
        for r in self.results:
            key = (r.strategy, r.chunking, r.embedder)
            if key not in unique_strategies:
                unique_strategies[key] = r

        sorted_by_index = sorted(
            unique_strategies.values(), key=lambda r: r.index_time_s
        )
        sorted_by_latency = sorted(
            unique_strategies.values(), key=lambda r: r.avg_latency_ms
        )

        lines.extend(
            [
                "",
                "### Fastest Indexing (Top 5)",
                "",
                "| Strategy | Index Time | Chunks |",
                "|----------|------------|--------|",
            ]
        )

        for r in sorted_by_index[:5]:
            lines.append(f"| {r.short_name} | {r.index_time_s:.2f}s | {r.num_chunks} |")

        lines.extend(
            [
                "",
                "### Fastest Query Latency (Top 5)",
                "",
                "| Strategy | Avg Latency | P95 Latency |",
                "|----------|-------------|-------------|",
            ]
        )

        for r in sorted_by_latency[:5]:
            lines.append(
                f"| {r.short_name} | {r.avg_latency_ms:.1f}ms | {r.p95_latency_ms:.1f}ms |"
            )

        return "\n".join(lines)

    def _full_results_table(self) -> str:
        lines = ["## Full Results"]

        exact_results = [r for r in self.results if r.metric == "exact_match"]
        if not exact_results:
            exact_results = self.results

        if not exact_results:
            return "\n".join(lines + ["\nNo results available."])

        sorted_results = sorted(exact_results, key=lambda r: (r.k, -r.coverage))

        lines.extend(
            [
                "",
                "<details>",
                "<summary>Click to expand full results table</summary>",
                "",
                "| K | Strategy | Chunking | Embedder | Coverage | Found | Total | Latency |",
                "|---|----------|----------|----------|----------|-------|-------|---------|",
            ]
        )

        for r in sorted_results:
            emb_short = r.embedder.split("/")[-1]
            lines.append(
                f"| {r.k} | {r.strategy} | {r.chunking} | {emb_short} | "
                f"{r.coverage:.1%} | {r.found} | {r.total} | {r.avg_latency_ms:.1f}ms |"
            )

        lines.extend(["", "</details>"])

        return "\n".join(lines)


def find_latest_results(results_dir: Path) -> Path | None:
    json_files = list(results_dir.glob("*_detailed.json"))
    if not json_files:
        json_files = list(results_dir.glob("*.json"))

    if not json_files:
        return None

    return max(json_files, key=lambda f: f.stat().st_mtime)


def load_results(file_path: Path) -> list[BenchmarkResult]:
    with open(file_path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return [BenchmarkResult.from_dict(d) for d in data]

    if isinstance(data, dict) and "results" in data:
        return [BenchmarkResult.from_dict(d) for d in data["results"]]

    raise ValueError(f"Unknown results format in {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark reports")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Input results JSON file (default: latest in results/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output markdown file (default: results/reports/<timestamp>_report.md)",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Generate reports for all result files",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    reports_dir = results_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        json_files = list(results_dir.glob("*_detailed.json"))
        if not json_files:
            print("No detailed result files found in results/")
            sys.exit(1)

        for json_file in json_files:
            print(f"Processing {json_file.name}...")
            try:
                results = load_results(json_file)
                generator = ReportGenerator(results, json_file.name)
                report = generator.generate_markdown()

                report_name = json_file.stem.replace("_detailed", "_report") + ".md"
                report_path = reports_dir / report_name
                report_path.write_text(report)
                print(f"  -> {report_path}")
            except Exception as e:
                print(f"  ERROR: {e}")

        return

    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = base_dir / args.input
    else:
        input_path = find_latest_results(results_dir)

    if not input_path or not input_path.exists():
        print("No results file found. Run benchmark first or specify --input.")
        sys.exit(1)

    print(f"Loading results from {input_path}...")
    results = load_results(input_path)
    print(f"Loaded {len(results)} evaluation results")

    generator = ReportGenerator(results, input_path.name)
    report = generator.generate_markdown()

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = base_dir / args.output
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_path = reports_dir / f"{timestamp}_report.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
