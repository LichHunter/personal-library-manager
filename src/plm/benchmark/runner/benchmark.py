"""Benchmark runner for evaluating search service performance.

Calls production search service HTTP API and calculates retrieval metrics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from plm.benchmark.runner.metrics import (
    first_relevant_rank,
    hit_at_k,
    hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    percentile,
    reciprocal_rank,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class PerQueryResult:
    case_id: str
    query: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    hit_at_1: bool
    hit_at_5: bool
    hit_at_10: bool
    first_relevant_rank: int | None
    reciprocal_rank: float
    ndcg_at_10: float
    request_id: str
    response_time_ms: float
    debug_info: list[dict[str, Any]]
    api_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResults:
    run_id: str
    run_timestamp: str
    dataset_path: str
    service_url: str
    k: int
    total_queries: int
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    mrr: float
    ndcg_at_10: float
    mean_response_time_ms: float
    p95_response_time_ms: float
    per_query_results: list[PerQueryResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["per_query_results"] = [r.to_dict() for r in self.per_query_results]
        return result


@dataclass
class BenchmarkCase:
    id: str
    query: str
    relevant_chunk_ids: list[str]
    tier: str | None = None
    confidence_score: float | None = None


def load_benchmark_cases(
    dataset_path: Path,
    tier_filter: str | None = None,
    min_confidence: float = 0.0,
) -> list[BenchmarkCase]:
    with dataset_path.open() as f:
        data = json.load(f)

    cases = []
    for item in data:
        case = BenchmarkCase(
            id=item["id"],
            query=item["query"],
            relevant_chunk_ids=item["relevant_chunk_ids"],
            tier=item.get("tier"),
            confidence_score=item.get("confidence_score"),
        )

        if tier_filter and case.tier != tier_filter:
            continue

        if case.confidence_score is not None and case.confidence_score < min_confidence:
            continue

        cases.append(case)

    log.info(f"Loaded {len(cases)} cases (tier={tier_filter}, min_conf={min_confidence})")
    return cases


async def query_api(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    k: int,
    explain: bool = True,
) -> dict[str, Any]:
    response = await client.post(
        f"{url}/query",
        json={
            "query": query,
            "k": k,
            "explain": explain,
        },
    )
    response.raise_for_status()
    return response.json()


async def evaluate_case(
    client: httpx.AsyncClient,
    url: str,
    case: BenchmarkCase,
    k: int,
) -> PerQueryResult | None:
    start_time = time.time()
    
    try:
        response = await query_api(client, url, case.query, k, explain=True)
    except httpx.HTTPStatusError as e:
        log.error(f"API error for case {case.id}: {e}")
        return None
    except httpx.RequestError as e:
        log.error(f"Request error for case {case.id}: {e}")
        return None
    
    elapsed_ms = (time.time() - start_time) * 1000

    retrieved_ids = [r["chunk_id"] for r in response.get("results", [])]
    expected = case.relevant_chunk_ids

    debug_info_list = [
        r.get("debug_info", {}) for r in response.get("results", [])
    ]

    api_metadata = response.get("metadata") or {}

    return PerQueryResult(
        case_id=case.id,
        query=case.query,
        expected_chunk_ids=expected,
        retrieved_chunk_ids=retrieved_ids,
        hit_at_1=hit_at_k(expected, retrieved_ids, 1),
        hit_at_5=hit_at_k(expected, retrieved_ids, 5),
        hit_at_10=hit_at_k(expected, retrieved_ids, 10),
        first_relevant_rank=first_relevant_rank(expected, retrieved_ids),
        reciprocal_rank=reciprocal_rank(expected, retrieved_ids),
        ndcg_at_10=ndcg_at_k(expected, retrieved_ids, 10),
        request_id=response.get("request_id", ""),
        response_time_ms=elapsed_ms,
        debug_info=debug_info_list,
        api_metadata=api_metadata,
    )


async def run_benchmark(
    cases: list[BenchmarkCase],
    url: str,
    k: int,
    concurrency: int,
    timeout: int,
) -> list[PerQueryResult]:
    results: list[PerQueryResult] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_evaluate(
        client: httpx.AsyncClient, case: BenchmarkCase
    ) -> PerQueryResult | None:
        async with semaphore:
            return await evaluate_case(client, url, case, k)

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [bounded_evaluate(client, case) for case in cases]
        
        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                results.append(result)
            completed += 1
            if completed % 50 == 0:
                log.info(f"Progress: {completed}/{len(cases)}")

    return results


def calculate_aggregate_metrics(
    results: list[PerQueryResult],
    dataset_path: str,
    service_url: str,
    k: int,
) -> BenchmarkResults:
    if not results:
        return BenchmarkResults(
            run_id=str(uuid.uuid4()),
            run_timestamp=datetime.now(timezone.utc).isoformat(),
            dataset_path=dataset_path,
            service_url=service_url,
            k=k,
            total_queries=0,
            hit_at_1=0.0,
            hit_at_5=0.0,
            hit_at_10=0.0,
            mrr=0.0,
            ndcg_at_10=0.0,
            mean_response_time_ms=0.0,
            p95_response_time_ms=0.0,
            per_query_results=[],
        )

    hit_1_list = [r.hit_at_1 for r in results]
    hit_5_list = [r.hit_at_5 for r in results]
    hit_10_list = [r.hit_at_10 for r in results]
    rr_list = [r.reciprocal_rank for r in results]
    ndcg_list = [r.ndcg_at_10 for r in results]
    latencies = [r.response_time_ms for r in results]

    return BenchmarkResults(
        run_id=str(uuid.uuid4()),
        run_timestamp=datetime.now(timezone.utc).isoformat(),
        dataset_path=dataset_path,
        service_url=service_url,
        k=k,
        total_queries=len(results),
        hit_at_1=hit_rate(hit_1_list),
        hit_at_5=hit_rate(hit_5_list),
        hit_at_10=hit_rate(hit_10_list),
        mrr=mean_reciprocal_rank(rr_list),
        ndcg_at_10=sum(ndcg_list) / len(ndcg_list),
        mean_response_time_ms=sum(latencies) / len(latencies),
        p95_response_time_ms=percentile(latencies, 95),
        per_query_results=results,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate search service performance against benchmark dataset"
    )
    parser.add_argument(
        "command",
        choices=["evaluate"],
        help="Command to run",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to benchmark dataset JSON file",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Search service URL",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--tier-filter",
        type=str,
        default=None,
        help="Filter by tier (gold, silver, bronze)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence score filter",
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

    if args.command == "evaluate":
        if not args.dataset.exists():
            log.error(f"Dataset file not found: {args.dataset}")
            return 1

        cases = load_benchmark_cases(
            args.dataset,
            tier_filter=args.tier_filter,
            min_confidence=args.min_confidence,
        )

        if not cases:
            log.error("No cases to evaluate after filtering")
            return 1

        log.info(f"Evaluating {len(cases)} cases against {args.url}")

        results = asyncio.run(
            run_benchmark(
                cases=cases,
                url=args.url,
                k=args.k,
                concurrency=args.concurrency,
                timeout=args.timeout,
            )
        )

        benchmark_results = calculate_aggregate_metrics(
            results=results,
            dataset_path=str(args.dataset),
            service_url=args.url,
            k=args.k,
        )

        output_json = json.dumps(benchmark_results.to_dict(), indent=2)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output_json)
            log.info(f"Results written to {args.output}")
        else:
            print(output_json)

        log.info(
            f"Summary: Hit@1={benchmark_results.hit_at_1:.3f}, "
            f"Hit@5={benchmark_results.hit_at_5:.3f}, "
            f"Hit@10={benchmark_results.hit_at_10:.3f}, "
            f"MRR={benchmark_results.mrr:.3f}, "
            f"NDCG@10={benchmark_results.ndcg_at_10:.3f}"
        )

        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
