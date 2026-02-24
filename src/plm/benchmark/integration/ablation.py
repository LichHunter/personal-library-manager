"""Component ablation analysis for retrieval pipeline.

Runs benchmark with components disabled to measure their individual contribution.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any

import httpx

from plm.benchmark.runner import (
    BenchmarkCase,
    hit_rate,
    mean_reciprocal_rank,
    reciprocal_rank,
)
from plm.benchmark.runner.metrics import hit_at_k

log = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of single ablation configuration.

    Attributes:
        config_name: Name of the configuration (e.g., 'full', 'no_rerank', 'no_rewrite').
        hit_at_5: Hit rate at k=5 for this configuration.
        mrr: Mean Reciprocal Rank for this configuration.
        delta_vs_full: MRR difference compared to full configuration (negative = worse).
    """

    config_name: str
    hit_at_5: float
    mrr: float
    delta_vs_full: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AblationConfig:
    """Configuration for a single ablation run."""

    name: str
    use_rerank: bool
    use_rewrite: bool


ABLATION_CONFIGS = [
    AblationConfig(name="full", use_rerank=True, use_rewrite=True),
    AblationConfig(name="no_rerank", use_rerank=False, use_rewrite=True),
    AblationConfig(name="no_rewrite", use_rerank=True, use_rewrite=False),
    AblationConfig(name="baseline", use_rerank=False, use_rewrite=False),
]


async def _query_api(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    k: int,
    use_rerank: bool,
    use_rewrite: bool,
) -> dict[str, Any]:
    response = await client.post(
        f"{url}/query",
        json={
            "query": query,
            "k": k,
            "use_rerank": use_rerank,
            "use_rewrite": use_rewrite,
            "explain": False,
        },
    )
    response.raise_for_status()
    return response.json()


async def _run_config(
    cases: list[BenchmarkCase],
    url: str,
    k: int,
    config: AblationConfig,
    concurrency: int,
    timeout: int,
) -> tuple[float, float]:
    """Run benchmark with specific config, return (hit_at_5, mrr)."""
    semaphore = asyncio.Semaphore(concurrency)
    hit_5_list: list[bool] = []
    rr_list: list[float] = []

    async def evaluate_case(
        client: httpx.AsyncClient, case: BenchmarkCase
    ) -> tuple[bool, float] | None:
        async with semaphore:
            try:
                response = await _query_api(
                    client, url, case.query, k, config.use_rerank, config.use_rewrite
                )
                retrieved = [r["chunk_id"] for r in response.get("results", [])]
                expected = case.relevant_chunk_ids
                return (
                    hit_at_k(expected, retrieved, 5),
                    reciprocal_rank(expected, retrieved),
                )
            except Exception as e:
                log.warning(f"Ablation query failed for {case.id}: {e}")
                return None

    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [evaluate_case(client, case) for case in cases]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result is not None:
                hit_5_list.append(result[0])
                rr_list.append(result[1])

    return hit_rate(hit_5_list), mean_reciprocal_rank(rr_list)


async def run_ablation_async(
    cases: list[BenchmarkCase],
    url: str,
    k: int = 10,
    concurrency: int = 4,
    timeout: int = 30,
) -> list[AblationResult]:
    """Run ablation analysis across all configurations.

    Args:
        cases: Benchmark cases to evaluate.
        url: Search service URL.
        k: Number of results to retrieve.
        concurrency: Number of parallel requests.
        timeout: Request timeout in seconds.

    Returns:
        List of AblationResult, one per configuration.
    """
    results: list[AblationResult] = []
    full_mrr: float | None = None

    for config in ABLATION_CONFIGS:
        log.info(f"Running ablation config: {config.name}")
        hit_5, mrr = await _run_config(cases, url, k, config, concurrency, timeout)

        if config.name == "full":
            full_mrr = mrr

        delta = mrr - (full_mrr or mrr)

        results.append(
            AblationResult(
                config_name=config.name,
                hit_at_5=hit_5,
                mrr=mrr,
                delta_vs_full=delta,
            )
        )

    return results


def run_ablation(
    cases: list[BenchmarkCase],
    url: str,
    k: int = 10,
    concurrency: int = 4,
    timeout: int = 30,
) -> list[AblationResult]:
    """Synchronous wrapper for run_ablation_async."""
    return asyncio.run(
        run_ablation_async(cases, url, k, concurrency, timeout)
    )
