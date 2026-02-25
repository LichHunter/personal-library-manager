from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TraceEntry:
    stage: str
    message: str


@dataclass
class RequestTrace:
    request_id: str
    entries: list[TraceEntry] = field(default_factory=list)


@dataclass
class QueryResult:
    question_id: str
    question: str
    target_doc_id: str
    dataset: str
    hit: bool
    rank: int | None
    retrieved_doc_ids: list[str]
    latency_ms: float
    k: int
    request_id: str | None = None
    trace: RequestTrace | None = None


@dataclass
class BenchmarkMetrics:
    total_queries: int
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    mrr: float
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    per_dataset: dict[str, dict] = field(default_factory=dict)


def calculate_metrics(results: list[QueryResult], k: int = 10) -> BenchmarkMetrics:
    if not results:
        return BenchmarkMetrics(
            total_queries=0,
            hit_at_1=0.0, hit_at_3=0.0, hit_at_5=0.0, hit_at_10=0.0,
            mrr=0.0,
            mean_latency_ms=0.0, p50_latency_ms=0.0, p95_latency_ms=0.0,
        )
    
    def hit_at_k(k_val: int) -> float:
        hits = sum(1 for r in results if r.rank is not None and r.rank <= k_val)
        return hits / len(results)
    
    reciprocal_ranks = []
    for r in results:
        if r.rank is not None:
            reciprocal_ranks.append(1.0 / r.rank)
        else:
            reciprocal_ranks.append(0.0)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    latencies = sorted([r.latency_ms for r in results])
    mean_latency = sum(latencies) / len(latencies)
    p50_latency = latencies[len(latencies) // 2]
    p95_idx = int(len(latencies) * 0.95)
    p95_latency = latencies[min(p95_idx, len(latencies) - 1)]
    
    per_dataset: dict[str, dict] = {}
    datasets = set(r.dataset for r in results)
    for dataset in datasets:
        dataset_results = [r for r in results if r.dataset == dataset]
        if dataset_results:
            dataset_hits = sum(1 for r in dataset_results if r.rank is not None and r.rank <= k)
            dataset_mrr_sum = sum(1.0 / r.rank if r.rank else 0.0 for r in dataset_results)
            per_dataset[dataset] = {
                "total": len(dataset_results),
                "hit_at_k": dataset_hits / len(dataset_results),
                "mrr": dataset_mrr_sum / len(dataset_results),
            }
    
    return BenchmarkMetrics(
        total_queries=len(results),
        hit_at_1=hit_at_k(1),
        hit_at_3=hit_at_k(3),
        hit_at_5=hit_at_k(5),
        hit_at_10=hit_at_k(10),
        mrr=mrr,
        mean_latency_ms=mean_latency,
        p50_latency_ms=p50_latency,
        p95_latency_ms=p95_latency,
        per_dataset=per_dataset,
    )


def get_worst_questions(results: list[QueryResult], n: int = 10) -> list[QueryResult]:
    def sort_key(r: QueryResult) -> tuple[bool, int]:
        return (r.rank is None, r.rank if r.rank else 0)
    
    sorted_results = sorted(results, key=sort_key, reverse=True)
    return sorted_results[:n]


def get_best_questions(results: list[QueryResult], n: int = 10) -> list[QueryResult]:
    found_results = [r for r in results if r.rank is not None]
    sorted_results = sorted(found_results, key=lambda r: r.rank or 0)
    return sorted_results[:n]
