from plm.benchmark.runner.metrics import (
    hit_at_k,
    first_relevant_rank,
    reciprocal_rank,
    ndcg_at_k,
    mean_reciprocal_rank,
    hit_rate
)
from plm.benchmark.runner.benchmark import calculate_aggregate_metrics, PerQueryResult

def test_hit_at_k():
    expected = ["c1", "c2"]
    retrieved = ["c3", "c1", "c4"]
    assert hit_at_k(expected, retrieved, 1) is False
    assert hit_at_k(expected, retrieved, 2) is True
    assert hit_at_k(expected, retrieved, 5) is True

def test_first_relevant_rank():
    expected = ["c1", "c2"]
    retrieved = ["c3", "c1", "c4"]
    assert first_relevant_rank(expected, retrieved) == 2
    assert first_relevant_rank(["c5"], retrieved) is None

def test_reciprocal_rank():
    expected = ["c1", "c2"]
    retrieved = ["c3", "c1", "c4"]
    assert reciprocal_rank(expected, retrieved) == 0.5
    assert reciprocal_rank(["c5"], retrieved) == 0.0

def test_ndcg_at_k():
    expected = ["c1", "c2"]
    retrieved = ["c1", "c3", "c2"]
    score = ndcg_at_k(expected, retrieved, 3)
    assert 0.91 < score < 0.93

def test_calculate_aggregate_metrics():
    results = [
        PerQueryResult(
            case_id="1", query="q1", expected_chunk_ids=["c1"], retrieved_chunk_ids=["c1"],
            hit_at_1=True, hit_at_5=True, hit_at_10=True, first_relevant_rank=1,
            reciprocal_rank=1.0, ndcg_at_10=1.0, request_id="r1", response_time_ms=100.0,
            debug_info=[], api_metadata={}
        ),
        PerQueryResult(
            case_id="2", query="q2", expected_chunk_ids=["c2"], retrieved_chunk_ids=["c3", "c2"],
            hit_at_1=False, hit_at_5=True, hit_at_10=True, first_relevant_rank=2,
            reciprocal_rank=0.5, ndcg_at_10=0.63, request_id="r2", response_time_ms=200.0,
            debug_info=[], api_metadata={}
        )
    ]
    
    agg = calculate_aggregate_metrics(results, "dataset.json", "http://localhost", 10)
    
    assert agg.total_queries == 2
    assert agg.hit_at_1 == 0.5
    assert agg.hit_at_5 == 1.0
    assert agg.mrr == 0.75
    assert agg.mean_response_time_ms == 150.0
