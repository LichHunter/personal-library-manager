from plm.benchmark.integration.analysis import generate_recommendations
from plm.benchmark.integration.complementarity import ComplementarityResult
from plm.benchmark.integration.cascade import CascadeResult
from plm.benchmark.integration.ablation import AblationResult

def test_generate_recommendations_all_good():
    comp = ComplementarityResult(
        overlap_at_10=0.5,
        overlap_at_50=0.7,
        bm25_unique_hits=10,
        semantic_unique_hits=10,
        error_correlation=0.2,
        fusion_potential=0.5
    )
    cascade = CascadeResult(
        bm25_recall_at_100=0.8,
        semantic_recall_at_100=0.75,
        rrf_recall_at_50=0.85,
        rrf_mrr=0.6,
        rerank_mrr=0.65,
        stage_contributions={"bm25": 0.4, "rrf": 0.2, "rrf_to_rerank": 0.05}
    )
    ablation = [
        AblationResult(config_name="no_rerank", hit_at_5=0.7, mrr=0.6, delta_vs_full=-0.05),
        AblationResult(config_name="no_rewrite", hit_at_5=0.72, mrr=0.62, delta_vs_full=-0.03)
    ]
    
    recs = generate_recommendations(comp, cascade, ablation)
    assert len(recs) == 1
    assert "effectively" in recs[0]

def test_generate_recommendations_high_correlation():
    comp = ComplementarityResult(
        overlap_at_10=0.8,
        overlap_at_50=0.9,
        bm25_unique_hits=1,
        semantic_unique_hits=1,
        error_correlation=0.5,
        fusion_potential=0.2
    )
    recs = generate_recommendations(comp, None, [])
    assert any("High error correlation" in r for r in recs)
    assert any("High overlap" in r for r in recs)
    assert any("Low fusion potential" in r for r in recs)

def test_generate_recommendations_low_rerank_contribution():
    cascade = CascadeResult(
        bm25_recall_at_100=0.8,
        semantic_recall_at_100=0.75,
        rrf_recall_at_50=0.85,
        rrf_mrr=0.6,
        rerank_mrr=0.605,
        stage_contributions={"bm25": 0.4, "rrf": 0.2, "rrf_to_rerank": 0.005}
    )
    recs = generate_recommendations(None, cascade, [])
    assert any("Reranker contributes less than 2%" in r for r in recs)

def test_generate_recommendations_low_bm25_recall():
    cascade = CascadeResult(
        bm25_recall_at_100=0.4,
        semantic_recall_at_100=0.75,
        rrf_recall_at_50=0.85,
        rrf_mrr=0.3,
        rerank_mrr=0.35,
        stage_contributions={"bm25": 0.1, "rrf": 0.2, "rrf_to_rerank": 0.05}
    )
    recs = generate_recommendations(None, cascade, [])
    assert any("Low BM25 recall" in r for r in recs)
