"""Tests for search pipeline components."""

import tempfile
import os

import pytest


class TestContentEnricher:
    """Tests for ContentEnricher component."""

    def test_format_with_keywords_and_entities(self):
        """ContentEnricher produces correct format with keywords and entities."""
        from plm.search.components.enricher import ContentEnricher

        enricher = ContentEnricher()
        result = enricher.process({
            'content': 'Test content here',
            'keywords': ['key1', 'key2', 'key3'],
            'entities': {'library': ['React', 'Vue'], 'framework': ['Next.js']}
        })
        expected = 'key1, key2, key3 | React, Vue, Next.js\n\nTest content here'
        assert result == expected

    def test_empty_keywords_and_entities(self):
        """ContentEnricher handles empty keywords and entities."""
        from plm.search.components.enricher import ContentEnricher

        enricher = ContentEnricher()
        result = enricher.process({
            'content': 'Just content',
            'keywords': [],
            'entities': {}
        })
        assert result == 'Just content'


class TestBM25Index:
    """Tests for BM25Index component."""

    def test_ranking(self):
        """BM25Index ranks matching content higher."""
        from plm.search.components.bm25 import BM25Index

        index = BM25Index()
        index.index(['kubernetes pod definition', 'docker container', 'kubernetes deployment'])
        results = index.search('kubernetes pod', k=3)
        assert results[0]['content'] == 'kubernetes pod definition'

    def test_persistence(self):
        """BM25Index persists and loads correctly."""
        from plm.search.components.bm25 import BM25Index

        index = BM25Index()
        index.index(['kubernetes pod definition', 'docker container'])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'bm25_index')
            index.save(path)
            assert os.path.exists(path)

            loaded = BM25Index.load(path)
            results = loaded.search('kubernetes', k=1)
            assert len(results) > 0


class TestQueryExpander:
    """Tests for QueryExpander component."""

    def test_expansion_with_known_term(self):
        """QueryExpander expands known domain terms."""
        from plm.search.types import Query, RewrittenQuery
        from plm.search.components.expander import QueryExpander

        expander = QueryExpander()
        query = Query(text='token authentication')
        rewritten = RewrittenQuery(original=query, rewritten='token auth', model='none')
        result = expander.process(rewritten)
        # token should trigger JWT expansion
        assert 'token' in result.expansions
        assert 'JWT' in result.expanded or 'jwt' in result.expanded.lower()

    def test_no_expansion_for_unknown_terms(self):
        """QueryExpander doesn't expand unknown terms."""
        from plm.search.types import Query, RewrittenQuery
        from plm.search.components.expander import QueryExpander

        expander = QueryExpander()
        query = Query(text='random query')
        rewritten = RewrittenQuery(original=query, rewritten='random query', model='none')
        result = expander.process(rewritten)
        assert result.expanded == 'random query'
        assert result.expansions == ()


class TestRRFFuser:
    """Tests for RRFFuser component."""

    def test_rrf_score_calculation(self):
        """RRFFuser calculates correct RRF scores."""
        from plm.search.types import ScoredChunk, FusionConfig
        from plm.search.components.rrf import RRFFuser

        config = FusionConfig(k=60, bm25_weight=1.0, semantic_weight=1.0)
        fuser = RRFFuser(config)
        bm25 = [ScoredChunk(chunk_id='a', content='text', score=10.0, source='bm25', rank=1)]
        semantic = [ScoredChunk(chunk_id='a', content='text', score=0.9, source='semantic', rank=1)]
        result = fuser.process([bm25, semantic])

        # RRF score for chunk 'a' = 1.0/(60+1) + 1.0/(60+1) = 2/61
        expected_score = 2 / 61
        assert abs(result[0].score - expected_score) < 0.001
        assert result[0].source == 'rrf'


class TestSimilarityScorer:
    """Tests for SimilarityScorer component."""

    def test_identical_vectors_score_one(self):
        """SimilarityScorer gives score ~1.0 for identical vectors."""
        from plm.search.components.semantic import SimilarityScorer

        scorer = SimilarityScorer()
        results = scorer.process({
            'query_embedding': (1.0, 0.0, 0.0),
            'chunk_embeddings': [
                {'id': 'c1', 'content': 'text1', 'embedding': (1.0, 0.0, 0.0)},
                {'id': 'c2', 'content': 'text2', 'embedding': (0.0, 1.0, 0.0)},
            ]
        })
        assert abs(results[0].score - 1.0) < 0.01
        assert results[0].source == 'semantic'
