"""Tests for Pipeline and Component base classes."""

import pytest


class TestPipeline:
    """Tests for Pipeline class."""

    def test_pipeline_chains_components(self):
        """Pipeline chains components correctly."""
        from plm.search.pipeline import Pipeline

        class Upper:
            def process(self, data: str) -> str:
                return data.upper()

        class Exclaim:
            def process(self, data: str) -> str:
                return data + '!'

        pipeline = Pipeline().add(Upper()).add(Exclaim())
        result = pipeline.run('hello')
        assert result == 'HELLO!'

    def test_pipeline_empty_raises_error(self):
        """Empty pipeline raises error on run."""
        from plm.search.pipeline import Pipeline

        pipeline = Pipeline()
        with pytest.raises(ValueError, match="Cannot run empty pipeline"):
            pipeline.run('test')

    def test_pipeline_len(self):
        """Pipeline reports correct length."""
        from plm.search.pipeline import Pipeline

        class Noop:
            def process(self, data):
                return data

        pipeline = Pipeline().add(Noop()).add(Noop()).add(Noop())
        assert len(pipeline) == 3

    def test_pipeline_error_propagation(self):
        """Pipeline propagates errors correctly."""
        from plm.search.pipeline import Pipeline, PipelineError

        class FailingComponent:
            def process(self, data):
                raise ValueError("Intentional failure")

        pipeline = Pipeline().add(FailingComponent())

        with pytest.raises(PipelineError):
            pipeline.run('test')
