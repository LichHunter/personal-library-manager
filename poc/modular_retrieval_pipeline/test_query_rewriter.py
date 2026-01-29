"""Test QueryRewriter component."""

import sys
from pathlib import Path

# Add parent directory to path so we can import the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from modular_retrieval_pipeline.types import Query, RewrittenQuery
from modular_retrieval_pipeline.components.query_rewriter import QueryRewriter
from modular_retrieval_pipeline.base import Component


def test_component_protocol():
    """Test that QueryRewriter implements Component protocol."""
    print("Test 1: Component protocol implementation")
    rewriter = QueryRewriter(timeout=5.0)
    assert isinstance(rewriter, Component), (
        "QueryRewriter should implement Component protocol"
    )
    assert hasattr(rewriter, "process"), "QueryRewriter should have process() method"
    print("✓ QueryRewriter implements Component protocol")


def test_type_transformation():
    """Test that process() accepts Query and returns RewrittenQuery."""
    print("\nTest 2: Type transformation")
    rewriter = QueryRewriter(timeout=5.0)
    query = Query(text="why does my token expire")
    result = rewriter.process(query)

    assert isinstance(result, RewrittenQuery), (
        f"Expected RewrittenQuery, got {type(result)}"
    )
    assert isinstance(result.original, Query), "RewrittenQuery.original should be Query"
    assert result.original == query, "Original query should be preserved"
    assert isinstance(result.rewritten, str), "Rewritten should be string"
    assert result.model == "claude-3-haiku", "Model should be claude-3-haiku"

    print(f"✓ Query → RewrittenQuery transformation works")
    print(f"  Original: {query.text}")
    print(f"  Rewritten: {result.rewritten}")
    print(f"  Model: {result.model}")


def test_timeout_configuration():
    """Test that timeout parameter is configurable."""
    print("\nTest 3: Timeout configuration")
    rewriter_5s = QueryRewriter(timeout=5.0)
    rewriter_10s = QueryRewriter(timeout=10.0)

    assert rewriter_5s.timeout == 5.0, "Timeout should be 5.0"
    assert rewriter_10s.timeout == 10.0, "Timeout should be 10.0"
    print("✓ Timeout parameter is configurable")


def test_immutability():
    """Test that RewrittenQuery is immutable."""
    print("\nTest 4: Immutability of RewrittenQuery")
    rewriter = QueryRewriter(timeout=5.0)
    query = Query(text="test")
    result = rewriter.process(query)

    try:
        result.rewritten = "modified"
        print("✗ RewrittenQuery should be immutable")
        sys.exit(1)
    except AttributeError:
        print("✓ RewrittenQuery is immutable (frozen dataclass)")


def test_pure_function():
    """Test that component is a pure function (stateless)."""
    print("\nTest 5: Pure function interface")
    rewriter = QueryRewriter(timeout=5.0)
    query1 = Query(text="test query")

    result1 = rewriter.process(query1)
    result2 = rewriter.process(query1)

    # Both results should have the same original query and model
    # (LLM output may vary slightly due to temperature, but structure is same)
    assert result1.original == result2.original, "Original query should be preserved"
    assert result1.model == result2.model, "Model should be consistent"
    assert isinstance(result1.rewritten, str), "Rewritten should be string"
    assert isinstance(result2.rewritten, str), "Rewritten should be string"
    print("✓ Pure function interface (stateless)")


def test_no_stored_state():
    """Test that component has no stored LLM client."""
    print("\nTest 6: No stored state")
    rewriter = QueryRewriter(timeout=5.0)

    # Check that only timeout is stored
    assert hasattr(rewriter, "timeout"), "Should have timeout attribute"
    assert not hasattr(rewriter, "llm_client"), "Should not have llm_client attribute"
    assert not hasattr(rewriter, "client"), "Should not have client attribute"

    print("✓ Component is stateless (no stored LLM client)")


if __name__ == "__main__":
    test_component_protocol()
    test_type_transformation()
    test_timeout_configuration()
    test_immutability()
    test_pure_function()
    test_no_stored_state()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
