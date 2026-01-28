"""Query rewriting component for modular retrieval pipeline.

This module wraps the existing rewrite_query function from enriched_hybrid_llm
to fit the Component protocol. It transforms user queries into better search
queries using Claude Haiku, with configurable timeout handling.

The component is stateless (no stored LLM client) and implements the Component
protocol for pipeline compatibility.

Example:
    >>> from types import Query, RewrittenQuery
    >>> rewriter = QueryRewriter(timeout=5.0)
    >>> query = Query("why does my token expire")
    >>> result = rewriter.process(query)
    >>> print(result.rewritten)  # "token expiration TTL lifetime"
"""

import sys
import time
from pathlib import Path
from typing import Optional

# Import types from parent module
from ..types import Query, RewrittenQuery
from ..base import Component

# Import the existing rewrite_query function
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "chunking_benchmark_v2"))
from retrieval.query_rewrite import rewrite_query


class QueryRewriter(Component[Query, RewrittenQuery]):
    """Component that rewrites queries for improved retrieval.

    Wraps the existing rewrite_query function from enriched_hybrid_llm to fit
    the Component protocol. Transforms user queries into documentation-aligned
    queries using Claude Haiku.

    The component is stateless: no LLM client is stored as an instance variable.
    Each call to process() independently invokes the rewrite_query function.

    Features:
    - Accepts Query objects, returns RewrittenQuery objects
    - Configurable timeout (default 5.0 seconds)
    - Preserves original query in result for provenance tracking
    - Tracks which model performed the rewriting
    - Pure function interface (no state mutation)

    Example:
        >>> rewriter = QueryRewriter(timeout=5.0)
        >>> query = Query("why does my token expire")
        >>> result = rewriter.process(query)
        >>> assert isinstance(result, RewrittenQuery)
        >>> assert result.original == query
        >>> assert result.model == "claude-3-haiku"
    """

    def __init__(self, timeout: float = 5.0):
        """Initialize QueryRewriter component.

        Args:
            timeout: Timeout in seconds for LLM call (default 5.0).
                    If the LLM takes longer than this, the original query
                    is returned unchanged.
        """
        self.timeout = timeout

    def process(self, data: Query) -> RewrittenQuery:
        """Rewrite a query for better retrieval.

        Transforms the input Query into a RewrittenQuery by:
        1. Extracting the query text from the Query object
        2. Calling rewrite_query() with the configured timeout
        3. Creating a RewrittenQuery that preserves the original query
        4. Tracking the model used ("claude-3-haiku")

        The rewrite_query function handles:
        - Converting problem descriptions to feature questions
        - Expanding abbreviations and technical jargon
        - Aligning with documentation terminology
        - Returning original query if rewriting fails or times out

        Args:
            data: Query object containing the text to rewrite

        Returns:
            RewrittenQuery object with:
            - original: The input Query (preserved for provenance)
            - rewritten: The rewritten query text
            - model: "claude-3-haiku" (the model used)

        Raises:
            Any exception raised by rewrite_query() will propagate
            (fail-fast behavior for pipeline error handling)
        """
        # Call the existing rewrite_query function
        rewritten_text = rewrite_query(data.text, timeout=self.timeout)

        # Create RewrittenQuery preserving the original
        return RewrittenQuery(
            original=data,
            rewritten=rewritten_text,
            model="claude-3-haiku",
        )
