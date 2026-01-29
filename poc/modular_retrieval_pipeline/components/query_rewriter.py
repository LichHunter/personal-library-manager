"""Query rewriting component for modular retrieval pipeline.

This module provides query rewriting functionality using Claude Haiku.
It transforms user queries into better search queries for documentation retrieval,
with configurable timeout handling.

The component is stateless (no stored LLM client) and implements the Component
protocol for pipeline compatibility.

Example:
    >>> from types import Query, RewrittenQuery
    >>> rewriter = QueryRewriter(timeout=5.0)
    >>> query = Query("why does my token expire")
    >>> result = rewriter.process(query)
    >>> print(result.rewritten)  # "token expiration TTL lifetime"
"""

import time
from typing import Optional

# Import types from parent module
from ..types import Query, RewrittenQuery
from ..base import Component
from ..utils.llm_provider import call_llm
from ..utils.logger import get_logger


QUERY_REWRITE_PROMPT = """You are a technical documentation search expert. Your task is to rewrite user questions as direct documentation lookup queries.

Guidelines:
1. Convert problem descriptions to feature/capability questions
2. Expand abbreviations and acronyms to full terms
3. Replace casual language with technical terminology
4. Align with documentation vocabulary and structure
5. Keep the rewritten query concise (one line)

Examples:
- "Why can't I schedule workflows every 30 seconds?" → "workflow scheduling minimum interval frequency constraints"
- "Why does my token stop working after 3600 seconds?" → "token expiration TTL lifetime 3600 seconds"
- "What's the RPO and RTO?" → "recovery point objective recovery time objective disaster recovery"

User question: {query}

Rewritten query (one line, no explanation):"""


class QueryRewriter(Component[Query, RewrittenQuery]):
    """Component that rewrites queries for improved retrieval.

    Transforms user queries into documentation-aligned queries using Claude Haiku.
    Converts problem descriptions to feature questions, expands abbreviations,
    and aligns with documentation terminology.

    The component is stateless: no LLM client is stored as an instance variable.
    Each call to process() independently invokes the LLM provider.

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
        self._log = get_logger()

    def _rewrite_query(self, query: str) -> str:
        """Rewrite query using Claude Haiku for better documentation retrieval.

        Converts user queries into documentation-aligned queries by:
        - Converting problem descriptions to feature questions
        - Expanding abbreviations and technical jargon
        - Aligning with documentation terminology

        Args:
            query: Original user query

        Returns:
            Rewritten query string, or original query if rewriting fails
        """
        if not query or not query.strip():
            return query

        self._log.debug(f"[query-rewrite] Rewriting query: {query}")

        start_time = time.time()

        try:
            prompt = QUERY_REWRITE_PROMPT.format(query=query)
            rewritten = call_llm(
                prompt, model="claude-haiku", timeout=int(self.timeout)
            )
            elapsed = time.time() - start_time

            if rewritten and rewritten.strip():
                rewritten = rewritten.strip()
                self._log.debug(
                    f"[query-rewrite] SUCCESS in {elapsed:.3f}s: {query} → {rewritten}"
                )
                return rewritten
            else:
                self._log.debug(
                    f"[query-rewrite] Empty response in {elapsed:.3f}s, using original"
                )
                return query

        except Exception as e:
            elapsed = time.time() - start_time
            self._log.warn(
                f"[query-rewrite] ERROR after {elapsed:.3f}s: {type(e).__name__}: {e}"
            )
            return query

    def process(self, data: Query) -> RewrittenQuery:
        """Rewrite a query for better retrieval.

        Transforms the input Query into a RewrittenQuery by:
        1. Extracting the query text from the Query object
        2. Calling _rewrite_query() with the configured timeout
        3. Creating a RewrittenQuery that preserves the original query
        4. Tracking the model used ("claude-3-haiku")

        The rewriting handles:
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
            Any exception raised by _rewrite_query() will propagate
            (fail-fast behavior for pipeline error handling)
        """
        # Call the internal rewrite method
        rewritten_text = self._rewrite_query(data.text)

        # Create RewrittenQuery preserving the original
        return RewrittenQuery(
            original=data,
            rewritten=rewritten_text,
            model="claude-3-haiku",
        )
