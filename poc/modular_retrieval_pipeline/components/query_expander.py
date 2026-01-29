"""Query expansion component for modular retrieval pipeline.

This module implements domain-specific query expansion to address vocabulary mismatch,
the #1 failure mode in realistic retrieval benchmarks (59% failure rate).

Query expansion adds synonyms and related terms to queries, improving retrieval when
users describe problems with natural language while documentation uses technical terms.

The component ports the expand_query logic from enriched_hybrid_llm to fit the
Component protocol. It is stateless and implements pure function semantics.

Example:
    >>> from types import Query, RewrittenQuery, ExpandedQuery
    >>> expander = QueryExpander()
    >>> query = Query("token authentication")
    >>> rewritten = RewrittenQuery(query, "token auth", "claude-3-haiku")
    >>> result = expander.process(rewritten)
    >>> print(result.expanded)  # "token auth JWT authentication iat exp issued claims expiration"
    >>> print(result.expansions)  # ("token",)
"""

from ..types import RewrittenQuery, ExpandedQuery
from ..base import Component
from ..utils.logger import get_logger


# Domain expansion dictionary for query expansion
# Addresses VOCABULARY_MISMATCH and ACRONYM_GAP root causes
# Sourced from enriched_hybrid_llm.py
DOMAIN_EXPANSIONS = {
    # RPO/RTO acronyms - 100% miss rate on disaster recovery queries
    "rpo": "recovery point objective RPO data loss backup",
    "recovery point objective": "RPO data loss backup disaster recovery",
    "rto": "recovery time objective RTO downtime recovery restore",
    "recovery time objective": "RTO downtime recovery restore disaster",
    # JWT terminology - "iat" is JWT-specific
    "jwt": "JSON web token JWT authentication iat exp issued claims",
    "token": "JWT authentication token iat exp issued claims expiration",
    # Database stack vocabulary
    "database stack": "PostgreSQL Redis Kafka database storage data layer",
    "database": "PostgreSQL Redis Kafka storage data layer",
    # Monitoring stack vocabulary
    "monitoring stack": "Prometheus Grafana Jaeger observability metrics tracing",
    "monitoring": "Prometheus Grafana Jaeger observability metrics tracing",
    "observability": "Prometheus Grafana Jaeger monitoring metrics tracing",
    # HPA/autoscaling terms
    "hpa": "horizontal pod autoscaler HPA scaling replicas CPU utilization",
    "autoscaling": "horizontal pod autoscaler HPA scaling replicas CPU utilization",
    "autoscaler": "horizontal pod autoscaler HPA scaling replicas CPU",
}


class QueryExpander(Component[RewrittenQuery, ExpandedQuery]):
    """Component that expands queries with domain-specific terms.

    Addresses vocabulary mismatch by adding synonyms and related terms to queries.
    This is critical for retrieval: realistic benchmarks show 59% failure rate due to
    vocabulary mismatch between user queries and documentation.

    The component is stateless: DOMAIN_EXPANSIONS is a class constant, not instance state.
    Each call to process() independently expands the query.

    Features:
    - Accepts RewrittenQuery objects, returns ExpandedQuery objects
    - Tracks which expansion terms were applied (tuple of expansion keys)
    - Deduplicates expansion terms (avoids adding terms already in query)
    - Case-insensitive matching (checks lowercase query)
    - Pure function interface (no state mutation)

    Example:
        >>> expander = QueryExpander()
        >>> query = Query("token authentication")
        >>> rewritten = RewrittenQuery(query, "token auth", "claude-3-haiku")
        >>> result = expander.process(rewritten)
        >>> assert isinstance(result, ExpandedQuery)
        >>> assert result.query == rewritten
        >>> assert "token" in result.expansions
        >>> assert "JWT" in result.expanded
    """

    # Class constant: domain expansions dictionary
    # Not instance state - shared across all instances
    DOMAIN_EXPANSIONS = DOMAIN_EXPANSIONS

    def __init__(self):
        """Initialize QueryExpander with logger."""
        self._log = get_logger()
        self._log.debug("[QueryExpander] Initialized")

    def process(self, data: RewrittenQuery) -> ExpandedQuery:
        """Expand a rewritten query with domain-specific terms.

        Transforms the input RewrittenQuery into an ExpandedQuery by:
        1. Checking which expansion terms appear in the rewritten query (case-insensitive)
        2. Collecting all expansion terms for matched keys
        3. Deduplicating: removing terms already in the query
        4. Appending new terms to the query
        5. Tracking which expansion keys were applied

        The expansion process:
        - Matches expansion keys case-insensitively against the rewritten query
        - Collects all expansion terms for matched keys
        - Removes duplicates and terms already in the query
        - Appends remaining terms in sorted order

        Args:
            data: RewrittenQuery object containing the query to expand

        Returns:
            ExpandedQuery object with:
            - query: The input RewrittenQuery (preserved for provenance)
            - expanded: The expanded query text with new terms appended
            - expansions: Tuple of expansion keys that were applied
            - method: "domain_specific" (the expansion method used)

        Example:
            >>> rewritten = RewrittenQuery(
            ...     Query("token authentication"),
            ...     "token auth",
            ...     "claude-3-haiku"
            ... )
            >>> expander = QueryExpander()
            >>> result = expander.process(rewritten)
            >>> # result.expanded = "token auth JWT authentication iat exp issued claims expiration"
            >>> # result.expansions = ("token",)
        """
        self._log.debug(f"[QueryExpander] Processing query: {data.rewritten}")

        # Get the rewritten query text and convert to lowercase for matching
        query_text = data.rewritten
        query_lower = query_text.lower()

        # Find which expansion terms appear in the query
        expansions_applied = []
        for term, expansion in self.DOMAIN_EXPANSIONS.items():
            if term in query_lower:
                expansions_applied.append((term, expansion))
                self._log.trace(f"[QueryExpander] Matched expansion term: {term}")

        # If no expansions found, return query unchanged
        if not expansions_applied:
            self._log.debug(
                "[QueryExpander] No expansions matched, returning original query"
            )
            return ExpandedQuery(
                query=data,
                expanded=query_text,
                expansions=(),
                method="domain_specific",
            )

        # Collect all expansion terms from matched keys
        expansion_terms = set()
        for _, expansion in expansions_applied:
            expansion_terms.update(expansion.split())

        # Remove terms already in the query (avoid duplication)
        # Use lowercase comparison to catch case-insensitive duplicates
        query_terms = set(query_lower.split())
        expansion_terms_lower = {term.lower() for term in expansion_terms}
        new_terms_lower = expansion_terms_lower - query_terms

        # Map back to original case from expansion_terms
        new_terms = {
            term for term in expansion_terms if term.lower() in new_terms_lower
        }

        # Build expanded query: original + new terms in sorted order
        expanded_text = f"{query_text} {' '.join(sorted(new_terms))}"

        # Extract just the expansion keys (not the full expansion text)
        expansion_keys = tuple(term for term, _ in expansions_applied)

        self._log.debug(
            f"[QueryExpander] Completed expansion with {len(expansion_keys)} terms: {expansion_keys}"
        )

        return ExpandedQuery(
            query=data,
            expanded=expanded_text,
            expansions=expansion_keys,
            method="domain_specific",
        )
