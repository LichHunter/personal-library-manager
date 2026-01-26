"""Hybrid GEMS retrieval strategy with query-type routing.

Routes queries to the best specialist strategy based on query type classification:
- Temporal queries → BM25F Hybrid (7.3/10)
- All other queries → Synthetic Variants (6.6-7.5/10)

Expected performance: ~6.8-7.0/10 (+0.2-0.4 improvement over baseline)
"""

from typing import Optional

from strategies import Chunk, Document
from .base import RetrievalStrategy, StructuredDocument
from .synthetic_variants import SyntheticVariantsRetrieval
from .bm25f_hybrid import BM25FHybridRetrieval


class HybridGemsRetrieval(RetrievalStrategy):
    """Hybrid routing strategy that selects best specialist per query type.

    Implements query-type routing based on phase3-decision.md analysis:
    - Classifies queries into 5 types: temporal, multi-hop, comparative, negation, implicit
    - Routes temporal queries to BM25F Hybrid (specialist advantage)
    - Routes all other queries to Synthetic Variants (best overall performer)

    Expected improvement: +0.2-0.4 points over baseline (6.6 → 6.8-7.0)
    """

    def __init__(self, name: str = "hybrid_gems", **kwargs):
        """Initialize hybrid strategy with two sub-strategies.

        Args:
            name: Strategy name (default: "hybrid_gems")
            **kwargs: Additional configuration passed to parent
        """
        super().__init__(name, **kwargs)

        # Initialize sub-strategies
        self.synthetic_strategy = SyntheticVariantsRetrieval(name="synthetic_variants")
        self.bm25f_strategy = BM25FHybridRetrieval(name="bm25f_hybrid")

        # Routing table for strategy selection
        self.strategies = {
            "synthetic_variants": self.synthetic_strategy,
            "bm25f_hybrid": self.bm25f_strategy,
        }

    def set_embedder(self, embedder, use_prefix: bool = False):
        """Set embedder for both sub-strategies.

        Args:
            embedder: Embedding model
            use_prefix: Whether to use prefix for query encoding
        """
        self.synthetic_strategy.set_embedder(embedder, use_prefix)
        self.bm25f_strategy.set_embedder(embedder, use_prefix)

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Index chunks in both sub-strategies.

        Args:
            chunks: List of chunks to index
            documents: Optional list of source documents
            structured_docs: Optional structured documents with sections
        """
        self.synthetic_strategy.index(chunks, documents, structured_docs)
        self.bm25f_strategy.index(chunks, documents, structured_docs)

    def _classify_query_type(self, query: str) -> str:
        """Classify query into one of 5 types.

        Classification priority (most specific first):
        1. Negation: 'not', 'without', 'except', 'exclude', 'never'
        2. Comparative: 'vs', 'versus', 'difference', 'compare', 'better', 'worse'
        3. Temporal: 'when', 'sequence', 'order', 'timeline', 'after', 'before', 'during'
        4. Multi-hop: 'compare', 'relate', 'both', 'and', 'between'
        5. Implicit: Default for queries that don't match other categories

        Args:
            query: Search query string

        Returns:
            Query type: 'temporal', 'multi-hop', 'comparative', 'negation', or 'implicit'
        """
        temporal_keywords = [
            "when",
            "sequence",
            "order",
            "timeline",
            "after",
            "before",
            "during",
        ]
        multihop_keywords = ["compare", "relate", "both", "and", "between"]
        negation_keywords = ["not", "without", "except", "exclude", "never"]
        comparative_keywords = [
            "vs",
            "versus",
            "difference",
            "compare",
            "better",
            "worse",
        ]

        query_lower = query.lower()

        # Priority order (most specific first)
        if any(kw in query_lower for kw in negation_keywords):
            return "negation"
        if any(kw in query_lower for kw in comparative_keywords):
            return "comparative"
        if any(kw in query_lower for kw in temporal_keywords):
            return "temporal"
        if any(kw in query_lower for kw in multihop_keywords):
            return "multi-hop"

        return "implicit"  # Default

    def _route_query(self, query_type: str) -> str:
        """Route query to best strategy based on type.

        Routing table based on phase3-decision.md analysis:
        - Temporal: BM25F Hybrid (7.3/10) - specialist advantage
        - Multi-hop: Synthetic Variants (7.5/10)
        - Comparative: Synthetic Variants (7.0/10)
        - Negation: Synthetic Variants (5.4/10)
        - Implicit: Synthetic Variants (7.0/10)

        Args:
            query_type: Query type from classification

        Returns:
            Strategy name: 'bm25f_hybrid' or 'synthetic_variants'
        """
        routing_table = {
            "temporal": "bm25f_hybrid",  # 7.3/10 - specialist advantage
            "multi-hop": "synthetic_variants",  # 7.5/10
            "comparative": "synthetic_variants",  # 7.0/10
            "negation": "synthetic_variants",  # 5.4/10 (best of weak options)
            "implicit": "synthetic_variants",  # 7.0/10
        }

        return routing_table.get(query_type, "synthetic_variants")

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve top-k chunks by classifying query and routing to best strategy.

        Process:
        1. Classify query type using keyword matching
        2. Route to appropriate strategy (temporal → BM25F, others → Synthetic)
        3. Execute retrieval on selected strategy

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of top-k chunks from selected strategy
        """
        # Classify query type
        query_type = self._classify_query_type(query)

        # Route to best strategy for this type
        strategy_name = self._route_query(query_type)
        strategy = self.strategies[strategy_name]

        # Execute retrieval on selected strategy
        return strategy.retrieve(query, k)

    def get_index_stats(self) -> dict:
        """Return statistics about both sub-strategy indexes.

        Returns:
            Dict with num_chunks and stats from both strategies
        """
        return {
            "num_chunks": len(self.synthetic_strategy.chunks)
            if self.synthetic_strategy.chunks
            else 0,
            "strategies": {
                "synthetic_variants": self.synthetic_strategy.get_index_stats(),
                "bm25f_hybrid": self.bm25f_strategy.get_index_stats(),
            },
        }
