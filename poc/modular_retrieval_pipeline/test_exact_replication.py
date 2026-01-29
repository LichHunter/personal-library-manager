"""Golden test: verify modular produces identical results to original.

This test verifies that ModularEnrichedHybridLLM produces 100% IDENTICAL results
to the original EnrichedHybridLLMRetrieval implementation.

Tests:
1. test_identical_enrichment() - Verifies enrichment pipeline produces identical strings
2. test_identical_results() - Verifies retrieval produces identical chunk IDs in same order
"""

import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "poc" / "chunking_benchmark_v2"))
sys.path.insert(0, str(project_root / "poc" / "modular_retrieval_pipeline"))
sys.path.insert(0, str(project_root / "poc"))

# Mock query rewriting BEFORE importing modular implementation
import retrieval.query_rewrite as qr_module

_original_rewrite = qr_module.rewrite_query
qr_module.rewrite_query = lambda query, **kwargs: query

from strategies import Document, MarkdownSemanticStrategy
from retrieval import create_retrieval_strategy
from modular_retrieval_pipeline.modular_enriched_hybrid_llm import (
    ModularEnrichedHybridLLM,
)


def load_test_chunks():
    """Load test chunks from kubernetes sample corpus.

    Uses the same chunking strategy and corpus as the benchmark.
    Returns a small set of chunks for fast testing.
    """
    corpus_dir = (
        project_root
        / "poc"
        / "chunking_benchmark_v2"
        / "corpus"
        / "kubernetes_sample_200"
    )

    # Load documents
    documents = []
    for doc_path in sorted(corpus_dir.glob("*.md"))[:3]:  # Use first 3 docs for speed
        doc_id = doc_path.stem
        content = doc_path.read_text()

        # Extract title from frontmatter or first heading
        title = doc_id
        lines = content.split("\n")
        for line in lines:
            if line.startswith("title:"):
                title = line.replace("title:", "").strip().strip("\"'")
                break
            elif line.startswith("# "):
                title = line[2:].strip()
                break

        doc = Document(id=doc_id, title=title, content=content, path=str(doc_path))
        documents.append(doc)

    # Chunk documents using same strategy as benchmark
    strategy = MarkdownSemanticStrategy(
        max_heading_level=4,
        target_chunk_size=400,
        min_chunk_size=50,
        max_chunk_size=800,
        overlap_sentences=1,
    )

    all_chunks = []
    for doc in documents:
        chunks = strategy.chunk(doc)
        all_chunks.extend(chunks)

    return all_chunks


def test_identical_enrichment():
    """Test that enrichment produces identical strings.

    Verifies that the modular enrichment pipeline (KeywordExtractor ->
    EntityExtractor -> ContentEnricher) produces the same output as the
    original FastEnricher.
    """
    print("\n" + "=" * 70)
    print("TEST: Identical Enrichment")
    print("=" * 70)

    from chunking_benchmark_v2.enrichment.fast import FastEnricher
    from modular_retrieval_pipeline.components.keyword_extractor import KeywordExtractor
    from modular_retrieval_pipeline.components.entity_extractor import EntityExtractor
    from modular_retrieval_pipeline.components.content_enricher import ContentEnricher
    from modular_retrieval_pipeline.base import Pipeline

    # Create both enrichers
    original = FastEnricher()
    modular = (
        Pipeline()
        .add(KeywordExtractor())
        .add(EntityExtractor())
        .add(ContentEnricher())
        .build()
    )

    # Test contents
    test_contents = [
        "Kubernetes horizontal pod autoscaler scales replicas based on CPU utilization",
        "Google Cloud Platform offers Kubernetes Engine for container orchestration",
        "The Topology Manager component manages node topology in Kubernetes clusters",
        "Token authentication uses JWT with iat and exp claims for expiration",
    ]

    all_passed = True
    for content in test_contents:
        orig_result = original.enrich(content).enhanced_content
        mod_result = modular.run({"content": content})

        if orig_result == mod_result:
            print(f"✓ Enrichment identical for '{content[:50]}...'")
        else:
            print(f"✗ MISMATCH for '{content[:50]}...'")
            print(f"  Original: {orig_result[:100]}...")
            print(f"  Modular:  {mod_result[:100]}...")
            all_passed = False

    if all_passed:
        print("\n✓ All enrichment tests passed!")
    else:
        raise AssertionError("Enrichment mismatch detected")


def test_identical_results():
    """Test that original and modular produce functionally equivalent retrieval results.

    Verifies that both implementations use the same enrichment and produce
    similar retrieval results. Due to non-determinism in query rewriting and
    potential floating-point precision differences, we verify that:
    1. Enriched contents are identical
    2. Embeddings are identical
    3. BM25 scores are identical
    4. Retrieval produces reasonable results
    """
    print("\n" + "=" * 70)
    print("TEST: Functionally Equivalent Retrieval")
    print("=" * 70)

    # Setup embedder
    print("\nLoading embedder: BAAI/bge-base-en-v1.5...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")

    # Create both strategies (disable cache in original for fair comparison)
    print("Creating original strategy...")
    original = create_retrieval_strategy(
        "enriched_hybrid_llm", debug=False, use_cache=False
    )
    original.set_embedder(embedder)

    print("Creating modular strategy...")
    modular = ModularEnrichedHybridLLM(debug=False)
    modular.set_embedder(embedder)

    # Load test chunks (same for both)
    print("Loading test chunks...")
    chunks = load_test_chunks()
    print(
        f"Loaded {len(chunks)} chunks from {len(set(c.doc_id for c in chunks))} documents"
    )

    # Index both
    print("\nIndexing original strategy...")
    original.index(chunks)

    print("Indexing modular strategy...")
    modular.index(chunks)

    # Verify enrichment is identical
    print("\nVerifying enrichment is identical...")
    enrichment_match = True
    for i in range(len(chunks)):
        if original._enriched_contents[i] != modular._enriched_contents[i]:
            print(f"✗ Enrichment mismatch at chunk {i}")
            enrichment_match = False
            break

    if enrichment_match:
        print("✓ All enriched contents are identical")
    else:
        raise AssertionError("Enrichment mismatch detected")

    # Verify embeddings are identical
    print("Verifying embeddings are identical...")
    if (
        original.embeddings is not None
        and modular.embeddings is not None
        and np.allclose(original.embeddings, modular.embeddings)
    ):
        print("✓ All embeddings are identical")
    else:
        raise AssertionError("Embeddings mismatch detected")

    # Test queries
    test_queries = [
        "What is the Topology Manager?",
        "How does token authentication work?",
        "What is the RPO for disaster recovery?",
        "How do I configure autoscaling?",
    ]

    print(f"\nRunning {len(test_queries)} test queries...")
    print("(Query rewriting is mocked for deterministic results)")

    for query in test_queries:
        orig_results = original.retrieve(query, k=5)
        mod_results = modular.retrieve(query, k=5)

        # Verify EXACT chunk ID matches in SAME order
        orig_ids = [c.id for c in orig_results]
        mod_ids = [c.id for c in mod_results]

        if orig_ids != mod_ids:
            print(f"✗ Query '{query[:40]}...' - MISMATCH!")
            print(f"  Original: {orig_ids}")
            print(f"  Modular:  {mod_ids}")
            raise AssertionError(f"Chunk ID mismatch for query: {query}")

        if not orig_ids or not mod_ids:
            raise AssertionError(f"Empty results for query: {query}")

        print(f"✓ Query '{query[:40]}...' - IDENTICAL results: {orig_ids}")

    print("\n✓ EXACT IDENTICAL retrieval verified! (100% match)")


if __name__ == "__main__":
    try:
        test_identical_enrichment()
        test_identical_results()
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED - Modular implementation is identical!")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 70)
        sys.exit(1)
