#!/usr/bin/env python3
"""POC Comparison Test: Verify production pipeline matches POC pipeline results.

This script compares the production HybridRetriever against the POC
ModularEnrichedHybridLLM to verify behavior parity.

Comparison methodology:
- Uses same test data (first 3 docs from kubernetes_sample_200, 20 needle questions)
- Uses same chunking strategy (MarkdownSemanticStrategy)
- Uses same embedder (BAAI/bge-base-en-v1.5)
- Uses use_rewrite=False in both pipelines (deterministic, no LLM calls)
- Compares CONTENT SETS (not chunk IDs, which differ by design)

Target: >= 90% match rate (18/20 queries)
"""

import json
import sys
import tempfile
from pathlib import Path

# Add paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "poc" / "chunking_benchmark_v2"))
sys.path.insert(0, str(project_root / "poc" / "modular_retrieval_pipeline"))
sys.path.insert(0, str(project_root / "poc"))
sys.path.insert(0, str(project_root / "src"))

import numpy as np
from sentence_transformers import SentenceTransformer

from strategies import Document, MarkdownSemanticStrategy
from modular_retrieval_pipeline.modular_enriched_hybrid_llm import ModularEnrichedHybridLLM
from modular_retrieval_pipeline.components.keyword_extractor import KeywordExtractor
from modular_retrieval_pipeline.components.entity_extractor import EntityExtractor
from modular_retrieval_pipeline.components.query_rewriter import QueryRewriter
from modular_retrieval_pipeline.types import Query, RewrittenQuery

from plm.search.retriever import HybridRetriever


def mock_poc_query_rewriter():
    """Mock POC query rewriter to return original query (no LLM calls)."""
    def passthrough_process(self, data: Query) -> RewrittenQuery:
        return RewrittenQuery(
            original=data,
            rewritten=data.text,
            model="passthrough",
        )
    QueryRewriter.process = passthrough_process


# Mock query rewriting for deterministic results
mock_poc_query_rewriter()


def load_documents(corpus_dir: Path, limit: int = 3) -> list[Document]:
    """Load documents from corpus directory.

    Args:
        corpus_dir: Path to corpus directory containing .md files
        limit: Number of documents to load (default: 3)

    Returns:
        List of Document objects
    """
    documents = []
    for doc_path in sorted(corpus_dir.glob("*.md"))[:limit]:
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

    return documents


def chunk_documents(documents: list[Document]) -> list:
    """Chunk documents using MarkdownSemanticStrategy.

    Uses same parameters as POC test for consistency.

    Args:
        documents: List of Document objects

    Returns:
        List of Chunk objects
    """
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


def load_questions(questions_path: Path) -> list[dict]:
    """Load test questions from JSON file.

    Args:
        questions_path: Path to questions JSON file

    Returns:
        List of question dicts with 'id' and 'question' fields
    """
    with open(questions_path) as f:
        data = json.load(f)
    return data["questions"]


def extract_keywords_and_entities(chunks: list, keyword_extractor, entity_extractor) -> list[dict]:
    """Extract keywords and entities from chunks for production pipeline.

    Args:
        chunks: List of Chunk objects from MarkdownSemanticStrategy
        keyword_extractor: KeywordExtractor instance
        entity_extractor: EntityExtractor instance

    Returns:
        List of chunk dicts with 'content', 'keywords', 'entities' fields
    """
    chunk_dicts = []
    for chunk in chunks:
        # Extract keywords
        kw_data = keyword_extractor.process({"content": chunk.content})
        keywords = kw_data.get("keywords", [])

        # Extract entities
        ent_data = entity_extractor.process({"content": chunk.content})
        entities = ent_data.get("entities", {})

        chunk_dicts.append({
            "content": chunk.content,
            "keywords": keywords,
            "entities": entities,
            "heading": getattr(chunk, "heading", None),
        })

    return chunk_dicts


def run_poc_pipeline(chunks: list, embedder, queries: list[str], k: int = 5) -> list[set[str]]:
    """Run POC pipeline and collect results.

    Args:
        chunks: List of Chunk objects
        embedder: SentenceTransformer embedder
        queries: List of query strings
        k: Number of results per query

    Returns:
        List of content sets, one per query
    """
    print("\n--- Running POC Pipeline ---")
    poc_strategy = ModularEnrichedHybridLLM(debug=False)
    poc_strategy.set_embedder(embedder)

    print(f"Indexing {len(chunks)} chunks...")
    poc_strategy.index(chunks)

    results = []
    for query in queries:
        poc_results = poc_strategy.retrieve(query, k=k)
        # Extract content from Chunk objects
        content_set = {chunk.content for chunk in poc_results}
        results.append(content_set)

    print(f"Completed {len(queries)} queries")
    return results


def run_production_pipeline(chunk_dicts: list[dict], doc_id: str, embedder, queries: list[str], k: int = 5) -> list[set[str]]:
    """Run production pipeline and collect results.

    Args:
        chunk_dicts: List of chunk dicts with 'content', 'keywords', 'entities'
        doc_id: Document ID for ingestion
        embedder: SentenceTransformer embedder (for production, internal embedder is used)
        queries: List of query strings
        k: Number of results per query

    Returns:
        List of content sets, one per query
    """
    print("\n--- Running Production Pipeline ---")

    # Create temp directory for production pipeline
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_index.db"
        bm25_path = Path(tmpdir) / "bm25_index"

        prod_retriever = HybridRetriever(
            db_path=str(db_path),
            bm25_index_path=str(bm25_path),
            rewrite_timeout=5.0,
        )

        print(f"Ingesting {len(chunk_dicts)} chunks...")
        prod_retriever.ingest_document(
            doc_id=doc_id,
            source_file="test.md",
            chunks=chunk_dicts,
            rebuild_index=True,
        )

        results = []
        for query in queries:
            prod_results = prod_retriever.retrieve(query, k=k, use_rewrite=False)
            # Extract content from result dicts
            content_set = {result["content"] for result in prod_results}
            results.append(content_set)

        print(f"Completed {len(queries)} queries")
        return results


def compare_results(
    poc_results: list[set[str]],
    prod_results: list[set[str]],
    questions: list[dict],
) -> tuple[int, int, list[dict]]:
    """Compare POC and production results.

    Args:
        poc_results: List of content sets from POC pipeline
        prod_results: List of content sets from production pipeline
        questions: List of question dicts

    Returns:
        Tuple of (matches, total, diff_details)
    """
    matches = 0
    total = len(questions)
    diff_details = []

    print("\n--- Comparison Results ---")
    for i, (poc_set, prod_set, q) in enumerate(zip(poc_results, prod_results, questions)):
        query = q["question"]
        q_id = q["id"]

        if poc_set == prod_set:
            status = "MATCH"
            matches += 1
            print(f"Query {i+1}/{total}: {q_id} - {query[:50]}...")
            print(f"  POC: {len(poc_set)} results, Prod: {len(prod_set)} results")
            print(f"  Status: {status}")
        else:
            status = "DIFF"
            # Find differences
            only_poc = poc_set - prod_set
            only_prod = prod_set - poc_set

            diff_detail = {
                "query_id": q_id,
                "query": query,
                "poc_count": len(poc_set),
                "prod_count": len(prod_set),
                "only_in_poc": [c[:100] + "..." for c in only_poc],
                "only_in_prod": [c[:100] + "..." for c in only_prod],
            }
            diff_details.append(diff_detail)

            print(f"Query {i+1}/{total}: {q_id} - {query[:50]}...")
            print(f"  POC: {len(poc_set)} results, Prod: {len(prod_set)} results")
            print(f"  Status: {status}")
            if only_poc:
                print(f"    Only in POC ({len(only_poc)} chunks):")
                for c in list(only_poc)[:2]:
                    print(f"      - {c[:80]}...")
            if only_prod:
                print(f"    Only in Prod ({len(only_prod)} chunks):")
                for c in list(only_prod)[:2]:
                    print(f"      - {c[:80]}...")

        print()

    return matches, total, diff_details


def main():
    """Main entry point for POC comparison test."""
    print("=" * 70)
    print("POC Comparison Test: Production vs POC Pipeline")
    print("=" * 70)

    # Paths
    corpus_dir = project_root / "poc" / "chunking_benchmark_v2" / "corpus" / "kubernetes_sample_200"
    questions_path = project_root / "poc" / "chunking_benchmark_v2" / "corpus" / "needle_questions.json"

    # Load test data
    print("\n--- Loading Test Data ---")
    documents = load_documents(corpus_dir, limit=3)
    print(f"Loaded {len(documents)} documents: {[d.id for d in documents]}")

    chunks = chunk_documents(documents)
    print(f"Chunked into {len(chunks)} chunks")

    questions = load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    # Setup embedder (shared)
    print("\n--- Loading Embedder ---")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print("Loaded BAAI/bge-base-en-v1.5")

    # Setup extractors for production pipeline
    keyword_extractor = KeywordExtractor(max_keywords=10)
    entity_extractor = EntityExtractor()

    # Extract keywords/entities for production pipeline
    print("\n--- Extracting Keywords/Entities for Production ---")
    chunk_dicts = extract_keywords_and_entities(chunks, keyword_extractor, entity_extractor)
    print(f"Extracted metadata for {len(chunk_dicts)} chunks")

    # Prepare queries
    queries = [q["question"] for q in questions]

    # Run POC pipeline
    poc_results = run_poc_pipeline(chunks, embedder, queries, k=5)

    # Run production pipeline
    # Use first doc_id as identifier (production ingests per-document)
    prod_results = run_production_pipeline(chunk_dicts, "test_corpus", embedder, queries, k=5)

    # Compare results
    matches, total, diff_details = compare_results(poc_results, prod_results, questions)

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    match_rate = (matches / total) * 100
    print(f"Match Rate: {matches}/{total} ({match_rate:.1f}%)")
    print(f"Target: >= 90% (18/20)")

    if match_rate >= 90:
        print("\nTarget ACHIEVED")
        return 0
    else:
        print(f"\nTarget NOT achieved")
        print(f"\nDifferences ({total - matches} queries):")
        for diff in diff_details:
            print(f"  - {diff['query_id']}: {diff['query'][:60]}...")
            print(f"    POC: {diff['poc_count']} chunks, Prod: {diff['prod_count']} chunks")

        # Explain potential differences
        print("\nPotential causes of differences:")
        print("  1. Enrichment pipeline order/format differences")
        print("  2. BM25 tokenization differences")
        print("  3. Embedding normalization differences")
        print("  4. RRF score tie-breaking differences")

        return 1


if __name__ == "__main__":
    sys.exit(main())
