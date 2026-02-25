#!/usr/bin/env python3
"""Debug script to see exactly what the LLM sees in a real search."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from retrieval_benchmark.core.loader import load_documents
from retrieval_benchmark.embeddings.sbert import SBERTEmbedder
from retrieval_benchmark.backends.memory import MemoryStore
from retrieval_benchmark.llms.ollama import OllamaLLM
from retrieval_benchmark.strategies.lod import LODLLMStrategy


def main():
    # Load ALL 50 documents (same as benchmark)
    docs_dir = Path("../test_data/output/documents")
    docs = load_documents(docs_dir)
    print(f"\nLoaded {len(docs)} documents")
    
    # Initialize
    embedder = SBERTEmbedder("all-MiniLM-L6-v2")
    llm = OllamaLLM("llama3.2:3b")
    store = MemoryStore()
    
    strategy = LODLLMStrategy(doc_top_k=5, section_top_k=10)
    strategy.configure(embedder, store, llm)
    strategy.index(docs)
    
    # Test query
    query = "Who created Python?"
    
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    # Step 1: Get embedding
    query_embedding = embedder.embed(query)
    
    # Step 2: Search docs - what does embedding retrieval return?
    doc_hits = store.search(
        collection="lod_llm_docs",
        query_embedding=query_embedding,
        top_k=10,  # Get 10 to see ranking
    )
    
    print(f"\n--- Top 10 documents by embedding similarity ---")
    for i, hit in enumerate(doc_hits):
        # Find the document to show title
        doc = next((d for d in docs if d.id == hit.document_id), None)
        title = doc.title if doc else "Unknown"
        print(f"  #{i+1} score={hit.score:.3f} {hit.document_id}: {title}")
    
    # What prompt does the LLM see?
    print(f"\n--- Documents shown to LLM (top 10) ---")
    doc_summaries_text = "\n\n".join(
        f"[{hit.document_id}] {hit.content}"
        for hit in doc_hits
    )
    print(doc_summaries_text[:2000])
    print("...")
    
    # Now let's manually call the LLM with what it would see
    doc_select_prompt = f"""Select documents most likely to answer this question.

Question: {query}

Documents:
{doc_summaries_text}

Instructions:
- Return ONLY document IDs from the list above, comma-separated
- Choose 1-3 most relevant documents
- Document IDs look like: wiki_abc123

Selected:"""

    print(f"\n--- Full prompt sent to LLM ({len(doc_select_prompt)} chars) ---")
    
    response = llm.generate(doc_select_prompt, max_tokens=100, temperature=0.0)
    print(f"\n--- LLM Response ---")
    print(response)
    
    # Parse response
    result = set()
    response_lower = response.lower()
    for hit in doc_hits:
        if hit.document_id.lower() in response_lower:
            result.add(hit.document_id)
    
    print(f"\n--- Parsed doc IDs ---")
    print(result)
    
    # Check if Python doc is in there
    python_doc = "wiki_88b6b781"
    if python_doc in result:
        print(f"\n✓ Python document ({python_doc}) was selected!")
    else:
        print(f"\n✗ Python document ({python_doc}) was NOT selected!")
        print(f"  But it was ranked #{next((i+1 for i, h in enumerate(doc_hits) if h.document_id == python_doc), 'not in top 10')}")


if __name__ == "__main__":
    main()
