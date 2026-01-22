#!/usr/bin/env python3
"""Debug script to manually test LOD LLM strategy."""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Setup
from retrieval_benchmark.core.loader import load_documents
from retrieval_benchmark.embeddings.sbert import SBERTEmbedder
from retrieval_benchmark.backends.memory import MemoryStore
from retrieval_benchmark.llms.ollama import OllamaLLM
from retrieval_benchmark.strategies.lod import LODLLMStrategy, LODEmbedStrategy
from retrieval_benchmark.strategies.flat import FlatStrategy


def main():
    # Load just the 4 documents used in ground truth
    docs_dir = Path("../test_data/output/documents")
    all_docs = load_documents(docs_dir)
    
    target_doc_ids = {"wiki_88b6b781", "wiki_e2723b0d", "wiki_d83f147f", "wiki_20a8b555"}
    docs = [d for d in all_docs if d.id in target_doc_ids]
    print(f"\nLoaded {len(docs)} documents:")
    for d in docs:
        print(f"  - {d.id}: {d.title}")
    
    # Test queries that failed
    test_queries = [
        ("Who created Python?", "wiki_88b6b781", "sec_1"),
        ("What does BDFL stand for in the Python community?", "wiki_88b6b781", "sec_1"),
        ("When was the Rust Foundation formed?", "wiki_e2723b0d", "sec_1_5"),
    ]
    
    # Initialize components
    embedder = SBERTEmbedder("all-MiniLM-L6-v2")
    
    # Test with different models
    models_to_test = [
        "llama3.2:3b",
        # Add more models here if available
    ]
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"Testing with model: {model_name}")
        print(f"{'='*60}")
        
        try:
            llm = OllamaLLM(model_name)
        except Exception as e:
            print(f"  Failed to load model: {e}")
            continue
        
        # Test LOD LLM
        store = MemoryStore()
        strategy = LODLLMStrategy(doc_top_k=5, section_top_k=10)
        strategy.configure(embedder, store, llm)
        strategy.index(docs)
        
        for query, expected_doc, expected_section in test_queries:
            print(f"\n--- Query: {query}")
            print(f"    Expected: {expected_doc} / {expected_section}")
            
            result = strategy.search(query, top_k=5)
            
            if not result.hits:
                print(f"    RESULT: NO HITS!")
            else:
                print(f"    RESULT: {len(result.hits)} hits")
                for i, hit in enumerate(result.hits[:3]):
                    status = "CORRECT" if hit.document_id == expected_doc and hit.section_id == expected_section else "wrong"
                    print(f"      #{i+1} [{status}] {hit.document_id} / {hit.section_id}")
                    print(f"          {hit.content[:100]}...")
        
        # Compare with flat strategy
        print(f"\n--- Flat strategy comparison ---")
        flat_store = MemoryStore()
        flat_strategy = FlatStrategy()
        flat_strategy.configure(embedder, flat_store)
        flat_strategy.index(docs)
        
        for query, expected_doc, expected_section in test_queries:
            result = flat_strategy.search(query, top_k=5)
            hit = result.hits[0] if result.hits else None
            if hit:
                status = "CORRECT" if hit.document_id == expected_doc and hit.section_id == expected_section else "wrong"
                print(f"  {query[:40]}... -> [{status}] {hit.document_id}/{hit.section_id}")
            else:
                print(f"  {query[:40]}... -> NO HITS")


def test_llm_directly():
    """Test the LLM prompts directly to see what's happening."""
    print("\n" + "="*60)
    print("Direct LLM prompt testing")
    print("="*60)
    
    llm = OllamaLLM("llama3.2:3b")
    
    # Simulate the doc selection prompt
    doc_summaries = """[wiki_88b6b781] Python (programming language)

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms.

[wiki_e2723b0d] Rust (programming language)

Rust is a general-purpose programming language emphasizing performance, type safety, and concurrency. It enforces memory safety, meaning that all references point to valid memory.

[wiki_d83f147f] Operating system

An operating system (OS) is system software that manages computer hardware and software resources, and provides common services for computer programs."""

    query = "Who created Python?"
    
    prompt = f"""Select documents most likely to answer this question.

Question: {query}

Documents:
{doc_summaries}

Instructions:
- Return ONLY document IDs from the list above, comma-separated
- Choose 1-3 most relevant documents
- Document IDs look like: wiki_abc123

Selected:"""

    print(f"\nPrompt:\n{prompt}")
    print("\n" + "-"*40)
    
    response = llm.generate(prompt, max_tokens=100, temperature=0.0)
    print(f"Response: {response}")
    
    # Now test section selection
    section_summaries = """[sec_1] History

Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica (CWI) in the Netherlands. It was designed as a successor to the ABC programming language.

[sec_2] Design philosophy and features

Python is a multi-paradigm programming language. Object-oriented programming and structured programming are fully supported.

[sec_6] Implementations

Most Python implementations (including CPython) include a read–eval–print loop (REPL).

[sec_8] Development

Python's development is conducted mostly through the Python Enhancement Proposal (PEP) process."""

    section_prompt = f"""Select sections most likely to answer this question.

Question: {query}

Sections:
{section_summaries}

Instructions:
- Return ONLY section IDs from the list above, comma-separated
- Choose 2-4 most relevant sections
- Section IDs look like: sec_2, sec_7_6, sec_3_1

Selected:"""

    print(f"\nSection prompt:\n{section_prompt}")
    print("\n" + "-"*40)
    
    response = llm.generate(section_prompt, max_tokens=100, temperature=0.0)
    print(f"Response: {response}")


if __name__ == "__main__":
    test_llm_directly()
    print("\n\n")
    main()
