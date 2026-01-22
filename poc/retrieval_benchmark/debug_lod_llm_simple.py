#!/usr/bin/env python3
"""Simple focused test of LLM document selection."""

from retrieval_benchmark.llms.ollama import OllamaLLM

def test_doc_selection():
    llm = OllamaLLM("llama3.2:3b")
    
    # The EXACT documents that would be retrieved
    docs = """[wiki_88b6b781] Python (programming language)

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.

[wiki_e2723b0d] Rust (programming language)

Rust is a general-purpose programming language emphasizing performance, type safety, and concurrency. It enforces memory safety.

[wiki_d83f147f] Operating system

An operating system (OS) is system software that manages computer hardware and software resources.

[wiki_20a8b555] Computer network

A computer network is a set of computers sharing resources located on or provided by network nodes."""

    queries = [
        ("Who created Python?", "wiki_88b6b781"),
        ("What is Rust's ownership system?", "wiki_e2723b0d"),
        ("What are operating system kernels?", "wiki_d83f147f"),
        ("What is the TCP/IP protocol?", "wiki_20a8b555"),
    ]
    
    print("=" * 60)
    print("Testing document selection prompts")
    print("=" * 60)
    
    for query, expected in queries:
        # Original prompt style
        prompt1 = f"""Select documents most likely to answer this question.

Question: {query}

Documents:
{docs}

Instructions:
- Return ONLY document IDs from the list above, comma-separated
- Choose 1-3 most relevant documents
- Document IDs look like: wiki_abc123

Selected:"""

        # Alternative: More direct prompt
        prompt2 = f"""Which document answers: "{query}"

Documents:
{docs}

Reply with ONLY the document ID (e.g., wiki_88b6b781). No explanation."""

        # Alternative: Even simpler
        prompt3 = f"""Question: {query}

Available documents:
- wiki_88b6b781: Python (programming language)  
- wiki_e2723b0d: Rust (programming language)
- wiki_d83f147f: Operating system
- wiki_20a8b555: Computer network

Which document ID would answer this question? Reply with just the ID."""

        print(f"\n--- Query: {query}")
        print(f"    Expected: {expected}")
        print()
        
        print("Original prompt style:")
        resp1 = llm.generate(prompt1, max_tokens=50, temperature=0.0)
        print(f"  Response: {resp1}")
        correct1 = expected in resp1
        print(f"  Correct: {correct1}")
        
        print("\nDirect prompt style:")
        resp2 = llm.generate(prompt2, max_tokens=50, temperature=0.0)
        print(f"  Response: {resp2}")
        correct2 = expected in resp2
        print(f"  Correct: {correct2}")
        
        print("\nSimple prompt style:")
        resp3 = llm.generate(prompt3, max_tokens=50, temperature=0.0)
        print(f"  Response: {resp3}")
        correct3 = expected in resp3
        print(f"  Correct: {correct3}")
        
        print("-" * 40)


def test_section_selection():
    llm = OllamaLLM("llama3.2:3b")
    
    sections = """[sec_1] History
Python was conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands.

[sec_2] Design philosophy and features
Python is a multi-paradigm programming language.

[sec_6] Implementations
Most Python implementations (including CPython) include a REPL.

[sec_8] Development
Python's development is conducted through the PEP process."""

    query = "Who created Python?"
    expected = "sec_1"
    
    print("\n" + "=" * 60)
    print("Testing section selection prompts")
    print("=" * 60)
    
    # Original style
    prompt1 = f"""Select sections most likely to answer this question.

Question: {query}

Sections:
{sections}

Instructions:
- Return ONLY section IDs from the list above, comma-separated
- Choose 2-4 most relevant sections
- Section IDs look like: sec_2, sec_7_6, sec_3_1

Selected:"""

    # Simpler style
    prompt2 = f"""Question: {query}

Sections:
{sections}

Which section ID answers this question? Reply with just the section ID (e.g., sec_1)."""

    print(f"\nQuery: {query}")
    print(f"Expected: {expected}")
    
    print("\nOriginal prompt:")
    resp1 = llm.generate(prompt1, max_tokens=50, temperature=0.0)
    print(f"  Response: {resp1}")
    
    print("\nSimple prompt:")
    resp2 = llm.generate(prompt2, max_tokens=50, temperature=0.0)
    print(f"  Response: {resp2}")


if __name__ == "__main__":
    test_doc_selection()
    test_section_selection()
