#!/usr/bin/env python3
"""Test potential fixes for LOD LLM document selection."""

from retrieval_benchmark.llms.ollama import OllamaLLM

def main():
    llm = OllamaLLM("llama3.2:3b")
    
    # Simulate what embedding retrieval returns for "Who created Python?"
    # The ACTUAL top 10 from embedding search
    docs_full = """[wiki_88b6b781] Python (programming language)

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming.

[wiki_03f339ae] Ada Lovelace

Augusta Ada King, Countess of Lovelace (née Byron; 10 December 1815 – 27 November 1852), also known as Ada Lovelace, was an English mathematician and writer chiefly known for work on Charles Babbage's proposed mechanical general-purpose computer.

[wiki_99aeefd0] Go (programming language)

Go is a high-level, general-purpose programming language that is statically-typed and compiled. It was designed at Google in 2007 by Robert Griesemer, Rob Pike, and Ken Thompson.

[wiki_40a703ca] Nikola Tesla

Nikola Tesla (10 July 1856 – 7 January 1943) was a Serbian-American engineer, futurist, and inventor.

[wiki_464e3cbf] Leonardo da Vinci

Leonardo di ser Piero da Vinci (15 April 1452 – 2 May 1519) was an Italian polymath of the High Renaissance.

[wiki_e2723b0d] Rust (programming language)

Rust is a general-purpose programming language emphasizing performance, type safety, and concurrency.

[wiki_7d929874] Renaissance

The Renaissance is a period of history and a European cultural movement covering the 15th and 16th centuries.

[wiki_5b7438c3] Theory of relativity

The theory of relativity usually encompasses two interrelated physics theories by Albert Einstein.

[wiki_20a8b555] Computer network

A computer network is a set of computers sharing resources located on or provided by network nodes.

[wiki_56f15824] Albert Einstein

Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist."""

    # Reduced set (top 5 only)
    docs_reduced = """[wiki_88b6b781] Python (programming language)

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.

[wiki_03f339ae] Ada Lovelace

Augusta Ada King, Countess of Lovelace, was an English mathematician known for work on Charles Babbage's analytical engine.

[wiki_99aeefd0] Go (programming language)

Go is a high-level programming language designed at Google in 2007 by Robert Griesemer, Rob Pike, and Ken Thompson.

[wiki_40a703ca] Nikola Tesla

Nikola Tesla was a Serbian-American engineer and inventor known for the modern AC electricity system.

[wiki_464e3cbf] Leonardo da Vinci

Leonardo da Vinci was an Italian polymath of the High Renaissance."""

    query = "Who created Python?"
    expected = "wiki_88b6b781"
    
    print("="*60)
    print(f"Query: {query}")
    print(f"Expected: {expected} (Python)")
    print("="*60)
    
    # Test 1: Original prompt with 10 docs
    print("\n--- Test 1: Original prompt, 10 docs ---")
    prompt1 = f"""Select documents most likely to answer this question.

Question: {query}

Documents:
{docs_full}

Instructions:
- Return ONLY document IDs from the list above, comma-separated
- Choose 1-3 most relevant documents
- Document IDs look like: wiki_abc123

Selected:"""

    resp1 = llm.generate(prompt1, max_tokens=100, temperature=0.0)
    print(f"Response: {resp1}")
    print(f"Correct: {expected in resp1}")
    
    # Test 2: Original prompt with 5 docs
    print("\n--- Test 2: Original prompt, 5 docs ---")
    prompt2 = f"""Select documents most likely to answer this question.

Question: {query}

Documents:
{docs_reduced}

Instructions:
- Return ONLY document IDs from the list above, comma-separated
- Choose 1-3 most relevant documents
- Document IDs look like: wiki_abc123

Selected:"""

    resp2 = llm.generate(prompt2, max_tokens=100, temperature=0.0)
    print(f"Response: {resp2}")
    print(f"Correct: {expected in resp2}")
    
    # Test 3: Simpler prompt with titles only
    print("\n--- Test 3: Simpler prompt, titles only ---")
    prompt3 = f"""Question: {query}

Which document answers this? Pick ONE.

1. wiki_88b6b781 - Python (programming language)
2. wiki_03f339ae - Ada Lovelace
3. wiki_99aeefd0 - Go (programming language)
4. wiki_40a703ca - Nikola Tesla  
5. wiki_464e3cbf - Leonardo da Vinci

Answer with just the document ID:"""

    resp3 = llm.generate(prompt3, max_tokens=50, temperature=0.0)
    print(f"Response: {resp3}")
    print(f"Correct: {expected in resp3}")
    
    # Test 4: More explicit prompt
    print("\n--- Test 4: Explicit ranking prompt ---")
    prompt4 = f"""I will give you a question and a list of documents. Return the ID of the FIRST document that is relevant to the question.

Question: {query}

Documents (in order of embedding similarity):
1. wiki_88b6b781: Python (programming language) - A high-level programming language emphasizing code readability.
2. wiki_03f339ae: Ada Lovelace - An English mathematician known for work on Babbage's analytical engine.
3. wiki_99aeefd0: Go (programming language) - A programming language designed at Google.
4. wiki_40a703ca: Nikola Tesla - A Serbian-American engineer and inventor.
5. wiki_464e3cbf: Leonardo da Vinci - An Italian Renaissance polymath.

The question asks about who created Python. Look at document #1 - it's about Python programming language.

Which document ID answers the question? Reply with ONLY the ID:"""

    resp4 = llm.generate(prompt4, max_tokens=50, temperature=0.0)
    print(f"Response: {resp4}")
    print(f"Correct: {expected in resp4}")
    
    # Test 5: Just trust the embedding and pick top 3
    print("\n--- Test 5: Hybrid approach (no LLM for doc selection) ---")
    print("If we just trust embedding similarity and take top 3:")
    print("  wiki_88b6b781 (Python) - CORRECT")
    print("  wiki_03f339ae (Ada Lovelace)")  
    print("  wiki_99aeefd0 (Go)")
    print("This would include the correct document!")
    
    # Test 6: Chain of thought
    print("\n--- Test 6: Chain of thought ---")
    prompt6 = f"""Question: {query}

Available documents:
- wiki_88b6b781: Python (programming language)
- wiki_03f339ae: Ada Lovelace  
- wiki_99aeefd0: Go (programming language)

Think step by step:
1. What is the question asking about? 
2. Which document topic matches?
3. What is the document ID?

Answer:"""

    resp6 = llm.generate(prompt6, max_tokens=150, temperature=0.0)
    print(f"Response: {resp6}")
    print(f"Correct: {expected in resp6}")


if __name__ == "__main__":
    main()
