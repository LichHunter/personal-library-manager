#!/usr/bin/env python3
"""Test with ACTUAL document content from real search."""

from retrieval_benchmark.llms.ollama import OllamaLLM

def main():
    llm = OllamaLLM("llama3.2:3b")
    
    # This is the ACTUAL content shown to LLM from the real search
    # (copy-pasted from the debug output)
    docs_actual = """[wiki_88b6b781] Python (programming language)

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including structured (particularly procedural), object-oriented and functional programming.

[wiki_03f339ae] Ada Lovelace

Augusta Ada King, Countess of Lovelace (née Byron; 10 December 1815 – 27 November 1852), also known as Ada Lovelace, was an English mathematician and writer chiefly known for work on Charles Babbage's proposed mechanical general-purpose computer, the analytical engine. She was the first to recognise the machine had applications beyond pure calculation. Lovelace is often considered the first computer programmer.

[wiki_99aeefd0] Go (programming language)

Go is a high-level, general-purpose programming language that is statically-typed and compiled. It is known for the simplicity of its syntax and the efficiency of development that it enables through the inclusion of a large standard library supplying many needs for common projects. It was designed at Google in 2007 by Robert Griesemer, Rob Pike, and Ken Thompson, and publicly announced in November 2009. It is syntactically similar to C, but also has garbage collection, structural typing, and CSP-style concurrency. It is often referred to as Golang to avoid ambiguity and because of its former domain name, golang.org, but its proper name is Go.

[wiki_40a703ca] Nikola Tesla

Nikola Tesla (10 July 1856 – 7 January 1943) was a Serbian-American engineer, futurist, and inventor. He is known for his contributions to the design of the modern alternating current (AC) electricity supply system.

[wiki_464e3cbf] Leonardo da Vinci

Leonardo di ser Piero da Vinci (15 April 1452 – 2 May 1519) was an Italian polymath of the High Renaissance who was active as a painter, draughtsman, engineer, scientist, theorist, sculptor, and architect. While his fame initially rested on his achievements as a painter, he has also become known for his notebooks, in which he made drawings and notes on a variety of subjects, including anatomy, astronomy, botany, cartography, painting, and paleontology. Leonardo is widely regarded to have been a genius who epitomized the Renaissance humanist ideal.

[wiki_e2723b0d] Rust (programming language)

Rust is a general-purpose programming language emphasizing performance, type safety, and concurrency. It enforces memory safety, meaning that all references point to valid memory. It does so without a traditional garbage collector; instead, memory safety errors and data races are prevented by the "borrow checker", which tracks the object lifetime of references at compile time.

[wiki_7d929874] Renaissance

The Renaissance is a period of history and a European cultural movement covering the 15th and 16th centuries. It marked the transition from the Late Middle Ages to modernity and was characterized by an effort to revive and surpass the ideas and achievements of classical antiquity.

[wiki_5b7438c3] Theory of relativity

The theory of relativity usually encompasses two interrelated physics theories by Albert Einstein: special relativity and general relativity, proposed and published in 1905 and 1915, respectively.

[wiki_20a8b555] Computer network

A computer network is a set of computers sharing resources located on or provided by network nodes. Computers use common communication protocols over digital interconnections to communicate with each other.

[wiki_56f15824] Albert Einstein

Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held as one of the most influential scientists in history."""

    query = "Who created Python?"
    expected = "wiki_88b6b781"
    
    print("="*60)
    print("Testing with ACTUAL content from real search")
    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print(f"Prompt length: {len(docs_actual)} chars")
    print("="*60)
    
    # Original prompt
    prompt = f"""Select documents most likely to answer this question.

Question: {query}

Documents:
{docs_actual}

Instructions:
- Return ONLY document IDs from the list above, comma-separated
- Choose 1-3 most relevant documents
- Document IDs look like: wiki_abc123

Selected:"""

    print(f"\nFull prompt length: {len(prompt)} chars")
    
    # Run multiple times to check consistency
    print("\n--- Running 5 times to check consistency ---")
    for i in range(5):
        resp = llm.generate(prompt, max_tokens=100, temperature=0.0)
        correct = expected in resp
        print(f"  Run {i+1}: {resp[:60]}... | Correct: {correct}")


if __name__ == "__main__":
    main()
