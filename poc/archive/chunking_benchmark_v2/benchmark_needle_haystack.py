#!/usr/bin/env python3
"""Needle-in-Haystack Benchmark for enriched_hybrid_llm retrieval strategy.

Tests whether the retrieval strategy can find content from a single "needle"
document buried among 200 Kubernetes documentation files.

Usage:
    python benchmark_needle_haystack.py --generate-questions
    python benchmark_needle_haystack.py --run-benchmark
    python benchmark_needle_haystack.py --all
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from strategies import Document, Chunk, MarkdownSemanticStrategy
from retrieval import create_retrieval_strategy
from enrichment import call_llm
from logger import BenchmarkLogger, set_logger


CORPUS_DIR = Path("corpus/kubernetes")
NEEDLE_SELECTION_FILE = Path("corpus/needle_selection.json")
NEEDLE_QUESTIONS_FILE = Path("corpus/needle_questions.json")
RETRIEVAL_RESULTS_FILE = Path("results/needle_haystack_retrieval.json")


QUESTION_GENERATION_PROMPT = """You are a technical documentation expert. Your task is to generate questions that test retrieval of specific facts from a Kubernetes documentation page.

Below is the FULL content of a Kubernetes documentation page about Topology Manager. Generate exactly 20 questions that:
1. Can ONLY be answered using information from THIS document
2. Test specific facts, numbers, names, or technical details (not vague concepts)
3. Vary in difficulty: 5 easy, 10 medium, 5 hard
4. Cover different sections of the document
5. Each question MUST have a clear, specific expected answer that appears in the document

DOCUMENT CONTENT:
---
{document_content}
---

Output format: Return ONLY a JSON array with exactly 20 objects, each with these fields:
- "id": "q_001" through "q_020"
- "question": The question text
- "expected_answer": The specific answer from the document (copy exact text when possible)
- "difficulty": "easy", "medium", or "hard"
- "section": Which section of the document contains the answer

Example format:
[
  {{"id": "q_001", "question": "What is the default scope for Topology Manager?", "expected_answer": "container", "difficulty": "easy", "section": "Topology manager scopes"}}
]

Generate the 20 questions now as a JSON array:"""


def load_needle_selection() -> dict:
    """Load the selected needle document info."""
    if not NEEDLE_SELECTION_FILE.exists():
        raise FileNotFoundError(
            f"Needle selection not found: {NEEDLE_SELECTION_FILE}\n"
            "Run task 1 first to select a needle document."
        )
    with open(NEEDLE_SELECTION_FILE) as f:
        return json.load(f)


def load_needle_document(selection: dict) -> str:
    """Load the full content of the needle document."""
    doc_path = CORPUS_DIR / selection["filename"]
    if not doc_path.exists():
        raise FileNotFoundError(f"Needle document not found: {doc_path}")
    return doc_path.read_text()


def generate_questions():
    """Generate 20 questions from the needle document using Claude Sonnet."""
    print("=" * 60)
    print("TASK 2: Generate Questions from Needle Document")
    print("=" * 60)

    # Load needle selection
    selection = load_needle_selection()
    print(f"Needle document: {selection['filename']}")
    print(f"Word count: {selection['word_count']}")

    # Load full document content
    doc_content = load_needle_document(selection)
    print(f"Document loaded: {len(doc_content)} characters")

    # Generate questions using Claude Sonnet
    print("\nCalling Claude Sonnet to generate 20 questions...")
    print("(This may take 30-60 seconds)")

    prompt = QUESTION_GENERATION_PROMPT.format(document_content=doc_content)

    start_time = time.time()
    response = call_llm(prompt, model="claude-sonnet", timeout=120)
    elapsed = time.time() - start_time

    print(f"Response received in {elapsed:.1f}s")

    # Parse JSON from response
    # Find the JSON array in the response
    try:
        # Try to find JSON array in response
        start_idx = response.find("[")
        end_idx = response.rfind("]") + 1
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No JSON array found in response")

        json_str = response[start_idx:end_idx]
        questions = json.loads(json_str)

        if len(questions) != 20:
            print(f"Warning: Got {len(questions)} questions instead of 20")

        # Add metadata
        output = {
            "needle_doc_id": selection["doc_id"],
            "needle_filename": selection["filename"],
            "generated_at": datetime.now().isoformat(),
            "generation_time_s": elapsed,
            "question_count": len(questions),
            "questions": questions,
        }

        # Save to file
        NEEDLE_QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(NEEDLE_QUESTIONS_FILE, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nGenerated {len(questions)} questions")
        print(f"Saved to: {NEEDLE_QUESTIONS_FILE}")

        # Print summary
        print("\nQuestion Summary:")
        difficulty_counts = {}
        for q in questions:
            d = q.get("difficulty", "unknown")
            difficulty_counts[d] = difficulty_counts.get(d, 0) + 1
        for d, c in sorted(difficulty_counts.items()):
            print(f"  {d}: {c}")

        print("\nSample questions:")
        for q in questions[:3]:
            print(f"  [{q['difficulty']}] {q['question']}")
            print(f"    Answer: {q['expected_answer'][:80]}...")

        return questions

    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON response: {e}")
        print(f"Response preview: {response[:500]}...")
        raise


def load_all_documents() -> list[Document]:
    """Load all 200 documents from the kubernetes sample corpus."""
    documents = []

    for doc_path in sorted(CORPUS_DIR.glob("*.md")):
        doc_id = doc_path.stem  # filename without .md
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


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunk documents using MarkdownSemanticStrategy."""
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


def run_benchmark(questions_file=None):
    """Run the benchmark: index all docs and retrieve for all questions."""
    # Initialize logger with TRACE level to capture query rewrites
    logger = BenchmarkLogger(min_level="TRACE")
    set_logger(logger)

    print("=" * 60)
    print("TASK 3: Run Benchmark - Index and Retrieve")
    print("=" * 60)

    # Load questions
    qfile = Path(questions_file) if questions_file else NEEDLE_QUESTIONS_FILE
    if not qfile.exists():
        raise FileNotFoundError(
            f"Questions file not found: {qfile}\n"
            "Run --generate-questions first or provide --questions flag."
        )

    with open(qfile) as f:
        questions_data = json.load(f)

    questions = questions_data["questions"]
    needle_doc_id = questions_data["needle_doc_id"]
    print(f"Loaded {len(questions)} questions for needle: {needle_doc_id}")

    # Load all documents
    print("\nLoading 200 documents...")
    documents = load_all_documents()
    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    print("\nChunking documents with MarkdownSemanticStrategy...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # Count chunks from needle doc
    needle_chunks = [c for c in chunks if c.doc_id == needle_doc_id]
    print(f"Needle document chunks: {len(needle_chunks)}")

    # Initialize retrieval strategy
    print("\nInitializing enriched_hybrid_llm strategy...")
    strategy = create_retrieval_strategy("enriched_hybrid_llm", debug=True)

    # Set up embedder
    print("Loading BGE embedder...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    strategy.set_embedder(embedder)

    # Index chunks
    print("\nIndexing chunks (this may take a minute)...")
    index_start = time.time()
    strategy.index(chunks, documents)
    index_time = time.time() - index_start
    print(f"Indexing complete in {index_time:.1f}s")

    # Get index stats
    index_stats = strategy.get_index_stats()
    print(f"Index stats: {json.dumps(index_stats, indent=2, default=str)}")

    # Run retrieval for each question
    print(f"\nRunning retrieval for {len(questions)} questions...")
    results = []

    for i, q in enumerate(questions):
        query_start = time.time()
        retrieved = strategy.retrieve(q["question"], k=5)
        latency = (time.time() - query_start) * 1000  # ms

        # Check if any retrieved chunk is from needle doc
        needle_found = any(c.doc_id == needle_doc_id for c in retrieved)

        result = {
            "question_id": q["id"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "category": q.get("category", ""),
            "latency_ms": round(latency, 1),
            "needle_found": needle_found,
            "retrieved_chunks": [
                {
                    "chunk_id": c.id,
                    "doc_id": c.doc_id,
                    "content": c.content,
                    "is_needle": c.doc_id == needle_doc_id,
                }
                for c in retrieved
            ],
        }
        results.append(result)

        status = "FOUND" if needle_found else "MISS"
        print(f"  [{i + 1:2d}/20] {status} ({latency:.0f}ms) {q['question'][:50]}...")

    # Calculate summary stats
    found_count = sum(1 for r in results if r["needle_found"])
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    output = {
        "benchmark_run_at": datetime.now().isoformat(),
        "needle_doc_id": needle_doc_id,
        "total_documents": len(documents),
        "total_chunks": len(chunks),
        "needle_chunks": len(needle_chunks),
        "index_time_s": round(index_time, 1),
        "index_stats": index_stats,
        "summary": {
            "total_questions": len(questions),
            "needle_found_count": found_count,
            "needle_found_rate": round(found_count / len(questions) * 100, 1),
            "avg_latency_ms": round(avg_latency, 1),
        },
        "results": results,
    }

    # Save results
    # Use custom output file if custom questions file provided
    if questions_file and questions_file != str(NEEDLE_QUESTIONS_FILE):
        # Derive output filename from questions file
        questions_path = Path(questions_file)
        output_filename = questions_path.stem + "_retrieval.json"
        results_file = RETRIEVAL_RESULTS_FILE.parent / output_filename
    else:
        results_file = RETRIEVAL_RESULTS_FILE

    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 60}")
    print(f"Documents indexed: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Needle chunks: {len(needle_chunks)}")
    print(f"Questions tested: {len(questions)}")
    print(
        f"Needle found rate: {found_count}/{len(questions)} ({output['summary']['needle_found_rate']}%)"
    )
    print(f"Average latency: {avg_latency:.0f}ms")
    print(f"\nResults saved to: {results_file}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Needle-in-Haystack Benchmark for enriched_hybrid_llm"
    )
    parser.add_argument(
        "--generate-questions",
        action="store_true",
        help="Generate 20 questions from the needle document",
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Run the benchmark: index and retrieve",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tasks: generate questions, then run benchmark",
    )
    parser.add_argument(
        "--questions",
        type=str,
        help="Path to custom questions file (e.g., corpus/needle_questions_adversarial.json)",
    )

    args = parser.parse_args()

    if not any([args.generate_questions, args.run_benchmark, args.all]):
        parser.print_help()
        return

    if args.all or args.generate_questions:
        generate_questions()

    if args.all or args.run_benchmark:
        run_benchmark(questions_file=args.questions)


if __name__ == "__main__":
    main()
