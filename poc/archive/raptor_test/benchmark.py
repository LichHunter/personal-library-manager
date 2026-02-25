#!/usr/bin/env python3
"""
RAPTOR Benchmark Script

Tests:
1. Indexing performance (time to build tree)
2. Retrieval accuracy (can we find the right information?)
3. QA accuracy (are answers correct and grounded?)

Usage:
    python benchmark.py [--model MODEL] [--document FILE]
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


# ============================================================================
# Test Document with Ground Truth Q&A pairs
# ============================================================================

TEST_DOCUMENT = """
# The History and Impact of Python Programming Language

## Origins and Creation

Python was created by Guido van Rossum and was first released in 1991. Van Rossum began working on Python in the late 1980s at Centrum Wiskunde & Informatica (CWI) in the Netherlands. He wanted to create a successor to the ABC programming language that would appeal to Unix/C hackers. The name "Python" was inspired by the British comedy group Monty Python, not the snake.

Python 2.0 was released in 2000 and introduced new features like list comprehensions and a garbage collection system. Python 3.0 was released in 2008 and was a major revision that was not completely backward-compatible with Python 2. The transition from Python 2 to Python 3 took many years, with Python 2 reaching end of life on January 1, 2020.

## Design Philosophy

Python's design philosophy emphasizes code readability and simplicity. The language's core philosophy is summarized in "The Zen of Python" (PEP 20), which includes aphorisms such as:
- "Beautiful is better than ugly"
- "Explicit is better than implicit"  
- "Simple is better than complex"
- "Readability counts"

Python uses significant whitespace (indentation) to delimit code blocks rather than curly braces or keywords. This enforces a consistent visual structure and makes Python code generally easier to read than code in other languages.

## Key Features

Python is a high-level, interpreted programming language with dynamic typing and automatic memory management. It supports multiple programming paradigms including procedural, object-oriented, and functional programming.

The language features a comprehensive standard library, often referred to as "batteries included." This library provides modules for string operations, file I/O, networking, database interaction, web services, and much more.

Python's package ecosystem, centered around the Python Package Index (PyPI), contains over 400,000 packages as of 2023. Popular packages include NumPy for numerical computing, Pandas for data analysis, Django and Flask for web development, and TensorFlow and PyTorch for machine learning.

## Applications

Python is widely used in various domains:

**Data Science and Machine Learning**: Python has become the dominant language for data science due to libraries like NumPy, Pandas, Matplotlib, Scikit-learn, TensorFlow, and PyTorch. Companies like Google, Netflix, and Spotify use Python for their data pipelines and ML models.

**Web Development**: Frameworks like Django and Flask make Python a popular choice for backend web development. Instagram, Pinterest, and Dropbox are built primarily with Python.

**Automation and Scripting**: Python's simple syntax makes it ideal for writing scripts to automate repetitive tasks, system administration, and DevOps workflows.

**Scientific Computing**: Python is extensively used in academic research and scientific computing, particularly in fields like physics, biology, and astronomy.

## Performance Considerations

Python is generally slower than compiled languages like C++ or Rust due to its interpreted nature and dynamic typing. However, this performance gap can be mitigated through:
- Using optimized libraries like NumPy that are implemented in C
- Just-In-Time compilation with tools like PyPy or Numba
- Writing performance-critical sections in C/C++ as Python extensions
- Using multiprocessing for CPU-bound tasks

For most applications, Python's development speed and ease of maintenance outweigh the runtime performance costs.

## Community and Governance

Python has a large and active community. The Python Software Foundation (PSF) is a non-profit organization that manages the Python programming language and its ecosystem. After Guido van Rossum stepped down as "Benevolent Dictator For Life" (BDFL) in 2018, Python adopted a governance model with a five-member steering council elected by core developers.

The language continues to evolve with annual releases. Python 3.12, released in October 2023, introduced improvements to error messages, performance optimizations, and new syntax features.
"""

# Ground truth Q&A pairs for accuracy testing
GROUND_TRUTH_QA = [
    {
        "question": "Who created Python?",
        "expected_answer": "Guido van Rossum",
        "keywords": ["guido", "van rossum"],
    },
    {
        "question": "When was Python first released?",
        "expected_answer": "1991",
        "keywords": ["1991"],
    },
    {
        "question": "What inspired the name Python?",
        "expected_answer": "Monty Python (the comedy group)",
        "keywords": ["monty python", "comedy"],
    },
    {
        "question": "When did Python 2 reach end of life?",
        "expected_answer": "January 1, 2020",
        "keywords": ["2020", "january"],
    },
    {
        "question": "What is PEP 20?",
        "expected_answer": "The Zen of Python",
        "keywords": ["zen of python", "zen"],
    },
    {
        "question": "How many packages are on PyPI?",
        "expected_answer": "Over 400,000",
        "keywords": ["400,000", "400000"],
    },
    {
        "question": "What web framework does Instagram use?",
        "expected_answer": "Django (or Python)",
        "keywords": ["django", "python"],
    },
    {
        "question": "Who manages Python after Guido stepped down?",
        "expected_answer": "A five-member steering council",
        "keywords": ["steering council", "five"],
    },
    {
        "question": "What is Python's design philosophy?",
        "expected_answer": "Readability and simplicity",
        "keywords": ["readability", "simplicity", "zen"],
    },
    {
        "question": "What is PyPy?",
        "expected_answer": "A Just-In-Time compiler for Python",
        "keywords": ["jit", "just-in-time", "compiler", "performance"],
    },
]


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    # Indexing metrics
    num_chunks: int
    num_nodes: int
    num_layers: int
    num_summaries: int
    
    # Timing
    chunking_time: float
    embedding_time: float
    clustering_time: float
    summarization_time: float
    total_indexing_time: float
    
    # Accuracy metrics
    retrieval_accuracy: float  # % of questions where relevant chunk was retrieved
    qa_accuracy: float  # % of questions answered correctly
    qa_details: List[Dict]  # Detailed results per question
    
    # Config
    model_name: str
    embedding_model: str
    chunk_size: int
    
    def to_dict(self):
        return asdict(self)


def check_answer_accuracy(answer: str, expected: str, keywords: List[str]) -> Tuple[bool, str]:
    """
    Check if answer contains expected information.
    
    Returns:
        (is_correct, reason)
    """
    answer_lower = answer.lower()
    
    # Check for keywords
    keywords_found = sum(1 for kw in keywords if kw.lower() in answer_lower)
    
    if keywords_found >= len(keywords) * 0.5:  # At least half the keywords
        return True, f"Found {keywords_found}/{len(keywords)} keywords"
    
    # Check for "cannot find" or similar
    negative_phrases = ["cannot find", "not in the context", "don't have", "no information"]
    if any(phrase in answer_lower for phrase in negative_phrases):
        return False, "Model indicated information not found"
    
    return False, f"Only found {keywords_found}/{len(keywords)} keywords"


def run_benchmark(
    model: str = "llama3.2:3b",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 100,
    document: str = TEST_DOCUMENT,
) -> BenchmarkResult:
    """Run the full benchmark."""
    
    from local_models import (
        LocalEmbeddingModel,
        OllamaSummarizationModel,
        OllamaQAModel,
        check_ollama_available,
    )
    from raptor_lite import RaptorLite
    
    console.print("\n[bold blue]RAPTOR Benchmark[/bold blue]\n")
    
    # Check Ollama
    if not check_ollama_available(model):
        console.print(f"[red]Error: Ollama model '{model}' not available[/red]")
        console.print(f"Run: [cyan]ollama pull {model}[/cyan]")
        raise RuntimeError(f"Model {model} not available")
    
    # Initialize models
    console.print("[yellow]Initializing models...[/yellow]")
    embed_model = LocalEmbeddingModel(embedding_model)
    summarizer = OllamaSummarizationModel(model)
    qa_model = OllamaQAModel(model)
    
    # Build RAPTOR tree
    console.print("\n[yellow]Building RAPTOR tree...[/yellow]")
    raptor = RaptorLite(
        embedding_model=embed_model,
        summarization_model=summarizer,
        qa_model=qa_model,
        max_tokens_per_chunk=chunk_size,
        max_layers=3,
        clustering_threshold=0.1,
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building tree...", total=None)
        tree = raptor.build_tree(document)
        progress.update(task, completed=True)
    
    stats = raptor.get_stats()
    
    # Print tree stats
    table = Table(title="Tree Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Chunks", str(stats.num_chunks))
    table.add_row("Total Nodes", str(stats.num_nodes))
    table.add_row("Layers", str(stats.num_layers))
    table.add_row("Summaries Created", str(stats.num_summaries))
    console.print(table)
    
    # Print timing
    table = Table(title="Timing Breakdown")
    table.add_column("Phase", style="cyan")
    table.add_column("Time (s)", style="green")
    table.add_column("% of Total", style="yellow")
    
    total = stats.total_seconds
    table.add_row("Chunking", f"{stats.chunking_seconds:.2f}", f"{stats.chunking_seconds/total*100:.1f}%")
    table.add_row("Embedding", f"{stats.embedding_seconds:.2f}", f"{stats.embedding_seconds/total*100:.1f}%")
    table.add_row("Clustering", f"{stats.clustering_seconds:.2f}", f"{stats.clustering_seconds/total*100:.1f}%")
    table.add_row("Summarization", f"{stats.summarization_seconds:.2f}", f"{stats.summarization_seconds/total*100:.1f}%")
    table.add_row("[bold]Total[/bold]", f"[bold]{total:.2f}[/bold]", "[bold]100%[/bold]")
    console.print(table)
    
    # Test retrieval and QA accuracy
    console.print("\n[yellow]Testing retrieval and QA accuracy...[/yellow]\n")
    
    qa_details = []
    retrieval_correct = 0
    qa_correct = 0
    
    for i, qa in enumerate(GROUND_TRUTH_QA):
        console.print(f"[dim]Q{i+1}: {qa['question']}[/dim]")
        
        # Test retrieval
        nodes = raptor.retrieve(qa["question"], top_k=3)
        retrieved_text = " ".join(n.text for n in nodes).lower()
        
        # Check if any keyword is in retrieved text
        retrieval_hit = any(kw.lower() in retrieved_text for kw in qa["keywords"])
        if retrieval_hit:
            retrieval_correct += 1
        
        # Test QA
        answer, sources = raptor.answer_question(qa["question"])
        is_correct, reason = check_answer_accuracy(answer, qa["expected_answer"], qa["keywords"])
        
        if is_correct:
            qa_correct += 1
            console.print(f"  [green]OK[/green] - {reason}")
        else:
            console.print(f"  [red]FAIL[/red] - {reason}")
            console.print(f"    [dim]Expected: {qa['expected_answer']}[/dim]")
            console.print(f"    [dim]Got: {answer[:100]}...[/dim]")
        
        qa_details.append({
            "question": qa["question"],
            "expected": qa["expected_answer"],
            "actual": answer,
            "retrieval_hit": retrieval_hit,
            "qa_correct": is_correct,
            "reason": reason,
        })
    
    retrieval_accuracy = retrieval_correct / len(GROUND_TRUTH_QA) * 100
    qa_accuracy = qa_correct / len(GROUND_TRUTH_QA) * 100
    
    # Print accuracy summary
    console.print("\n")
    table = Table(title="Accuracy Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="green")
    table.add_row("Retrieval Accuracy", f"{retrieval_accuracy:.1f}% ({retrieval_correct}/{len(GROUND_TRUTH_QA)})")
    table.add_row("QA Accuracy", f"{qa_accuracy:.1f}% ({qa_correct}/{len(GROUND_TRUTH_QA)})")
    console.print(table)
    
    return BenchmarkResult(
        num_chunks=stats.num_chunks,
        num_nodes=stats.num_nodes,
        num_layers=stats.num_layers,
        num_summaries=stats.num_summaries,
        chunking_time=stats.chunking_seconds,
        embedding_time=stats.embedding_seconds,
        clustering_time=stats.clustering_seconds,
        summarization_time=stats.summarization_seconds,
        total_indexing_time=stats.total_seconds,
        retrieval_accuracy=retrieval_accuracy,
        qa_accuracy=qa_accuracy,
        qa_details=qa_details,
        model_name=model,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
    )


def main():
    parser = argparse.ArgumentParser(description="RAPTOR Benchmark")
    parser.add_argument("--model", default="llama3.2:3b", help="Ollama model to use")
    parser.add_argument("--embedding", default="sentence-transformers/all-MiniLM-L6-v2", 
                       help="Embedding model to use")
    parser.add_argument("--chunk-size", type=int, default=100, help="Tokens per chunk")
    parser.add_argument("--document", type=Path, help="Custom document file to use")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Load custom document if provided
    document = TEST_DOCUMENT
    if args.document:
        document = args.document.read_text()
        console.print(f"[cyan]Using custom document: {args.document}[/cyan]")
    
    try:
        result = run_benchmark(
            model=args.model,
            embedding_model=args.embedding,
            chunk_size=args.chunk_size,
            document=document,
        )
        
        # Save results if requested
        if args.output:
            args.output.write_text(json.dumps(result.to_dict(), indent=2))
            console.print(f"\n[green]Results saved to {args.output}[/green]")
        
        # Print summary
        console.print(Panel(
            f"[bold green]Benchmark Complete[/bold green]\n\n"
            f"Indexing Time: {result.total_indexing_time:.2f}s\n"
            f"Retrieval Accuracy: {result.retrieval_accuracy:.1f}%\n"
            f"QA Accuracy: {result.qa_accuracy:.1f}%",
            title="Summary",
        ))
        
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        raise


if __name__ == "__main__":
    main()
