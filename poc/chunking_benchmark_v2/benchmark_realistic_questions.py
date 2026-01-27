#!/usr/bin/env python3
"""
Benchmark using realistic questions from the kubefix dataset.

This script loads questions from the kubefix HuggingFace dataset and validates
that we have matching documents in our corpus for each question.
"""

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from datasets import load_dataset

from enrichment.provider import call_llm


# Path to the corpus directory
CORPUS_DIR = Path(__file__).parent / "corpus" / "kubernetes"


TRANSFORMATION_PROMPT_V2 = """You are a technical documentation expert. Your task is to transform bot-like questions into realistic user questions.

Transform the following question into TWO realistic user questions that a mid-level developer/DevOps engineer would ask when facing a real problem.

Target user profile:
- Mid-level developer/DevOps
- Familiar with containers and basic Kubernetes
- Has a REAL problem or task
- Does NOT know exact Kubernetes terminology
- Describes symptoms, behaviors, goals - not technical terms

Guidelines:
1. Each question should describe a REAL problem, symptom, or goal
2. Use natural language, not technical jargon
3. Include specific symptoms or error behaviors when relevant
4. Keep questions concise (<120 characters each)
5. Avoid starting with "What is" - use problem-oriented language instead
6. Questions should sound like Stack Overflow posts, not documentation queries

Examples:

Original: "What is the purpose of a LimitRange object?"
Transformed:
{{
  "q1": "how to set default memory limits for all pods in a namespace without editing each deployment",
  "q2": "prevent developers from requesting too much memory per container"
}}

Original: "What is the difference between a Deployment and a StatefulSet?"
Transformed:
{{
  "q1": "my database pods keep getting new hostnames after restart and can't find each other",
  "q2": "which controller should I use for a postgresql cluster that needs stable network identity"
}}

Original: "How does the kubelet handle container health checks?"
Transformed:
{{
  "q1": "kubernetes not restarting my container even though the app inside is frozen",
  "q2": "how to make k8s automatically restart pod when my health endpoint stops responding"
}}

Original: "What is a NetworkPolicy in Kubernetes?"
Transformed:
{{
  "q1": "block all traffic between namespaces except for specific services",
  "q2": "my pods in namespace A can reach pods in namespace B but I want to restrict that"
}}

Original: "What is the purpose of Pod anti-affinity?"
Transformed:
{{
  "q1": "spread my replicas across different nodes so one node failure doesn't kill everything",
  "q2": "kubernetes keeps scheduling all my pods on the same node even though I have 5 nodes"
}}

Original: "What are the access modes for PersistentVolumes?"
Transformed:
{{
  "q1": "can multiple pods write to the same volume at the same time",
  "q2": "my second pod can't mount the volume because first pod is using it"
}}

Original: "What is a ConfigMap?"
Transformed:
{{
  "q1": "how to pass config file to container without rebuilding image",
  "q2": "change application settings without redeploying pods"
}}

Original: "What is RBAC in Kubernetes?"
Transformed:
{{
  "q1": "user getting forbidden error when trying to list pods in specific namespace",
  "q2": "how to give developer access to only their team's namespace"
}}

Now transform this question:

Original: "{original_question}"

Return ONLY a JSON object with two fields: "q1" and "q2". No explanation, no markdown code blocks."""


def kubefix_to_our_path(kubefix_source: str) -> str:
    """
    Convert kubefix source path to our corpus filename.

    Example:
        /content/en/docs/concepts/architecture/cgroups.md
        → concepts_architecture_cgroups.md
    """
    path = kubefix_source.replace("/content/en/docs/", "")
    path = path.replace("/", "_")
    return path


def doc_path_to_doc_id(doc_path: str) -> str:
    """
    Convert document path to document ID (remove .md extension).

    Example:
        concepts_architecture_cgroups.md → concepts_architecture_cgroups
    """
    return doc_path.replace(".md", "")


def transform_question(original_question: str, max_retries: int = 3) -> Optional[dict]:
    """
    Transform a bot-like question into realistic user questions using Claude Haiku.

    Args:
        original_question: The original bot-like question to transform
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Dict with 'q1' and 'q2' fields containing transformed questions,
        or None if transformation fails after all retries

    Example:
        >>> result = transform_question("What is a ConfigMap?")
        >>> result
        {
            "q1": "how to pass config file to container without rebuilding image",
            "q2": "change application settings without redeploying pods"
        }
    """
    prompt = TRANSFORMATION_PROMPT_V2.format(original_question=original_question)

    for attempt in range(max_retries):
        try:
            content = call_llm(prompt, model="claude-haiku", timeout=30)

            if not content:
                print(
                    f"[transform_question] Attempt {attempt + 1}: Empty response from LLM"
                )
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue

            content = content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                # Remove markdown code block markers
                lines = content.split("\n")
                # Remove first line (```json or ```)
                lines = lines[1:]
                # Remove last line (```)
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines).strip()

            # Parse JSON response
            result = json.loads(content)

            # Validate response has required fields
            if "q1" in result and "q2" in result:
                return result
            else:
                print(
                    f"[transform_question] Attempt {attempt + 1}: Missing required fields (q1, q2)"
                )

        except json.JSONDecodeError as e:
            print(f"[transform_question] Attempt {attempt + 1}: JSON parse error: {e}")
        except Exception as e:
            print(
                f"[transform_question] Attempt {attempt + 1}: Error: {type(e).__name__}: {e}"
            )

        # Wait before retry (except on last attempt)
        if attempt < max_retries - 1:
            time.sleep(1)

    print(
        f"[transform_question] Failed after {max_retries} attempts for: {original_question}"
    )
    return None


def evaluate_transformation_quality(q1: str, q2: str, original: str) -> dict:
    """
    Evaluate the quality of transformed questions using 5 heuristics.

    Heuristics:
    1. Originality: <70% word overlap with original
    2. Phrasing: Starts with problem language (how, why, my, can, etc.)
    3. Conciseness: <120 characters
    4. Realism: Contains symptoms (error, can't, doesn't, need, etc.)
    5. Not "What is": Doesn't start with "What is"

    Args:
        q1: First transformed question
        q2: Second transformed question (not currently evaluated)
        original: Original question

    Returns:
        Dict with:
        - scores: Dict of float scores (0-1) for each heuristic
        - issues: List of issue descriptions
        - pass: Boolean indicating if transformation passes quality check

    Example:
        >>> evaluate_transformation_quality(
        ...     "how to set default memory limits",
        ...     "prevent developers from requesting too much memory",
        ...     "What is the purpose of a LimitRange object?"
        ... )
        {
            "scores": {"originality": 0.9, "phrasing": 1.0, "conciseness": 1.0, "realism": 0.0, "overall": 0.7},
            "issues": [],
            "pass": True
        }
    """
    issues = []
    scores = {}

    # 1. Originality: <70% word overlap
    original_words = set(original.lower().split())
    q1_words = set(q1.lower().split())
    overlap = len(original_words & q1_words) / max(len(original_words), 1)
    if overlap > 0.7:
        issues.append("q1 too similar to original")
    scores["originality"] = 1 - overlap

    # 2. Phrasing: starts with problem language
    good_starts = [
        "how",
        "why",
        "my",
        "can",
        "what",
        "when",
        "which",
        "is",
        "does",
        "getting",
        "error",
    ]
    q1_good = any(q1.lower().startswith(s) for s in good_starts)
    if not q1_good:
        issues.append("q1 doesn't start with problem language")
    scores["phrasing"] = float(q1_good)

    # 3. Conciseness: <120 chars
    if len(q1) > 120:
        issues.append("q1 too long")
    scores["conciseness"] = min(1.0, 120 / max(len(q1), 1))

    # 4. Realism: contains symptoms
    realistic_markers = [
        "error",
        "not",
        "can't",
        "doesn't",
        "won't",
        "keeps",
        "need",
        "want",
        "how to",
    ]
    q1_realistic = any(m in q1.lower() for m in realistic_markers)
    scores["realism"] = float(q1_realistic)

    # 5. Not "What is"
    if q1.lower().startswith("what is"):
        issues.append("q1 uses 'What is' pattern")

    # Overall
    scores["overall"] = sum(scores.values()) / len(scores)

    return {
        "scores": scores,
        "issues": issues,
        "pass": scores["overall"] >= 0.5 and len(issues) <= 2,
    }


def load_diverse_test_samples(n: int = 20) -> list:
    """
    Load diverse test samples from kubefix dataset across multiple topics.

    Selects samples across diverse Kubernetes topics to ensure comprehensive
    testing of the transformation prompt.

    Topics covered: cgroup, service, volume, deployment, autoscal, configmap,
    rbac, probe, affinity, network

    Args:
        n: Number of samples to load (default: 20)

    Returns:
        List of dicts with keys: instruction, source, our_doc_path
    """
    print(f"Loading {n} diverse test samples...")

    # Load kubefix dataset
    ds = load_dataset("andyburgin/kubefix", split="train")

    # Topic keywords for diversity
    topic_keywords = [
        "cgroup",
        "service",
        "volume",
        "deployment",
        "autoscal",
        "configmap",
        "rbac",
        "probe",
        "affinity",
        "network",
    ]

    # Group questions by topic
    topic_groups = {topic: [] for topic in topic_keywords}
    other_questions = []

    for example in ds:
        source = example["source"]
        instruction = example["instruction"]

        # Check if doc exists in our corpus
        our_path = kubefix_to_our_path(source)
        doc_file = CORPUS_DIR / our_path

        if not doc_file.exists():
            continue

        # Categorize by topic
        found_topic = False
        for topic in topic_keywords:
            if topic.lower() in instruction.lower() or topic.lower() in source.lower():
                topic_groups[topic].append(
                    {
                        "instruction": instruction,
                        "source": source,
                        "our_doc_path": our_path,
                    }
                )
                found_topic = True
                break

        if not found_topic:
            other_questions.append(
                {
                    "instruction": instruction,
                    "source": source,
                    "our_doc_path": our_path,
                }
            )

    # Select diverse samples: try to get at least 1-2 from each topic
    samples = []
    samples_per_topic = max(1, n // len(topic_keywords))

    for topic in topic_keywords:
        if len(topic_groups[topic]) > 0:
            selected = random.sample(
                topic_groups[topic], min(samples_per_topic, len(topic_groups[topic]))
            )
            samples.extend(selected)

    # Fill remaining slots with other questions
    remaining = n - len(samples)
    if remaining > 0 and other_questions:
        additional = random.sample(
            other_questions, min(remaining, len(other_questions))
        )
        samples.extend(additional)

    # Ensure we have exactly n samples
    if len(samples) > n:
        samples = random.sample(samples, n)
    elif len(samples) < n:
        # If we don't have enough diverse samples, just take any n
        all_valid = []
        for example in ds:
            source = example["source"]
            our_path = kubefix_to_our_path(source)
            doc_file = CORPUS_DIR / our_path
            if doc_file.exists():
                all_valid.append(
                    {
                        "instruction": example["instruction"],
                        "source": source,
                        "our_doc_path": our_path,
                    }
                )
        samples = random.sample(all_valid, min(n, len(all_valid)))

    return samples[:n]


def run_prompt_iteration_test() -> dict:
    """
    Run automated prompt iteration test on diverse samples.

    Transforms N samples and evaluates quality of each transformation.
    Returns summary with pass rate.

    Returns:
        Dict with keys: total, successful, passing, pass_rate, issues_summary
    """
    samples = load_diverse_test_samples(n=20)
    n = len(samples)

    print(f"\nRunning prompt iteration test on {n} samples...\n")

    successful_transforms = 0
    passing_quality = 0
    all_issues = []

    for i, sample in enumerate(samples, 1):
        original = sample["instruction"]
        print(f"[{i}/{n}] Transforming: {original[:60]}...")

        # Transform question
        result = transform_question(original)

        if result is None:
            print(f"  ❌ Transformation failed")
            continue

        successful_transforms += 1
        q1 = result["q1"]
        q2 = result["q2"]

        # Evaluate quality
        quality = evaluate_transformation_quality(q1, q2, original)

        print(f"  Q1: {q1}")
        print(f"  Q2: {q2}")
        print(
            f"  Quality: {quality['scores']['overall']:.2f} | Pass: {quality['pass']}"
        )

        if quality["pass"]:
            passing_quality += 1
        else:
            if quality["issues"]:
                print(f"  Issues: {', '.join(quality['issues'])}")
                all_issues.extend(quality["issues"])

    # Calculate summary
    pass_rate = (passing_quality / n * 100) if n > 0 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successful transforms: {successful_transforms}/{n}")
    print(f"Passing quality: {passing_quality}/{n} ({pass_rate:.1f}%)")

    if pass_rate >= 80:
        print("✅ PROMPT QUALITY ACCEPTABLE")
    else:
        print("❌ PROMPT NEEDS IMPROVEMENT")

    # Count common issues
    if all_issues:
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        print("\nCommon issues:")
        for issue, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            print(f"  - {issue}: {count} occurrences")

    # Append to notepad
    notepad_path = (
        Path(__file__).parent.parent.parent
        / ".sisyphus/notepads/realistic-questions-benchmark/learnings.md"
    )
    with open(notepad_path, "a") as f:
        f.write(
            f"\n## [{datetime.now().strftime('%Y-%m-%d')}] Task 3: Prompt Iteration Test\n"
        )
        f.write(f"- Tested on {n} diverse samples\n")
        f.write(f"- Successful transforms: {successful_transforms}/{n}\n")
        f.write(f"- Pass rate: {passing_quality}/{n} ({pass_rate:.1f}%)\n")
        if all_issues:
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
            f.write(
                f"- Common issues: {', '.join(f'{k} ({v})' for k, v in sorted(issue_counts.items(), key=lambda x: -x[1]))}\n"
            )

    return {
        "total": n,
        "successful": successful_transforms,
        "passing": passing_quality,
        "pass_rate": pass_rate,
        "issues_summary": all_issues,
    }


def generate_realistic_questions(
    n: int = 200, output_path: Optional[str] = None
) -> dict:
    """
    Generate realistic questions from kubefix dataset.

    Samples n questions randomly from valid kubefix entries (seed=42),
    transforms each with progress logging, and saves to JSON file.

    Args:
        n: Number of questions to generate (default: 200)
        output_path: Path to save output JSON (default: corpus/realistic_questions.json)

    Returns:
        Dict with keys: total, high_quality, output_path, questions_list

    Example:
        >>> result = generate_realistic_questions(n=200)
        >>> result["high_quality"]
        180
    """
    # Set random seed for reproducibility
    random.seed(42)

    # Default output path
    if output_path is None:
        output_path = str(CORPUS_DIR / "realistic_questions.json")

    print(f"Generating {n} realistic questions...")
    print(f"Output: {output_path}\n")

    # Load kubefix dataset
    print("Loading kubefix dataset...")
    ds = load_dataset("andyburgin/kubefix", split="train")

    # Filter to questions with matching corpus docs
    print("Filtering to questions with matching docs...")
    valid_questions = []
    for example in ds:
        source = example["source"]
        our_path = kubefix_to_our_path(source)
        doc_file = CORPUS_DIR / our_path

        if doc_file.exists():
            valid_questions.append(
                {
                    "instruction": example["instruction"],
                    "source": source,
                    "our_doc_path": our_path,
                }
            )

    print(f"Found {len(valid_questions)} questions with matching docs\n")

    # Sample n questions randomly
    if len(valid_questions) < n:
        print(
            f"Warning: Only {len(valid_questions)} valid questions available, using all"
        )
        sampled = valid_questions
    else:
        sampled = random.sample(valid_questions, n)

    # Transform and evaluate each question
    questions = []
    high_quality_count = 0

    print(f"Transforming {len(sampled)} questions...\n")

    for i, sample in enumerate(sampled, 1):
        original = sample["instruction"]
        source = sample["source"]
        our_path = sample["our_doc_path"]
        doc_id = doc_path_to_doc_id(our_path)

        # Progress logging every 20
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(sampled)} questions processed...")

        # Transform question
        result = transform_question(original)

        if result is None:
            # Skip failed transformations
            continue

        q1 = result["q1"]
        q2 = result["q2"]

        # Evaluate quality
        quality = evaluate_transformation_quality(q1, q2, original)

        # Build question dict
        question_dict = {
            "original_instruction": original,
            "original_source": source,
            "our_doc_path": our_path,
            "doc_id": doc_id,
            "realistic_q1": q1,
            "realistic_q2": q2,
            "quality_score": quality["scores"]["overall"],
            "quality_pass": quality["pass"],
        }

        questions.append(question_dict)

        if quality["pass"]:
            high_quality_count += 1

    # Build metadata
    metadata = {
        "source": "kubefix",
        "model": "claude-3-5-haiku-latest",
        "prompt_version": "v2",
        "total": len(questions),
        "high_quality": high_quality_count,
    }

    # Build output
    output = {
        "metadata": metadata,
        "questions": questions,
    }

    # Save to JSON file
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total generated: {len(questions)}")
    print(
        f"High quality: {high_quality_count} ({100 * high_quality_count / len(questions):.1f}%)"
    )
    print(f"Output: {output_path}")

    # Append to notepad
    notepad_path = (
        Path(__file__).parent.parent.parent
        / ".sisyphus/notepads/realistic-questions-benchmark/learnings.md"
    )
    with open(notepad_path, "a") as f:
        f.write(
            f"\n## [{datetime.now().strftime('%Y-%m-%d')}] Task 4: Generate Questions\n"
        )
        f.write(f"- Generated {len(questions)} questions\n")
        f.write(
            f"- High quality: {high_quality_count} ({100 * high_quality_count / len(questions):.1f}%)\n"
        )
        f.write(f"- Output: {output_path}\n")

    return {
        "total": len(questions),
        "high_quality": high_quality_count,
        "output_path": output_path,
        "questions_list": questions,
    }


def run_retrieval_benchmark(questions_file: Optional[str] = None) -> dict:
    """
    Run retrieval benchmark against full K8s corpus using realistic questions.

    Loads the full corpus (1,569 docs), chunks with MarkdownSemanticStrategy,
    indexes with enriched_hybrid_llm strategy, and evaluates retrieval for
    each question.

    Args:
        questions_file: Path to questions JSON file (default: corpus/realistic_questions.json)

    Returns:
        Dict with metadata, summary metrics, and per-question results
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent))

    from sentence_transformers import SentenceTransformer

    from retrieval import create_retrieval_strategy
    from strategies import Chunk, Document, MarkdownSemanticStrategy

    print("=" * 60)
    print("RETRIEVAL BENCHMARK - Realistic Questions")
    print("=" * 60)

    # Load questions
    if questions_file is None:
        questions_path = Path(__file__).parent / "corpus" / "realistic_questions.json"
    else:
        questions_path = Path(questions_file)

    if not questions_path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {questions_path}\n"
            "Run --generate N first to generate realistic questions."
        )

    with open(questions_path) as f:
        data = json.load(f)
        questions = data["questions"]

    print(f"Loaded {len(questions)} questions from {questions_path}")

    # Load full corpus
    corpus_dir = Path(__file__).parent / "corpus" / "kubernetes"
    print(f"\nLoading corpus from {corpus_dir}...")

    documents = []
    for doc_file in sorted(corpus_dir.glob("*.md")):
        with open(doc_file) as f:
            content = f.read()
        doc_id = doc_file.stem  # filename without .md
        documents.append(Document(id=doc_id, content=content, metadata={}))

    print(f"Loaded {len(documents)} documents")

    # Chunk documents
    print("\nChunking documents with MarkdownSemanticStrategy (target=400)...")
    chunker = MarkdownSemanticStrategy(
        target_chunk_size=400, min_chunk_size=50, max_chunk_size=800
    )

    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc.content, doc.id)
        all_chunks.extend(chunks)

    print(f"Created {len(all_chunks)} chunks")

    # Initialize retrieval strategy
    print("\nInitializing enriched_hybrid_llm strategy...")
    strategy = create_retrieval_strategy("enriched_hybrid_llm")

    print("Loading BGE embedder...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    strategy.set_embedder(embedder)

    # Index chunks
    print("\nIndexing chunks (this may take several minutes)...")
    index_start = time.time()
    strategy.index(all_chunks, documents)
    index_time = time.time() - index_start
    print(f"Indexing complete in {index_time:.1f}s")

    # Run retrieval for each question
    print(f"\nRunning retrieval for {len(questions)} questions (2 variants each)...")
    results = []

    for i, q_data in enumerate(questions):
        expected_doc = q_data["doc_id"]

        for variant in ["realistic_q1", "realistic_q2"]:
            query = q_data[variant]

            # Retrieve top-5
            start_time = time.time()
            retrieved = strategy.retrieve(query, k=5)
            latency_ms = (time.time() - start_time) * 1000

            # Check if expected doc appears
            retrieved_doc_ids = [chunk.doc_id for chunk in retrieved]
            hit_at_5 = expected_doc in retrieved_doc_ids
            hit_at_1 = (
                retrieved_doc_ids[0] == expected_doc if retrieved_doc_ids else False
            )
            rank = retrieved_doc_ids.index(expected_doc) + 1 if hit_at_5 else None

            results.append(
                {
                    "question": query,
                    "variant": variant,
                    "expected_doc": expected_doc,
                    "retrieved_docs": retrieved_doc_ids,
                    "hit_at_1": hit_at_1,
                    "hit_at_5": hit_at_5,
                    "rank": rank,
                    "latency_ms": round(latency_ms, 1),
                }
            )

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(questions)} questions processed...")

    # Calculate summary metrics
    hit_at_1_count = sum(1 for r in results if r["hit_at_1"])
    hit_at_5_count = sum(1 for r in results if r["hit_at_5"])
    hit_at_1_rate = hit_at_1_count / len(results)
    hit_at_5_rate = hit_at_5_count / len(results)

    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = [1 / r["rank"] if r["rank"] else 0 for r in results]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    avg_latency = sum(r["latency_ms"] for r in results) / len(results)

    # Create timestamped results folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    results_dir = Path(__file__).parent / "results" / f"realistic_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Build output
    output = {
        "metadata": {
            "corpus_size": len(documents),
            "chunk_count": len(all_chunks),
            "timestamp": timestamp,
            "strategy": "enriched_hybrid_llm",
            "chunking": "MarkdownSemanticStrategy",
            "chunking_params": {"target": 400, "min": 50, "max": 800},
            "questions_file": str(questions_path),
            "total_queries": len(results),
            "index_time_s": round(index_time, 1),
        },
        "summary": {
            "hit_at_1": round(hit_at_1_rate, 4),
            "hit_at_5": round(hit_at_5_rate, 4),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_latency, 1),
        },
        "results": results,
    }

    # Save results
    output_file = results_dir / "retrieval_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Corpus: {len(documents)} documents, {len(all_chunks)} chunks")
    print(f"Questions: {len(questions)} ({len(results)} queries total)")
    print(f"Index time: {index_time:.1f}s")
    print(f"\nResults:")
    print(f"  Hit@1: {hit_at_1_count}/{len(results)} ({hit_at_1_rate:.2%})")
    print(f"  Hit@5: {hit_at_5_count}/{len(results)} ({hit_at_5_rate:.2%})")
    print(f"  MRR:   {mrr:.4f}")
    print(f"  Avg latency: {avg_latency:.1f}ms")
    print(f"\n✅ Results saved to: {output_file}")

    # Append to notepad
    notepad_path = (
        Path(__file__).parent.parent.parent
        / ".sisyphus/notepads/realistic-questions-benchmark/learnings.md"
    )
    try:
        with open(notepad_path, "a") as f:
            f.write(
                f"\n## [{datetime.now().strftime('%Y-%m-%d')}] Task 5: Retrieval Benchmark\n"
            )
            f.write(f"- Corpus: {len(documents)} docs, {len(all_chunks)} chunks\n")
            f.write(
                f"- Hit@1: {hit_at_1_rate:.2%}, Hit@5: {hit_at_5_rate:.2%}, MRR: {mrr:.4f}\n"
            )
            f.write(f"- Results: {output_file}\n")
    except Exception as e:
        print(f"Warning: Could not append to notepad: {e}")

    return output


def generate_report(results_folder: str):
    """
    Generate failure analysis report from retrieval benchmark results.

    Args:
        results_folder: Path to results folder containing retrieval_results.json
    """
    results_path = Path(results_folder) / "retrieval_results.json"

    if not results_path.exists():
        print(f"❌ Error: Results file not found: {results_path}")
        return

    print(f"Loading results from {results_path}...")
    with open(results_path) as f:
        data = json.load(f)

    metadata = data["metadata"]
    summary = data["summary"]
    results = data["results"]

    # Analyze Q1 vs Q2 performance
    # Assume alternating: even indices = Q1, odd indices = Q2
    q1_results = [r for i, r in enumerate(results) if i % 2 == 0]
    q2_results = [r for i, r in enumerate(results) if i % 2 == 1]

    q1_hit5 = (
        sum(1 for r in q1_results if r["hit_at_5"]) / len(q1_results)
        if q1_results
        else 0
    )
    q2_hit5 = (
        sum(1 for r in q2_results if r["hit_at_5"]) / len(q2_results)
        if q2_results
        else 0
    )

    # Categorize failures
    failures = [r for r in results if not r["hit_at_5"]]

    failure_categories = {
        "VOCABULARY_MISMATCH": [],
        "RANKING_ERROR": [],
        "CHUNKING_ISSUE": [],
        "EMBEDDING_BLIND": [],
    }

    for failure in failures:
        rank = failure.get("rank")
        if rank is None:
            # Not found at all - likely vocabulary mismatch
            failure_categories["VOCABULARY_MISMATCH"].append(failure)
        elif rank > 5:
            # Found but ranked too low
            failure_categories["RANKING_ERROR"].append(failure)
        else:
            # Other issues (chunking, embedding)
            failure_categories["CHUNKING_ISSUE"].append(failure)

    # Find worst failures (top 10)
    worst_failures = sorted(failures, key=lambda r: r.get("rank", 999))[:10]

    # Generate markdown report
    report_lines = []
    report_lines.append("# Realistic Questions Benchmark Report\n\n")
    report_lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    report_lines.append(f"Results folder: `{results_folder}`\n\n")
    report_lines.append("---\n\n")

    report_lines.append("## Summary\n\n")
    report_lines.append(f"- **Corpus size**: {metadata['corpus_size']} documents\n")
    report_lines.append(f"- **Chunk count**: {metadata['chunk_count']} chunks\n")
    report_lines.append(f"- **Total queries**: {len(results)}\n")
    report_lines.append(f"- **Strategy**: {metadata['strategy']}\n")
    report_lines.append(f"- **Chunking**: {metadata['chunking']}\n")
    report_lines.append(f"- **Hit@1**: {summary['hit_at_1']:.1%}\n")
    report_lines.append(f"- **Hit@5**: {summary['hit_at_5']:.1%}\n")
    report_lines.append(f"- **MRR**: {summary['mrr']:.3f}\n\n")

    report_lines.append("---\n\n")
    report_lines.append("## Q1 vs Q2 Performance\n\n")
    report_lines.append(f"- **Q1 (realistic variant 1) Hit@5**: {q1_hit5:.1%}\n")
    report_lines.append(f"- **Q2 (realistic variant 2) Hit@5**: {q2_hit5:.1%}\n\n")

    report_lines.append("---\n\n")
    report_lines.append("## Failure Analysis\n\n")
    report_lines.append("| Category | Count | % |\n")
    report_lines.append("|----------|-------|---|\n")
    for category, items in failure_categories.items():
        pct = len(items) / len(failures) * 100 if failures else 0
        report_lines.append(f"| {category} | {len(items)} | {pct:.1f}% |\n")
    report_lines.append("\n")

    report_lines.append("---\n\n")
    report_lines.append("## Worst Failures\n\n")
    for i, failure in enumerate(worst_failures, 1):
        report_lines.append(f'### {i}. Question: "{failure["question"]}"\n\n')
        report_lines.append(f"- **Expected**: `{failure['expected_doc']}`\n")
        retrieved_docs = failure["retrieved_docs"][:3]
        report_lines.append(
            f"- **Retrieved**: {', '.join(f'`{d}`' for d in retrieved_docs)}\n"
        )
        rank = failure.get("rank")
        rank_str = str(rank) if rank else "Not found"
        report_lines.append(f"- **Rank**: {rank_str}\n\n")

    report_content = "".join(report_lines)

    # Save report
    report_path = Path(results_folder) / "benchmark_report.md"
    with open(report_path, "w") as f:
        f.write(report_content)

    print(f"\n✅ Report saved to: {report_path}")

    # Append to notepad
    try:
        notepad_path = (
            Path(__file__).parent.parent.parent
            / ".sisyphus/notepads/realistic-questions-benchmark/learnings.md"
        )
        with open(notepad_path, "a") as f:
            f.write(
                f"\n## [{datetime.now().strftime('%Y-%m-%d')}] Task 6: Failure Analysis Report\n"
            )
            f.write(f"- Report generated for: {results_folder}\n")
            f.write(f"- Total failures: {len(failures)}\n")
            f.write(
                f"- Categories: {', '.join(f'{k}={len(v)}' for k, v in failure_categories.items())}\n"
            )
    except Exception as e:
        print(f"Warning: Could not append to notepad: {e}")


def validate_mapping():
    """Validate path mapping coverage against kubefix dataset."""
    print("Loading kubefix dataset...")
    ds = load_dataset("andyburgin/kubefix", split="train")

    total_questions = len(ds)
    matched_count = 0
    missing_docs = []

    print(f"Processing {total_questions} questions...")

    for i, example in enumerate(ds):
        source = example["source"]
        our_path = kubefix_to_our_path(source)
        doc_file = CORPUS_DIR / our_path

        if doc_file.exists():
            matched_count += 1
        else:
            missing_docs.append(
                {
                    "source": source,
                    "expected_path": our_path,
                    "question": example["instruction"][:100],  # First 100 chars
                }
            )

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{total_questions}...")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total kubefix questions: {total_questions}")
    print(
        f"Questions with matching docs: {matched_count} ({100 * matched_count / total_questions:.1f}%)"
    )
    print(f"Missing docs: {len(missing_docs)}")

    if missing_docs:
        print("\nMissing documents:")
        for item in missing_docs[:10]:  # Show first 10
            print(f"  - {item['source']}")
            print(f"    Expected: {item['expected_path']}")
            print(f"    Question: {item['question']}...")
        if len(missing_docs) > 10:
            print(f"  ... and {len(missing_docs) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Realistic question benchmark using kubefix dataset"
    )
    parser.add_argument(
        "--validate-mapping", action="store_true", help="Validate path mapping coverage"
    )
    parser.add_argument(
        "--test-prompt", action="store_true", help="Run automated prompt iteration test"
    )
    parser.add_argument(
        "--generate", type=int, metavar="N", help="Generate N realistic questions"
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Run retrieval benchmark against full corpus",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        metavar="PATH",
        help="Path to questions JSON file (for --run-benchmark)",
    )
    parser.add_argument(
        "--report",
        type=str,
        metavar="PATH",
        help="Generate failure analysis report from results folder",
    )

    args = parser.parse_args()

    if args.validate_mapping:
        validate_mapping()
    elif args.test_prompt:
        run_prompt_iteration_test()
    elif args.generate:
        generate_realistic_questions(n=args.generate)
    elif args.run_benchmark:
        run_retrieval_benchmark(questions_file=args.questions_file)
    elif args.report:
        generate_report(args.report)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
