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

import anthropic
from datasets import load_dataset


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
    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    prompt = TRANSFORMATION_PROMPT_V2.format(original_question=original_question)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.content[0].text.strip()

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

    args = parser.parse_args()

    if args.validate_mapping:
        validate_mapping()
    elif args.test_prompt:
        run_prompt_iteration_test()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
