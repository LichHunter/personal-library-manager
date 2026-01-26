#!/usr/bin/env python3
"""
Benchmark using realistic questions from the kubefix dataset.

This script loads questions from the kubefix HuggingFace dataset and validates
that we have matching documents in our corpus for each question.
"""

import argparse
from pathlib import Path
from datasets import load_dataset


# Path to the corpus directory
CORPUS_DIR = Path(__file__).parent / "corpus" / "kubernetes"


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

    args = parser.parse_args()

    if args.validate_mapping:
        validate_mapping()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
