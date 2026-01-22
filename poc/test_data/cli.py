import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .generator import TestDataGenerator

console = Console()


def cmd_generate(args: argparse.Namespace) -> None:
    generator = TestDataGenerator(ollama_model=args.model)
    output_dir = Path(args.output)

    if args.source == "wikipedia":
        if not args.titles:
            console.print("[red]Error: --titles required for wikipedia source[/red]")
            sys.exit(1)
        docs = generator.add_wikipedia(args.titles)
        console.print(f"[green]Fetched {len(docs)} Wikipedia articles[/green]")

    elif args.source == "synthetic":
        if not args.topics:
            console.print("[red]Error: --topics required for synthetic source[/red]")
            sys.exit(1)
        docs = generator.add_synthetic(args.topics, num_sections=args.num_sections)
        console.print(f"[green]Generated {len(docs)} synthetic documents[/green]")

    if args.generate_gt:
        gts = generator.generate_ground_truth(questions_per_doc=args.questions_per_doc)
        console.print(f"[green]Generated {len(gts)} ground truth Q&A pairs[/green]")

    generator.save(output_dir)
    console.print(f"[green]Saved dataset to {output_dir}[/green]")


def cmd_ground_truth(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    generator = TestDataGenerator.load(input_dir, ollama_model=args.model)

    if not generator.documents:
        console.print("[red]Error: No documents found in input directory[/red]")
        sys.exit(1)

    gts = generator.generate_ground_truth(questions_per_doc=args.questions_per_doc)
    console.print(f"[green]Generated {len(gts)} ground truth Q&A pairs[/green]")

    generator.save(input_dir)
    console.print(f"[green]Updated dataset at {input_dir}[/green]")


def cmd_stats(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    manifest_path = input_dir / "manifest.json"

    if not manifest_path.exists():
        console.print("[red]Error: manifest.json not found[/red]")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    console.print("\n[bold]Dataset Statistics[/bold]\n")

    doc_table = Table(title="Documents")
    doc_table.add_column("ID")
    doc_table.add_column("Title")
    doc_table.add_column("Source")
    doc_table.add_column("Sections")

    for doc in manifest.get("documents", []):
        doc_table.add_row(
            doc["id"],
            doc["title"][:40] + "..." if len(doc["title"]) > 40 else doc["title"],
            doc["source"],
            str(doc["num_sections"]),
        )

    console.print(doc_table)

    gt_stats = manifest.get("ground_truth_stats", {})
    console.print(f"\n[bold]Ground Truth: {gt_stats.get('total', 0)} entries[/bold]")

    if gt_stats:
        diff_table = Table(title="By Difficulty")
        diff_table.add_column("Difficulty")
        diff_table.add_column("Count")
        for diff, count in gt_stats.get("by_difficulty", {}).items():
            diff_table.add_row(diff, str(count))
        console.print(diff_table)

        type_table = Table(title="By Query Type")
        type_table.add_column("Type")
        type_table.add_column("Count")
        for qtype, count in gt_stats.get("by_query_type", {}).items():
            type_table.add_row(qtype, str(count))
        console.print(type_table)

    console.print(f"\n[dim]Created: {manifest.get('created_at', 'unknown')}[/dim]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test data generation and evaluation for RAG systems",
        prog="python -m poc.test_data.cli",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate test dataset")
    gen_parser.add_argument(
        "--source",
        choices=["wikipedia", "synthetic"],
        required=True,
        help="Data source type",
    )
    gen_parser.add_argument(
        "--titles",
        nargs="+",
        help="Wikipedia article titles (for wikipedia source)",
    )
    gen_parser.add_argument(
        "--topics",
        nargs="+",
        help="Topics for synthetic generation",
    )
    gen_parser.add_argument(
        "--output",
        default="./output",
        help="Output directory (default: ./output)",
    )
    gen_parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model for generation (default: llama3.2:3b)",
    )
    gen_parser.add_argument(
        "--num-sections",
        type=int,
        default=4,
        help="Number of sections for synthetic docs (default: 4)",
    )
    gen_parser.add_argument(
        "--generate-gt",
        action="store_true",
        help="Also generate ground truth Q&A pairs",
    )
    gen_parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=5,
        help="Questions per document for ground truth (default: 5)",
    )
    gen_parser.set_defaults(func=cmd_generate)

    gt_parser = subparsers.add_parser("ground-truth", help="Generate ground truth for existing docs")
    gt_parser.add_argument(
        "--input",
        required=True,
        help="Input directory with documents",
    )
    gt_parser.add_argument(
        "--questions-per-doc",
        type=int,
        default=5,
        help="Questions per document (default: 5)",
    )
    gt_parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model for generation (default: llama3.2:3b)",
    )
    gt_parser.set_defaults(func=cmd_ground_truth)

    stats_parser = subparsers.add_parser("stats", help="Show dataset statistics")
    stats_parser.add_argument(
        "--input",
        required=True,
        help="Input directory with dataset",
    )
    stats_parser.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
