"""Parallel LLM Generator for benchmark case generation.

Processes SignalBundles to generate benchmark cases using LLM with self-checks.
Uses ThreadPoolExecutor for parallel generation. Outputs JSONL batches.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from plm.benchmark.generation.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    build_generator_prompt,
    build_signals_summary,
)
from plm.shared.llm import call_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class GeneratedCase:
    """Output from generator for a single SignalBundle."""

    case_id: str
    bundle_id: str
    query: str
    matched_quote: str | None
    evidence_text: str
    reasoning: str
    chunk_ids: list[str]
    tier_from_signals: Literal["gold", "silver", "bronze"]
    generation_timestamp: str
    generator_model: str
    internal_retries: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationStats:
    """Statistics for a generation run."""

    total_bundles_input: int = 0
    cases_generated: int = 0
    failed_count: int = 0
    pass_first_try: int = 0
    required_retry: int = 0
    tier_breakdown: dict[str, int] | None = None
    avg_query_words: float = 0.0
    generation_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace for quote matching."""
    return " ".join(text.split())


def validate_quote_in_chunks(quote: str, chunk_contents: list[str]) -> bool:
    """Check if quote appears verbatim in any chunk (whitespace normalized)."""
    if not quote:
        return True
    normalized_quote = normalize_whitespace(quote)
    for chunk in chunk_contents:
        normalized_chunk = normalize_whitespace(chunk)
        if normalized_quote in normalized_chunk:
            return True
    return False


def parse_llm_response(response: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not response:
        return None

    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        start_idx = 1
        end_idx = len(lines)
        for i, line in enumerate(lines[1:], 1):
            if line.startswith("```"):
                end_idx = i
                break
        text = "\n".join(lines[start_idx:end_idx])

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
    return None


def validate_generated_case(
    parsed: dict,
    tier: str,
    chunk_contents: list[str],
) -> tuple[bool, str]:
    """Validate a generated case passes self-checks.

    Returns:
        (is_valid, error_message)
    """
    query = parsed.get("query", "")
    matched_quote = parsed.get("matched_quote")

    word_count = count_words(query)
    if word_count < 5:
        return False, f"Query too short: {word_count} words (min 5)"
    if word_count > 100:
        return False, f"Query too long: {word_count} words (max 100)"

    if matched_quote:
        if not validate_quote_in_chunks(matched_quote, chunk_contents):
            return False, "Quote not found verbatim in chunks"
        if tier == "gold" and len(matched_quote) < 30:
            return False, f"GOLD quote too short: {len(matched_quote)} chars (min 30)"

    if not parsed.get("evidence_text"):
        return False, "Missing evidence_text"

    if not parsed.get("reasoning"):
        return False, "Missing reasoning"

    return True, ""


def generate_single_case(
    bundle: dict,
    model: str,
    max_retries: int = 2,
) -> tuple[GeneratedCase | None, str | None, int]:
    """Generate a single case from a SignalBundle.

    Returns:
        (case, error_message, retries_used)
    """
    bundle_id = bundle["bundle_id"]
    tier = bundle["max_possible_tier"]
    chunk_contents = bundle["chunk_contents"]
    chunk_ids = bundle["chunk_ids"]

    signals_summary = build_signals_summary(
        quote_matches=bundle.get("quote_matches", []),
        reciprocal_matches=bundle.get("reciprocal_matches", []),
        fragment_matches_heading=bundle.get("fragment_matches_heading", False),
        url_fragment=bundle.get("url_fragment"),
        answer_upvotes=bundle.get("answer_upvotes", 0),
        is_accepted=bundle.get("is_accepted", False),
    )

    prompt = build_generator_prompt(
        question_title=bundle["question_title"],
        question_body=bundle["question_body"],
        answer_body=bundle["answer_body"],
        chunk_contents=chunk_contents,
        signals_summary=signals_summary,
        tier=tier,
    )

    full_prompt = f"{GENERATOR_SYSTEM_PROMPT}\n\n{prompt}"

    retries = 0
    last_error = ""

    for attempt in range(max_retries + 1):
        response = call_llm(
            full_prompt,
            model=model,
            max_tokens=1024,
            temperature=0.3,
        )

        if not response:
            last_error = "Empty LLM response"
            retries = attempt
            continue

        parsed = parse_llm_response(response)
        if not parsed:
            last_error = "Failed to parse JSON response"
            retries = attempt
            continue

        is_valid, error = validate_generated_case(parsed, tier, chunk_contents)
        if is_valid:
            case = GeneratedCase(
                case_id=str(uuid.uuid4()),
                bundle_id=bundle_id,
                query=parsed["query"],
                matched_quote=parsed.get("matched_quote"),
                evidence_text=parsed["evidence_text"],
                reasoning=parsed["reasoning"],
                chunk_ids=chunk_ids,
                tier_from_signals=tier,
                generation_timestamp=datetime.now(timezone.utc).isoformat(),
                generator_model=model,
                internal_retries=attempt,
            )
            return case, None, attempt

        last_error = error
        retries = attempt

    return None, last_error, retries


def load_signal_bundles(signals_path: Path) -> list[dict]:
    """Load signal bundles from JSONL file."""
    bundles = []
    with signals_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                bundle = json.loads(line)
                if bundle.get("max_possible_tier") != "exclude":
                    bundles.append(bundle)
    return bundles


def process_batch(
    bundles: list[dict],
    model: str,
    workers: int,
    output_dir: Path,
    batch_num: int,
) -> tuple[list[GeneratedCase], list[dict]]:
    """Process a batch of bundles in parallel.

    Returns:
        (cases, errors)
    """
    cases: list[GeneratedCase] = []
    errors: list[dict] = []

    def process_wrapper(bundle: dict) -> tuple[GeneratedCase | None, dict | None]:
        case, error, retries = generate_single_case(bundle, model)
        if case:
            return case, None
        return None, {
            "bundle_id": bundle["bundle_id"],
            "so_answer_id": bundle.get("so_answer_id"),
            "error": error,
            "retries": retries,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_wrapper, b): b for b in bundles}
        processed = 0

        for future in as_completed(futures):
            processed += 1
            if processed % 50 == 0:
                log.info(f"Batch {batch_num}: {processed}/{len(bundles)} processed...")

            case, error = future.result()
            if case:
                cases.append(case)
            elif error:
                errors.append(error)

    return cases, errors


def write_batch(cases: list[GeneratedCase], output_path: Path) -> None:
    """Write cases to JSONL file."""
    with output_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate benchmark cases from SignalBundles using LLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m plm.benchmark.generation.generator \\
    --signals artifacts/benchmark/signals/signal_bundles.jsonl \\
    --output artifacts/benchmark/generated/ \\
    --workers 4 \\
    --batch-size 500
""",
    )
    parser.add_argument(
        "--signals",
        required=True,
        type=Path,
        help="Path to signal_bundles.jsonl",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for generated batches",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Bundles per batch (default: 500)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model (default: PLM_LLM_MODEL env or claude-haiku-4-5)",
    )

    args = parser.parse_args(argv)

    if not args.signals.exists():
        log.error(f"Signals file not found: {args.signals}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    model = args.model or os.getenv("PLM_LLM_MODEL", "claude-haiku-4-5")

    log.info("=" * 60)
    log.info("Benchmark Case Generator")
    log.info("=" * 60)
    log.info(f"Signals: {args.signals}")
    log.info(f"Output: {args.output}")
    log.info(f"Workers: {args.workers}")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Model: {model}")

    log.info("-" * 60)
    log.info("Loading signal bundles...")

    bundles = load_signal_bundles(args.signals)
    log.info(f"Loaded {len(bundles)} bundles (excluding 'exclude' tier)")

    if not bundles:
        log.warning("No bundles to process")
        return 0

    log.info("-" * 60)
    log.info("Starting generation...")

    start_time = time.time()

    all_cases: list[GeneratedCase] = []
    all_errors: list[dict] = []
    pass_first_try = 0
    required_retry = 0

    num_batches = (len(bundles) + args.batch_size - 1) // args.batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(bundles))
        batch_bundles = bundles[start_idx:end_idx]

        log.info(f"Processing batch {batch_num + 1}/{num_batches} ({len(batch_bundles)} bundles)")

        cases, errors = process_batch(
            batch_bundles,
            model,
            args.workers,
            args.output,
            batch_num + 1,
        )

        for case in cases:
            if case.internal_retries == 0:
                pass_first_try += 1
            else:
                required_retry += 1

        all_cases.extend(cases)
        all_errors.extend(errors)

        batch_file = args.output / f"batch_{batch_num + 1}.jsonl"
        write_batch(cases, batch_file)
        log.info(f"Wrote {len(cases)} cases to {batch_file}")

    duration = time.time() - start_time

    tier_breakdown: dict[str, int] = {}
    total_words = 0
    for case in all_cases:
        tier = case.tier_from_signals
        tier_breakdown[tier] = tier_breakdown.get(tier, 0) + 1
        total_words += count_words(case.query)

    avg_query_words = total_words / len(all_cases) if all_cases else 0.0

    stats = GenerationStats(
        total_bundles_input=len(bundles),
        cases_generated=len(all_cases),
        failed_count=len(all_errors),
        pass_first_try=pass_first_try,
        required_retry=required_retry,
        tier_breakdown=tier_breakdown,
        avg_query_words=avg_query_words,
        generation_duration_seconds=duration,
    )

    stats_file = args.output / "generation_stats.json"
    stats_file.write_text(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    log.info(f"Wrote stats to {stats_file}")

    if all_errors:
        errors_file = args.output / "generation_errors.log"
        with errors_file.open("w") as f:
            for err in all_errors:
                f.write(
                    f"bundle_id={err['bundle_id']} "
                    f"so_answer_id={err.get('so_answer_id')} "
                    f"error={err['error']} "
                    f"retries={err['retries']}\n"
                )
        log.info(f"Wrote {len(all_errors)} errors to {errors_file}")

    log.info("-" * 60)
    log.info("GENERATION STATISTICS")
    log.info("-" * 60)
    log.info(f"Total bundles input:    {stats.total_bundles_input}")
    log.info(f"Cases generated:        {stats.cases_generated}")
    log.info(f"Failed count:           {stats.failed_count}")
    log.info(f"Passed first try:       {stats.pass_first_try}")
    log.info(f"Required retry:         {stats.required_retry}")
    log.info(f"Avg query words:        {stats.avg_query_words:.1f}")
    log.info(f"Duration:               {stats.generation_duration_seconds:.1f}s")

    for tier, count in sorted(tier_breakdown.items()):
        log.info(f"  {tier}: {count}")

    log.info("=" * 60)
    log.info("GENERATION COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
