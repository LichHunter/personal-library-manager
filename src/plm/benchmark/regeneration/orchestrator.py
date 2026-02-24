"""Regeneration orchestrator for failed benchmark cases.

Processes verified_failed.jsonl, attempts regeneration with structured feedback,
and outputs regenerated_passed, unmappable, and quarantine files.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from plm.benchmark.generation.generator import (
    GeneratedCase,
    normalize_whitespace,
    parse_llm_response,
    validate_generated_case,
)
from plm.benchmark.generation.prompts import (
    GENERATOR_SYSTEM_PROMPT,
    build_generator_prompt,
    build_signals_summary,
)
from plm.benchmark.verification.verifier import (
    VerificationFailure,
    VerificationResult,
    load_chunk_corpus,
    verify,
)
from plm.shared.llm import call_llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


TerminationDecision = Literal["PASSED", "RETRY", "UNMAPPABLE", "QUARANTINE"]


@dataclass
class RegenerationAttempt:
    """Record of a single regeneration attempt."""

    case_id: str
    attempt_number: int
    previous_failures: list[dict]
    feedback_prompt: str
    new_case: dict | None
    verification_result: dict | None
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegenerationResult:
    """Final result for a regenerated case."""

    original_case_id: str
    bundle_id: str
    final_status: Literal["passed", "unmappable", "quarantine"]
    termination_reason: str
    total_attempts: int
    final_case: dict | None
    all_attempts: list[RegenerationAttempt]

    def to_dict(self) -> dict:
        return {
            "original_case_id": self.original_case_id,
            "bundle_id": self.bundle_id,
            "final_status": self.final_status,
            "termination_reason": self.termination_reason,
            "total_attempts": self.total_attempts,
            "final_case": self.final_case,
            "all_attempts": [a.to_dict() for a in self.all_attempts],
        }


@dataclass
class RegenerationStats:
    """Statistics for a regeneration run."""

    total_failed_input: int = 0
    regenerated_passed: int = 0
    unmappable_count: int = 0
    quarantine_count: int = 0
    recovery_rate: float = 0.0
    quarantine_rate: float = 0.0
    attempts_distribution: dict[int, int] = field(default_factory=dict)
    regeneration_duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def create_regeneration_prompt(
    original_case: dict,
    failures: list[VerificationFailure],
) -> str:
    """Create structured feedback prompt based on failure types."""
    lines = ["Previous attempt failed verification:"]

    for f in failures:
        if f.check_name == "quote_exists":
            lines.append(f"- Quote '{f.expected}' NOT FOUND in chunk.")
            lines.append("  FIX: Find a DIFFERENT quote that appears VERBATIM.")
        elif f.check_name == "quote_length":
            lines.append(f"- Quote only {f.found} chars, need >=30 for GOLD.")
            lines.append("  FIX: Find a longer quote or accept lower tier.")
        elif f.check_name == "tier_match":
            lines.append(f"- Claimed tier '{f.found}' but signals support '{f.expected}'.")
            lines.append(f"  FIX: Accept tier '{f.expected}'.")
        elif f.check_name == "query_length":
            word_count = int(f.found) if f.found.isdigit() else 0
            fix = "Expand" if word_count < 5 else "Condense"
            lines.append(f"- Query has {f.found} words, need 5-100.")
            lines.append(f"  FIX: {fix} the query.")
        elif f.check_name == "chunk_exists":
            lines.append(f"- Chunk '{f.expected}' NOT FOUND in corpus.")
            lines.append("  FIX: This is a corpus gap - cannot be fixed by regeneration.")

    lines.append("")
    lines.append("Generate a NEW case that avoids these failures.")

    return "\n".join(lines)


def get_termination_decision(
    failures: list[VerificationFailure],
    retry_count: int,
    previous_failures_history: list[list[VerificationFailure]],
) -> tuple[TerminationDecision, str | None]:
    """Determine whether to retry, terminate as unmappable, or quarantine.

    Args:
        failures: Current verification failures
        retry_count: Number of retries already attempted (0-based)
        previous_failures_history: List of failure lists from previous attempts

    Returns:
        (decision, reason) tuple
    """
    if not failures:
        return "PASSED", None

    corpus_gap = any(
        f.check_name == "chunk_exists" and not f.recoverable for f in failures
    )
    if corpus_gap:
        return "UNMAPPABLE", "corpus_gap"

    if retry_count >= 3:
        return "QUARANTINE", "max_retries_exceeded"

    if retry_count >= 2 and previous_failures_history:
        current_checks = {f.check_name for f in failures}
        prev_checks = {f.check_name for f in previous_failures_history[-1]}
        if current_checks == prev_checks:
            primary_failure = list(current_checks)[0] if current_checks else "unknown"
            return "UNMAPPABLE", f"persistent_failure_{primary_failure}"

    return "RETRY", None


def load_failed_cases(failed_path: Path) -> list[dict]:
    """Load failed cases from verified_failed.jsonl."""
    cases = []
    with failed_path.open() as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                cases.append(entry)
    return cases


def load_signal_bundles_map(signals_path: Path) -> dict[str, dict]:
    """Load signal bundles into a map keyed by bundle_id."""
    bundles = {}
    with signals_path.open() as f:
        for line in f:
            if line.strip():
                bundle = json.loads(line)
                bundles[bundle["bundle_id"]] = bundle
    return bundles


def regenerate_case(
    bundle: dict,
    feedback: str,
    model: str,
) -> tuple[GeneratedCase | None, str | None]:
    """Attempt to regenerate a case with feedback.

    Uses a fresh LLM call (no conversation context).

    Returns:
        (case, error_message)
    """
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

    base_prompt = build_generator_prompt(
        question_title=bundle["question_title"],
        question_body=bundle["question_body"],
        answer_body=bundle["answer_body"],
        chunk_contents=chunk_contents,
        signals_summary=signals_summary,
        tier=tier,
    )

    full_prompt = f"{GENERATOR_SYSTEM_PROMPT}\n\n{base_prompt}\n\n## REGENERATION FEEDBACK\n{feedback}"

    response = call_llm(
        full_prompt,
        model=model,
        max_tokens=1024,
        temperature=0.3,
    )

    if not response:
        return None, "Empty LLM response"

    parsed = parse_llm_response(response)
    if not parsed:
        return None, "Failed to parse JSON response"

    is_valid, error = validate_generated_case(parsed, tier, chunk_contents)
    if not is_valid:
        return None, f"Self-check failed: {error}"

    case = GeneratedCase(
        case_id=str(uuid.uuid4()),
        bundle_id=bundle["bundle_id"],
        query=parsed["query"],
        matched_quote=parsed.get("matched_quote"),
        evidence_text=parsed["evidence_text"],
        reasoning=parsed["reasoning"],
        chunk_ids=chunk_ids,
        tier_from_signals=tier,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        generator_model=model,
        internal_retries=0,
    )

    return case, None


def process_failed_case(
    failed_entry: dict,
    bundles_map: dict[str, dict],
    chunk_corpus: dict[str, str],
    model: str,
) -> RegenerationResult:
    """Process a single failed case through regeneration loop.

    Returns:
        RegenerationResult with final status
    """
    original_case = failed_entry["case"]
    original_verification = failed_entry["verification"]

    case_id = original_case["case_id"]
    bundle_id = original_case["bundle_id"]

    if bundle_id not in bundles_map:
        return RegenerationResult(
            original_case_id=case_id,
            bundle_id=bundle_id,
            final_status="unmappable",
            termination_reason="bundle_not_found",
            total_attempts=0,
            final_case=None,
            all_attempts=[],
        )

    bundle = bundles_map[bundle_id]

    initial_failures = [
        VerificationFailure(
            check_name=f["check_name"],
            expected=f["expected"],
            found=f["found"],
            recoverable=f["recoverable"],
        )
        for f in original_verification["failures"]
    ]

    corpus_gap = any(
        f.check_name == "chunk_exists" and not f.recoverable
        for f in initial_failures
    )
    if corpus_gap:
        return RegenerationResult(
            original_case_id=case_id,
            bundle_id=bundle_id,
            final_status="unmappable",
            termination_reason="corpus_gap",
            total_attempts=0,
            final_case=None,
            all_attempts=[],
        )

    attempts: list[RegenerationAttempt] = []
    failures_history: list[list[VerificationFailure]] = [initial_failures]
    current_failures = initial_failures

    for attempt_num in range(1, 4):  # Max 3 attempts
        feedback = create_regeneration_prompt(original_case, current_failures)

        new_case, gen_error = regenerate_case(bundle, feedback, model)

        if gen_error:
            attempt = RegenerationAttempt(
                case_id=case_id,
                attempt_number=attempt_num,
                previous_failures=[f.to_dict() for f in current_failures],
                feedback_prompt=feedback,
                new_case=None,
                verification_result={"error": gen_error},
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            attempts.append(attempt)
            continue

        assert new_case is not None
        verification = verify(new_case, chunk_corpus)

        attempt = RegenerationAttempt(
            case_id=case_id,
            attempt_number=attempt_num,
            previous_failures=[f.to_dict() for f in current_failures],
            feedback_prompt=feedback,
            new_case=new_case.to_dict(),
            verification_result=verification.to_dict(),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        attempts.append(attempt)

        if verification.passed:
            return RegenerationResult(
                original_case_id=case_id,
                bundle_id=bundle_id,
                final_status="passed",
                termination_reason=f"passed_attempt_{attempt_num}",
                total_attempts=attempt_num,
                final_case=new_case.to_dict(),
                all_attempts=attempts,
            )

        current_failures = verification.failures
        failures_history.append(current_failures)

        decision, reason = get_termination_decision(
            current_failures, attempt_num, failures_history[:-1]
        )

        if decision == "UNMAPPABLE":
            return RegenerationResult(
                original_case_id=case_id,
                bundle_id=bundle_id,
                final_status="unmappable",
                termination_reason=reason or "unknown",
                total_attempts=attempt_num,
                final_case=None,
                all_attempts=attempts,
            )

    return RegenerationResult(
        original_case_id=case_id,
        bundle_id=bundle_id,
        final_status="quarantine",
        termination_reason="max_retries_exceeded",
        total_attempts=3,
        final_case=None,
        all_attempts=attempts,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate failed benchmark cases with structured feedback.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m plm.benchmark.regeneration.orchestrator \\
    --failed artifacts/benchmark/verification/verified_failed.jsonl \\
    --signals artifacts/benchmark/signals/signal_bundles.jsonl \\
    --corpus-db /path/to/index.db \\
    --output artifacts/benchmark/regeneration/
""",
    )
    parser.add_argument(
        "--failed",
        required=True,
        type=Path,
        help="Path to verified_failed.jsonl",
    )
    parser.add_argument(
        "--signals",
        required=True,
        type=Path,
        help="Path to signal_bundles.jsonl",
    )
    parser.add_argument(
        "--corpus-db",
        required=True,
        type=Path,
        help="Path to SQLite corpus database",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model (default: PLM_LLM_MODEL env or claude-haiku-4-5)",
    )

    args = parser.parse_args(argv)

    if not args.failed.exists():
        log.error(f"Failed cases file not found: {args.failed}")
        return 1

    if not args.signals.exists():
        log.error(f"Signals file not found: {args.signals}")
        return 1

    if not args.corpus_db.exists():
        log.error(f"Corpus database not found: {args.corpus_db}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    model = args.model or os.getenv("PLM_LLM_MODEL", "claude-haiku-4-5")

    log.info("=" * 60)
    log.info("Regeneration Orchestrator")
    log.info("=" * 60)
    log.info(f"Failed cases: {args.failed}")
    log.info(f"Signals: {args.signals}")
    log.info(f"Corpus DB: {args.corpus_db}")
    log.info(f"Output: {args.output}")
    log.info(f"Model: {model}")

    log.info("-" * 60)
    log.info("Loading inputs...")

    failed_cases = load_failed_cases(args.failed)
    log.info(f"Loaded {len(failed_cases)} failed cases")

    bundles_map = load_signal_bundles_map(args.signals)
    log.info(f"Loaded {len(bundles_map)} signal bundles")

    chunk_corpus = load_chunk_corpus(args.corpus_db)

    if not failed_cases:
        log.warning("No failed cases to process")
        return 0

    log.info("-" * 60)
    log.info("Processing failed cases...")

    start_time = time.time()

    passed_results: list[RegenerationResult] = []
    unmappable_results: list[RegenerationResult] = []
    quarantine_results: list[RegenerationResult] = []
    all_results: list[RegenerationResult] = []
    attempts_distribution: dict[int, int] = {}

    for i, entry in enumerate(failed_cases):
        if (i + 1) % 10 == 0:
            log.info(f"Processing {i + 1}/{len(failed_cases)}...")

        result = process_failed_case(entry, bundles_map, chunk_corpus, model)
        all_results.append(result)

        attempts_distribution[result.total_attempts] = (
            attempts_distribution.get(result.total_attempts, 0) + 1
        )

        if result.final_status == "passed":
            passed_results.append(result)
        elif result.final_status == "unmappable":
            unmappable_results.append(result)
        else:
            quarantine_results.append(result)

    duration = time.time() - start_time

    passed_file = args.output / "regenerated_passed.jsonl"
    with passed_file.open("w") as f:
        for result in passed_results:
            if result.final_case:
                f.write(json.dumps(result.final_case, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(passed_results)} passed cases to {passed_file}")

    unmappable_file = args.output / "unmappable.jsonl"
    with unmappable_file.open("w") as f:
        for result in unmappable_results:
            entry = {
                "original_case_id": result.original_case_id,
                "bundle_id": result.bundle_id,
                "termination_reason": result.termination_reason,
                "total_attempts": result.total_attempts,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(unmappable_results)} unmappable cases to {unmappable_file}")

    quarantine_file = args.output / "quarantine.jsonl"
    with quarantine_file.open("w") as f:
        for result in quarantine_results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(quarantine_results)} quarantine cases to {quarantine_file}")

    log_file = args.output / "regeneration_log.jsonl"
    with log_file.open("w") as f:
        for result in all_results:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(all_results)} results to {log_file}")

    total_generated = len(passed_results) + len(unmappable_results) + len(quarantine_results)
    recovery_rate = len(passed_results) / len(failed_cases) if failed_cases else 0.0
    quarantine_rate = len(quarantine_results) / total_generated if total_generated else 0.0

    stats = RegenerationStats(
        total_failed_input=len(failed_cases),
        regenerated_passed=len(passed_results),
        unmappable_count=len(unmappable_results),
        quarantine_count=len(quarantine_results),
        recovery_rate=recovery_rate,
        quarantine_rate=quarantine_rate,
        attempts_distribution=attempts_distribution,
        regeneration_duration_seconds=duration,
    )

    stats_file = args.output / "regeneration_stats.json"
    stats_file.write_text(json.dumps(stats.to_dict(), indent=2, ensure_ascii=False))
    log.info(f"Wrote stats to {stats_file}")

    log.info("-" * 60)
    log.info("REGENERATION STATISTICS")
    log.info("-" * 60)
    log.info(f"Total failed input:     {stats.total_failed_input}")
    log.info(f"Regenerated passed:     {stats.regenerated_passed}")
    log.info(f"Unmappable count:       {stats.unmappable_count}")
    log.info(f"Quarantine count:       {stats.quarantine_count}")
    log.info(f"Recovery rate:          {stats.recovery_rate * 100:.1f}%")
    log.info(f"Quarantine rate:        {stats.quarantine_rate * 100:.1f}%")
    log.info(f"Duration:               {stats.regeneration_duration_seconds:.1f}s")
    log.info(f"Attempts distribution:  {stats.attempts_distribution}")

    quality_pass = True
    if stats.quarantine_rate > 0.05:
        log.warning(f"Quarantine rate {stats.quarantine_rate * 100:.1f}% exceeds 5% target")
        quality_pass = False
    if stats.recovery_rate < 0.50:
        log.warning(f"Recovery rate {stats.recovery_rate * 100:.1f}% below 50% target")
        quality_pass = False

    if quality_pass:
        log.info("Quality targets MET")
    else:
        log.warning("Quality targets NOT MET")

    log.info("=" * 60)
    log.info("REGENERATION COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
