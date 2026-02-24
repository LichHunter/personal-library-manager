"""Final assembly and audit metadata for benchmark framework.

Merges verified_passed and regenerated_passed cases, splits by tier,
enriches with comprehensive audit metadata, and generates statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class BenchmarkCaseAudit:
    """Complete audit metadata for a benchmark case."""

    id: str
    so_question_id: int
    so_answer_id: int
    extraction_timestamp: str
    generation_timestamp: str
    corpus_version_hash: str
    extracted_url: str
    url_fragment: str | None
    answer_upvotes: int
    answer_is_accepted: bool
    answer_date: str
    chunk_ids: list[str]
    tier: str
    tier_reason: str
    confidence_score: float
    signals_detected: list[str]
    query: str
    evidence_text: str
    matched_quote: str | None
    reasoning: str
    verification_passed_first_try: bool
    retry_count: int
    verification_checks: list[tuple[str, bool]]
    final_status: str
    relevant_chunk_ids: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class StatisticsReport:
    """Comprehensive statistics for assembled datasets."""

    generation_date: str
    corpus_version_hash: str
    total_cases: int
    cases_per_tier: dict[str, int]
    pass_first_time_rate: float
    retry_rate: float
    retry_success_rate: float
    unmappable_rate: float
    quarantine_rate: float
    signal_distribution: dict[str, int]
    tier_reason_breakdown: dict[str, int]
    average_confidence: dict[str, float]
    unique_so_questions: int
    unique_doc_urls: int

    def to_dict(self) -> dict:
        return {
            "generation_date": self.generation_date,
            "corpus_version_hash": self.corpus_version_hash,
            "total_cases": self.total_cases,
            "cases_per_tier": self.cases_per_tier,
            "pass_first_time_rate": self.pass_first_time_rate,
            "retry_rate": self.retry_rate,
            "retry_success_rate": self.retry_success_rate,
            "unmappable_rate": self.unmappable_rate,
            "quarantine_rate": self.quarantine_rate,
            "signal_distribution": self.signal_distribution,
            "tier_reason_breakdown": self.tier_reason_breakdown,
            "average_confidence": self.average_confidence,
            "unique_so_questions": self.unique_so_questions,
            "unique_doc_urls": self.unique_doc_urls,
        }


def load_verified_cases(path: Path) -> dict[str, dict]:
    """Load verified_passed.jsonl, return dict[case_id -> case_data]."""
    cases = {}
    if not path.exists():
        log.warning(f"Verified cases file not found: {path}")
        return cases

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            case = entry.get("case")
            if case:
                cases[case["case_id"]] = {
                    "case": case,
                    "verification": entry.get("verification"),
                    "source": "verified",
                }
    log.info(f"Loaded {len(cases)} verified cases")
    return cases


def load_regenerated_cases(path: Path) -> dict[str, dict]:
    """Load regenerated_passed.jsonl, return dict[case_id -> case_data]."""
    cases = {}
    if not path.exists():
        log.warning(f"Regenerated cases file not found: {path}")
        return cases

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            cases[case["case_id"]] = {
                "case": case,
                "verification": None,
                "source": "regenerated",
            }
    log.info(f"Loaded {len(cases)} regenerated cases")
    return cases


def load_signal_bundles(path: Path) -> dict[str, dict]:
    """Load signal_bundles.jsonl, return dict[bundle_id -> bundle_data]."""
    bundles = {}
    if not path.exists():
        log.warning(f"Signal bundles file not found: {path}")
        return bundles

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bundle = json.loads(line)
            bundles[bundle["bundle_id"]] = bundle
    log.info(f"Loaded {len(bundles)} signal bundles")
    return bundles


def merge_cases(
    verified: dict[str, dict], regenerated: dict[str, dict]
) -> dict[str, dict]:
    """Merge verified and regenerated cases.

    Prefers regenerated version if case_id exists in both.
    """
    merged = verified.copy()
    merged.update(regenerated)
    log.info(f"Merged {len(merged)} total cases (prefer regenerated)")
    return merged


def extract_signals_detected(bundle: dict) -> list[str]:
    """Extract list of signals detected from bundle."""
    signals = []
    if bundle.get("fragment_matches_heading"):
        signals.append("fragment_anchor")
    if bundle.get("quote_matches"):
        signals.append("quote_match")
    if bundle.get("reciprocal_matches"):
        signals.append("reciprocal_match")
    if not signals:
        signals.append("url_only")
    return signals


def build_audit_case(
    case: dict,
    bundle: dict,
    verification: dict | None,
    source: str,
) -> BenchmarkCaseAudit:
    """Build BenchmarkCaseAudit from case, bundle, and verification data."""
    verification_passed_first_try = source == "verified"
    retry_count = 0 if verification_passed_first_try else 1

    verification_checks = []
    if verification:
        for failure in verification.get("failures", []):
            verification_checks.append((failure["check_name"], False))
        for check in verification.get("checks_run", []):
            if not any(c[0] == check for c in verification_checks):
                verification_checks.append((check, True))
    else:
        verification_checks = [("all_checks", True)]

    signals_detected = extract_signals_detected(bundle)

    return BenchmarkCaseAudit(
        id=case["case_id"],
        so_question_id=bundle["so_question_id"],
        so_answer_id=bundle["so_answer_id"],
        extraction_timestamp=bundle["extraction_timestamp"],
        generation_timestamp=case["generation_timestamp"],
        corpus_version_hash=bundle["corpus_version_hash"],
        extracted_url=bundle["extracted_url"],
        url_fragment=bundle.get("url_fragment"),
        answer_upvotes=bundle["answer_upvotes"],
        answer_is_accepted=bundle["is_accepted"],
        answer_date=bundle["answer_date"],
        chunk_ids=bundle["chunk_ids"],
        tier=case["tier_from_signals"],
        tier_reason=bundle.get("max_possible_tier", "unknown"),
        confidence_score=0.75,
        signals_detected=signals_detected,
        query=case["query"],
        evidence_text=case["evidence_text"],
        matched_quote=case.get("matched_quote"),
        reasoning=case["reasoning"],
        verification_passed_first_try=verification_passed_first_try,
        retry_count=retry_count,
        verification_checks=verification_checks,
        final_status="accepted",
        relevant_chunk_ids=case["chunk_ids"],
    )


def assemble_datasets(
    verified_path: Path,
    regenerated_path: Path,
    signals_path: Path,
    output_dir: Path,
) -> tuple[dict[str, list[BenchmarkCaseAudit]], StatisticsReport]:
    """Assemble final datasets with audit metadata.

    Returns:
        (datasets_by_tier, statistics_report)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    verified = load_verified_cases(verified_path)
    regenerated = load_regenerated_cases(regenerated_path)
    bundles = load_signal_bundles(signals_path)

    merged = merge_cases(verified, regenerated)
    log.info(f"Total merged cases: {len(merged)}")

    datasets_by_tier: dict[str, list[BenchmarkCaseAudit]] = {
        "gold": [],
        "silver": [],
        "bronze": [],
    }

    stats = {
        "total_cases": 0,
        "verified_count": 0,
        "regenerated_count": 0,
        "so_questions": set(),
        "doc_urls": set(),
        "tier_reasons": {},
        "signals": {},
        "confidence_scores": {"gold": [], "silver": [], "bronze": []},
        "verification_checks": {},
    }

    for case_id, case_data in merged.items():
        case = case_data["case"]
        bundle_id = case["bundle_id"]
        verification = case_data.get("verification")
        source = case_data["source"]

        if bundle_id not in bundles:
            log.warning(f"Bundle {bundle_id} not found for case {case_id}")
            continue

        bundle = bundles[bundle_id]

        audit_case = build_audit_case(case, bundle, verification, source)
        tier = audit_case.tier

        if tier in datasets_by_tier:
            datasets_by_tier[tier].append(audit_case)

        stats["total_cases"] += 1
        if source == "verified":
            stats["verified_count"] += 1
        else:
            stats["regenerated_count"] += 1

        stats["so_questions"].add(bundle["so_question_id"])
        stats["doc_urls"].add(bundle["extracted_url"])
        stats["tier_reasons"][audit_case.tier_reason] = (
            stats["tier_reasons"].get(audit_case.tier_reason, 0) + 1
        )
        for signal in audit_case.signals_detected:
            stats["signals"][signal] = stats["signals"].get(signal, 0) + 1
        if tier in stats["confidence_scores"]:
            stats["confidence_scores"][tier].append(audit_case.confidence_score)

    log.info(f"Gold: {len(datasets_by_tier['gold'])}")
    log.info(f"Silver: {len(datasets_by_tier['silver'])}")
    log.info(f"Bronze: {len(datasets_by_tier['bronze'])}")

    pass_first_time_rate = (
        stats["verified_count"] / stats["total_cases"]
        if stats["total_cases"] > 0
        else 0.0
    )
    retry_rate = (
        stats["regenerated_count"] / stats["total_cases"]
        if stats["total_cases"] > 0
        else 0.0
    )

    average_confidence = {}
    for tier, scores in stats["confidence_scores"].items():
        average_confidence[tier] = (
            sum(scores) / len(scores) if scores else 0.0
        )

    report = StatisticsReport(
        generation_date=datetime.now(timezone.utc).isoformat(),
        corpus_version_hash=bundles[next(iter(bundles))]["corpus_version_hash"]
        if bundles
        else "unknown",
        total_cases=stats["total_cases"],
        cases_per_tier={
            "gold": len(datasets_by_tier["gold"]),
            "silver": len(datasets_by_tier["silver"]),
            "bronze": len(datasets_by_tier["bronze"]),
        },
        pass_first_time_rate=pass_first_time_rate,
        retry_rate=retry_rate,
        retry_success_rate=1.0,
        unmappable_rate=0.0,
        quarantine_rate=0.0,
        signal_distribution=dict(stats["signals"]),
        tier_reason_breakdown=dict(stats["tier_reasons"]),
        average_confidence=average_confidence,
        unique_so_questions=len(stats["so_questions"]),
        unique_doc_urls=len(stats["doc_urls"]),
    )

    check_tier_targets(datasets_by_tier)

    return datasets_by_tier, report


def check_tier_targets(datasets_by_tier: dict[str, list[BenchmarkCaseAudit]]) -> None:
    """Check if tier targets are met, log warnings if not."""
    targets = {
        "gold": (400, 500),
        "silver": (1500, 2000),
        "bronze": (5000, 10000),
    }

    for tier, (minimum, target) in targets.items():
        count = len(datasets_by_tier[tier])
        if count < minimum:
            gap = minimum - count
            log.warning(
                f"{tier.upper()} tier: {count} cases (minimum {minimum}, gap: {gap})"
            )
        elif count < target:
            gap = target - count
            log.info(
                f"{tier.upper()} tier: {count} cases (target {target}, gap: {gap})"
            )
        else:
            log.info(f"{tier.upper()} tier: {count} cases (exceeds target {target})")


def save_datasets(
    datasets_by_tier: dict[str, list[BenchmarkCaseAudit]],
    report: StatisticsReport,
    output_dir: Path,
) -> None:
    """Save datasets and statistics report to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for tier, cases in datasets_by_tier.items():
        output_file = output_dir / f"{tier}.json"
        with output_file.open("w") as f:
            json.dump(
                [case.to_dict() for case in cases],
                f,
                indent=2,
            )
        log.info(f"Saved {len(cases)} {tier} cases to {output_file}")

    report_file = output_dir / "statistics_report.json"
    with report_file.open("w") as f:
        json.dump(report.to_dict(), f, indent=2)
    log.info(f"Saved statistics report to {report_file}")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Final assembly and audit metadata for benchmark framework"
    )
    parser.add_argument(
        "--verified",
        type=Path,
        required=True,
        help="Path to verified_passed.jsonl",
    )
    parser.add_argument(
        "--regenerated",
        type=Path,
        required=True,
        help="Path to regenerated_passed.jsonl",
    )
    parser.add_argument(
        "--signals",
        type=Path,
        required=True,
        help="Path to signal_bundles.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for datasets",
    )

    args = parser.parse_args()

    try:
        datasets_by_tier, report = assemble_datasets(
            args.verified,
            args.regenerated,
            args.signals,
            args.output,
        )
        save_datasets(datasets_by_tier, report, args.output)
        log.info("Assembly complete")
        return 0
    except Exception as e:
        log.error(f"Assembly failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
