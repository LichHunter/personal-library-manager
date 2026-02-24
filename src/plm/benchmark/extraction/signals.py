"""Deterministic Signal Extraction Pipeline for benchmark framework.

Processes StackOverflow posts and extracts signal bundles with:
- URL mapping to documentation chunks
- Quote and reciprocal matching
- Tier assignment
- Corpus version hashing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal
from urllib.parse import urlparse

from plm.benchmark.matching.fragment_matcher import (
    find_reciprocal_matches,
    normalize_anchor,
)
from plm.benchmark.matching.quote_matcher import find_quote_matches
from plm.benchmark.tier.engine import (
    QuoteMatch,
    ReciprocalMatch,
    TierAssignment,
    TierAssignmentInput,
    assign_tier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class SignalBundle:
    """Complete signal bundle for a single SO answer."""

    bundle_id: str
    so_question_id: int
    so_answer_id: int
    question_title: str
    question_body: str
    answer_body: str
    extracted_url: str
    url_fragment: str | None
    answer_upvotes: int
    is_accepted: bool
    answer_date: str
    chunk_ids: list[str]
    chunk_contents: list[str]
    quote_matches: list[dict] = field(default_factory=list)
    reciprocal_matches: list[dict] = field(default_factory=list)
    fragment_matches_heading: bool = False
    max_possible_tier: Literal["gold", "silver", "bronze", "exclude"] = "exclude"
    extraction_timestamp: str = ""
    corpus_version_hash: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)


def load_mappings(mappings_dir: Path) -> tuple[dict, dict, dict]:
    """Load URL-to-chunk mappings from directory.

    Returns:
        (url_to_chunks, anchor_to_heading, url_to_docid)
    """
    url_to_chunks_file = mappings_dir / "url_to_chunks.json"
    anchor_to_heading_file = mappings_dir / "anchor_to_heading.json"
    url_to_docid_file = mappings_dir / "url_to_docid.json"

    if not url_to_chunks_file.exists():
        raise FileNotFoundError(f"Missing mapping file: {url_to_chunks_file}")

    url_to_chunks = json.loads(url_to_chunks_file.read_text())
    anchor_to_heading = (
        json.loads(anchor_to_heading_file.read_text())
        if anchor_to_heading_file.exists()
        else {}
    )
    url_to_docid = (
        json.loads(url_to_docid_file.read_text())
        if url_to_docid_file.exists()
        else {}
    )

    log.info(f"Loaded {len(url_to_chunks)} URL mappings")
    log.info(f"Loaded {len(anchor_to_heading)} anchor mappings")

    return url_to_chunks, anchor_to_heading, url_to_docid


def load_chunks_from_db(
    db_path: Path, chunk_ids: list[str]
) -> dict[str, str]:
    """Load chunk contents from database.

    Returns:
        {chunk_id: content}
    """
    if not chunk_ids:
        return {}

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    try:
        placeholders = ",".join("?" * len(chunk_ids))
        query = f"SELECT id, content FROM chunks WHERE id IN ({placeholders})"
        cursor = conn.execute(query, chunk_ids)

        chunks = {}
        for row in cursor.fetchall():
            chunks[row["id"]] = row["content"]

        return chunks
    finally:
        conn.close()


def compute_corpus_version_hash(chunk_ids: list[str], chunks: dict[str, str]) -> str:
    """Compute SHA256 hash of corpus state.

    Hash is based on sorted chunk_id:content_hash pairs.
    """
    hash_parts = []
    for chunk_id in sorted(chunk_ids):
        content = chunks.get(chunk_id, "")
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        hash_parts.append(f"{chunk_id}:{content_hash}")

    hash_input = "".join(hash_parts)
    return hashlib.sha256(hash_input.encode()).hexdigest()


def extract_url_path(url: str) -> str:
    """Extract normalized path from URL."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") if parsed.path != "/" else parsed.path
    return path


def extract_signals(
    so_post: dict,
    url_to_chunks: dict,
    anchor_to_heading: dict,
    url_to_docid: dict,
    chunks_db: Path,
    corpus_version_hash: str,
) -> SignalBundle | None:
    """Extract signals from a single SO post.

    Returns:
        SignalBundle if successful, None if unmappable
    """
    question_id = so_post["question_id"]
    answer_id = so_post["answer_id"]
    doc_url = so_post["doc_url"]
    answer_body = so_post["answer_body"]
    question_body = so_post["question_body"]
    question_title = so_post["question_title"]
    answer_score = so_post["answer_score"]
    is_accepted = so_post["is_accepted"]
    answer_date = so_post["answer_date"]

    # Extract URL path and fragment
    parsed = urlparse(doc_url)
    url_path = extract_url_path(doc_url)
    url_fragment = parsed.fragment if parsed.fragment else None

    # Map URL to chunks
    if url_path not in url_to_chunks:
        return None

    chunk_ids = url_to_chunks[url_path]
    if not chunk_ids:
        return None

    # Load chunk contents
    chunks = load_chunks_from_db(chunks_db, chunk_ids)
    chunk_contents = [chunks.get(cid, "") for cid in chunk_ids]

    # Find quote matches
    quote_matches: list[QuoteMatch] = []
    for chunk_id in chunk_ids:
        chunk_content = chunks.get(chunk_id, "")
        matches = find_quote_matches(answer_body, chunk_content, chunk_id)
        for m in matches:
            quote_matches.append(
                QuoteMatch(
                    matched_text=m.matched_text,
                    match_length=m.match_length,
                    source_type=m.source_type,
                    chunk_id=m.chunk_id,
                    chunk_offset=m.chunk_offset,
                    answer_offset=m.answer_offset,
                )
            )

    # Find reciprocal matches
    reciprocal_matches: list[ReciprocalMatch] = []
    for chunk_id in chunk_ids:
        chunk_content = chunks.get(chunk_id, "")
        matches = find_reciprocal_matches(answer_body, chunk_content, chunk_id)
        for m in matches:
            reciprocal_matches.append(
                ReciprocalMatch(
                    matched_words=m.matched_words,
                    word_count=m.word_count,
                    chunk_id=m.chunk_id,
                    direction=m.direction,
                )
            )

    # Check fragment anchor match
    fragment_matches_heading = False
    if url_fragment:
        normalized_fragment = normalize_anchor(url_fragment)
        if normalized_fragment in anchor_to_heading:
            fragment_matches_heading = True

    # Assign tier
    tier_input = TierAssignmentInput(
        so_answer_id=answer_id,
        url_match=True,
        fragment_anchor=url_fragment,
        fragment_matches_heading=fragment_matches_heading,
        quote_matches=quote_matches,
        reciprocal_matches=reciprocal_matches,
        upvotes=answer_score,
        is_accepted=is_accepted,
        multiple_answers_same_url=len(url_to_docid.get(url_path, [])),
    )

    tier_assignment = assign_tier(tier_input)

    # Create signal bundle
    now = datetime.now(timezone.utc).isoformat()

    bundle = SignalBundle(
        bundle_id=str(uuid.uuid4()),
        so_question_id=question_id,
        so_answer_id=answer_id,
        question_title=question_title,
        question_body=question_body,
        answer_body=answer_body,
        extracted_url=doc_url,
        url_fragment=url_fragment,
        answer_upvotes=answer_score,
        is_accepted=is_accepted,
        answer_date=answer_date,
        chunk_ids=chunk_ids,
        chunk_contents=chunk_contents,
        quote_matches=[asdict(m) for m in quote_matches],
        reciprocal_matches=[asdict(m) for m in reciprocal_matches],
        fragment_matches_heading=fragment_matches_heading,
        max_possible_tier=tier_assignment.tier,
        extraction_timestamp=now,
        corpus_version_hash=corpus_version_hash,
    )

    return bundle


def load_so_posts(so_data_path: Path) -> Iterator[dict]:
    """Load SO posts from JSON file."""
    data = json.loads(so_data_path.read_text())
    for post in data:
        yield post


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract signal bundles from StackOverflow posts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m plm.benchmark.extraction.signals \\
    --so-data artifacts/benchmark/raw/so_k8s_answers.json \\
    --mappings artifacts/benchmark/mappings/ \\
    --corpus-db /path/to/index.db \\
    --output artifacts/benchmark/signals/ \\
    --workers 8
""",
    )
    parser.add_argument(
        "--so-data",
        required=True,
        type=Path,
        help="Path to SO posts JSON file",
    )
    parser.add_argument(
        "--mappings",
        required=True,
        type=Path,
        help="Directory containing URL mapping JSON files",
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
        help="Output directory for signal bundles",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args(argv)

    # Validate inputs
    if not args.so_data.exists():
        log.error(f"SO data file not found: {args.so_data}")
        return 1

    if not args.mappings.exists():
        log.error(f"Mappings directory not found: {args.mappings}")
        return 1

    if not args.corpus_db.exists():
        log.error(f"Corpus database not found: {args.corpus_db}")
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Signal Extraction Pipeline")
    log.info("=" * 60)
    log.info(f"SO Data: {args.so_data}")
    log.info(f"Mappings: {args.mappings}")
    log.info(f"Corpus DB: {args.corpus_db}")
    log.info(f"Output: {args.output}")
    log.info(f"Workers: {args.workers}")

    log.info("-" * 60)
    log.info("Loading mappings...")

    try:
        url_to_chunks, anchor_to_heading, url_to_docid = load_mappings(args.mappings)
    except FileNotFoundError as e:
        log.error(f"Failed to load mappings: {e}")
        return 1

    log.info("-" * 60)
    log.info("Computing corpus version hash...")

    # Load all chunks to compute hash
    all_chunk_ids = list(set(cid for chunk_list in url_to_chunks.values() for cid in chunk_list))
    chunks_for_hash = load_chunks_from_db(args.corpus_db, all_chunk_ids)
    corpus_version_hash = compute_corpus_version_hash(all_chunk_ids, chunks_for_hash)
    log.info(f"Corpus version hash: {corpus_version_hash}")

    log.info("-" * 60)
    log.info("Loading SO posts...")

    so_posts = list(load_so_posts(args.so_data))
    log.info(f"Loaded {len(so_posts)} SO posts")

    log.info("-" * 60)
    log.info("Extracting signals (parallel processing)...")

    start_time = datetime.now(timezone.utc)

    bundles: list[SignalBundle] = []
    unmappable: list[dict] = []

    def extract_wrapper(post: dict) -> tuple[SignalBundle | None, dict | None]:
        """Wrapper for parallel execution."""
        try:
            bundle = extract_signals(
                post,
                url_to_chunks,
                anchor_to_heading,
                url_to_docid,
                args.corpus_db,
                corpus_version_hash,
            )
            if bundle is None:
                return None, {
                    "so_answer_id": post["answer_id"],
                    "so_question_id": post["question_id"],
                    "reason": "no_chunks",
                    "doc_url": post["doc_url"],
                }
            return bundle, None
        except Exception as e:
            log.error(f"Error processing answer {post['answer_id']}: {e}")
            return None, {
                "so_answer_id": post["answer_id"],
                "so_question_id": post["question_id"],
                "reason": "processing_error",
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_wrapper, post): post for post in so_posts}

        for i, future in enumerate(as_completed(futures)):
            if (i + 1) % 100 == 0:
                log.info(f"Processed {i + 1}/{len(so_posts)} posts...")

            bundle, unmappable_entry = future.result()
            if bundle:
                bundles.append(bundle)
            elif unmappable_entry:
                unmappable.append(unmappable_entry)

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    log.info("-" * 60)
    log.info("EXTRACTION STATISTICS")
    log.info("-" * 60)
    log.info(f"Total posts processed:  {len(so_posts)}")
    log.info(f"Bundles created:        {len(bundles)}")
    log.info(f"Unmappable posts:       {len(unmappable)}")

    # Count by tier
    tier_counts = {
        "gold": sum(1 for b in bundles if b.max_possible_tier == "gold"),
        "silver": sum(1 for b in bundles if b.max_possible_tier == "silver"),
        "bronze": sum(1 for b in bundles if b.max_possible_tier == "bronze"),
        "exclude": sum(1 for b in bundles if b.max_possible_tier == "exclude"),
    }

    log.info(f"Gold potential:         {tier_counts['gold']}")
    log.info(f"Silver potential:       {tier_counts['silver']}")
    log.info(f"Bronze potential:       {tier_counts['bronze']}")
    log.info(f"Excluded:               {tier_counts['exclude']}")
    log.info(f"Duration:               {duration:.2f}s")

    log.info("-" * 60)
    log.info("Writing output files...")

    # Write signal bundles JSONL
    bundles_file = args.output / "signal_bundles.jsonl"
    with bundles_file.open("w") as f:
        for bundle in bundles:
            f.write(json.dumps(bundle.to_dict(), ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(bundles)} bundles to {bundles_file}")

    # Write unmappable log
    if unmappable:
        unmappable_file = args.output / "unmappable.log"
        with unmappable_file.open("w") as f:
            for entry in unmappable:
                f.write(
                    f"answer_id={entry['so_answer_id']} "
                    f"question_id={entry['so_question_id']} "
                    f"reason={entry['reason']}"
                )
                if "error" in entry:
                    f.write(f" error={entry['error']}")
                f.write("\n")
        log.info(f"Wrote {len(unmappable)} unmappable entries to {unmappable_file}")

    # Write extraction stats
    stats = {
        "total_posts_processed": len(so_posts),
        "bundles_created": len(bundles),
        "unmappable_count": len(unmappable),
        "gold_potential": tier_counts["gold"],
        "silver_potential": tier_counts["silver"],
        "bronze_potential": tier_counts["bronze"],
        "excluded_count": tier_counts["exclude"],
        "corpus_version_hash": corpus_version_hash,
        "extraction_duration_seconds": duration,
    }

    stats_file = args.output / "extraction_stats.json"
    stats_file.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    log.info(f"Wrote stats to {stats_file}")

    log.info("=" * 60)
    log.info("EXTRACTION COMPLETE")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
