from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

K8S_DOCS_URL_PATTERN = "%kubernetes.io/docs%"
K8S_TAG_PATTERNS = ["%kubernetes%", "%k8s%"]
MIN_ANSWER_SCORE = 5


@dataclass
class SOAnswer:
    question_id: int
    question_title: str
    question_body: str
    question_tags: list[str]
    answer_id: int
    answer_body: str
    answer_score: int
    is_accepted: bool
    doc_url: str
    answer_date: str

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question_title": self.question_title[:150],
            "question_body": self.question_body,
            "question_tags": self.question_tags,
            "answer_id": self.answer_id,
            "answer_body": self.answer_body,
            "answer_score": self.answer_score,
            "is_accepted": self.is_accepted,
            "doc_url": self.doc_url,
            "answer_date": self.answer_date,
        }


EXTRACT_K8S_ANSWERS_SQL = """
SELECT 
    q.Id as question_id,
    q.Title as question_title,
    q.Body as question_body,
    q.Tags as question_tags,
    a.Id as answer_id,
    a.Body as answer_body,
    a.Score as answer_score,
    CASE WHEN q.AcceptedAnswerId = a.Id THEN 1 ELSE 0 END as is_accepted,
    urls.Url as doc_url,
    a.CreationDate as answer_date
FROM Posts q
JOIN Posts a ON a.ParentId = q.Id AND a.PostTypeId = 2
JOIN PostVersionUrl urls ON urls.PostId = a.Id
WHERE (q.Tags LIKE ? OR q.Tags LIKE ?)
  AND urls.Url LIKE ?
  AND a.Score >= ?
ORDER BY a.Score DESC
"""


def parse_tags(tags_str: str | None) -> list[str]:
    if not tags_str:
        return []
    tags_str = tags_str.strip()
    if tags_str.startswith("<") and tags_str.endswith(">"):
        return [t for t in tags_str[1:-1].split("><") if t]
    return [t.strip() for t in tags_str.split(",") if t.strip()]


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    
    if parsed.scheme == "http":
        url = url.replace("http://", "https://", 1)
    
    path = parsed.path.rstrip("/") if parsed.path != "/" else parsed.path
    
    normalized = f"https://{parsed.netloc}{path}"
    if parsed.fragment:
        normalized += f"#{parsed.fragment}"
    
    return normalized


def extract_k8s_answers(db_path: Path, min_score: int = MIN_ANSWER_SCORE) -> Iterator[SOAnswer]:
    log.info(f"Opening database: {db_path}")
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.execute(
            EXTRACT_K8S_ANSWERS_SQL,
            (K8S_TAG_PATTERNS[0], K8S_TAG_PATTERNS[1], K8S_DOCS_URL_PATTERN, min_score),
        )
        
        batch_size = 1000
        row_count = 0
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                row_count += 1
                if row_count % 5000 == 0:
                    log.info(f"Processed {row_count} rows...")
                
                yield SOAnswer(
                    question_id=row["question_id"],
                    question_title=row["question_title"] or "",
                    question_body=row["question_body"] or "",
                    question_tags=parse_tags(row["question_tags"]),
                    answer_id=row["answer_id"],
                    answer_body=row["answer_body"] or "",
                    answer_score=row["answer_score"],
                    is_accepted=bool(row["is_accepted"]),
                    doc_url=normalize_url(row["doc_url"]),
                    answer_date=row["answer_date"] or "",
                )
        
        log.info(f"Total rows extracted: {row_count}")
    finally:
        conn.close()


def compute_statistics(answers: list[SOAnswer]) -> dict:
    unique_questions = len({a.question_id for a in answers})
    unique_answers = len({a.answer_id for a in answers})
    unique_urls = len({a.doc_url for a in answers})
    accepted_count = sum(1 for a in answers if a.is_accepted)
    
    score_dist = Counter(a.answer_score for a in answers)
    score_buckets = {
        "5-9": sum(v for k, v in score_dist.items() if 5 <= k <= 9),
        "10-24": sum(v for k, v in score_dist.items() if 10 <= k <= 24),
        "25-49": sum(v for k, v in score_dist.items() if 25 <= k <= 49),
        "50-99": sum(v for k, v in score_dist.items() if 50 <= k <= 99),
        "100+": sum(v for k, v in score_dist.items() if k >= 100),
    }
    
    url_paths = Counter()
    for a in answers:
        parsed = urlparse(a.doc_url)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 2:
            url_paths[f"/docs/{parts[1]}/"] += 1
    
    return {
        "total_entries": len(answers),
        "unique_questions": unique_questions,
        "unique_answers": unique_answers,
        "unique_doc_urls": unique_urls,
        "accepted_answers": accepted_count,
        "score_distribution": score_buckets,
        "top_doc_sections": dict(url_paths.most_common(10)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract Kubernetes Q&A pairs with doc links from SOTorrent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset Setup:
  1. Download from https://zenodo.org/record/4415593
  2. Required files: Posts.sql.7z, PostVersionUrl.sql.7z
  3. Extract and load into SQLite (or use MySQL then export)
  4. Run this script with --db pointing to the SQLite file

Example:
  python -m plm.benchmark.extraction.so_extractor \\
    --db /path/to/sotorrent.db \\
    --output artifacts/benchmark/raw/so_k8s_answers.json
""",
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Path to SOTorrent SQLite database file",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=MIN_ANSWER_SCORE,
        help=f"Minimum answer score (default: {MIN_ANSWER_SCORE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of results (for testing)",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only compute and print statistics, don't write output",
    )
    
    args = parser.parse_args(argv)
    
    if not args.db.exists():
        log.error(f"Database file not found: {args.db}")
        return 1
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("SOTorrent K8s Answer Extraction")
    log.info("=" * 60)
    log.info(f"Database: {args.db}")
    log.info(f"Output: {args.output}")
    log.info(f"Min score: {args.min_score}")
    if args.limit:
        log.info(f"Limit: {args.limit}")
    
    log.info("-" * 60)
    log.info("Extracting answers...")
    
    answers: list[SOAnswer] = []
    try:
        for answer in extract_k8s_answers(args.db, args.min_score):
            answers.append(answer)
            if args.limit and len(answers) >= args.limit:
                log.info(f"Reached limit of {args.limit} entries")
                break
    except sqlite3.OperationalError as e:
        log.error(f"Database error: {e}")
        log.error("Make sure the database has Posts and PostVersionUrl tables")
        return 1
    
    if not answers:
        log.warning("No matching answers found!")
        log.warning("Check that the database contains K8s questions with doc links")
        return 1
    
    log.info("-" * 60)
    log.info("Computing statistics...")
    stats = compute_statistics(answers)
    
    log.info("-" * 60)
    log.info("STATISTICS")
    log.info("-" * 60)
    log.info(f"Total entries:      {stats['total_entries']}")
    log.info(f"Unique questions:   {stats['unique_questions']}")
    log.info(f"Unique answers:     {stats['unique_answers']}")
    log.info(f"Unique doc URLs:    {stats['unique_doc_urls']}")
    log.info(f"Accepted answers:   {stats['accepted_answers']}")
    log.info("")
    log.info("Score distribution:")
    for bucket, count in stats["score_distribution"].items():
        log.info(f"  {bucket:>8}: {count}")
    log.info("")
    log.info("Top documentation sections:")
    for section, count in stats["top_doc_sections"].items():
        log.info(f"  {section:40}: {count}")
    
    if args.stats_only:
        log.info("-" * 60)
        log.info("Stats-only mode, skipping output file")
        return 0
    
    log.info("-" * 60)
    log.info(f"Writing {len(answers)} entries to {args.output}...")
    
    output_data = [a.to_dict() for a in answers]
    args.output.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    
    log.info("=" * 60)
    log.info("EXTRACTION COMPLETE")
    log.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
