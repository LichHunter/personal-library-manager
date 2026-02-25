#!/usr/bin/env python3
"""Fetch Kubernetes StackOverflow Q&A with doc links via API (two-phase fetch)."""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

K8S_DOCS_PATTERN = re.compile(r'href=["\']?(https?://kubernetes\.io/docs[^"\'>\s]*)["\']?', re.IGNORECASE)
MIN_ANSWER_SCORE = 5
API_BASE = "https://api.stackexchange.com/2.3"


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


def extract_k8s_doc_urls(html_body: str) -> list[str]:
    urls = K8S_DOCS_PATTERN.findall(html_body)
    normalized = []
    seen = set()
    for url in urls:
        url = html.unescape(url).rstrip('.,;:')
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") if parsed.path != "/" else parsed.path
        normalized_url = f"https://kubernetes.io{path}"
        if parsed.fragment:
            normalized_url += f"#{parsed.fragment}"
        if normalized_url not in seen:
            seen.add(normalized_url)
            normalized.append(normalized_url)
    return normalized


def fetch_questions(client: httpx.Client, page: int, pagesize: int = 100) -> tuple[list[dict], bool, int]:
    params = {
        "order": "desc",
        "sort": "votes",
        "tagged": "kubernetes",
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": pagesize,
        "page": page,
    }
    
    response = client.get(f"{API_BASE}/questions", params=params)
    response.raise_for_status()
    data = response.json()
    
    quota = data.get("quota_remaining", 0)
    has_more = data.get("has_more", False)
    
    return data.get("items", []), has_more, quota


def fetch_answers_batch(client: httpx.Client, question_ids: list[int]) -> tuple[list[dict], int]:
    if not question_ids:
        return [], 300
    
    ids_str = ";".join(str(qid) for qid in question_ids)
    params = {
        "order": "desc",
        "sort": "votes",
        "site": "stackoverflow",
        "filter": "withbody",
        "pagesize": 100,
    }
    
    response = client.get(f"{API_BASE}/questions/{ids_str}/answers", params=params)
    response.raise_for_status()
    data = response.json()
    
    return data.get("items", []), data.get("quota_remaining", 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Kubernetes Q&A with doc links from StackOverflow API.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--min-score", type=int, default=MIN_ANSWER_SCORE)
    parser.add_argument("--question-pages", type=int, default=10)
    
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("StackOverflow K8s Q&A Extraction (API - Two Phase)")
    log.info("=" * 60)
    
    all_results: list[SOAnswer] = []
    questions_with_links = 0
    
    with httpx.Client(timeout=30.0) as client:
        page = 1
        has_more = True
        
        while has_more and page <= args.question_pages and len(all_results) < args.limit:
            log.info(f"Fetching questions page {page}...")
            questions, has_more, quota = fetch_questions(client, page, pagesize=100)
            log.info(f"  Got {len(questions)} questions, quota={quota}")
            
            if quota < 20:
                log.warning("Low API quota, stopping")
                break
            
            question_ids = [q["question_id"] for q in questions]
            question_map = {q["question_id"]: q for q in questions}
            
            batch_size = 30
            for i in range(0, len(question_ids), batch_size):
                batch_ids = question_ids[i:i+batch_size]
                
                log.info(f"  Fetching answers for questions {i+1}-{i+len(batch_ids)}...")
                answers, quota = fetch_answers_batch(client, batch_ids)
                log.info(f"    Got {len(answers)} answers, quota={quota}")
                
                if quota < 20:
                    log.warning("Low API quota, stopping")
                    break
                
                for a in answers:
                    if a.get("score", 0) < args.min_score:
                        continue
                    
                    doc_urls = extract_k8s_doc_urls(a.get("body", ""))
                    if not doc_urls:
                        continue
                    
                    qid = a.get("question_id")
                    if qid is None:
                        continue
                    q = question_map.get(qid, {})
                    
                    for doc_url in doc_urls:
                        creation_date = a.get("creation_date", 0)
                        answer_date = datetime.utcfromtimestamp(creation_date).isoformat() + "Z" if creation_date else ""
                        
                        all_results.append(SOAnswer(
                            question_id=qid,
                            question_title=q.get("title", ""),
                            question_body=q.get("body", ""),
                            question_tags=q.get("tags", []),
                            answer_id=a["answer_id"],
                            answer_body=a.get("body", ""),
                            answer_score=a.get("score", 0),
                            is_accepted=(a["answer_id"] == q.get("accepted_answer_id")),
                            doc_url=doc_url,
                            answer_date=answer_date,
                        ))
                        questions_with_links += 1
                
                time.sleep(0.3)
            
            log.info(f"  Total entries so far: {len(all_results)}")
            page += 1
            time.sleep(0.5)
    
    if not all_results:
        log.warning("No matching answers found!")
        return 1
    
    unique_questions = len({r.question_id for r in all_results})
    unique_answers = len({r.answer_id for r in all_results})
    unique_urls = len({r.doc_url for r in all_results})
    
    log.info("-" * 60)
    log.info(f"Total entries:      {len(all_results)}")
    log.info(f"Unique questions:   {unique_questions}")
    log.info(f"Unique answers:     {unique_answers}")
    log.info(f"Unique doc URLs:    {unique_urls}")
    
    output_data = [r.to_dict() for r in all_results[:args.limit]]
    args.output.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    log.info(f"Wrote {len(output_data)} entries to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
