from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from plm.shared.logger import get_logger


@dataclass
class BenchmarkQuestion:
    id: str
    question: str
    target_doc_id: str
    dataset: str


def load_questions(corpus_dir: Path, datasets: list[str] | None = None) -> list[BenchmarkQuestion]:
    log = get_logger()
    
    if datasets is None:
        datasets = ["needle", "realistic", "informed"]
    
    questions = []
    
    for dataset in datasets:
        file_path = corpus_dir / f"{dataset}_benchmark.json"
        if not file_path.exists():
            log.warn(f"Dataset file not found: {file_path}")
            continue
        
        log.info(f"Loading {dataset} dataset from {file_path}")
        
        with open(file_path) as f:
            data = json.load(f)
        
        for item in data.get("per_question", []):
            questions.append(BenchmarkQuestion(
                id=item["id"],
                question=item["question"],
                target_doc_id=item["target_doc_id"],
                dataset=dataset,
            ))
        
        log.info(f"Loaded {len(data.get('per_question', []))} questions from {dataset}")
    
    log.info(f"Total questions loaded: {len(questions)}")
    return questions
