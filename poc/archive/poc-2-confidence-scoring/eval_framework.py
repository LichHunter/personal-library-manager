#!/usr/bin/env python3
"""
Evaluation framework for NER model comparison on technical entity extraction.

Tests NER models on SO NER dataset + synthetic OOD dataset.
Measures: accuracy (P/R/F1), confidence calibration, and extraction speed.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "poc-1c-scalable-ner"))
from scoring import many_to_many_score


@dataclass
class ExtractedEntity:
    text: str
    confidence: float
    label: str = ""


@dataclass
class DocResult:
    doc_id: str
    extracted: list[ExtractedEntity]
    gt_terms: list[str]
    precision: float
    recall: float
    f1: float
    hallucination: float
    avg_confidence: float
    extraction_time_ms: float
    tp_confidences: list[float] = field(default_factory=list)
    fp_confidences: list[float] = field(default_factory=list)


@dataclass
class ModelReport:
    model_name: str
    dataset_name: str
    total_docs: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_hallucination: float
    avg_confidence: float
    avg_time_ms: float
    tp_mean_confidence: float
    fp_mean_confidence: float
    confidence_separation: float
    results: list[DocResult]


class NERModel(Protocol):
    @property
    def name(self) -> str: ...
    def extract(self, text: str) -> list[ExtractedEntity]: ...


def evaluate_document(model: NERModel, doc: dict) -> DocResult:
    text = doc["text"]
    gt_terms = doc["gt_terms"]

    start = time.perf_counter()
    entities = model.extract(text)
    elapsed_ms = (time.perf_counter() - start) * 1000

    extracted_terms = [e.text for e in entities]
    scores = many_to_many_score(extracted_terms, gt_terms)

    fp_set = set(scores.get("fp_terms", []))
    tp_confidences = []
    fp_confidences = []
    for e in entities:
        if e.text in fp_set:
            fp_confidences.append(e.confidence)
        else:
            tp_confidences.append(e.confidence)

    avg_conf = sum(e.confidence for e in entities) / len(entities) if entities else 0.0

    return DocResult(
        doc_id=doc.get("doc_id", "unknown"),
        extracted=entities,
        gt_terms=gt_terms,
        precision=scores["precision"],
        recall=scores["recall"],
        f1=scores["f1"],
        hallucination=scores["hallucination"],
        avg_confidence=avg_conf,
        extraction_time_ms=elapsed_ms,
        tp_confidences=tp_confidences,
        fp_confidences=fp_confidences,
    )


def evaluate_model(model: NERModel, documents: list[dict], dataset_name: str = "unknown") -> ModelReport:
    results = []
    for i, doc in enumerate(documents, 1):
        if i % 10 == 0:
            print(f"    [{i}/{len(documents)}] {doc.get('doc_id', '?')}")
        result = evaluate_document(model, doc)
        results.append(result)

    n = len(results)
    if n == 0:
        raise ValueError("No documents to evaluate")

    all_tp_conf = [c for r in results for c in r.tp_confidences]
    all_fp_conf = [c for r in results for c in r.fp_confidences]
    tp_mean = sum(all_tp_conf) / len(all_tp_conf) if all_tp_conf else 0.0
    fp_mean = sum(all_fp_conf) / len(all_fp_conf) if all_fp_conf else 0.0

    return ModelReport(
        model_name=model.name,
        dataset_name=dataset_name,
        total_docs=n,
        avg_precision=sum(r.precision for r in results) / n,
        avg_recall=sum(r.recall for r in results) / n,
        avg_f1=sum(r.f1 for r in results) / n,
        avg_hallucination=sum(r.hallucination for r in results) / n,
        avg_confidence=sum(r.avg_confidence for r in results) / n,
        avg_time_ms=sum(r.extraction_time_ms for r in results) / n,
        tp_mean_confidence=tp_mean,
        fp_mean_confidence=fp_mean,
        confidence_separation=tp_mean - fp_mean,
        results=results,
    )


def load_so_ner_test(n_docs: int = 100, seed: int = 42) -> list[dict]:
    import random
    test_path = (
        Path(__file__).parent.parent
        / "poc-1c-scalable-ner"
        / "artifacts"
        / "test_documents.json"
    )
    with open(test_path) as f:
        all_docs = json.load(f)
    random.seed(seed)
    return random.sample(all_docs, min(n_docs, len(all_docs)))


def load_ood_dataset() -> list[dict]:
    ood_path = Path(__file__).parent / "artifacts" / "ood_dataset.json"
    if not ood_path.exists():
        raise FileNotFoundError(
            f"OOD dataset not found at {ood_path}. Run generate_ood_dataset.py first."
        )
    with open(ood_path) as f:
        return json.load(f)


def report_to_dict(report: ModelReport) -> dict:
    return {
        "model_name": report.model_name,
        "dataset_name": report.dataset_name,
        "total_docs": report.total_docs,
        "avg_precision": report.avg_precision,
        "avg_recall": report.avg_recall,
        "avg_f1": report.avg_f1,
        "avg_hallucination": report.avg_hallucination,
        "avg_confidence": report.avg_confidence,
        "avg_time_ms": report.avg_time_ms,
        "tp_mean_confidence": report.tp_mean_confidence,
        "fp_mean_confidence": report.fp_mean_confidence,
        "confidence_separation": report.confidence_separation,
        "per_doc": [
            {
                "doc_id": r.doc_id,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "hallucination": r.hallucination,
                "avg_confidence": r.avg_confidence,
                "extraction_time_ms": r.extraction_time_ms,
                "n_extracted": len(r.extracted),
                "n_gt": len(r.gt_terms),
            }
            for r in report.results
        ],
    }


def print_report(report: ModelReport) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {report.model_name} | {report.dataset_name} ({report.total_docs} docs)")
    print(f"{'=' * 60}")
    print(f"  Precision:       {report.avg_precision:.3f}")
    print(f"  Recall:          {report.avg_recall:.3f}")
    print(f"  F1:              {report.avg_f1:.3f}")
    print(f"  Hallucination:   {report.avg_hallucination:.3f}")
    print(f"  Avg Confidence:  {report.avg_confidence:.3f}")
    print(f"  Avg Time (ms):   {report.avg_time_ms:.1f}")
    print(f"  ---")
    print(f"  TP Mean Conf:    {report.tp_mean_confidence:.3f}")
    print(f"  FP Mean Conf:    {report.fp_mean_confidence:.3f}")
    print(f"  Conf Separation: {report.confidence_separation:+.3f}")
    print(f"{'=' * 60}")
