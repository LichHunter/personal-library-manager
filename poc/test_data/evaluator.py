import logging
from typing import List, Set

from .models import (
    EvaluationReport,
    GroundTruth,
    Retriever,
    RetrievalResult,
    SearchResult,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def word_overlap(text1: str, text2: str) -> float:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    return jaccard_similarity(words1, words2)


class RetrievalEvaluator:
    def __init__(self, ground_truths: List[GroundTruth]):
        self.ground_truths = ground_truths

    def _compute_evidence_overlap(
        self,
        expected_evidence: List[str],
        retrieved_content: List[str],
    ) -> float:
        if not expected_evidence or not retrieved_content:
            return 0.0

        combined_retrieved = " ".join(retrieved_content)
        overlaps = []
        for evidence in expected_evidence:
            overlap = word_overlap(evidence, combined_retrieved)
            overlaps.append(overlap)

        return sum(overlaps) / len(overlaps) if overlaps else 0.0

    def _evaluate_single(
        self,
        gt: GroundTruth,
        results: List[SearchResult],
    ) -> RetrievalResult:
        retrieved_docs = [r.document_id for r in results]
        retrieved_sections = [r.section_id for r in results]
        retrieved_content = [r.content for r in results]

        doc_found = gt.document_id in retrieved_docs
        doc_rank = -1
        if doc_found:
            doc_rank = retrieved_docs.index(gt.document_id) + 1

        section_found = any(sid in retrieved_sections for sid in gt.section_ids)

        evidence_overlap = self._compute_evidence_overlap(gt.evidence, retrieved_content)

        return RetrievalResult(
            test_id=gt.id,
            question=gt.question,
            expected_doc=gt.document_id,
            retrieved_docs=retrieved_docs,
            doc_found=doc_found,
            doc_rank=doc_rank,
            expected_sections=gt.section_ids,
            retrieved_sections=retrieved_sections,
            section_found=section_found,
            evidence_overlap=evidence_overlap,
        )

    def evaluate(self, retriever: Retriever, top_k: int = 5) -> EvaluationReport:
        logging.info(f"Evaluating {len(self.ground_truths)} test cases")

        results: List[RetrievalResult] = []
        for gt in self.ground_truths:
            search_results = retriever.search(gt.question, top_k=top_k)
            result = self._evaluate_single(gt, search_results)
            results.append(result)

        total = len(results)
        if total == 0:
            return EvaluationReport(
                total_tests=0,
                doc_recall_at_1=0.0,
                doc_recall_at_3=0.0,
                doc_recall_at_5=0.0,
                section_recall_at_1=0.0,
                section_recall_at_3=0.0,
                section_recall_at_5=0.0,
                mean_evidence_overlap=0.0,
                failures=[],
            )

        doc_recall_1 = sum(1 for r in results if r.doc_rank == 1) / total
        doc_recall_3 = sum(1 for r in results if 0 < r.doc_rank <= 3) / total
        doc_recall_5 = sum(1 for r in results if 0 < r.doc_rank <= 5) / total

        def section_in_top_k(result: RetrievalResult, k: int) -> bool:
            top_sections = result.retrieved_sections[:k]
            return any(sid in top_sections for sid in result.expected_sections)

        section_recall_1 = sum(1 for r in results if section_in_top_k(r, 1)) / total
        section_recall_3 = sum(1 for r in results if section_in_top_k(r, 3)) / total
        section_recall_5 = sum(1 for r in results if section_in_top_k(r, 5)) / total

        mean_overlap = sum(r.evidence_overlap for r in results) / total

        failures = [r for r in results if not r.doc_found or not r.section_found]

        report = EvaluationReport(
            total_tests=total,
            doc_recall_at_1=doc_recall_1,
            doc_recall_at_3=doc_recall_3,
            doc_recall_at_5=doc_recall_5,
            section_recall_at_1=section_recall_1,
            section_recall_at_3=section_recall_3,
            section_recall_at_5=section_recall_5,
            mean_evidence_overlap=mean_overlap,
            failures=failures,
        )

        logging.info(f"Evaluation complete: {total} tests, {len(failures)} failures")
        return report

    def print_report(self, report: EvaluationReport) -> None:
        print("\n" + "=" * 60)
        print("RETRIEVAL EVALUATION REPORT")
        print("=" * 60)
        print(f"\nTotal Tests: {report.total_tests}")
        print("\nDocument Recall:")
        print(f"  @1: {report.doc_recall_at_1:.1%}")
        print(f"  @3: {report.doc_recall_at_3:.1%}")
        print(f"  @5: {report.doc_recall_at_5:.1%}")
        print("\nSection Recall:")
        print(f"  @1: {report.section_recall_at_1:.1%}")
        print(f"  @3: {report.section_recall_at_3:.1%}")
        print(f"  @5: {report.section_recall_at_5:.1%}")
        print(f"\nMean Evidence Overlap: {report.mean_evidence_overlap:.1%}")

        if report.failures:
            print(f"\nFailures ({len(report.failures)}):")
            for f in report.failures[:5]:
                print(f"  - [{f.test_id}] {f.question[:50]}...")
                print(f"    Expected: {f.expected_doc}, Got: {f.retrieved_docs[:3]}")
            if len(report.failures) > 5:
                print(f"  ... and {len(report.failures) - 5} more")

        print("=" * 60)
