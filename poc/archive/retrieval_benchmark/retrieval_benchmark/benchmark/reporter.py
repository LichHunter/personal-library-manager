"""Report generation for benchmark results."""

import csv
import logging
from dataclasses import asdict, fields
from pathlib import Path
from typing import TextIO

from ..core.types import IndexStats, QueryResult, StrategySummary

logger = logging.getLogger(__name__)

MAX_CONTENT_LENGTH = 200


def _truncate(text: str, max_len: int = MAX_CONTENT_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _escape_md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


class Reporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._index_stats_file: TextIO | None = None
        self._search_results_file: TextIO | None = None
        self._index_stats_writer: csv.DictWriter | None = None
        self._search_results_writer: csv.DictWriter | None = None
        
        self._all_query_results: list[QueryResult] = []
        self._all_index_stats: list[IndexStats] = []
    
    def start(self) -> None:
        self._index_stats_file = open(
            self.output_dir / "index_stats.csv", "w", newline=""
        )
        self._index_stats_writer = csv.DictWriter(
            self._index_stats_file,
            fieldnames=[f.name for f in fields(IndexStats)],
        )
        self._index_stats_writer.writeheader()
        
        self._search_results_file = open(
            self.output_dir / "search_results.csv", "w", newline=""
        )
        
        search_fieldnames = [
            "query_id", "run_number", "strategy", "backend",
            "embedding_model", "llm_model", "question",
            "expected_doc_id", "expected_section_ids", "expected_answer",
            "top_k", "doc_found", "doc_rank",
            "section_found", "section_rank",
            "evidence_recall", "answer_token_overlap", "rouge_l",
            "search_time_ms", "embed_calls", "llm_calls",
            "retrieved_chunks",
        ]
        self._search_results_writer = csv.DictWriter(
            self._search_results_file,
            fieldnames=search_fieldnames,
        )
        self._search_results_writer.writeheader()
    
    def write_index_stats(self, stats: IndexStats) -> None:
        if self._index_stats_writer is None:
            raise RuntimeError("Reporter not started. Call start() first.")
        
        self._all_index_stats.append(stats)
        self._index_stats_writer.writerow(asdict(stats))
        if self._index_stats_file:
            self._index_stats_file.flush()
    
    def write_query_result(self, result: QueryResult) -> None:
        if self._search_results_writer is None:
            raise RuntimeError("Reporter not started. Call start() first.")
        
        self._all_query_results.append(result)
        
        cm = result.content_metrics
        row = {
            "query_id": result.query_id,
            "run_number": result.run_number,
            "strategy": result.strategy,
            "backend": result.backend,
            "embedding_model": result.embedding_model,
            "llm_model": result.llm_model,
            "question": result.question,
            "expected_doc_id": result.expected_doc_id,
            "expected_section_ids": ",".join(result.expected_section_ids),
            "expected_answer": result.expected_answer,
            "top_k": result.top_k,
            "doc_found": result.doc_found,
            "doc_rank": result.doc_rank,
            "section_found": result.section_found,
            "section_rank": result.section_rank,
            "evidence_recall": f"{cm.evidence_recall:.3f}" if cm else "",
            "answer_token_overlap": f"{cm.answer_token_overlap:.3f}" if cm else "",
            "rouge_l": f"{cm.rouge_l:.3f}" if cm else "",
            "search_time_ms": result.search_time_ms,
            "embed_calls": result.embed_calls,
            "llm_calls": result.llm_calls,
            "retrieved_chunks": self._serialize_chunks(result),
        }
        
        self._search_results_writer.writerow(row)
        if self._search_results_file:
            self._search_results_file.flush()
    
    def _serialize_chunks(self, result: QueryResult) -> str:
        parts = []
        for chunk in result.retrieved_chunks:
            marker = ""
            if chunk.is_correct_doc and chunk.is_correct_section:
                marker = "[CORRECT] "
            elif chunk.is_correct_doc:
                marker = "[DOC OK] "
            parts.append(
                f"#{chunk.rank} {marker}(score={chunk.score:.3f}, "
                f"doc={chunk.document_id}, sec={chunk.section_id}): "
                f"{_truncate(chunk.content, 100)}"
            )
        return " ||| ".join(parts)
    
    def write_summary(self, summaries: list[StrategySummary]) -> None:
        summary_path = self.output_dir / "summary.csv"
        
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[f.name for f in fields(StrategySummary)],
            )
            writer.writeheader()
            
            for summary in summaries:
                writer.writerow(asdict(summary))
        
        logger.info(f"Wrote summary to {summary_path}")
        
        self._write_markdown_report(summaries)
    
    def _write_markdown_report(self, summaries: list[StrategySummary]) -> None:
        report_path = self.output_dir / "report.md"
        
        with open(report_path, "w") as f:
            f.write("# Benchmark Report\n\n")
            
            f.write("## Summary\n\n")
            f.write("| Strategy | Backend | Model | Doc@5 | Sec@5 | MRR | Avg Time |\n")
            f.write("|----------|---------|-------|-------|-------|-----|----------|\n")
            
            for s in summaries:
                model = s.embedding_model.split("/")[-1]
                f.write(
                    f"| {s.strategy} | {s.backend} | {model} | "
                    f"{s.doc_recall_at_5:.1%} | {s.section_recall_at_5:.1%} | "
                    f"{s.doc_mrr:.3f} | {s.avg_search_time_ms:.1f}ms |\n"
                )
            
            f.write("\n## Index Statistics\n\n")
            f.write("| Strategy | Backend | Documents | Chunks | Vectors | Index Time |\n")
            f.write("|----------|---------|-----------|--------|---------|------------|\n")
            
            for stats in self._all_index_stats:
                f.write(
                    f"| {stats.strategy} | {stats.backend} | {stats.num_documents} | "
                    f"{stats.num_chunks} | {stats.num_vectors} | {stats.duration_sec:.2f}s |\n"
                )
            
            f.write("\n## Query Results\n\n")
            
            results_by_query = self._group_results_by_query()
            for query_id, strategy_results in results_by_query.items():
                self._write_query_comparison(f, query_id, strategy_results)
        
        logger.info(f"Wrote markdown report to {report_path}")
    
    def _group_results_by_query(self) -> dict[str, dict[str, QueryResult]]:
        """Group results by query_id, then by strategy. Only top_k=5 results."""
        results_by_query: dict[str, dict[str, QueryResult]] = {}
        
        for result in self._all_query_results:
            if result.top_k != 5:
                continue
            
            if result.query_id not in results_by_query:
                results_by_query[result.query_id] = {}
            
            strategy_key = f"{result.strategy}"
            if strategy_key not in results_by_query[result.query_id]:
                results_by_query[result.query_id][strategy_key] = result
        
        return results_by_query
    
    def _write_query_comparison(
        self, 
        f: TextIO, 
        query_id: str, 
        strategy_results: dict[str, QueryResult],
    ) -> None:
        """Write comparison of all strategies for a single query."""
        if not strategy_results:
            return
        
        first_result = next(iter(strategy_results.values()))
        strategies = list(strategy_results.keys())
        
        all_pass = all(
            r.doc_found and r.section_found 
            for r in strategy_results.values()
        )
        all_fail = all(
            not r.doc_found or not r.section_found 
            for r in strategy_results.values()
        )
        
        if all_pass:
            status = "ALL PASS"
        elif all_fail:
            status = "ALL FAIL"
        else:
            status = "MIXED"
        
        f.write(f"### Query: {query_id} [{status}]\n\n")
        
        f.write("#### Question\n\n")
        f.write(f"> {_escape_md(first_result.question)}\n\n")
        
        f.write("#### Expected\n\n")
        f.write(f"- **Document:** `{first_result.expected_doc_id}`\n")
        f.write(f"- **Sections:** `{', '.join(first_result.expected_section_ids)}`\n")
        f.write(f"- **Answer:** {_escape_md(first_result.expected_answer)}\n\n")
        
        f.write("#### Strategy Comparison\n\n")
        
        header = "| Metric |"
        separator = "|--------|"
        for strategy in strategies:
            header += f" {strategy} |"
            separator += "--------|"
        f.write(header + "\n")
        f.write(separator + "\n")
        
        def status_icon(r: QueryResult) -> str:
            if r.doc_found and r.section_found:
                return "PASS"
            elif r.doc_found:
                return "DOC ONLY"
            else:
                return "FAIL"
        
        row = "| Status |"
        for strategy in strategies:
            r = strategy_results[strategy]
            row += f" {status_icon(r)} |"
        f.write(row + "\n")
        
        row = "| Doc Rank |"
        for strategy in strategies:
            r = strategy_results[strategy]
            rank = r.doc_rank if r.doc_rank > 0 else "-"
            row += f" {rank} |"
        f.write(row + "\n")
        
        row = "| Section Rank |"
        for strategy in strategies:
            r = strategy_results[strategy]
            rank = r.section_rank if r.section_rank > 0 else "-"
            row += f" {rank} |"
        f.write(row + "\n")
        
        row = "| Search Time |"
        for strategy in strategies:
            r = strategy_results[strategy]
            row += f" {r.search_time_ms:.1f}ms |"
        f.write(row + "\n")
        
        has_content_metrics = any(
            r.content_metrics for r in strategy_results.values()
        )
        if has_content_metrics:
            row = "| Evidence Recall |"
            for strategy in strategies:
                r = strategy_results[strategy]
                if r.content_metrics:
                    row += f" {r.content_metrics.evidence_recall:.1%} |"
                else:
                    row += " - |"
            f.write(row + "\n")
            
            row = "| Answer Overlap |"
            for strategy in strategies:
                r = strategy_results[strategy]
                if r.content_metrics:
                    row += f" {r.content_metrics.answer_token_overlap:.1%} |"
                else:
                    row += " - |"
            f.write(row + "\n")
            
            row = "| ROUGE-L |"
            for strategy in strategies:
                r = strategy_results[strategy]
                if r.content_metrics:
                    row += f" {r.content_metrics.rouge_l:.3f} |"
                else:
                    row += " - |"
            f.write(row + "\n")
        
        f.write("\n")
        
        f.write("#### Retrieved Chunks by Strategy\n\n")
        for strategy in strategies:
            result = strategy_results[strategy]
            f.write(f"**{strategy}**\n\n")
            
            if not result.retrieved_chunks:
                f.write("_No chunks retrieved_\n\n")
                continue
            
            for chunk in result.retrieved_chunks:
                if chunk.is_correct_doc and chunk.is_correct_section:
                    chunk_status = "CORRECT"
                elif chunk.is_correct_doc:
                    chunk_status = "doc ok"
                else:
                    chunk_status = "-"
                
                f.write(
                    f"{chunk.rank}. [{chunk_status}] (score: {chunk.score:.3f}) "
                    f"`{chunk.document_id}` / `{chunk.section_id or '-'}`\n"
                )
                f.write(f"   > {_escape_md(_truncate(chunk.content, 200))}\n\n")
            
            f.write("\n")
        
        f.write("---\n\n")
    
    def _write_query_section(self, f: TextIO, result: QueryResult) -> None:
        status = "PASS" if result.doc_found and result.section_found else "FAIL"
        f.write(f"### Query: {result.query_id} [{status}]\n\n")
        
        f.write("#### Question\n\n")
        f.write(f"> {_escape_md(result.question)}\n\n")
        
        f.write("#### Expected\n\n")
        f.write(f"- **Document:** `{result.expected_doc_id}`\n")
        f.write(f"- **Sections:** `{', '.join(result.expected_section_ids)}`\n")
        f.write(f"- **Answer:** {_escape_md(result.expected_answer)}\n\n")
        
        f.write("#### Results\n\n")
        f.write(f"- **doc_rank:** {result.doc_rank}\n")
        f.write(f"- **section_rank:** {result.section_rank}\n")
        f.write(f"- **search_time:** {result.search_time_ms:.1f}ms\n\n")
        
        if result.content_metrics:
            cm = result.content_metrics
            f.write("#### Content Quality\n\n")
            f.write(f"| Metric | Score |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Evidence Recall | {cm.evidence_recall:.1%} |\n")
            f.write(f"| Answer Token Overlap | {cm.answer_token_overlap:.1%} |\n")
            f.write(f"| ROUGE-L | {cm.rouge_l:.3f} |\n")
            if cm.max_evidence_similarity is not None:
                f.write(f"| Max Evidence Similarity | {cm.max_evidence_similarity:.3f} |\n")
            if cm.answer_context_similarity is not None:
                f.write(f"| Answer-Context Similarity | {cm.answer_context_similarity:.3f} |\n")
            f.write("\n")
        
        f.write("#### Retrieved Chunks\n\n")
        
        for chunk in result.retrieved_chunks:
            if chunk.is_correct_doc and chunk.is_correct_section:
                chunk_status = "CORRECT"
            elif chunk.is_correct_doc:
                chunk_status = "doc ok"
            else:
                chunk_status = "-"
            
            f.write(f"**#{chunk.rank}** [{chunk_status}] (score: {chunk.score:.3f}) ")
            f.write(f"`{chunk.document_id}` / `{chunk.section_id or '-'}`\n\n")
            f.write(f"> {_escape_md(_truncate(chunk.content, 300))}\n\n")
        
        f.write("---\n\n")
    
    def finish(self) -> None:
        if self._index_stats_file:
            self._index_stats_file.close()
            self._index_stats_file = None
            self._index_stats_writer = None
        
        if self._search_results_file:
            self._search_results_file.close()
            self._search_results_file = None
            self._search_results_writer = None
        
        logger.info(f"Reports written to {self.output_dir}")
    
    def __enter__(self) -> "Reporter":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()
