from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from plm.benchmark.loader import BenchmarkQuestion
from plm.benchmark.metrics import QueryResult
from plm.shared.logger import PipelineLogger, get_logger


@dataclass
class RunnerConfig:
    service_url: str = "http://localhost:8000"
    k: int = 10
    use_rewrite: bool = False
    use_rerank: bool = False
    timeout: float = 30.0


class BenchmarkRunner:
    def __init__(self, config: RunnerConfig, log: PipelineLogger | None = None):
        self.config = config
        self.log = log or get_logger()
        self.client = httpx.Client(timeout=config.timeout)
    
    def run_query(self, question: BenchmarkQuestion) -> QueryResult:
        self.log.trace(f"Running query: {question.id} - {question.question[:50]}...")
        
        start_time = time.perf_counter()
        
        try:
            response = self.client.post(
                f"{self.config.service_url}/query",
                json={
                    "query": question.question,
                    "k": self.config.k,
                    "use_rewrite": self.config.use_rewrite,
                    "use_rerank": self.config.use_rerank,
                },
            )
            response.raise_for_status()
            data = response.json()
            
        except httpx.HTTPError as e:
            self.log.error(f"HTTP error for query {question.id}: {e}")
            return QueryResult(
                question_id=question.id,
                question=question.question,
                target_doc_id=question.target_doc_id,
                dataset=question.dataset,
                hit=False,
                rank=None,
                retrieved_doc_ids=[],
                latency_ms=(time.perf_counter() - start_time) * 1000,
                k=self.config.k,
            )
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        retrieved_doc_ids = [r["doc_id"] for r in data.get("results", [])]
        
        rank = None
        hit = False
        for i, doc_id in enumerate(retrieved_doc_ids, start=1):
            if doc_id.startswith(question.target_doc_id):
                rank = i
                hit = True
                break
        
        self.log.trace(
            f"Query {question.id}: hit={hit}, rank={rank}, "
            f"latency={latency_ms:.1f}ms, target={question.target_doc_id}"
        )
        
        if not hit:
            self.log.trace(
                f"MISS: {question.id} | target={question.target_doc_id} | "
                f"retrieved={retrieved_doc_ids[:5]}"
            )
        
        return QueryResult(
            question_id=question.id,
            question=question.question,
            target_doc_id=question.target_doc_id,
            dataset=question.dataset,
            hit=hit,
            rank=rank,
            retrieved_doc_ids=retrieved_doc_ids,
            latency_ms=latency_ms,
            k=self.config.k,
        )
    
    def run_all(self, questions: list[BenchmarkQuestion]) -> list[QueryResult]:
        self.log.info(f"Starting benchmark run with {len(questions)} questions")
        self.log.info(f"Config: k={self.config.k}, rewrite={self.config.use_rewrite}, rerank={self.config.use_rerank}")
        
        results = []
        for i, question in enumerate(questions, start=1):
            if i % 50 == 0:
                self.log.progress(i, len(questions), "queries completed")
            
            result = self.run_query(question)
            results.append(result)
        
        self.log.info(f"Benchmark run complete: {len(results)} queries evaluated")
        return results
    
    def close(self):
        self.client.close()
