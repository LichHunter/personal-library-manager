"""Main benchmark executor."""

import gc
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from ..backends import create_backend
from ..config.schema import (
    BackendConfig,
    EmbeddingConfig,
    ExperimentConfig,
    LLMConfig,
    StrategyConfig,
)
from ..core.loader import load_documents, load_ground_truth
from ..core.protocols import Embedder, LLM, VectorStore
from ..core.types import Document, GroundTruth, IndexStats, StrategySummary
from ..embeddings import create_embedder
from ..strategies import create_strategy
from ..llms import create_llm

from .evaluator import Evaluator, EvaluationContext
from .reporter import Reporter

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Main benchmark executor.
    
    Orchestrates the complete benchmark workflow:
    1. Load documents and ground truth
    2. Generate configuration combinations
    3. For each combination:
       - Create components (embedder, store, LLM)
       - Index documents
       - Run queries
       - Evaluate results
    4. Generate reports
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.evaluator = Evaluator()
        
        # Setup output directory
        self.output_dir = Path(config.output_dir) / config.id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Save config copy
        self._save_config()
    
    def _setup_logging(self) -> None:
        """Configure logging to file and console."""
        log_file = self.output_dir / "run.log"
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        root_logger.setLevel(logging.DEBUG)
        
        logger.info(f"Logging to {log_file}")
    
    def _save_config(self) -> None:
        """Save experiment config to output directory."""
        config_path = self.output_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self.config.model_dump(), f, default_flow_style=False)
        logger.info(f"Config saved to {config_path}")
    
    def run(self) -> None:
        """Execute the benchmark."""
        logger.info(f"Starting benchmark: {self.config.id}")
        logger.info(f"Description: {self.config.description}")
        
        # Load data
        documents = self._load_documents()
        ground_truth = self._load_ground_truth()
        
        logger.info(f"Loaded {len(documents)} documents, {len(ground_truth)} queries")
        
        # Generate configuration combinations
        combinations = self._generate_combinations()
        logger.info(f"Generated {len(combinations)} configurations to test")
        
        # Track summaries for final report
        summaries: list[StrategySummary] = []
        
        # Execute benchmarks
        with Reporter(self.output_dir) as reporter:
            for i, combo in enumerate(combinations, 1):
                strategy_cfg, backend_cfg, embedding_cfg, llm_cfg = combo
                
                config_name = self._config_name(
                    strategy_cfg, backend_cfg, embedding_cfg, llm_cfg
                )
                logger.info(f"[{i}/{len(combinations)}] Running: {config_name}")
                
                try:
                    summary = self._run_single_config(
                        strategy_cfg=strategy_cfg,
                        backend_cfg=backend_cfg,
                        embedding_cfg=embedding_cfg,
                        llm_cfg=llm_cfg,
                        documents=documents,
                        ground_truth=ground_truth,
                        reporter=reporter,
                    )
                    summaries.append(summary)
                except Exception as e:
                    logger.error(f"Configuration failed: {config_name}", exc_info=True)
                    # Continue with next configuration
                finally:
                    # Force garbage collection to free GPU memory
                    gc.collect()
            
            # Write summary report
            reporter.write_summary(summaries)
        
        logger.info("Benchmark complete!")
        logger.info(f"Results: {self.output_dir}")
    
    def _load_documents(self) -> list[Document]:
        """Load documents from configured path."""
        docs_dir = Path(self.config.data.documents_dir)
        
        # Handle relative paths
        if not docs_dir.is_absolute():
            # Assume relative to benchmark directory
            docs_dir = Path(__file__).parent.parent / docs_dir
        
        documents = load_documents(docs_dir)
        
        # Apply max_documents limit
        if self.config.data.max_documents:
            documents = documents[:self.config.data.max_documents]
        
        return documents
    
    def _load_ground_truth(self) -> list[GroundTruth]:
        """Load ground truth queries from configured path."""
        gt_path = Path(self.config.data.ground_truth_path)
        
        # Handle relative paths
        if not gt_path.is_absolute():
            gt_path = Path(__file__).parent.parent / gt_path
        
        queries = load_ground_truth(gt_path)
        
        # Apply max_queries limit
        if self.config.data.max_queries:
            queries = queries[:self.config.data.max_queries]
        
        return queries
    
    def _generate_combinations(
        self,
    ) -> list[tuple[
        StrategyConfig,
        BackendConfig, 
        EmbeddingConfig,
        Optional[LLMConfig],
    ]]:
        """Generate all valid configuration combinations."""
        combinations = []
        
        for strategy_cfg in self.config.strategies:
            for backend_cfg in self.config.backends:
                for embedding_cfg in self.config.embeddings:
                    # Check if strategy requires LLM
                    if strategy_cfg.name in ("lod_llm", "raptor"):
                        # Strategies that require LLM
                        for llm_cfg in self.config.llms:
                            combinations.append((
                                strategy_cfg,
                                backend_cfg,
                                embedding_cfg,
                                llm_cfg,
                            ))
                    else:
                        # Strategies without LLM
                        combinations.append((
                            strategy_cfg,
                            backend_cfg,
                            embedding_cfg,
                            None,
                        ))
        
        return combinations
    
    def _config_name(
        self,
        strategy_cfg: StrategyConfig,
        backend_cfg: BackendConfig,
        embedding_cfg: EmbeddingConfig,
        llm_cfg: Optional[LLMConfig],
    ) -> str:
        """Generate a human-readable config name."""
        parts = [
            strategy_cfg.name,
            backend_cfg.name,
            embedding_cfg.model.split("/")[-1],  # Just model name
        ]
        if llm_cfg:
            parts.append(llm_cfg.model)
        return " / ".join(parts)
    
    def _run_single_config(
        self,
        strategy_cfg: StrategyConfig,
        backend_cfg: BackendConfig,
        embedding_cfg: EmbeddingConfig,
        llm_cfg: Optional[LLMConfig],
        documents: list[Document],
        ground_truth: list[GroundTruth],
        reporter: Reporter,
    ) -> StrategySummary:
        """Run benchmark for a single configuration."""
        # Create components
        embedder = create_embedder(embedding_cfg)
        store = create_backend(backend_cfg)
        llm: Optional[LLM] = None
        if llm_cfg:
            llm = create_llm(llm_cfg)
        
        # Create strategy
        strategy = create_strategy(strategy_cfg, embedder, store, llm)
        
        # Create evaluation context
        context = EvaluationContext(
            strategy=strategy_cfg.name,
            backend=backend_cfg.name,
            embedding_model=embedding_cfg.model,
            llm_model=llm_cfg.model if llm_cfg else None,
        )
        
        try:
            # Index documents
            logger.info("Indexing documents...")
            index_stats = strategy.index(documents)
            reporter.write_index_stats(index_stats)
            
            logger.info(
                f"Indexed {index_stats.num_vectors} vectors in "
                f"{index_stats.duration_sec:.2f}s"
            )
            
            # Run queries
            logger.info(f"Running {len(ground_truth)} queries...")
            all_results = []
            
            for gt in ground_truth:
                for run in range(1, self.config.benchmark.runs_per_query + 1):
                    for top_k in self.config.benchmark.top_k_values:
                        # Execute search
                        response = strategy.search(gt.question, top_k=top_k)
                        
                        # Evaluate
                        result = self.evaluator.evaluate_query(
                            query_id=gt.id,
                            run_number=run,
                            ground_truth=gt,
                            response=response,
                            context=context,
                            top_k=top_k,
                        )
                        
                        all_results.append(result)
                        reporter.write_query_result(result)
            
            # Compute summary
            summary = self.evaluator.compute_summary(
                results=all_results,
                index_time_sec=index_stats.duration_sec,
                num_vectors=index_stats.num_vectors,
                runs_per_query=self.config.benchmark.runs_per_query,
            )
            
            logger.info(
                f"doc_recall@5={summary.doc_recall_at_5:.2%}, "
                f"section_recall@5={summary.section_recall_at_5:.2%}, "
                f"avg_search={summary.avg_search_time_ms:.1f}ms"
            )
            
            return summary
        
        finally:
            # Cleanup
            strategy.clear()
            store.close()
