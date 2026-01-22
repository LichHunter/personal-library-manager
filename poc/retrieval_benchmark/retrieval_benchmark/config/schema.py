"""Pydantic configuration schema for experiments."""

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""
    model: str = "all-MiniLM-L6-v2"
    provider: Literal["sentence_transformers"] = "sentence_transformers"


class LLMConfig(BaseModel):
    """Configuration for LLM."""
    model: str = "llama3.2:3b"
    provider: Literal["ollama"] = "ollama"
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 500


class BackendConfig(BaseModel):
    """Configuration for vector store backend."""
    name: Literal["memory", "chromadb", "sqlite_vec"] = "memory"
    path: Optional[str] = None  # For persistent backends


class StrategyConfig(BaseModel):
    """Configuration for retrieval strategy."""
    name: Literal["flat", "raptor", "lod_embed", "lod_llm"] = "flat"
    
    # Chunking parameters
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # LOD-specific parameters
    doc_top_k: int = 5
    section_top_k: int = 10
    
    # RAPTOR-specific parameters
    max_layers: int = 3
    cluster_threshold: float = 0.1


class DataConfig(BaseModel):
    """Configuration for input data."""
    documents_dir: str = "../test_data/output/documents"
    ground_truth_path: str = "../test_data/output/ground_truth.json"
    max_documents: Optional[int] = None
    max_queries: Optional[int] = None


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""
    runs_per_query: int = 3
    top_k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    random_seed: int = 42


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    id: str
    description: str = ""
    
    data: DataConfig = Field(default_factory=DataConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    
    # Components to test
    embeddings: list[EmbeddingConfig] = Field(
        default_factory=lambda: [EmbeddingConfig()]
    )
    llms: list[LLMConfig] = Field(
        default_factory=lambda: [LLMConfig()]
    )
    backends: list[BackendConfig] = Field(
        default_factory=lambda: [BackendConfig()]
    )
    strategies: list[StrategyConfig] = Field(
        default_factory=lambda: [StrategyConfig()]
    )
    
    # Output configuration
    output_dir: str = "./results"


def load_config(config_path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    config_path = Path(config_path)
    
    with open(config_path) as f:
        data = yaml.safe_load(f)
    
    return ExperimentConfig(**data)
