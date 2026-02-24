"""Configuration for search retrieval system.

Supports switching between BM25 and SPLADE sparse retrievers via
environment variables or explicit configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plm.search.components.sparse.base import SparseRetriever


class SparseRetrieverType(str, Enum):
    BM25 = "bm25"
    SPLADE = "splade"


@dataclass
class SPLADESettings:
    model: str = "naver/splade-cocondenser-ensembledistil"
    device: str | None = None
    max_length: int = 512
    batch_size: int = 32


@dataclass
class SemanticSettings:
    enabled: bool = True
    model: str = "BAAI/bge-base-en-v1.5"


@dataclass
class RetrievalConfig:
    sparse_retriever: SparseRetrieverType = SparseRetrieverType.BM25
    splade: SPLADESettings = field(default_factory=SPLADESettings)
    semantic: SemanticSettings = field(default_factory=SemanticSettings)

    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        sparse_type_str = os.environ.get("PLM_SPARSE_RETRIEVER", "bm25").lower()
        try:
            sparse_type = SparseRetrieverType(sparse_type_str)
        except ValueError:
            sparse_type = SparseRetrieverType.BM25

        splade = SPLADESettings(
            model=os.environ.get("PLM_SPLADE_MODEL", SPLADESettings.model),
            device=os.environ.get("PLM_SPLADE_DEVICE"),
            max_length=int(os.environ.get("PLM_SPLADE_MAX_LENGTH", "512")),
            batch_size=int(os.environ.get("PLM_SPLADE_BATCH_SIZE", "32")),
        )

        semantic_enabled = os.environ.get("PLM_SEMANTIC_ENABLED", "true").lower() == "true"
        semantic = SemanticSettings(enabled=semantic_enabled)

        return cls(sparse_retriever=sparse_type, splade=splade, semantic=semantic)


def create_sparse_retriever(
    config: RetrievalConfig | None = None,
) -> "SparseRetriever":
    if config is None:
        config = RetrievalConfig.from_env()

    if config.sparse_retriever == SparseRetrieverType.SPLADE:
        from plm.search.components.sparse import SPLADEConfig, SPLADERetriever

        splade_config = SPLADEConfig(
            model_name=config.splade.model,
            device=config.splade.device,
            max_length=config.splade.max_length,
            batch_size=config.splade.batch_size,
        )
        return SPLADERetriever(config=splade_config)
    else:
        from plm.search.components.sparse import BM25Retriever

        return BM25Retriever()
