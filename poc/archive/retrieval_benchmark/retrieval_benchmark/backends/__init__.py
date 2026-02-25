"""Vector store backend implementations."""

from ..config.schema import BackendConfig
from ..core.protocols import VectorStore
from .memory import MemoryStore


def create_backend(config: BackendConfig) -> VectorStore:
    """Factory function to create a vector store backend from config."""
    if config.name == "memory":
        return MemoryStore()
    elif config.name == "chromadb":
        from .chromadb_store import ChromaDBStore
        return ChromaDBStore(path=config.path)
    elif config.name == "sqlite_vec":
        from .sqlite_vec_store import SqliteVecStore
        return SqliteVecStore(path=config.path)
    else:
        raise ValueError(f"Unknown backend: {config.name}")


__all__ = ["create_backend", "MemoryStore"]
