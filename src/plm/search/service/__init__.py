"""FastAPI service for search with directory watcher for auto-ingestion."""

from plm.search.service.app import app
from plm.search.service.watcher import DirectoryWatcher

__all__ = ["app", "DirectoryWatcher"]
