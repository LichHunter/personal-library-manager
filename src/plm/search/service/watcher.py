"""Directory watcher for auto-ingestion with robustness features.

Features:
- Atomic file handling: ignores .tmp files (process only when renamed to final)
- Failure handling: moves bad files to failed/ with .error metadata file
- Batch debounce: collects files for 1s, then batch ingest with single rebuild
- SIGTERM handler: drains queue before shutdown
- Startup flag: --process-existing (default: ignore existing files on start)
- Success handling: moves processed files to processed/
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import signal
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    from plm.search.retriever import HybridRetriever

logger = logging.getLogger(__name__)


def json_to_chunks(doc_dict: dict) -> list[dict]:
    """Convert a JSON-serialized DocumentResult dict to chunk list.

    This handles the JSON structure where dataclasses are serialized as dicts.

    Args:
        doc_dict: Dict with 'headings' list, each containing 'heading', 'chunks'.
                  Each chunk has 'text', 'keywords', 'entities', 'start_char', 'end_char'.

    Returns:
        List of chunk dicts ready for HybridRetriever.ingest_document().
    """
    chunks = []
    for section in doc_dict.get("headings", []):
        heading = section.get("heading", "")
        for chunk in section.get("chunks", []):
            # Transform entities from list[{text, label, score}] to dict[label, list[text]]
            entities_list = chunk.get("entities", [])
            entities_dict: dict[str, list[str]] = {}
            seen: dict[str, set[str]] = {}
            for entity in entities_list:
                label = entity.get("label", "unknown")
                text = entity.get("text", "")
                if label not in seen:
                    seen[label] = set()
                    entities_dict[label] = []
                if text not in seen[label]:
                    entities_dict[label].append(text)
                    seen[label].add(text)

            chunks.append({
                "content": chunk.get("text", ""),
                "keywords": chunk.get("keywords", []),
                "entities": entities_dict,
                "heading": heading,
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
            })
    return chunks


@dataclass
class FileEvent:
    """Represents a file event to be processed."""
    path: Path
    timestamp: float


class IngestionHandler(FileSystemEventHandler):
    """Watchdog event handler that queues files for batch processing."""

    def __init__(
        self,
        on_files_ready: Callable[[list[Path]], None],
        debounce_seconds: float = 1.0,
    ) -> None:
        """Initialize handler.

        Args:
            on_files_ready: Callback when batch is ready to process.
            debounce_seconds: Wait time before processing batch.
        """
        super().__init__()
        self.on_files_ready = on_files_ready
        self.debounce_seconds = debounce_seconds
        self._pending: dict[str, Path] = {}  # path_str -> Path
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def on_created(self, event) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return
        src_path = event.src_path
        path_str: str = src_path.decode("utf-8") if isinstance(src_path, bytes) else src_path
        self._handle_file(Path(path_str))

    def on_moved(self, event) -> None:
        """Handle file move events (e.g., .tmp -> final)."""
        if event.is_directory:
            return
        dest_path = event.dest_path
        path_str: str = dest_path.decode("utf-8") if isinstance(dest_path, bytes) else dest_path
        self._handle_file(Path(path_str))

    def _handle_file(self, path: Path) -> None:
        """Queue a file for processing if valid."""
        # Ignore .tmp files (atomic write pattern)
        if path.suffix == ".tmp":
            logger.debug(f"[Watcher] Ignoring .tmp file: {path}")
            return

        # Only process JSON files
        if path.suffix != ".json":
            logger.debug(f"[Watcher] Ignoring non-JSON file: {path}")
            return

        # Ignore files in processed/ or failed/ subdirs
        if "processed" in path.parts or "failed" in path.parts:
            return

        with self._lock:
            self._pending[str(path)] = path
            self._reset_timer()

    def _reset_timer(self) -> None:
        """Reset the debounce timer."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce_seconds, self._flush_batch)
        self._timer.daemon = True
        self._timer.start()

    def _flush_batch(self) -> None:
        """Flush pending files to callback."""
        with self._lock:
            if not self._pending:
                return
            files = list(self._pending.values())
            self._pending.clear()
            self._timer = None

        logger.info(f"[Watcher] Processing batch of {len(files)} files")
        self.on_files_ready(files)

    def drain(self) -> None:
        """Drain any pending files immediately."""
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            if not self._pending:
                return
            files = list(self._pending.values())
            self._pending.clear()

        if files:
            logger.info(f"[Watcher] Draining {len(files)} pending files on shutdown")
            self.on_files_ready(files)


class DirectoryWatcher:
    """Watches a directory and auto-ingests new document JSON files.

    Features:
    - Ignores .tmp files (atomic write pattern)
    - Batch debounce: waits 1s after last file before processing
    - Moves successful files to processed/ subdirectory
    - Moves failed files to failed/ with .error metadata
    - Graceful SIGTERM handling
    """

    def __init__(
        self,
        watch_dir: str | Path,
        retriever: HybridRetriever,
        debounce_seconds: float = 1.0,
        process_existing: bool = False,
    ) -> None:
        """Initialize watcher.

        Args:
            watch_dir: Directory to watch for new files.
            retriever: HybridRetriever instance for ingestion.
            debounce_seconds: Seconds to wait before processing batch.
            process_existing: If True, process existing files on start.
        """
        self.watch_dir = Path(watch_dir)
        self.retriever = retriever
        self.debounce_seconds = debounce_seconds
        self.process_existing = process_existing

        # Create subdirectories
        self.processed_dir = self.watch_dir / "processed"
        self.failed_dir = self.watch_dir / "failed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize watchdog
        self._handler = IngestionHandler(
            on_files_ready=self._process_batch,
            debounce_seconds=debounce_seconds,
        )
        self._observer = Observer()
        self._running = False
        self._shutdown_event = threading.Event()

        # Track original signal handlers
        self._original_sigterm = None
        self._original_sigint = None

    def start(self) -> None:
        """Start watching the directory."""
        if self._running:
            return

        # Process existing files if requested
        if self.process_existing:
            existing = list(self.watch_dir.glob("*.json"))
            # Exclude processed/ and failed/ subdirs
            existing = [
                f for f in existing
                if "processed" not in f.parts and "failed" not in f.parts
            ]
            if existing:
                logger.info(f"[Watcher] Processing {len(existing)} existing files")
                self._process_batch(existing)

        # Set up signal handlers
        self._setup_signal_handlers()

        # Start observer
        self._observer.schedule(self._handler, str(self.watch_dir), recursive=False)
        self._observer.start()
        self._running = True
        logger.info(f"[Watcher] Started watching {self.watch_dir}")

    def stop(self) -> None:
        """Stop watching and drain pending files."""
        if not self._running:
            return

        logger.info("[Watcher] Stopping...")
        self._shutdown_event.set()

        # Drain pending files
        self._handler.drain()

        # Stop observer
        self._observer.stop()
        self._observer.join(timeout=5.0)
        self._running = False

        # Restore signal handlers
        self._restore_signal_handlers()

        logger.info("[Watcher] Stopped")

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown on SIGTERM/SIGINT."""
        def handler(signum, frame):
            logger.info(f"[Watcher] Received signal {signum}, initiating shutdown")
            self.stop()
            # Call original handler if exists
            if signum == signal.SIGTERM and self._original_sigterm:
                if callable(self._original_sigterm):
                    self._original_sigterm(signum, frame)
            elif signum == signal.SIGINT and self._original_sigint:
                if callable(self._original_sigint):
                    self._original_sigint(signum, frame)

        self._original_sigterm = signal.signal(signal.SIGTERM, handler)
        self._original_sigint = signal.signal(signal.SIGINT, handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)

    def _process_batch(self, files: list[Path]) -> None:
        """Process a batch of files.

        For each file:
        1. Load JSON and convert to chunks
        2. Ingest with rebuild_index=False
        3. Move to processed/ or failed/

        After all files, call rebuild_index() once.
        """
        successful = []
        for filepath in files:
            try:
                self._process_file(filepath)
                successful.append(filepath)
            except Exception as e:
                logger.error(f"[Watcher] Failed to process {filepath}: {e}")
                self._move_to_failed(filepath, e)

        # Rebuild index once after all files
        if successful:
            try:
                self.retriever.rebuild_index()
                logger.info(f"[Watcher] Rebuilt index after ingesting {len(successful)} files")
            except Exception as e:
                logger.error(f"[Watcher] Failed to rebuild index: {e}")

    def _process_file(self, filepath: Path) -> None:
        """Process a single file."""
        logger.debug(f"[Watcher] Processing {filepath}")

        with open(filepath, encoding="utf-8") as f:
            doc_dict = json.load(f)

        source_file = doc_dict.get("source_file", str(filepath))

        chunks = json_to_chunks(doc_dict)
        if not chunks:
            raise ValueError(f"No chunks extracted from {filepath}")

        all_content = "".join(c["content"] for c in chunks)
        content_hash = hashlib.sha256(all_content.encode()).hexdigest()[:12]
        doc_id = f"{filepath.stem}_{content_hash}"

        deleted = self.retriever.storage.delete_documents_by_content_hash(content_hash)
        if deleted > 0:
            logger.info(f"[Watcher] Deduplicated {deleted} existing document(s) with hash {content_hash}")

        self.retriever.ingest_document(
            doc_id=doc_id,
            source_file=source_file,
            chunks=chunks,
            rebuild_index=False,
        )

        # Move to processed
        self._move_to_processed(filepath)
        logger.info(f"[Watcher] Ingested {filepath.name}: {len(chunks)} chunks")

    def _move_to_processed(self, filepath: Path) -> None:
        """Move file to processed/ directory."""
        dest = self.processed_dir / filepath.name
        # Handle name collision
        if dest.exists():
            base = filepath.stem
            suffix = filepath.suffix
            counter = 1
            while dest.exists():
                dest = self.processed_dir / f"{base}_{counter}{suffix}"
                counter += 1
        shutil.move(str(filepath), str(dest))

    def _move_to_failed(self, filepath: Path, error: Exception) -> None:
        """Move file to failed/ and create .error metadata."""
        try:
            dest = self.failed_dir / filepath.name
            # Handle name collision
            if dest.exists():
                base = filepath.stem
                suffix = filepath.suffix
                counter = 1
                while dest.exists():
                    dest = self.failed_dir / f"{base}_{counter}{suffix}"
                    counter += 1

            shutil.move(str(filepath), str(dest))

            # Create .error metadata file
            error_file = dest.with_suffix(dest.suffix + ".error")
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                "original_path": str(filepath),
            }
            with open(error_file, "w", encoding="utf-8") as f:
                json.dump(error_info, f, indent=2)

            logger.debug(f"[Watcher] Moved {filepath.name} to failed/")
        except Exception as e:
            logger.error(f"[Watcher] Failed to move {filepath} to failed/: {e}")
