"""Watch mode for fast extraction service.

Monitors input directory for new files and processes them continuously.
Supports both file output (standalone) and queue output (pipeline mode).

Environment Variables:
    QUEUE_ENABLED: "true" to publish to Redis queue (default: "false")
    QUEUE_URL: Redis URL (default: "redis://localhost:6379")
    QUEUE_STREAM: Stream name (default: "plm:extraction")
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plm.extraction.fast.document_processor import process_document
from plm.shared.queue import MessageQueue, create_queue

logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_requested = False


def signal_handler(sig: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {sig}, initiating shutdown...")
    shutdown_requested = True


def serialize_result(result: Any) -> dict:
    """Serialize DocumentResult to dict."""
    data = asdict(result)
    for section in data["headings"]:
        for chunk in section["chunks"]:
            chunk["entities"] = [
                {"text": e["text"], "label": e["label"], "score": e["score"],
                 "start": e["start"], "end": e["end"]}
                for e in chunk["entities"]
            ]
    return data


def get_pending_files(input_dir: Path, processed: set[Path], patterns: list[str]) -> list[Path]:
    """Get files that haven't been processed yet."""
    all_files: set[Path] = set()
    for pattern in patterns:
        all_files.update(input_dir.glob(pattern))
    return sorted(all_files - processed)


def process_file(
    filepath: Path,
    output_dir: Path,
    queue: MessageQueue,
    queue_enabled: bool,
    queue_stream: str,
    confidence_threshold: float = 0.7,
    extraction_threshold: float = 0.3,
) -> bool:
    """Process a single file and output to queue or file.
    
    Returns True if successful, False on error.
    """
    try:
        logger.info(f"Processing: {filepath.name}")
        t0 = time.perf_counter()
        
        result = process_document(
            filepath,
            confidence_threshold=confidence_threshold,
            extraction_threshold=extraction_threshold,
        )
        
        elapsed = time.perf_counter() - t0
        
        if result.error:
            logger.error(f"Failed to process {filepath.name}: {result.error}")
            return False
        
        output_data = serialize_result(result)
        
        if queue_enabled:
            # Publish to queue with envelope
            envelope = {
                "message_id": str(uuid.uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_service": "fast-extraction",
                "payload": output_data,
            }
            queue.publish(queue_stream, envelope)
            logger.info(f"Published to queue: {filepath.name} ({result.total_entities} entities, {elapsed:.2f}s)")
        else:
            # Write to file
            output_path = output_dir / filepath.with_suffix(".json").name
            output_path.write_text(json.dumps(output_data, indent=2))
            logger.info(f"Wrote: {output_path.name} ({result.total_entities} entities, {elapsed:.2f}s)")
        
        return True
        
    except Exception as e:
        logger.exception(f"Error processing {filepath.name}: {e}")
        return False


def watch_loop(
    input_dir: Path,
    output_dir: Path,
    patterns: list[str],
    poll_interval: float = 5.0,
    process_existing: bool = True,
    confidence_threshold: float = 0.7,
    extraction_threshold: float = 0.3,
) -> int:
    """Main watch loop - monitors directory and processes new files.
    
    Returns exit code (0 for success, 1 for error).
    """
    global shutdown_requested
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Setup queue
    queue_enabled = os.environ.get("QUEUE_ENABLED", "false").lower() == "true"
    queue_stream = os.environ.get("QUEUE_STREAM", "plm:extraction")
    queue: MessageQueue = create_queue()
    
    if queue_enabled:
        if not queue.is_available():
            logger.error("QUEUE_ENABLED=true but Redis is unavailable")
            return 1
        logger.info(f"Queue mode: publishing to {queue_stream}")
    else:
        logger.info(f"File mode: writing to {output_dir}")
    
    # Ensure directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pre-load GLiNER model
    logger.info("Loading GLiNER model...")
    from plm.extraction.fast.gliner import get_model
    get_model()
    logger.info("Model loaded, starting watch loop")
    
    processed: set[Path] = set()
    
    # Optionally skip existing files
    if not process_existing:
        existing = get_pending_files(input_dir, processed, patterns)
        processed.update(existing)
        logger.info(f"Skipping {len(existing)} existing files")
    
    logger.info(f"Watching: {input_dir} (patterns: {patterns})")
    logger.info(f"Poll interval: {poll_interval}s")
    
    files_processed = 0
    files_errored = 0
    
    while not shutdown_requested:
        pending = get_pending_files(input_dir, processed, patterns)
        
        if pending:
            logger.info(f"Found {len(pending)} new file(s)")
            
            for filepath in pending:
                if shutdown_requested:
                    break
                
                success = process_file(
                    filepath=filepath,
                    output_dir=output_dir,
                    queue=queue,
                    queue_enabled=queue_enabled,
                    queue_stream=queue_stream,
                    confidence_threshold=confidence_threshold,
                    extraction_threshold=extraction_threshold,
                )
                
                processed.add(filepath)
                if success:
                    files_processed += 1
                else:
                    files_errored += 1
        
        if not shutdown_requested:
            time.sleep(poll_interval)
    
    logger.info(f"Shutdown complete. Processed: {files_processed}, Errors: {files_errored}")
    return 0


def main() -> int:
    """CLI entry point for watch mode."""
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    parser = argparse.ArgumentParser(
        description="Watch directory and extract entities from new documents.",
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Input directory to watch")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory for JSON files (used when queue disabled)")
    parser.add_argument("--pattern", type=str, default="*.md,*.txt",
                        help="Comma-separated glob patterns (default: *.md,*.txt)")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between directory scans (default: 5)")
    parser.add_argument("--process-existing", action="store_true",
                        help="Process existing files on startup (default: skip)")
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                        help="Confidence threshold for flagging (default: 0.7)")
    parser.add_argument("--extraction-threshold", type=float, default=0.3,
                        help="GLiNER extraction threshold (default: 0.3)")
    
    args = parser.parse_args()
    patterns = [p.strip() for p in args.pattern.split(",")]
    
    return watch_loop(
        input_dir=args.input,
        output_dir=args.output,
        patterns=patterns,
        poll_interval=args.poll_interval,
        process_existing=args.process_existing,
        confidence_threshold=args.confidence_threshold,
        extraction_threshold=args.extraction_threshold,
    )


if __name__ == "__main__":
    sys.exit(main())
