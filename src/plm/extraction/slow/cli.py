"""Docker entrypoint CLI using V6 extraction pipeline (exact POC-1c replication).

Environment Variables:
    INPUT_DIR: Input directory to watch (default: /data/input)
    OUTPUT_DIR: Output directory for JSON results (default: /data/output)
    LOG_DIR: Log directory for low-confidence terms (default: /data/logs)
    VOCAB_PATH: Path to auto_vocab.json (default: /data/vocabularies/auto_vocab.json)
    TRAIN_DOCS_PATH: Path to train_documents.json (default: /data/vocabularies/train_documents.json)
    POLL_INTERVAL: Polling interval in seconds (default: 30)
    PROCESS_ONCE: Process existing files and exit (default: false)
    DRY_RUN: Dry run mode, no output written (default: false)
"""

import json
import os
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plm.extraction.slow.format_transformer import transform_to_fast_format
from plm.shared.queue import MessageQueue, create_queue

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from plm.extraction.slow import (
    extract_hybrid_v5,
    build_term_index,
    get_strategy_config,
    build_retrieval_index,
)

shutdown_requested = False


def signal_handler(sig: int, frame: Any) -> None:
    global shutdown_requested
    print("\nShutting down gracefully...")
    shutdown_requested = True


def parse_bool(value: str) -> bool:
    return value.lower() in ("true", "1", "yes", "on")


def parse_env() -> dict[str, Any]:
    return {
        "input_dir": Path(os.getenv("INPUT_DIR", "/data/input")),
        "output_dir": Path(os.getenv("OUTPUT_DIR", "/data/output")),
        "log_dir": Path(os.getenv("LOG_DIR", "/data/logs")),
        "vocab_path": Path(os.getenv("VOCAB_PATH", "/data/vocabularies/auto_vocab.json")),
        "train_docs_path": Path(os.getenv("TRAIN_DOCS_PATH", "/data/vocabularies/train_documents.json")),
        "poll_interval": int(os.getenv("POLL_INTERVAL", "30")),
        "process_once": parse_bool(os.getenv("PROCESS_ONCE", "false")),
        "dry_run": parse_bool(os.getenv("DRY_RUN", "false")),
        # Queue configuration
        "queue_enabled": parse_bool(os.getenv("QUEUE_ENABLED", "false")),
        "queue_stream": os.getenv("QUEUE_STREAM", "plm:extraction"),
    }


def setup_directories(config: dict[str, Any]) -> None:
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["log_dir"].mkdir(parents=True, exist_ok=True)


def get_input_files(input_dir: Path, processed: set[Path]) -> list[Path]:
    if not input_dir.exists():
        return []
    all_files = [f for f in input_dir.iterdir() if f.is_file()]
    return [f for f in all_files if f not in processed]


def load_v6_resources(config: dict[str, Any]) -> dict[str, Any]:
    """Load all V6 extraction resources (vocab, train docs, index, model)."""
    print("Loading V6 resources...")
    
    with open(config["vocab_path"]) as f:
        auto_vocab = json.load(f)
    print(f"  Loaded auto_vocab: {auto_vocab.get('stats', {})}")
    
    with open(config["train_docs_path"]) as f:
        train_docs = json.load(f)
    print(f"  Loaded {len(train_docs)} training documents")
    
    print("  Building term index...")
    term_index = build_term_index(train_docs)
    print(f"  Built term index: {len(term_index)} terms")
    
    print("  Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print("  Building FAISS index...")
    train_texts = [d["text"] for d in train_docs]
    train_embeddings = model.encode(train_texts, show_progress_bar=False)
    index = faiss.IndexFlatL2(train_embeddings.shape[1])
    index.add(np.array(train_embeddings).astype('float32'))
    print(f"  Built FAISS index: {index.ntotal} vectors")
    
    strategy = get_strategy_config("strategy_v6")
    print(f"  Using strategy: {strategy.name}")
    
    return {
        "auto_vocab": auto_vocab,
        "train_docs": train_docs,
        "term_index": term_index,
        "model": model,
        "index": index,
        "strategy": strategy,
    }


def process_document(
    file_path: Path,
    resources: dict[str, Any],
) -> dict[str, Any]:
    """Process document using V6 extraction."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        return {
            "file": file_path.name,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }
    
    doc = {
        "doc_id": file_path.stem,
        "text": text,
        "gt_terms": [],
    }
    
    terms = extract_hybrid_v5(
        doc,
        resources["train_docs"],
        resources["index"],
        resources["model"],
        resources["auto_vocab"],
        term_index=resources["term_index"],
        strategy=resources["strategy"],
    )
    
    term_results = []
    for t in terms:
        term_results.append({
            "term": t,
            "confidence": 0.9,
            "level": "HIGH",
            "sources": ["v6"],
        })
    
    return {
        "file": file_path.name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [{
            "text": text,
            "chunk_index": 0,
            "heading": None,
            "terms": term_results,
        }],
        "stats": {
            "total_chunks": 1,
            "total_terms": len(terms),
            "high_confidence": len(terms),
            "medium_confidence": 0,
            "low_confidence": 0,
        },
    }


def write_output(result: dict[str, Any], output_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"  [DRY RUN] Would write: {output_path}")
        return
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {output_path}")


def watch_loop(config: dict[str, Any]) -> None:
    processed: set[Path] = set()
    
    resources = load_v6_resources(config)
    
    setup_directories(config)
    
    # Initialize queue if enabled
    queue: MessageQueue = create_queue()
    queue_enabled = config["queue_enabled"]
    queue_stream = config["queue_stream"]
    
    if queue_enabled:
        if not queue.is_available():
            print("Error: QUEUE_ENABLED=true but Redis is unavailable", file=sys.stderr)
            sys.exit(1)
        print(f"Queue: {queue_stream} (enabled)")
    else:
        print("Queue: disabled")
    
    print(f"\nWatching: {config['input_dir']}")
    print(f"Output: {config['output_dir']}")
    print(f"Process once: {config['process_once']}")
    print(f"Dry run: {config['dry_run']}")
    
    while not shutdown_requested:
        new_files = get_input_files(config["input_dir"], processed)
        
        if new_files:
            print(f"\nFound {len(new_files)} new file(s)")
            
            for file_path in new_files:
                if shutdown_requested:
                    break
                
                print(f"Processing: {file_path.name}")
                
                result = process_document(file_path, resources)
                
                if queue_enabled:
                    # Transform to fast format and publish to queue
                    fast_result = transform_to_fast_format(result, source_file=str(file_path))
                    envelope = {
                        "message_id": str(uuid.uuid4()),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "source_service": "slow-extraction",
                        "payload": fast_result,
                    }
                    queue.publish(queue_stream, envelope)
                    print(f"  Published to queue: {queue_stream}")
                else:
                    # Write to file (standalone mode)
                    output_path = config["output_dir"] / f"{file_path.stem}.json"
                    write_output(result, output_path, config["dry_run"])
                
                processed.add(file_path)
                
                if "stats" in result:
                    print(f"  Terms: {result['stats']['total_terms']}")
        
        if config["process_once"]:
            print("\nProcess-once mode: exiting")
            break
        
        time.sleep(config["poll_interval"])
    
    print("Shutdown complete")


def main() -> None:
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    config = parse_env()
    
    for required in ["vocab_path", "train_docs_path"]:
        path = config[required]
        if not path.exists():
            print(f"Fatal error: {required} not found: {path}", file=sys.stderr)
            sys.exit(1)
    
    watch_loop(config)


if __name__ == "__main__":
    main()
