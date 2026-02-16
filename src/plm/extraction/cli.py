"""Docker entrypoint CLI for document processing with folder watching.

Environment Variables:
    INPUT_DIR: Input directory to watch (default: /data/input)
    OUTPUT_DIR: Output directory for JSON results (default: /data/output)
    LOG_DIR: Log directory for low-confidence terms (default: /data/logs)
    VOCAB_NEGATIVES_PATH: Path to negatives vocabulary (default: /data/vocabularies/tech_domain_negatives.json)
    VOCAB_SEEDS_PATH: Path to seeds vocabulary (default: /data/vocabularies/auto_vocab.json)
    POLL_INTERVAL: Polling interval in seconds (default: 30)
    CONFIDENCE_THRESHOLD: Confidence threshold for filtering (default: 0.5)
    CHUNKING_STRATEGY: Chunking strategy (default: whole)
    CHUNK_MIN_TOKENS: Minimum tokens per chunk (default: 50)
    CHUNK_MAX_TOKENS: Maximum tokens per chunk (default: 256)
    PROCESS_ONCE: Process existing files and exit (default: false)
    DRY_RUN: Dry run mode, no output written (default: false)
    LLM_MODEL: LLM model to use (default: sonnet)

CLI Flags:
    --chunker STRATEGY: Override CHUNKING_STRATEGY
    --process-once: Override PROCESS_ONCE
    --dry-run: Override DRY_RUN
    --help: Show usage
"""

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from plm.extraction.chunking import get_chunker
from plm.extraction.fast.confidence import compute_confidence
from plm.extraction.slow import (
    extract_by_taxonomy,
    extract_candidates_heuristic,
    classify_candidates_llm,
    ground_candidates,
    filter_noise,
    load_negatives,
    validate_terms,
    expand_spans,
    suppress_subspans,
    final_dedup,
)


# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(sig: int, frame: Any) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    print("\nShutting down gracefully...")
    shutdown_requested = True


def parse_bool(value: str) -> bool:
    """Parse boolean from environment variable."""
    return value.lower() in ("true", "1", "yes", "on")


def parse_env() -> dict[str, Any]:
    """Parse environment variables with defaults."""
    return {
        "input_dir": Path(os.getenv("INPUT_DIR", "/data/input")),
        "output_dir": Path(os.getenv("OUTPUT_DIR", "/data/output")),
        "log_dir": Path(os.getenv("LOG_DIR", "/data/logs")),
        "vocab_negatives_path": Path(
            os.getenv(
                "VOCAB_NEGATIVES_PATH",
                "/data/vocabularies/tech_domain_negatives.json",
            )
        ),
        "vocab_seeds_path": Path(
            os.getenv("VOCAB_SEEDS_PATH", "/data/vocabularies/auto_vocab.json")
        ),
        "poll_interval": int(os.getenv("POLL_INTERVAL", "30")),
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", "0.5")),
        "chunking_strategy": os.getenv("CHUNKING_STRATEGY", "whole"),
        "chunk_min_tokens": int(os.getenv("CHUNK_MIN_TOKENS", "50")),
        "chunk_max_tokens": int(os.getenv("CHUNK_MAX_TOKENS", "256")),
        "process_once": parse_bool(os.getenv("PROCESS_ONCE", "false")),
        "dry_run": parse_bool(os.getenv("DRY_RUN", "false")),
        "llm_model": os.getenv("LLM_MODEL", "sonnet"),
    }


def parse_cli_args(config: dict[str, Any]) -> dict[str, Any]:
    """Parse CLI arguments and override config."""
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--help":
            print_help()
            sys.exit(0)
        elif arg == "--chunker":
            if i + 1 >= len(args):
                print("Error: --chunker requires an argument", file=sys.stderr)
                sys.exit(1)
            config["chunking_strategy"] = args[i + 1]
            i += 2
        elif arg == "--process-once":
            config["process_once"] = True
            i += 1
        elif arg == "--dry-run":
            config["dry_run"] = True
            i += 1
        else:
            print(f"Error: Unknown argument: {arg}", file=sys.stderr)
            print_help()
            sys.exit(1)
    return config


def print_help() -> None:
    """Print usage information."""
    print(__doc__)


def setup_directories(config: dict[str, Any]) -> None:
    """Create output directories if they don't exist."""
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    config["log_dir"].mkdir(parents=True, exist_ok=True)


def get_input_files(input_dir: Path, processed: set[Path]) -> list[Path]:
    """Get list of unprocessed files from input directory."""
    if not input_dir.exists():
        return []
    
    all_files = [f for f in input_dir.iterdir() if f.is_file()]
    return [f for f in all_files if f not in processed]


def extract_terms_from_chunk(
    chunk_text: str,
    negatives: set[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract terms from a single chunk using V6 pipeline.
    
    V6 Pipeline stages:
    1. Extract: Taxonomy + Candidate verify (heuristic + LLM)
    2. Ground: Span verification and deduplication
    3. Filter: Noise filtering (stop words, negatives)
    4. Validate: Context validation for ambiguous terms
    5. Postprocess: Expand spans, suppress subspans, final dedup
    
    Args:
        chunk_text: Text content of the chunk
        negatives: Set of negative terms to filter
        config: Configuration dictionary
        
    Returns:
        List of term dictionaries with confidence scores
    """
    # Stage 1a: Taxonomy extraction
    taxonomy_terms = extract_by_taxonomy(chunk_text, model=config["llm_model"])
    
    # Stage 1b: Candidate verify (heuristic + LLM)
    heuristic_candidates = extract_candidates_heuristic(chunk_text)
    verified_candidates = classify_candidates_llm(
        heuristic_candidates,
        chunk_text,
        model=config["llm_model"],
    )
    
    # Stage 2: Ground candidates (merge by source)
    candidates_by_source = {
        "taxonomy": taxonomy_terms,
        "heuristic": verified_candidates,
    }
    grounded = ground_candidates(candidates_by_source, chunk_text)
    
    # Extract terms from grounded dict
    grounded_terms = [entry["term"] for entry in grounded.values()]
    
    # Stage 3: Filter noise
    filtered = filter_noise(grounded_terms, negatives)
    
    # Stage 4: Validate terms (for medium confidence)
    validated = validate_terms(filtered, chunk_text, model=config["llm_model"])
    
    # Stage 5: Postprocess
    expanded = expand_spans(validated, chunk_text)
    suppressed = suppress_subspans(expanded)
    final_terms = final_dedup(suppressed)
    
    # Compute confidence for each term
    results = []
    for term in final_terms:
        # Find sources for this term from grounded data
        sources = []
        for key, entry in grounded.items():
            if entry["term"] == term or entry["term"].lower() == term.lower():
                sources = list(entry["sources"])
                break
        
        confidence, level = compute_confidence(term, sources)
        
        # Filter by confidence threshold
        if confidence >= config["confidence_threshold"]:
            results.append({
                "term": term,
                "confidence": confidence,
                "level": level,
                "sources": sources,
            })
    
    return results


def process_document(
    file_path: Path,
    config: dict[str, Any],
    negatives: set[str],
) -> dict[str, Any]:
    """Process a single document through the extraction pipeline.
    
    Args:
        file_path: Path to the document file
        config: Configuration dictionary
        negatives: Set of negative terms to filter
        
    Returns:
        Dictionary with extraction results
    """
    # Read file content
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return {
            "file": file_path.name,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(e),
        }
    
    if config["chunking_strategy"] == "heading":
        from plm.extraction.chunking.heading import HeadingChunker
        chunker = HeadingChunker(
            min_tokens=config["chunk_min_tokens"],
            max_tokens=config["chunk_max_tokens"],
        )
    else:
        chunker = get_chunker(config["chunking_strategy"])
    
    # Chunk the document
    chunks = chunker.chunk(text, file_path.name)
    
    # Process each chunk
    chunk_results = []
    total_high = 0
    total_medium = 0
    total_low = 0
    
    for chunk in chunks:
        terms = extract_terms_from_chunk(chunk.text, negatives, config)
        
        # Count by confidence level
        for term_data in terms:
            if term_data["level"] == "HIGH":
                total_high += 1
            elif term_data["level"] == "MEDIUM":
                total_medium += 1
            elif term_data["level"] == "LOW":
                total_low += 1
        
        chunk_results.append({
            "text": chunk.text,
            "chunk_index": chunk.index,
            "heading": chunk.heading,
            "terms": terms,
        })
    
    # Build result
    total_terms = total_high + total_medium + total_low
    result = {
        "file": file_path.name,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "chunks": chunk_results,
        "stats": {
            "total_chunks": len(chunks),
            "total_terms": total_terms,
            "high_confidence": total_high,
            "medium_confidence": total_medium,
            "low_confidence": total_low,
        },
    }
    
    return result


def write_output(
    result: dict[str, Any],
    output_path: Path,
    log_path: Path,
    dry_run: bool,
) -> None:
    """Write extraction results to JSON and log low-confidence terms.
    
    Args:
        result: Extraction result dictionary
        output_path: Path to output JSON file
        log_path: Path to low-confidence log file
        dry_run: If True, don't write files
    """
    if dry_run:
        print(f"[DRY RUN] Would write: {output_path}")
        return
    
    # Write main output JSON
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Wrote: {output_path}")
    except Exception as e:
        print(f"Error writing {output_path}: {e}", file=sys.stderr)
    
    # Log low-confidence terms
    try:
        with log_path.open("a", encoding="utf-8") as f:
            for chunk in result.get("chunks", []):
                for term_data in chunk.get("terms", []):
                    if term_data["level"] == "LOW":
                        log_entry = {
                            "file": result["file"],
                            "term": term_data["term"],
                            "confidence": term_data["confidence"],
                            "level": term_data["level"],
                            "context": chunk["text"][:200],  # First 200 chars
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing low-confidence log: {e}", file=sys.stderr)


def watch_loop(config: dict[str, Any]) -> None:
    """Main watch loop for processing documents.
    
    Args:
        config: Configuration dictionary
    """
    processed: set[Path] = set()
    
    # Load negatives vocabulary
    negatives: set[str] = set()
    try:
        negatives = load_negatives(config["vocab_negatives_path"])
        if negatives:
            print(f"Loaded {len(negatives)} negative terms")
    except Exception as e:
        print(f"Warning: Could not load negatives: {e}", file=sys.stderr)
    
    # Setup directories
    setup_directories(config)
    
    # Low-confidence log path
    low_conf_log = config["log_dir"] / "low_confidence.jsonl"
    
    print(f"Watching: {config['input_dir']}")
    print(f"Output: {config['output_dir']}")
    print(f"Chunking: {config['chunking_strategy']}")
    print(f"Process once: {config['process_once']}")
    print(f"Dry run: {config['dry_run']}")
    
    while not shutdown_requested:
        # Get unprocessed files
        new_files = get_input_files(config["input_dir"], processed)
        
        if new_files:
            print(f"\nFound {len(new_files)} new file(s)")
            
            for file_path in new_files:
                if shutdown_requested:
                    break
                
                print(f"Processing: {file_path.name}")
                
                # Process document
                result = process_document(file_path, config, negatives)
                
                # Write output
                output_path = config["output_dir"] / f"{file_path.stem}.json"
                write_output(result, output_path, low_conf_log, config["dry_run"])
                
                # Mark as processed
                processed.add(file_path)
                
                # Print stats
                if "stats" in result:
                    stats = result["stats"]
                    print(
                        f"  Terms: {stats['total_terms']} "
                        f"(H:{stats['high_confidence']}, "
                        f"M:{stats['medium_confidence']}, "
                        f"L:{stats['low_confidence']})"
                    )
        
        # Exit if process_once mode
        if config["process_once"]:
            print("\nProcess-once mode: exiting")
            break
        
        # Sleep before next poll
        if not shutdown_requested:
            time.sleep(config["poll_interval"])
    
    print("Shutdown complete")


def main() -> None:
    """Main entry point."""
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse configuration
    config = parse_env()
    config = parse_cli_args(config)
    
    # Run watch loop
    try:
        watch_loop(config)
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
