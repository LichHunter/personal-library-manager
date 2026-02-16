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
    load_auto_vocab,
    extract_seeds,
    get_bypass_set,
    get_seeds_set,
    get_contextual_seeds_set,
    load_term_index,
    AutoVocab,
    TermInfo,
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
        "vocab_term_index_path": Path(
            os.getenv("VOCAB_TERM_INDEX_PATH", "/data/vocabularies/term_index.json")
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
    seeds: list[str],
    contextual_seeds: list[str],
    bypass: set[str],
    seeds_set: set[str],
    contextual_seeds_set: set[str],
    term_index: dict[str, TermInfo],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract terms from a single chunk using V6 pipeline."""
    # Stage 1a: Taxonomy extraction (LLM)
    taxonomy_terms = extract_by_taxonomy(chunk_text, model=config["llm_model"])
    
    # Stage 1b: Candidate verify (heuristic + LLM)
    heuristic_candidates = extract_candidates_heuristic(chunk_text)
    verified_candidates = classify_candidates_llm(
        heuristic_candidates,
        chunk_text,
        model=config["llm_model"],
    )
    
    # Stage 1c: Seed extraction (zero-cost, high-recall)
    seed_terms = extract_seeds(chunk_text, seeds)
    contextual_seed_terms = extract_seeds(chunk_text, contextual_seeds)
    
    # Stage 2: Ground candidates (merge by source)
    candidates_by_source = {
        "taxonomy": taxonomy_terms,
        "heuristic": verified_candidates,
        "seeds": seed_terms,
        "contextual_seeds": contextual_seed_terms,
    }
    grounded = ground_candidates(candidates_by_source, chunk_text)
    
    # Stage 3: Filter noise
    after_noise: dict[str, dict] = {}
    for key, cand in grounded.items():
        term = cand["term"]
        grounded_terms_single = [term]
        filtered_single = filter_noise(grounded_terms_single, negatives, bypass=bypass)
        if filtered_single:
            after_noise[key] = cand
    
    # Stage 4: V6 confidence tier routing
    llm_sources = {"taxonomy", "heuristic"}
    high_confidence: list[str] = []
    needs_validation: list[str] = []
    
    protected_set = bypass | seeds_set | contextual_seeds_set
    
    for key, cand in after_noise.items():
        term = cand["term"]
        sources_set = cand["sources"]
        source_count = cand["source_count"]
        tl = term.lower().strip()
        
        info = term_index.get(tl)
        entity_ratio = info["entity_ratio"] if info else 0.5
        
        has_llm_vote = bool(sources_set & llm_sources)
        
        # HIGH: seed bypass (term in seeds with seeds source)
        if tl in seeds_set and "seeds" in sources_set:
            if source_count >= 1:
                high_confidence.append(term)
                continue
        
        # HIGH: contextual seed with multiple sources
        if tl in contextual_seeds_set and "contextual_seeds" in sources_set:
            if source_count >= 2:
                high_confidence.append(term)
                continue
        
        # HIGH: structural pattern with good entity_ratio
        if _has_structural_pattern(term):
            if entity_ratio > 0 or tl in seeds_set or tl in bypass:
                high_confidence.append(term)
            else:
                needs_validation.append(term)
            continue
        
        # HIGH: high entity_ratio (≥0.8)
        if entity_ratio >= 0.8:
            high_confidence.append(term)
            continue
        
        # HIGH: multiple sources including LLM
        if source_count >= 2 and has_llm_vote:
            high_confidence.append(term)
            continue
        
        # MEDIUM: single LLM vote or moderate entity_ratio - needs validation
        if has_llm_vote or entity_ratio >= 0.5:
            needs_validation.append(term)
            continue
        
        # Otherwise skip (low confidence)
    
    # Stage 5: Validate medium-confidence terms
    validated = validate_terms(
        needs_validation, chunk_text, model=config["llm_model"], bypass=bypass
    )
    all_terms = high_confidence + validated
    
    # Stage 6: Postprocess
    expanded = expand_spans(all_terms, chunk_text)
    suppressed = suppress_subspans(expanded, protected=protected_set)
    final_terms = final_dedup(suppressed)
    
    # Compute confidence for each term
    results = []
    for term in final_terms:
        sources: list[str] = []
        for key, entry in grounded.items():
            if entry["term"] == term or entry["term"].lower() == term.lower():
                sources = list(entry["sources"])
                break
        
        tl = term.lower()
        info = term_index.get(tl)
        entity_ratio = info["entity_ratio"] if info else 0.5
        is_bypass_term = tl in bypass or tl in seeds_set
        
        confidence, level = compute_confidence(
            term, sources, entity_ratio=entity_ratio, is_bypass=is_bypass_term
        )
        
        if confidence >= config["confidence_threshold"]:
            results.append({
                "term": term,
                "confidence": confidence,
                "level": level,
                "sources": sources,
            })
    
    return results


def _has_structural_pattern(term: str) -> bool:
    """Check if term has structural code patterns (CamelCase, dots, parens, etc.)."""
    import re
    if re.match(r"^[A-Z][a-z]+[A-Z]", term):
        return True
    if "." in term or "(" in term or "::" in term:
        return True
    if re.match(r"^[A-Z][A-Z0-9_]+$", term) and len(term) >= 2:
        return True
    return False


def process_document(
    file_path: Path,
    config: dict[str, Any],
    negatives: set[str],
    seeds: list[str],
    contextual_seeds: list[str],
    bypass: set[str],
    seeds_set: set[str],
    contextual_seeds_set: set[str],
    term_index: dict[str, TermInfo],
) -> dict[str, Any]:
    """Process a single document through the extraction pipeline."""
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
    
    chunks = chunker.chunk(text, file_path.name)
    
    chunk_results = []
    total_high = 0
    total_medium = 0
    total_low = 0
    
    for chunk in chunks:
        terms = extract_terms_from_chunk(
            chunk.text, negatives, seeds, contextual_seeds, bypass,
            seeds_set, contextual_seeds_set, term_index, config
        )
        
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
    
    negatives: set[str] = set()
    try:
        negatives = load_negatives(config["vocab_negatives_path"])
        if negatives:
            print(f"Loaded {len(negatives)} negative terms")
    except Exception as e:
        print(f"Warning: Could not load negatives: {e}", file=sys.stderr)
    
    seeds: list[str] = []
    contextual_seeds: list[str] = []
    bypass: set[str] = set()
    seeds_set: set[str] = set()
    contextual_seeds_set: set[str] = set()
    try:
        vocab = load_auto_vocab(config["vocab_seeds_path"])
        seeds = vocab["seeds"]
        contextual_seeds = vocab["contextual_seeds"]
        bypass = get_bypass_set(vocab)
        seeds_set = get_seeds_set(vocab)
        contextual_seeds_set = get_contextual_seeds_set(vocab)
        print(f"Loaded {len(seeds)} seed terms, {len(bypass)} bypass terms, {len(contextual_seeds)} contextual seeds")
    except Exception as e:
        print(f"Warning: Could not load seeds vocabulary: {e}", file=sys.stderr)
    
    term_index: dict[str, TermInfo] = {}
    try:
        term_index = load_term_index(config["vocab_term_index_path"])
        print(f"Loaded term index with {len(term_index)} terms")
    except Exception as e:
        print(f"Warning: Could not load term index: {e}", file=sys.stderr)
    
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
                
                result = process_document(
                    file_path, config, negatives, seeds, contextual_seeds,
                    bypass, seeds_set, contextual_seeds_set, term_index
                )
                
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
