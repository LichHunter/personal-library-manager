#!/usr/bin/env python3
"""Download all models required for the benchmark.

This script pre-downloads all embedding models, rerankers, and Ollama LLMs
so the benchmark can run without network delays.

Usage:
    python download_models.py                    # Use default config.yaml
    python download_models.py --config config_full.yaml
    python download_models.py --embedders-only   # Only download embedders
    python download_models.py --ollama-only      # Only pull Ollama models
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml


def log(msg: str, level: str = "INFO"):
    """Simple timestamped logging."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {level}: {msg}", flush=True)


def download_sentence_transformer(model_name: str) -> bool:
    """Download a sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        log(f"Downloading embedder: {model_name}")
        _ = SentenceTransformer(model_name)
        log(f"  OK: {model_name}")
        return True
    except Exception as e:
        log(f"  FAILED: {model_name} - {e}", level="ERROR")
        return False


def download_cross_encoder(model_name: str) -> bool:
    """Download a cross-encoder model."""
    try:
        from sentence_transformers import CrossEncoder
        log(f"Downloading reranker: {model_name}")
        _ = CrossEncoder(model_name)
        log(f"  OK: {model_name}")
        return True
    except Exception as e:
        log(f"  FAILED: {model_name} - {e}", level="ERROR")
        return False


def pull_ollama_model(model_name: str) -> bool:
    """Pull an Ollama model."""
    try:
        log(f"Pulling Ollama model: {model_name}")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes max
        )
        if result.returncode == 0:
            log(f"  OK: {model_name}")
            return True
        else:
            log(f"  FAILED: {model_name} - {result.stderr}", level="ERROR")
            return False
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT: {model_name}", level="ERROR")
        return False
    except FileNotFoundError:
        log(f"  FAILED: ollama not found in PATH", level="ERROR")
        return False
    except Exception as e:
        log(f"  FAILED: {model_name} - {e}", level="ERROR")
        return False


def check_ollama_running() -> bool:
    """Check if Ollama is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models for benchmark")
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--embedders-only",
        action="store_true",
        help="Only download embedding models"
    )
    parser.add_argument(
        "--rerankers-only",
        action="store_true",
        help="Only download reranker models"
    )
    parser.add_argument(
        "--ollama-only",
        action="store_true",
        help="Only pull Ollama models"
    )
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    config_path = base_dir / args.config
    
    if not config_path.exists():
        log(f"Config file not found: {config_path}", level="ERROR")
        sys.exit(1)
    
    log(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Determine what to download
    download_embedders = not args.rerankers_only and not args.ollama_only
    download_rerankers = not args.embedders_only and not args.ollama_only
    download_ollama = not args.embedders_only and not args.rerankers_only
    
    results = {"success": [], "failed": []}
    
    # Download embedders
    if download_embedders:
        log("\n" + "="*60)
        log("DOWNLOADING EMBEDDING MODELS")
        log("="*60)
        
        embedders = [e for e in config.get("embedding_models", []) if e.get("enabled", True)]
        for emb in embedders:
            if download_sentence_transformer(emb["name"]):
                results["success"].append(f"embedder:{emb['name']}")
            else:
                results["failed"].append(f"embedder:{emb['name']}")
    
    # Download rerankers
    if download_rerankers:
        log("\n" + "="*60)
        log("DOWNLOADING RERANKER MODELS")
        log("="*60)
        
        rerankers = [r for r in config.get("reranker_models", []) if r.get("enabled", True)]
        for reranker in rerankers:
            if download_cross_encoder(reranker["name"]):
                results["success"].append(f"reranker:{reranker['name']}")
            else:
                results["failed"].append(f"reranker:{reranker['name']}")
    
    # Pull Ollama models
    if download_ollama:
        log("\n" + "="*60)
        log("PULLING OLLAMA MODELS")
        log("="*60)
        
        if not check_ollama_running():
            log("Ollama is not running. Please start it with: ollama serve", level="WARNING")
            log("Skipping Ollama model downloads")
        else:
            llms = [l for l in config.get("llm_models", []) if l.get("enabled", True)]
            for llm in llms:
                if pull_ollama_model(llm["name"]):
                    results["success"].append(f"ollama:{llm['name']}")
                else:
                    results["failed"].append(f"ollama:{llm['name']}")
    
    # Summary
    log("\n" + "="*60)
    log("DOWNLOAD SUMMARY")
    log("="*60)
    log(f"Successful: {len(results['success'])}")
    for item in results["success"]:
        log(f"  OK: {item}")
    
    if results["failed"]:
        log(f"Failed: {len(results['failed'])}", level="WARNING")
        for item in results["failed"]:
            log(f"  FAILED: {item}", level="WARNING")
        sys.exit(1)
    else:
        log("\nAll models downloaded successfully!")
        log("You can now run the benchmark with:")
        log(f"  python run_benchmark.py --config {args.config}")


if __name__ == "__main__":
    main()
