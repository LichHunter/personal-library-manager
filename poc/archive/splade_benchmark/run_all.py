#!/usr/bin/env python3
"""Run all SPLADE benchmark phases.

Usage:
    cd poc/splade_benchmark
    .venv/bin/python run_all.py
    
    # Skip index building (if already done)
    .venv/bin/python run_all.py --skip-index
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


POC_DIR = Path(__file__).parent
ARTIFACTS_DIR = POC_DIR / "artifacts"
SPLADE_INDEX_PATH = ARTIFACTS_DIR / "splade_index"


def run_script(script: str, args: list[str] = None) -> bool:
    """Run a Python script."""
    cmd = [sys.executable, script]
    if args:
        cmd.extend(args)
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, cwd=POC_DIR)
    return result.returncode == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SPLADE Benchmark")
    parser.add_argument("--skip-index", action="store_true", help="Skip index building")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip BM25 baseline")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip hybrid evaluation")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for encoding")
    
    args = parser.parse_args()
    
    print("SPLADE Benchmark - Full Pipeline")
    print("="*60)
    
    if not args.skip_baseline:
        if not (ARTIFACTS_DIR / "baseline_bm25.json").exists():
            print("\nPhase 2: BM25 Baseline")
            if not run_script("baseline_bm25.py"):
                print("ERROR: BM25 baseline failed")
                return 1
        else:
            print("\nPhase 2: BM25 Baseline (already exists, skipping)")
    
    if not args.skip_index:
        if not SPLADE_INDEX_PATH.exists():
            print("\nPhase 3: Build SPLADE Index")
            print("WARNING: This may take 60-90 minutes on CPU")
            if not run_script("splade_index.py", [f"--batch-size={args.batch_size}"]):
                print("ERROR: SPLADE index building failed")
                return 1
        else:
            print("\nPhase 3: SPLADE Index (already exists, skipping)")
    
    if not SPLADE_INDEX_PATH.exists():
        print("\nERROR: SPLADE index not found. Run without --skip-index")
        return 1
    
    print("\nPhase 4: SPLADE-only Benchmark")
    if not run_script("splade_benchmark.py"):
        print("ERROR: SPLADE benchmark failed")
        return 1
    
    if not args.skip_hybrid:
        print("\nPhase 5: Hybrid SPLADE+Semantic Benchmark")
        if not run_script("hybrid_splade.py"):
            print("ERROR: Hybrid benchmark failed")
            return 1
    
    print("\nPhase 6: Technical Term Analysis")
    if not run_script("technical_analysis.py", ["--include-informed"]):
        print("WARNING: Technical analysis failed (non-fatal)")
    
    print("\nPhase 9: Generate Visualizations")
    if not run_script("visualize.py"):
        print("WARNING: Visualization failed (non-fatal)")
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {ARTIFACTS_DIR}")
    print("\nArtifacts:")
    for f in sorted(ARTIFACTS_DIR.glob("*.json")):
        print(f"  - {f.name}")
    
    print(f"\nPlots saved to: {ARTIFACTS_DIR / 'plots'}")
    for f in sorted((ARTIFACTS_DIR / "plots").glob("*.png")):
        print(f"  - {f.name}")
    
    print("\nSee RESULTS.md for analysis template")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
