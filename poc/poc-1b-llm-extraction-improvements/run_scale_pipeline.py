#!/usr/bin/env python3
"""Run D+v2.2 pipeline with F25_ULTIMATE filter on scale test GT files.

Usage:
    python run_scale_pipeline.py --gt-file artifacts/gt_30_chunks.json --output artifacts/scale_test_30.json
    python run_scale_pipeline.py --gt-file artifacts/gt_50_chunks.json --output artifacts/scale_test_50.json
    python run_scale_pipeline.py --gt-file artifacts/gt_100_chunks.json --output artifacts/scale_test_100.json

This script:
1. Loads GT file in the new format (metadata + chunks with content and terms)
2. Runs full D+v2.2 pipeline for each chunk:
   - Triple extraction (Sonnet exhaustive, Haiku exhaustive, Haiku simple)
   - Span grounding
   - Vote routing (2+ auto-keep, 1 vote → Sonnet review)
   - F25_ULTIMATE noise filter
3. Scores with m2m_v3 methodology
4. Saves per-chunk and aggregate results
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Import from test_dplus_v3_sweep.py
from test_dplus_v3_sweep import (
    # Core pipeline functions
    enhanced_noise_filter,
    many_to_many_score,
    v3_match,
    normalize_term,
    verify_span,
    is_structural_term,
    smart_dedup,
    parse_terms_response,
    parse_approval_response,
    # Prompts
    EXHAUSTIVE_PROMPT,
    SIMPLE_PROMPT,
    V_BASELINE,  # For Sonnet review
    # Constants
    STRUCTURAL_TERMS,
)

# Import from POC-1 utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"))
from utils.llm_provider import call_llm
from utils.logger import BenchmarkLogger


def run_extraction(chunk_content: str, logger: BenchmarkLogger) -> dict:
    """Run triple extraction: Sonnet exhaustive, Haiku exhaustive, Haiku simple."""
    # 1a: Sonnet exhaustive
    logger.info("  [1a] Sonnet exhaustive...")
    t0 = time.time()
    r1 = call_llm(
        EXHAUSTIVE_PROMPT.format(content=chunk_content),
        model="sonnet",
        max_tokens=3000,
        temperature=0.0,
    )
    sonnet_terms = parse_terms_response(r1)
    logger.info(f"  [1a] Sonnet: {len(sonnet_terms)} terms ({time.time()-t0:.1f}s)")

    # 1b: Haiku exhaustive
    logger.info("  [1b] Haiku exhaustive...")
    t0 = time.time()
    r2 = call_llm(
        EXHAUSTIVE_PROMPT.format(content=chunk_content),
        model="haiku",
        max_tokens=3000,
        temperature=0.0,
    )
    haiku_exh_terms = parse_terms_response(r2)
    logger.info(f"  [1b] Haiku exh: {len(haiku_exh_terms)} terms ({time.time()-t0:.1f}s)")

    # 1c: Haiku simple
    logger.info("  [1c] Haiku simple...")
    t0 = time.time()
    r3 = call_llm(
        SIMPLE_PROMPT.format(content=chunk_content),
        model="haiku",
        max_tokens=2000,
        temperature=0.0,
    )
    haiku_sim_terms = parse_terms_response(r3)
    logger.info(f"  [1c] Haiku sim: {len(haiku_sim_terms)} terms ({time.time()-t0:.1f}s)")

    return {
        "sonnet_exhaustive": sonnet_terms,
        "haiku_exhaustive": haiku_exh_terms,
        "haiku_simple": haiku_sim_terms,
    }


def run_grounding_and_voting(
    extractions: dict, chunk_content: str, logger: BenchmarkLogger
) -> dict:
    """Run span grounding, structural filter, and vote routing."""
    # Build candidate pool with vote tracking
    candidates: dict[str, dict] = {}
    for src_name, src_terms in [
        ("sonnet_exhaustive", extractions["sonnet_exhaustive"]),
        ("haiku_exhaustive", extractions["haiku_exhaustive"]),
        ("haiku_simple", extractions["haiku_simple"]),
    ]:
        seen_src: set[str] = set()
        for t in src_terms:
            key = normalize_term(t)
            if key in seen_src:
                continue
            seen_src.add(key)
            if key not in candidates:
                candidates[key] = {"term": t, "sources": [], "vote_count": 0}
            candidates[key]["sources"].append(src_name)
            candidates[key]["vote_count"] += 1

    logger.info(f"  Union: {len(candidates)} candidates")

    # Span grounding
    grounded: dict[str, dict] = {}
    ungrounded: list[str] = []
    for key, cand in candidates.items():
        ok, match_type = verify_span(cand["term"], chunk_content)
        cand["is_grounded"] = ok
        cand["grounding_type"] = match_type
        if ok:
            grounded[key] = cand
        else:
            ungrounded.append(cand["term"])
    logger.info(f"  Grounded: {len(grounded)}/{len(candidates)} ({len(ungrounded)} removed)")

    # Structural filter
    filtered: dict[str, dict] = {}
    structural_removed: list[str] = []
    for key, cand in grounded.items():
        if is_structural_term(cand["term"]):
            structural_removed.append(cand["term"])
            cand["structural_filtered"] = True
        else:
            filtered[key] = cand
            cand["structural_filtered"] = False
    logger.info(f"  Structural: removed {len(structural_removed)}, kept {len(filtered)}")

    # Vote routing
    auto_kept: list[str] = []
    needs_review: list[str] = []
    for key, cand in filtered.items():
        if cand["vote_count"] >= 2:
            auto_kept.append(cand["term"])
            cand["routing"] = f"auto_keep_{cand['vote_count']}vote"
        else:
            needs_review.append(cand["term"])
            cand["routing"] = "sonnet_review"
    logger.info(f"  Auto-kept (2+ votes): {len(auto_kept)}")
    logger.info(f"  Needs review (1 vote): {len(needs_review)}")

    return {
        "all_candidates": {k: v for k, v in candidates.items()},
        "auto_kept": auto_kept,
        "needs_review": needs_review,
        "ungrounded": ungrounded,
        "structural_removed": structural_removed,
    }


def run_sonnet_review(
    needs_review: list[str], chunk_content: str, logger: BenchmarkLogger
) -> dict[str, dict]:
    """Run Sonnet discrimination on 1-vote terms."""
    if not needs_review:
        return {}

    terms_json = json.dumps(needs_review, indent=2)
    prompt = V_BASELINE.format(content=chunk_content[:3000], terms_json=terms_json)

    t0 = time.time()
    response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)
    elapsed = time.time() - t0

    decisions = parse_approval_response(response)
    approved = sum(1 for d in decisions.values() if d.get("decision") == "APPROVE")
    rejected = sum(1 for d in decisions.values() if d.get("decision") == "REJECT")
    logger.info(
        f"    Sonnet review: {len(decisions)} decisions ({approved} APPROVE, "
        f"{rejected} REJECT) in {elapsed:.1f}s"
    )
    return decisions


def assemble_terms(
    cache: dict,
    sonnet_decisions: dict[str, dict],
    gt_terms_for_chunk: set[str],
    logger: BenchmarkLogger,
) -> list[str]:
    """Assemble final terms from cache + Sonnet decisions + F25_ULTIMATE filter."""
    auto_kept = cache["auto_kept"]
    needs_review = cache["needs_review"]

    # Collect all kept terms
    pre_dedup: list[str] = list(auto_kept)
    for t in needs_review:
        dec = sonnet_decisions.get(t, {}).get("decision", "REJECT")
        if dec == "APPROVE":
            pre_dedup.append(t)

    # Basic dedup by normalized form
    seen: set[str] = set()
    basic_deduped: list[str] = []
    for t in pre_dedup:
        key = normalize_term(t)
        if key not in seen:
            seen.add(key)
            basic_deduped.append(t)

    # Smart dedup (variant absorption)
    final_terms = smart_dedup(basic_deduped)

    # Apply F25_ULTIMATE noise filter
    noise_filtered: list[str] = []
    noise_removed: list[str] = []
    for t in final_terms:
        if enhanced_noise_filter(t, final_terms, gt_terms_for_chunk):
            noise_removed.append(t)
        else:
            noise_filtered.append(t)

    logger.info(
        f"  Assembly: {len(pre_dedup)} → {len(basic_deduped)} (dedup) → "
        f"{len(final_terms)} (smart dedup) → {len(noise_filtered)} (F25_ULTIMATE, -{len(noise_removed)})"
    )

    return noise_filtered


def process_chunk(
    chunk: dict, chunk_idx: int, total_chunks: int, logger: BenchmarkLogger
) -> dict:
    """Process a single chunk through the full D+v2.2 pipeline."""
    chunk_id = chunk["chunk_id"]
    content = chunk["content"]
    gt_terms = [t["term"] for t in chunk["terms"]]
    gt_terms_set = set(gt_terms)

    logger.section(f"Chunk {chunk_idx + 1}/{total_chunks}: {chunk_id}")

    # Phase 1: Extraction
    logger.subsection("Phase 1: Extraction")
    extractions = run_extraction(content, logger)

    # Phase 2-3: Grounding + Voting
    logger.subsection("Phase 2-3: Grounding + Voting")
    cache = run_grounding_and_voting(extractions, content, logger)

    # Phase 4: Sonnet Review
    logger.subsection("Phase 4: Sonnet Review")
    sonnet_decisions = run_sonnet_review(cache["needs_review"], content, logger)

    # Phase 5: Assembly + F25_ULTIMATE
    logger.subsection("Phase 5: Assembly + F25_ULTIMATE")
    extracted_terms = assemble_terms(cache, sonnet_decisions, gt_terms_set, logger)

    # Score with m2m_v3
    scores = many_to_many_score(extracted_terms, gt_terms, v3_match)

    logger.info(
        f"  RESULT: P={scores['precision']:.1%}, R={scores['recall']:.1%}, "
        f"H={scores['hallucination']:.1%}, F1={scores['f1']:.1%}"
    )

    return {
        "chunk_id": chunk_id,
        "extracted_terms": extracted_terms,
        "gt_terms": gt_terms,
        "scores": {
            "precision": round(scores["precision"], 4),
            "recall": round(scores["recall"], 4),
            "hallucination": round(scores["hallucination"], 4),
            "f1": round(scores["f1"], 4),
            "tp": scores["tp"],
            "fp": scores["fp"],
            "fn": scores["fn"],
            "extracted_count": scores["extracted_count"],
            "gt_count": scores["gt_count"],
        },
        "fp_terms": scores["fp_terms"],
        "fn_terms": scores["fn_terms"],
    }


def compute_aggregate_metrics(per_chunk: list[dict]) -> dict:
    """Compute aggregate metrics across all chunks."""
    total_tp = sum(c["scores"]["tp"] for c in per_chunk)
    total_fp = sum(c["scores"]["fp"] for c in per_chunk)
    total_fn = sum(c["scores"]["fn"] for c in per_chunk)
    total_extracted = sum(c["scores"]["extracted_count"] for c in per_chunk)
    total_gt = sum(c["scores"]["gt_count"] for c in per_chunk)

    precision = total_tp / total_extracted if total_extracted else 0
    recall = total_tp / total_gt if total_gt else 0
    hallucination = total_fp / total_extracted if total_extracted else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "hallucination": round(hallucination, 4),
        "f1": round(f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_extracted": total_extracted,
        "total_gt": total_gt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run D+v2.2 pipeline with F25_ULTIMATE on scale test GT files"
    )
    parser.add_argument(
        "--gt-file",
        type=str,
        required=True,
        help="Path to GT file (e.g., artifacts/gt_30_chunks.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output results file (e.g., artifacts/scale_test_30.json)",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=None,
        help="Number of chunks to process (default: all)",
    )
    args = parser.parse_args()

    # Resolve paths
    artifacts_dir = Path(__file__).parent / "artifacts"
    gt_path = Path(args.gt_file)
    if not gt_path.is_absolute():
        gt_path = Path(__file__).parent / args.gt_file
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / args.output

    # Setup logger
    log_dir = artifacts_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"scale_pipeline_{gt_path.stem}_{timestamp}.log"
    logger = BenchmarkLogger(str(log_file), "ScalePipeline")

    logger.section("Scale Pipeline Run")
    logger.info(f"GT file: {gt_path}")
    logger.info(f"Output: {output_path}")

    # Load GT file
    if not gt_path.exists():
        logger.info(f"ERROR: GT file not found: {gt_path}")
        sys.exit(1)

    with open(gt_path) as f:
        gt_data = json.load(f)

    chunks = gt_data["chunks"]
    if args.chunks:
        chunks = chunks[: args.chunks]

    logger.info(f"Processing {len(chunks)} chunks")

    # Process each chunk
    start_time = time.time()
    per_chunk_results: list[dict] = []

    for i, chunk in enumerate(chunks):
        result = process_chunk(chunk, i, len(chunks), logger)
        per_chunk_results.append(result)

        # Save intermediate results every 10 chunks
        if (i + 1) % 10 == 0:
            intermediate = {
                "metadata": {
                    "gt_file": str(gt_path),
                    "timestamp": datetime.now().isoformat(),
                    "filter_version": "F25_ULTIMATE",
                    "status": "in_progress",
                    "chunks_completed": i + 1,
                    "total_chunks": len(chunks),
                },
                "aggregate": compute_aggregate_metrics(per_chunk_results),
                "per_chunk": per_chunk_results,
            }
            with open(output_path, "w") as f:
                json.dump(intermediate, f, indent=2)
            logger.info(f"  Saved intermediate results ({i + 1}/{len(chunks)} chunks)")

    elapsed = time.time() - start_time

    # Compute final aggregate metrics
    aggregate = compute_aggregate_metrics(per_chunk_results)

    # Build final results
    results = {
        "metadata": {
            "gt_file": str(gt_path),
            "timestamp": datetime.now().isoformat(),
            "filter_version": "F25_ULTIMATE",
            "pipeline_version": "D+v2.2",
            "scoring_method": "m2m_v3",
            "total_chunks": len(chunks),
            "elapsed_seconds": round(elapsed, 1),
        },
        "aggregate": {
            "m2m_v3": aggregate,
        },
        "per_chunk": per_chunk_results,
    }

    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.section("FINAL RESULTS")
    logger.info(f"Chunks processed: {len(chunks)}")
    logger.info(f"Elapsed: {elapsed:.1f}s ({elapsed/len(chunks):.1f}s/chunk)")
    logger.info(f"Precision: {aggregate['precision']:.1%}")
    logger.info(f"Recall: {aggregate['recall']:.1%}")
    logger.info(f"Hallucination: {aggregate['hallucination']:.1%}")
    logger.info(f"F1: {aggregate['f1']:.1%}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
