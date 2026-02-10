#!/usr/bin/env python3
"""Scale test runner for LLM extraction evaluation at scale.

This module provides utilities for:
1. Random chunk sampling with reproducibility
2. Ground truth generation with checkpointing (API failure recovery)
3. Full pipeline execution with comprehensive audit logging

Usage:
    from scale_test_runner import sample_chunks, generate_gt_with_checkpointing, run_pipeline_with_logging

    # Sample 30 random chunks reproducibly
    chunks = sample_chunks(all_chunks, 30, seed=42)

    # Generate GT with checkpoint recovery
    generate_gt_with_checkpointing(chunks, output_path, checkpoint_dir)

    # Run full pipeline with audit trail
    run_pipeline_with_logging(gt_path, output_path, audit_dir)
"""

import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Import LLM provider from POC-1
sys.path.insert(0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"))
from utils.llm_provider import call_llm
from utils.logger import BenchmarkLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Rate limiting between API calls (seconds)
API_RATE_LIMIT_DELAY = 1.0

# Default model for GT generation
GT_MODEL = "opus"

# Cost tracking (approximate per million tokens)
MODEL_COSTS = {
    "claude-opus-4-5-20251101": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-latest": {"input": 0.25, "output": 1.25},
}

# ============================================================================
# OPUS GT PROMPT (from expand_ground_truth.py)
# ============================================================================

OPUS_GT_PROMPT = """You are building a gold-standard ground truth for evaluating term extraction systems. Your output will be used as the DEFINITIVE answer key - completeness and accuracy are paramount.

DOCUMENTATION CHUNK:
---
{chunk_content}
---

TASK: Extract EVERY technical term, concept, and domain-specific vocabulary from this chunk. Be EXHAUSTIVE - a missed term is a failure mode we cannot recover from.

TIER DEFINITIONS:
- Tier 1: Core domain resources, components, and proper nouns (e.g., "Pod", "kubelet", "etcd", "ReplicaSet", "PersistentVolume", "ConfigMap")
  -> Named K8s/infrastructure resources, API objects, CLI tools, specific component names
- Tier 2: Domain concepts, processes, and technical vocabulary (e.g., "control plane", "scheduling", "container", "namespace", "cluster", "authorization", "node")
  -> Technical concepts, architectural patterns, processes, roles, and operational terms
- Tier 3: Contextual/supporting technical terms (e.g., "HTTPS", "API", "Linux", "OOM", "host", "memory")
  -> General technical vocabulary that carries meaning in this context, protocols, OS concepts

WHAT TO EXTRACT:
- Named resources and API objects (Pod, Deployment, ConfigMap, PVC)
- Infrastructure components (kubelet, kube-apiserver, etcd, containerd)
- Domain concepts (control plane, scheduling, container orchestration, fault tolerance)
- Technical processes (garbage collection, reconciliation, rolling update)
- CLI flags and commands (--cloud-provider, kubectl)
- Protocols and standards (HTTPS, gRPC, CNI, CSI)
- Abbreviations and acronyms (PV, PVC, OOM, VM, CEL)
- Security/networking concepts when used in context (authorization, bearer token, TLS, ingress)
- Linux/OS concepts when relevant (cgroups, namespace, process, memory)
- Architectural terms (high availability, fault tolerance, load balancing)
- Feature gates and API versions (alpha, beta, stable - when describing feature lifecycle)

WHAT NOT TO EXTRACT:
- Structural/formatting words (title, section, overview, content, weight, description)
- YAML/Markdown syntax (---, ```, true, false)
- Common English words used non-technically (the, each, several, manages)
- Hugo/templating syntax (glossary_tooltip, feature-state)

RULES:
1. Every term MUST appear verbatim in the chunk text (exact string match, case-insensitive)
2. Extract the term as it appears in the text - don't normalize or paraphrase
3. If a term appears in both singular and plural forms, include the more common form
4. Include compound terms ("control plane", "worker node", "container runtime") AND their key individual components if independently meaningful
5. Err on the side of INCLUSION - it's far better to have borderline terms than to miss valid ones
6. A term used generically in English but carrying specific technical meaning in this context IS a valid term

OUTPUT FORMAT (strict JSON):
{{
  "terms": [
    {{"term": "exact term from text", "tier": 1, "reasoning": "Why this is a valid term (1 sentence)"}},
    {{"term": "another term", "tier": 2, "reasoning": "Why included"}}
  ]
}}

Be EXHAUSTIVE. Aim for 15-40 terms per chunk depending on content density. Missing a valid term is worse than including a borderline one.
"""


# ============================================================================
# SPAN VERIFICATION (from expand_ground_truth.py)
# ============================================================================


def verify_term_in_text(term: str, text: str) -> dict:
    """Verify a term exists in the chunk text. Returns match info."""
    text_lower = text.lower()
    term_lower = term.lower()

    # Exact match (case-insensitive)
    if term_lower in text_lower:
        return {"grounded": True, "match_type": "exact", "term": term}

    # Normalized match (- and _ as spaces)
    term_norm = term_lower.replace("-", " ").replace("_", " ")
    text_norm = text_lower.replace("-", " ").replace("_", " ")
    if term_norm in text_norm:
        return {"grounded": True, "match_type": "normalized", "term": term}

    # CamelCase split match
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in text_lower:
        return {"grounded": True, "match_type": "camelcase", "term": term}

    # Singular/plural match
    if term_lower.endswith("s") and term_lower[:-1] in text_lower:
        return {"grounded": True, "match_type": "singular_of_plural", "term": term}
    if not term_lower.endswith("s") and (term_lower + "s") in text_lower:
        return {"grounded": True, "match_type": "plural_of_singular", "term": term}
    if term_lower.endswith("es") and term_lower[:-2] in text_lower:
        return {"grounded": True, "match_type": "singular_of_plural_es", "term": term}

    return {"grounded": False, "match_type": "none", "term": term}


# ============================================================================
# RANDOM SAMPLING
# ============================================================================


def sample_chunks(
    pool: list[Any],
    n: int,
    seed: int = 42,
) -> list[Any]:
    """Reproducibly sample n random chunks from pool.

    Args:
        pool: List of chunks to sample from
        n: Number of chunks to sample
        seed: Random seed for reproducibility (default: 42)

    Returns:
        List of n randomly sampled chunks (preserves order relative to pool)

    Example:
        >>> s1 = sample_chunks(list(range(1000)), 30, seed=42)
        >>> s2 = sample_chunks(list(range(1000)), 30, seed=42)
        >>> assert s1 == s2, "Sampling not reproducible!"
    """
    if n >= len(pool):
        return list(pool)

    # Use seeded random.Random instance for reproducibility
    rng = random.Random(seed)

    # Sample indices and sort to preserve relative order
    indices = sorted(rng.sample(range(len(pool)), n))

    return [pool[i] for i in indices]


# ============================================================================
# GROUND TRUTH GENERATION WITH CHECKPOINTING
# ============================================================================


def generate_gt_with_checkpointing(
    chunks: list[dict],
    output_path: Path,
    checkpoint_dir: Path,
    model: str = GT_MODEL,
    logger: Optional[BenchmarkLogger] = None,
    seed: int = 42,
) -> dict:
    """Generate ground truth for chunks with checkpoint recovery.

    Saves checkpoint after EACH chunk to enable resume on API failure.
    Stores random seed and model version in all metadata.

    Args:
        chunks: List of chunk dicts with 'chunk_id' and 'content' keys
        output_path: Path to save final GT JSON
        checkpoint_dir: Directory for checkpoint files
        model: LLM model to use for GT generation
        logger: Optional BenchmarkLogger for logging
        seed: Random seed used for sampling (stored in metadata)

    Returns:
        GT data dict with metadata and chunks
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if logger is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = BenchmarkLogger(
            log_dir=LOG_DIR,
            log_file=f"gt_generation_{timestamp}.log",
            console=True,
            min_level="INFO",
        )

    logger.section("Ground Truth Generation with Checkpointing")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Model: {model}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}")

    # Check for existing checkpoint
    checkpoint_file = checkpoint_dir / "gt_checkpoint.json"
    completed_chunks: list[dict] = []
    completed_ids: set[str] = set()
    api_calls: list[dict] = []
    total_cost = 0.0

    if checkpoint_file.exists():
        logger.info(f"Found checkpoint: {checkpoint_file}")
        with open(checkpoint_file) as f:
            checkpoint_data = json.load(f)
        completed_chunks = checkpoint_data.get("chunks", [])
        completed_ids = {c["chunk_id"] for c in completed_chunks}
        api_calls = checkpoint_data.get("api_calls", [])
        total_cost = checkpoint_data.get("total_cost", 0.0)
        logger.info(f"Resuming from checkpoint: {len(completed_chunks)} chunks done")

    # Process remaining chunks
    remaining = [c for c in chunks if c["chunk_id"] not in completed_ids]
    logger.info(f"Remaining chunks to process: {len(remaining)}")

    for i, chunk in enumerate(remaining):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]

        logger.section(
            f"Chunk {len(completed_chunks) + 1}/{len(chunks)}: {chunk_id}"
        )
        logger.info(f"  Content length: {len(content)}")

        # Rate limiting
        if i > 0:
            logger.info(f"  Rate limiting: sleeping {API_RATE_LIMIT_DELAY}s...")
            time.sleep(API_RATE_LIMIT_DELAY)

        # Call LLM for term extraction
        prompt = OPUS_GT_PROMPT.format(chunk_content=content)

        call_start = time.time()
        response = call_llm(
            prompt=prompt,
            model=model,
            timeout=120,
            max_tokens=4000,
            temperature=0.0,
        )
        call_elapsed = time.time() - call_start

        # Track API call
        api_call = {
            "chunk_id": chunk_id,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": call_elapsed,
            "prompt_length": len(prompt),
            "response_length": len(response) if response else 0,
            "success": bool(response),
        }
        api_calls.append(api_call)

        if not response:
            logger.error(f"  ERROR: Empty response from {model} for {chunk_id}")
            # Save checkpoint even on error
            _save_checkpoint(
                checkpoint_file, completed_chunks, api_calls, total_cost, seed, model
            )
            continue

        logger.info(f"  LLM responded in {call_elapsed:.1f}s")

        # Parse response
        terms_data = _parse_llm_response(response, chunk_id, logger)
        if terms_data is None:
            _save_checkpoint(
                checkpoint_file, completed_chunks, api_calls, total_cost, seed, model
            )
            continue

        # Span verification
        grounded_terms = []
        ungrounded_terms = []
        for t in terms_data:
            result = verify_term_in_text(t["term"], content)
            if result["grounded"]:
                grounded_terms.append({"term": t["term"], "tier": t["tier"]})
            else:
                ungrounded_terms.append(t["term"])

        if ungrounded_terms:
            logger.warn(
                f"  Removed {len(ungrounded_terms)} ungrounded terms: {ungrounded_terms[:5]}{'...' if len(ungrounded_terms) > 5 else ''}"
            )

        logger.info(f"  Grounded terms: {len(grounded_terms)}")

        # Build chunk result
        chunk_result = {
            "chunk_id": chunk_id,
            "content": content,
            "terms": grounded_terms,
            "term_count": len(grounded_terms),
            "ungrounded_removed": len(ungrounded_terms),
        }
        # Preserve optional metadata from input chunk
        for key in ["doc_id", "heading", "source_file"]:
            if key in chunk:
                chunk_result[key] = chunk[key]

        completed_chunks.append(chunk_result)

        # Save checkpoint after EACH chunk
        _save_checkpoint(
            checkpoint_file, completed_chunks, api_calls, total_cost, seed, model
        )
        logger.info(f"  Checkpoint saved ({len(completed_chunks)}/{len(chunks)})")

    # Build final GT output
    total_terms = sum(c["term_count"] for c in completed_chunks)
    avg_terms = total_terms / len(completed_chunks) if completed_chunks else 0

    gt_data = {
        "metadata": {
            "total_chunks": len(completed_chunks),
            "total_terms": total_terms,
            "average_terms_per_chunk": round(avg_terms, 1),
            "source": "scale_test_runner",
            "anthropic_models": [model],
            "timestamp": datetime.now().isoformat(),
            "random_seed": seed,
            "model_version": model,
        },
        "chunks": completed_chunks,
        "api_calls": api_calls,
        "total_cost_estimate": total_cost,
    }

    # Save final output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(gt_data, f, indent=2)

    logger.section("SUMMARY")
    logger.info(f"Total chunks processed: {len(completed_chunks)}")
    logger.info(f"Total terms extracted: {total_terms}")
    logger.info(f"Average terms per chunk: {avg_terms:.1f}")
    logger.info(f"API calls made: {len(api_calls)}")
    logger.info(f"Output saved to: {output_path}")

    return gt_data


def _save_checkpoint(
    checkpoint_file: Path,
    chunks: list[dict],
    api_calls: list[dict],
    total_cost: float,
    seed: int,
    model: str,
) -> None:
    """Save checkpoint data to file."""
    checkpoint_data = {
        "chunks": chunks,
        "api_calls": api_calls,
        "total_cost": total_cost,
        "last_updated": datetime.now().isoformat(),
        "random_seed": seed,
        "model_version": model,
    }
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)


def _parse_llm_response(
    response: str, chunk_id: str, logger: BenchmarkLogger
) -> Optional[list[dict]]:
    """Parse LLM response to extract terms."""
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.error(f"  ERROR: No JSON found in response for {chunk_id}")
            return None

        data = json.loads(json_match.group())
        terms_data = data.get("terms", [])

        if not isinstance(terms_data, list):
            logger.error(f"  ERROR: 'terms' is not a list for {chunk_id}")
            return None

        terms = []
        for item in terms_data:
            if not isinstance(item, dict):
                continue
            term = item.get("term", "").strip()
            tier = item.get("tier", 2)
            if term and tier in (1, 2, 3):
                terms.append({"term": term, "tier": tier})

        logger.info(f"  Parsed {len(terms)} terms")
        return terms

    except json.JSONDecodeError as e:
        logger.error(f"  ERROR: JSON parse failed for {chunk_id}: {e}")
        return None


# ============================================================================
# PIPELINE EXECUTION WITH AUDIT LOGGING
# ============================================================================


def run_pipeline_with_logging(
    gt_path: Path,
    output_path: Path,
    audit_dir: Path,
    pipeline_config: Optional[dict] = None,
    logger: Optional[BenchmarkLogger] = None,
) -> dict:
    """Run full extraction pipeline with comprehensive audit logging.

    Logs every term's journey through the pipeline including:
    - Source extraction (which model/pass extracted it)
    - Grounding status (pass/fail, match type)
    - Voting results (which extractors agreed)
    - Filter decisions (kept/removed, reason)
    - Final disposition (true positive, false positive, missed)

    Args:
        gt_path: Path to ground truth JSON file
        output_path: Path to save pipeline results
        audit_dir: Directory for audit trail files
        pipeline_config: Optional pipeline configuration overrides
        logger: Optional BenchmarkLogger for logging

    Returns:
        Results dict with metrics and audit trail
    """
    gt_path = Path(gt_path)
    output_path = Path(output_path)
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if logger is None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = BenchmarkLogger(
            log_dir=LOG_DIR,
            log_file=f"pipeline_{timestamp}.log",
            console=True,
            min_level="INFO",
        )

    logger.section("Pipeline Execution with Audit Logging")
    logger.info(f"GT path: {gt_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Audit dir: {audit_dir}")

    # Load ground truth
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with open(gt_path) as f:
        gt_data = json.load(f)

    # Extract metadata
    metadata = gt_data.get("metadata", {})
    seed = metadata.get("random_seed", 42)
    model_version = metadata.get("model_version", "unknown")

    logger.info(f"GT seed: {seed}")
    logger.info(f"GT model: {model_version}")
    logger.info(f"GT chunks: {len(gt_data.get('chunks', []))}")

    # Initialize audit trail
    audit_trail: list[dict] = []
    api_calls: list[dict] = []
    total_cost = 0.0

    # Process each chunk through pipeline
    chunks = gt_data.get("chunks", [])
    results_chunks: list[dict] = []

    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        gt_terms = {t["term"].lower() for t in chunk.get("terms", [])}

        logger.section(f"Chunk {i+1}/{len(chunks)}: {chunk_id}")

        # Rate limiting
        if i > 0:
            time.sleep(API_RATE_LIMIT_DELAY)

        # Run extraction (simulated multi-model voting)
        chunk_audit = {
            "chunk_id": chunk_id,
            "gt_terms": list(gt_terms),
            "extractions": [],
            "term_journeys": {},
        }

        # Multi-model extraction (Sonnet exhaustive + Haiku simple + Haiku exhaustive)
        extractors = [
            ("sonnet_exhaustive", "sonnet", _get_exhaustive_prompt),
            ("haiku_simple", "haiku", _get_simple_prompt),
            ("haiku_exhaustive", "haiku", _get_exhaustive_prompt),
        ]

        all_terms: dict[str, dict] = {}

        for extractor_name, model, prompt_fn in extractors:
            logger.info(f"  Running {extractor_name}...")

            prompt = prompt_fn(content)
            call_start = time.time()
            response = call_llm(
                prompt=prompt,
                model=model,
                timeout=90,
                max_tokens=2000,
                temperature=0.0,
            )
            call_elapsed = time.time() - call_start

            # Track API call
            api_calls.append({
                "chunk_id": chunk_id,
                "extractor": extractor_name,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "elapsed_seconds": call_elapsed,
                "success": bool(response),
            })

            if not response:
                logger.warn(f"    Empty response from {extractor_name}")
                continue

            # Parse terms
            extracted = _parse_extraction_response(response)
            logger.info(f"    Extracted: {len(extracted)} terms")

            chunk_audit["extractions"].append({
                "extractor": extractor_name,
                "terms": extracted,
                "elapsed": call_elapsed,
            })

            # Aggregate terms with voting
            for term in extracted:
                term_key = term.lower()
                if term_key not in all_terms:
                    all_terms[term_key] = {
                        "term": term,
                        "sources": [],
                        "vote_count": 0,
                        "is_grounded": False,
                    }
                all_terms[term_key]["sources"].append(extractor_name)
                all_terms[term_key]["vote_count"] += 1

        # Grounding check for all terms
        for term_key, term_data in all_terms.items():
            result = verify_term_in_text(term_data["term"], content)
            term_data["is_grounded"] = result["grounded"]
            term_data["grounding_type"] = result["match_type"]

            # Track term journey
            is_gt = term_key in gt_terms
            chunk_audit["term_journeys"][term_key] = {
                "term": term_data["term"],
                "sources": term_data["sources"],
                "vote_count": term_data["vote_count"],
                "is_grounded": term_data["is_grounded"],
                "grounding_type": term_data["grounding_type"],
                "is_ground_truth": is_gt,
                "disposition": "pending",
            }

        # Apply voting threshold (2+ votes)
        kept_terms = []
        for term_key, term_data in all_terms.items():
            journey = chunk_audit["term_journeys"][term_key]

            if not term_data["is_grounded"]:
                journey["disposition"] = "rejected_ungrounded"
                continue

            if term_data["vote_count"] < 2:
                journey["disposition"] = "rejected_low_vote"
                continue

            journey["disposition"] = "kept"
            kept_terms.append(term_data["term"])

        # Calculate metrics
        kept_lower = {t.lower() for t in kept_terms}
        true_positives = kept_lower & gt_terms
        false_positives = kept_lower - gt_terms
        missed = gt_terms - kept_lower

        precision = len(true_positives) / len(kept_lower) if kept_lower else 0.0
        recall = len(true_positives) / len(gt_terms) if gt_terms else 0.0

        # Update dispositions
        for term_key in true_positives:
            if term_key in chunk_audit["term_journeys"]:
                chunk_audit["term_journeys"][term_key]["disposition"] = "true_positive"
        for term_key in false_positives:
            if term_key in chunk_audit["term_journeys"]:
                chunk_audit["term_journeys"][term_key]["disposition"] = "false_positive"
        for term_key in missed:
            chunk_audit["term_journeys"][term_key] = {
                "term": term_key,
                "sources": [],
                "vote_count": 0,
                "is_grounded": True,  # GT terms are grounded by definition
                "grounding_type": "ground_truth",
                "is_ground_truth": True,
                "disposition": "missed",
            }

        chunk_result = {
            "chunk_id": chunk_id,
            "gt_count": len(gt_terms),
            "extracted_count": len(kept_terms),
            "true_positives": len(true_positives),
            "false_positives": len(false_positives),
            "missed": len(missed),
            "precision": precision,
            "recall": recall,
            "kept_terms": kept_terms,
        }
        results_chunks.append(chunk_result)
        audit_trail.append(chunk_audit)

        logger.info(f"  Precision: {precision:.1%}, Recall: {recall:.1%}")
        logger.info(f"  TP: {len(true_positives)}, FP: {len(false_positives)}, Missed: {len(missed)}")

    # Calculate aggregate metrics
    total_tp = sum(c["true_positives"] for c in results_chunks)
    total_fp = sum(c["false_positives"] for c in results_chunks)
    total_missed = sum(c["missed"] for c in results_chunks)
    total_extracted = sum(c["extracted_count"] for c in results_chunks)
    total_gt = sum(c["gt_count"] for c in results_chunks)

    agg_precision = total_tp / total_extracted if total_extracted else 0.0
    agg_recall = total_tp / total_gt if total_gt else 0.0
    f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0.0
    hallucination_rate = total_fp / total_extracted if total_extracted else 0.0

    # Build final results
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "gt_path": str(gt_path),
            "random_seed": seed,
            "model_version": model_version,
            "num_chunks": len(chunks),
        },
        "aggregate_metrics": {
            "precision": agg_precision,
            "recall": agg_recall,
            "f1": f1,
            "hallucination_rate": hallucination_rate,
            "total_true_positives": total_tp,
            "total_false_positives": total_fp,
            "total_missed": total_missed,
            "total_extracted": total_extracted,
            "total_gt_terms": total_gt,
        },
        "chunks": results_chunks,
        "api_calls": api_calls,
        "total_cost_estimate": total_cost,
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save audit trail
    audit_file = audit_dir / f"audit_trail_{timestamp}.json"
    with open(audit_file, "w") as f:
        json.dump({
            "metadata": results["metadata"],
            "audit_trail": audit_trail,
        }, f, indent=2)

    logger.section("SUMMARY")
    logger.info(f"Aggregate Precision: {agg_precision:.1%}")
    logger.info(f"Aggregate Recall: {agg_recall:.1%}")
    logger.info(f"F1 Score: {f1:.1%}")
    logger.info(f"Hallucination Rate: {hallucination_rate:.1%}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Audit trail saved to: {audit_file}")

    return results


def _get_exhaustive_prompt(content: str) -> str:
    """Get exhaustive extraction prompt."""
    return f"""Extract ALL technical terms from this documentation chunk. Be exhaustive - include every concept, component, command, and domain-specific vocabulary.

CHUNK:
---
{content}
---

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""


def _get_simple_prompt(content: str) -> str:
    """Get simple extraction prompt."""
    return f"""Extract key technical terms from this documentation chunk.

CHUNK:
---
{content}
---

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""


def _parse_extraction_response(response: str) -> list[str]:
    """Parse extraction response to get term list."""
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            return []

        data = json.loads(json_match.group())
        terms = data.get("terms", [])

        if isinstance(terms, list):
            return [str(t).strip() for t in terms if t]

        return []

    except (json.JSONDecodeError, Exception):
        return []


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scale Test Runner")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # sample command
    sample_parser = subparsers.add_parser("sample", help="Sample chunks")
    sample_parser.add_argument("--pool-size", type=int, default=100, help="Pool size")
    sample_parser.add_argument("--n", type=int, default=30, help="Sample size")
    sample_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # generate-gt command
    gt_parser = subparsers.add_parser("generate-gt", help="Generate ground truth")
    gt_parser.add_argument("--input", type=str, required=True, help="Input chunks JSON")
    gt_parser.add_argument("--output", type=str, required=True, help="Output GT path")
    gt_parser.add_argument("--checkpoint-dir", type=str, required=True, help="Checkpoint directory")
    gt_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # run-pipeline command
    pipeline_parser = subparsers.add_parser("run-pipeline", help="Run pipeline")
    pipeline_parser.add_argument("--gt", type=str, required=True, help="GT file path")
    pipeline_parser.add_argument("--output", type=str, required=True, help="Output path")
    pipeline_parser.add_argument("--audit-dir", type=str, required=True, help="Audit directory")

    args = parser.parse_args()

    if args.command == "sample":
        # Demo sampling
        pool = list(range(args.pool_size))
        sampled = sample_chunks(pool, args.n, seed=args.seed)
        print(f"Sampled {len(sampled)} items with seed={args.seed}")
        print(f"First 10: {sampled[:10]}")

        # Verify reproducibility
        sampled2 = sample_chunks(pool, args.n, seed=args.seed)
        assert sampled == sampled2, "Sampling not reproducible!"
        print("Reproducibility verified!")

    elif args.command == "generate-gt":
        with open(args.input) as f:
            data = json.load(f)
        chunks = data.get("chunks", data) if isinstance(data, dict) else data
        generate_gt_with_checkpointing(
            chunks,
            Path(args.output),
            Path(args.checkpoint_dir),
            seed=args.seed,
        )

    elif args.command == "run-pipeline":
        run_pipeline_with_logging(
            Path(args.gt),
            Path(args.output),
            Path(args.audit_dir),
        )

    else:
        parser.print_help()
