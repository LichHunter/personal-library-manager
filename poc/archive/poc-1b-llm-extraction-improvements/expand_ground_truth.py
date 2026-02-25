#!/usr/bin/env python3
"""Expand ground truth using Opus for comprehensive term extraction.

The current GT has ~163 terms across 15 chunks (~10.9 terms/chunk avg), but
the GT audit proved it's ~30-50% incomplete. This script:

1. Uses Opus to generate comprehensive terms for each chunk
2. Deterministically verifies every term is grounded in source text (span check)
3. Merges with existing GT, deduplicates
4. Outputs small_chunk_ground_truth_v2.json

Usage:
    python expand_ground_truth.py [--dry-run] [--chunks N]
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm
from utils.logger import BenchmarkLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GT_V1_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"
GT_V2_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth_v2.json"
RAW_OPUS_PATH = ARTIFACTS_DIR / "gt_expansion_opus_raw.json"
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# PROMPT
# ============================================================================

OPUS_GT_PROMPT = """You are building a gold-standard ground truth for evaluating term extraction systems. Your output will be used as the DEFINITIVE answer key — completeness and accuracy are paramount.

DOCUMENTATION CHUNK:
---
{chunk_content}
---

TASK: Extract EVERY technical term, concept, and domain-specific vocabulary from this chunk. Be EXHAUSTIVE — a missed term is a failure mode we cannot recover from.

TIER DEFINITIONS:
- Tier 1: Core domain resources, components, and proper nouns (e.g., "Pod", "kubelet", "etcd", "ReplicaSet", "PersistentVolume", "ConfigMap")
  → Named K8s/infrastructure resources, API objects, CLI tools, specific component names
- Tier 2: Domain concepts, processes, and technical vocabulary (e.g., "control plane", "scheduling", "container", "namespace", "cluster", "authorization", "node")
  → Technical concepts, architectural patterns, processes, roles, and operational terms
- Tier 3: Contextual/supporting technical terms (e.g., "HTTPS", "API", "Linux", "OOM", "host", "memory")
  → General technical vocabulary that carries meaning in this context, protocols, OS concepts

WHAT TO EXTRACT:
✅ Named resources and API objects (Pod, Deployment, ConfigMap, PVC)
✅ Infrastructure components (kubelet, kube-apiserver, etcd, containerd)
✅ Domain concepts (control plane, scheduling, container orchestration, fault tolerance)
✅ Technical processes (garbage collection, reconciliation, rolling update)
✅ CLI flags and commands (--cloud-provider, kubectl)
✅ Protocols and standards (HTTPS, gRPC, CNI, CSI)
✅ Abbreviations and acronyms (PV, PVC, OOM, VM, CEL)
✅ Security/networking concepts when used in context (authorization, bearer token, TLS, ingress)
✅ Linux/OS concepts when relevant (cgroups, namespace, process, memory)
✅ Architectural terms (high availability, fault tolerance, load balancing)
✅ Feature gates and API versions (alpha, beta, stable — when describing feature lifecycle)

WHAT NOT TO EXTRACT:
❌ Structural/formatting words (title, section, overview, content, weight, description)
❌ YAML/Markdown syntax (---, ```, true, false)
❌ Common English words used non-technically (the, each, several, manages)
❌ Hugo/templating syntax (glossary_tooltip, feature-state)

RULES:
1. Every term MUST appear verbatim in the chunk text (exact string match, case-insensitive)
2. Extract the term as it appears in the text — don't normalize or paraphrase
3. If a term appears in both singular and plural forms, include the more common form
4. Include compound terms ("control plane", "worker node", "container runtime") AND their key individual components if independently meaningful
5. Err on the side of INCLUSION — it's far better to have borderline terms than to miss valid ones
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
# SPAN VERIFICATION
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
# DEDUPLICATION
# ============================================================================


def normalize_for_dedup(term: str) -> str:
    """Normalize a term for deduplication purposes."""
    t = term.lower().strip()
    t = t.replace("-", " ").replace("_", " ")
    # Remove trailing 's' for basic singular/plural normalization
    # But be careful with terms like "class", "process", "address"
    short_terms_keep_s = {"class", "process", "address", "access", "express", "less", "pass"}
    if t.endswith("s") and len(t) > 3 and t not in short_terms_keep_s:
        t_singular = t[:-1]
        # Don't strip if it would make the term meaningless
        if len(t_singular) > 2:
            t = t_singular
    return t


def deduplicate_terms(terms: list[dict]) -> list[dict]:
    """Deduplicate terms, keeping highest tier (lowest number)."""
    seen: dict[str, dict] = {}
    for t in terms:
        key = normalize_for_dedup(t["term"])
        if key not in seen or t["tier"] < seen[key]["tier"]:
            seen[key] = t
    return list(seen.values())


# ============================================================================
# OPUS EXTRACTION
# ============================================================================


def extract_terms_opus(
    chunk_content: str, chunk_id: str, logger: BenchmarkLogger
) -> list[dict]:
    """Use Opus to extract comprehensive terms from a chunk."""
    prompt = OPUS_GT_PROMPT.format(chunk_content=chunk_content)

    logger.info(f"  Calling Opus for {chunk_id} (content_len={len(chunk_content)})...")
    start = time.time()
    response = call_llm(
        prompt=prompt,
        model="opus",
        timeout=120,
        max_tokens=4000,
        temperature=0.0,
    )
    elapsed = time.time() - start
    logger.info(f"  Opus responded in {elapsed:.1f}s (response_len={len(response)})")

    if not response:
        logger.error(f"  ERROR: Empty response from Opus for {chunk_id}")
        return []

    # Parse response
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.error(f"  ERROR: No JSON found in Opus response for {chunk_id}")
            return []

        data = json.loads(json_match.group())
        terms_data = data.get("terms", [])

        if not isinstance(terms_data, list):
            logger.error(f"  ERROR: 'terms' is not a list for {chunk_id}")
            return []

        terms = []
        for item in terms_data:
            if not isinstance(item, dict):
                continue
            term = item.get("term", "").strip()
            tier = item.get("tier", 2)
            reasoning = item.get("reasoning", "")
            if term and tier in (1, 2, 3):
                terms.append(
                    {"term": term, "tier": tier, "reasoning": reasoning}
                )

        logger.info(f"  Parsed {len(terms)} terms from Opus for {chunk_id}")
        return terms

    except json.JSONDecodeError as e:
        logger.error(f"  ERROR: JSON parse failed for {chunk_id}: {e}")
        logger.debug(f"  Response: {response[:500]}")
        return []


# ============================================================================
# MERGE WITH EXISTING GT
# ============================================================================


def merge_with_existing(
    opus_terms: list[dict], existing_terms: list[dict], logger: BenchmarkLogger
) -> list[dict]:
    """Merge Opus terms with existing GT terms, keeping existing tier assignments
    where they exist (human-validated) and adding new ones from Opus."""
    # Index existing terms by normalized key
    existing_by_key: dict[str, dict] = {}
    for t in existing_terms:
        key = normalize_for_dedup(t["term"])
        existing_by_key[key] = t

    merged = []
    new_count = 0
    kept_existing = 0

    # First, keep all existing terms (they're human-validated)
    for t in existing_terms:
        merged.append({"term": t["term"], "tier": t["tier"]})
        kept_existing += 1

    # Then add Opus terms that aren't already in existing
    for t in opus_terms:
        key = normalize_for_dedup(t["term"])
        if key not in existing_by_key:
            merged.append({"term": t["term"], "tier": t["tier"]})
            new_count += 1

    logger.info(
        f"  Merge: kept {kept_existing} existing + {new_count} new from Opus = {len(merged)} total"
    )
    return merged


# ============================================================================
# MAIN
# ============================================================================


def run_expansion(num_chunks: int = 15, dry_run: bool = False):
    """Run GT expansion on all chunks."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = BenchmarkLogger(
        log_dir=LOG_DIR,
        log_file=f"gt_expansion_{timestamp}.log",
        console=True,
        min_level="INFO",
    )

    logger.section("Ground Truth Expansion with Opus")
    logger.info(f"Processing {num_chunks} chunks")
    logger.info(f"GT v1: {GT_V1_PATH}")
    logger.info(f"GT v2 output: {GT_V2_PATH}")
    logger.info(f"Dry run: {dry_run}")

    # Load existing GT
    with open(GT_V1_PATH) as f:
        gt_v1 = json.load(f)

    chunks = gt_v1["chunks"][:num_chunks]
    logger.info(f"Loaded {len(chunks)} chunks from v1 GT ({gt_v1['total_terms']} terms)")

    # Process each chunk
    all_raw_opus: list[dict] = []
    v2_chunks: list[dict] = []
    total_new_terms = 0
    total_removed_ungrounded = 0

    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        existing_terms = chunk["terms"]

        logger.section(f"Chunk {i+1}/{len(chunks)}: {chunk_id}")
        logger.info(f"  Content length: {len(content)}")
        logger.info(f"  Existing terms: {len(existing_terms)}")

        if dry_run:
            logger.info("  [DRY RUN] Skipping Opus call")
            v2_chunks.append(chunk)
            continue

        # Step 1: Extract with Opus
        opus_terms = extract_terms_opus(content, chunk_id, logger)
        all_raw_opus.append(
            {
                "chunk_id": chunk_id,
                "opus_terms": opus_terms,
                "opus_count": len(opus_terms),
            }
        )

        # Step 2: Span-verify ALL terms (Opus + existing)
        opus_grounded = []
        opus_ungrounded = []
        for t in opus_terms:
            result = verify_term_in_text(t["term"], content)
            if result["grounded"]:
                opus_grounded.append(t)
            else:
                opus_ungrounded.append(t)
                logger.warn(
                    f"  UNGROUNDED (Opus): '{t['term']}' — not found in chunk text"
                )

        if opus_ungrounded:
            logger.warn(
                f"  Removed {len(opus_ungrounded)} ungrounded Opus terms"
            )
            total_removed_ungrounded += len(opus_ungrounded)

        # Step 3: Merge with existing GT
        merged = merge_with_existing(opus_grounded, existing_terms, logger)

        # Step 4: Final deduplication
        deduped = deduplicate_terms(merged)
        logger.info(
            f"  After dedup: {len(deduped)} terms (was {len(merged)} before dedup)"
        )

        new_terms = len(deduped) - len(existing_terms)
        total_new_terms += max(0, new_terms)
        logger.info(
            f"  Net change: {'+' if new_terms >= 0 else ''}{new_terms} terms"
        )

        # Build v2 chunk
        v2_chunk = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "heading": chunk["heading"],
            "source_file": chunk["source_file"],
            "content": chunk["content"],
            "terms": [{"term": t["term"], "tier": t["tier"]} for t in deduped],
            "term_count": len(deduped),
        }
        v2_chunks.append(v2_chunk)

        # Log all terms for review
        logger.info(f"  Final terms for {chunk_id}:")
        for t in sorted(deduped, key=lambda x: (x["tier"], x["term"])):
            is_new = normalize_for_dedup(t["term"]) not in {
                normalize_for_dedup(et["term"]) for et in existing_terms
            }
            marker = " [NEW]" if is_new else ""
            logger.info(f"    T{t['tier']}: {t['term']}{marker}")

    # Save results
    if not dry_run:
        # Save raw Opus output
        with open(RAW_OPUS_PATH, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "model": "opus",
                    "chunks": all_raw_opus,
                },
                f,
                indent=2,
            )
        logger.info(f"Raw Opus output saved to {RAW_OPUS_PATH}")

        # Save v2 GT
        total_v2_terms = sum(c["term_count"] for c in v2_chunks)
        gt_v2 = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "based_on": "small_chunk_ground_truth.json (v1)",
            "expansion_model": "claude-opus-4-5",
            "expansion_method": "Opus extraction + span verification + merge with v1",
            "total_chunks": len(v2_chunks),
            "total_terms": total_v2_terms,
            "v1_total_terms": gt_v1["total_terms"],
            "new_terms_added": total_new_terms,
            "ungrounded_removed": total_removed_ungrounded,
            "chunks": v2_chunks,
        }
        with open(GT_V2_PATH, "w") as f:
            json.dump(gt_v2, f, indent=2)
        logger.info(f"GT v2 saved to {GT_V2_PATH}")

    # Summary
    logger.section("SUMMARY")
    if not dry_run:
        logger.info(f"V1 total terms: {gt_v1['total_terms']}")
        logger.info(f"V2 total terms: {total_v2_terms}")
        logger.info(f"New terms added: {total_new_terms}")
        logger.info(f"Ungrounded removed: {total_removed_ungrounded}")
        logger.info(f"Avg terms/chunk v1: {gt_v1['total_terms'] / len(chunks):.1f}")
        logger.info(f"Avg terms/chunk v2: {total_v2_terms / len(v2_chunks):.1f}")

        # Per-chunk comparison
        logger.info("")
        logger.info(f"{'Chunk':<55} {'V1':>4} {'V2':>4} {'Delta':>6}")
        logger.info("-" * 75)
        for v1c, v2c in zip(chunks, v2_chunks):
            delta = v2c["term_count"] - v1c["term_count"]
            logger.info(
                f"{v1c['chunk_id']:<55} {v1c['term_count']:>4} {v2c['term_count']:>4} {'+' if delta >= 0 else ''}{delta:>5}"
            )
    else:
        logger.info("Dry run — no changes made")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expand GT with Opus")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't call Opus, just show plan"
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=15,
        help="Number of chunks to process (default: all 15)",
    )
    args = parser.parse_args()

    run_expansion(num_chunks=args.chunks, dry_run=args.dry_run)
