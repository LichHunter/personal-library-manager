#!/usr/bin/env python3
"""Generate ground truth for scale testing (30/50/100 chunks).

This script generates GT files using Opus with:
- Random sampling (seed=42)
- Checkpointing after each chunk
- Strict span verification
- Cost tracking

Usage:
    python generate_scale_gt.py --num-chunks 30
    python generate_scale_gt.py --num-chunks 50
    python generate_scale_gt.py --num-chunks 100
"""

import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Setup paths
POC_DIR = Path(__file__).parent
ARTIFACTS_DIR = POC_DIR / "artifacts"
CORPUS_DIR = Path("/home/susano/Code/personal-library-manager/poc/chunking_benchmark_v2/corpus/kubernetes_sample_200")

# Import LLM provider
sys.path.insert(0, str(POC_DIR.parent / "poc-1-llm-extraction-guardrails"))
from utils.llm_provider import call_llm

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 42
CHECKPOINT_DIR = ARTIFACTS_DIR / "gt_checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Rate limiting (seconds between API calls)
API_DELAY = 1.0

# Cost per million tokens (approximate)
OPUS_COST = {"input": 15.0, "output": 75.0}

# ============================================================================
# OPUS GT PROMPT (EXACT COPY from expand_ground_truth.py)
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
# CHUNK LOADING
# ============================================================================

def load_all_chunks() -> list[dict]:
    """Load all available chunks from corpus."""
    chunks = []
    
    if not CORPUS_DIR.exists():
        print(f"ERROR: Corpus directory not found: {CORPUS_DIR}")
        return chunks
    
    md_files = list(CORPUS_DIR.glob("**/*.md"))
    print(f"Found {len(md_files)} markdown files in corpus")
    
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            doc_id = md_file.stem
            
            # Simple chunking: split by major headings
            sections = extract_sections(content)
            
            for i, section in enumerate(sections):
                chunk_id = f"{doc_id}_sec{i}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": section["content"],
                    "heading": section["heading"],
                    "source_file": str(md_file.name),
                })
                
        except Exception as e:
            print(f"  Warning: Failed to process {md_file}: {e}")
    
    print(f"Total chunks created: {len(chunks)}")
    return chunks


def extract_sections(content: str) -> list[dict]:
    """Extract sections from markdown content."""
    sections = []
    
    # Split by major headings (## or #)
    lines = content.split('\n')
    current_heading = "Introduction"
    current_content = []
    
    for line in lines:
        if line.startswith('# '):
            # Save previous section
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "content": '\n'.join(current_content).strip()
                })
            current_heading = line[2:].strip()
            current_content = []
        elif line.startswith('## '):
            # Save previous section
            if current_content:
                sections.append({
                    "heading": current_heading,
                    "content": '\n'.join(current_content).strip()
                })
            current_heading = line[3:].strip()
            current_content = []
        else:
            current_content.append(line)
    
    # Add final section
    if current_content:
        sections.append({
            "heading": current_heading,
            "content": '\n'.join(current_content).strip()
        })
    
    # Filter out very short sections
    sections = [s for s in sections if len(s["content"]) > 200]
    
    return sections


# ============================================================================
# RANDOM SAMPLING
# ============================================================================

def sample_chunks(chunks: list, n: int, seed: int = RANDOM_SEED) -> list:
    """Sample n chunks randomly with fixed seed for reproducibility."""
    rng = random.Random(seed)
    if n >= len(chunks):
        return chunks
    return rng.sample(chunks, n)


# ============================================================================
# GT GENERATION WITH CHECKPOINTING
# ============================================================================

def load_checkpoint(checkpoint_path: Path) -> dict | None:
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        with open(checkpoint_path) as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint_path: Path, data: dict):
    """Save checkpoint after each chunk."""
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2)


def verify_term_in_text(term: str, text: str) -> dict:
    """Verify a term exists in the chunk text. Returns match info.
    EXACT COPY from expand_ground_truth.py for consistency."""
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


def generate_gt_for_chunk(chunk: dict) -> dict:
    content = chunk["content"][:4000]
    prompt = OPUS_GT_PROMPT.format(chunk_content=content)
    
    start_time = time.time()
    
    try:
        response = call_llm(
            prompt,
            model="opus",
            temperature=0.0,
            max_tokens=4000,
            timeout=120,
        )
        
        elapsed = time.time() - start_time
        
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            terms = data.get("terms", [])
            
            validated_terms = []
            for t in terms:
                term = t.get("term", "").strip()
                tier = t.get("tier", 2)
                reasoning = t.get("reasoning", "")
                if term and tier in (1, 2, 3):
                    result = verify_term_in_text(term, chunk["content"])
                    if result["grounded"]:
                        validated_terms.append({
                            "term": term, 
                            "tier": tier, 
                            "reasoning": reasoning
                        })
            
            return {
                "success": True,
                "terms": validated_terms,
                "term_count": len(validated_terms),
                "elapsed_time": elapsed,
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(response) // 4,
            }
        else:
            return {
                "success": False,
                "terms": [],
                "term_count": 0,
                "error": "JSON parse error",
                "elapsed_time": elapsed,
            }
            
    except Exception as e:
        return {
            "success": False,
            "terms": [],
            "term_count": 0,
            "error": str(e),
            "elapsed_time": time.time() - start_time,
        }


def generate_gt_with_checkpointing(
    chunks: list,
    output_path: Path,
    checkpoint_path: Path,
    num_chunks: int
) -> dict:
    """Generate GT with checkpointing for API failure recovery."""
    
    # Load or initialize checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint is None:
        checkpoint = {
            "metadata": {
                "started_at": datetime.now().isoformat(),
                "random_seed": RANDOM_SEED,
                "target_chunks": num_chunks,
                "model": "claude-opus-4-5-20251101",
                "total_cost_usd": 0.0,
            },
            "completed_chunks": [],
            "chunks": [],
        }
    
    completed_ids = {c["chunk_id"] for c in checkpoint["completed_chunks"]}
    
    print(f"\nGenerating GT for {num_chunks} chunks...")
    print(f"Already completed: {len(completed_ids)}")
    print(f"Remaining: {num_chunks - len(completed_ids)}")
    
    for i, chunk in enumerate(chunks[:num_chunks]):
        cid = chunk["chunk_id"]
        
        # Skip if already done
        if cid in completed_ids:
            print(f"  [{i+1}/{num_chunks}] {cid} - SKIPPED (already done)")
            continue
        
        print(f"  [{i+1}/{num_chunks}] {cid}...", end=" ", flush=True)
        
        # Generate GT
        result = generate_gt_for_chunk(chunk)
        
        if result["success"]:
            print(f"✓ {result['term_count']} terms")
            
            # Add to checkpoint
            gt_entry = {
                "chunk_id": cid,
                "doc_id": chunk["doc_id"],
                "heading": chunk["heading"],
                "source_file": chunk["source_file"],
                "content": chunk["content"],
                "terms": result["terms"],
                "term_count": result["term_count"],
                "generated_at": datetime.now().isoformat(),
                "elapsed_time": result["elapsed_time"],
            }
            checkpoint["completed_chunks"].append(gt_entry)
            checkpoint["chunks"].append(gt_entry)
            
            # Update cost estimate
            input_cost = (result.get("input_tokens", 0) / 1_000_000) * OPUS_COST["input"]
            output_cost = (result.get("output_tokens", 0) / 1_000_000) * OPUS_COST["output"]
            checkpoint["metadata"]["total_cost_usd"] += input_cost + output_cost
            
        else:
            print(f"✗ ERROR: {result.get('error', 'Unknown')}")
        
        # Save checkpoint after each chunk
        checkpoint["metadata"]["last_saved"] = datetime.now().isoformat()
        save_checkpoint(checkpoint_path, checkpoint)
        
        # Rate limiting
        time.sleep(API_DELAY)
    
    # Finalize
    checkpoint["metadata"]["completed_at"] = datetime.now().isoformat()
    checkpoint["metadata"]["total_chunks"] = len(checkpoint["chunks"])
    checkpoint["metadata"]["total_terms"] = sum(c["term_count"] for c in checkpoint["chunks"])
    
    if checkpoint["chunks"]:
        checkpoint["metadata"]["average_terms_per_chunk"] = (
            checkpoint["metadata"]["total_terms"] / len(checkpoint["chunks"])
        )
    
    # Save final output
    with open(output_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    print(f"\n✓ GT generation complete!")
    print(f"  Total chunks: {checkpoint['metadata']['total_chunks']}")
    print(f"  Total terms: {checkpoint['metadata']['total_terms']}")
    print(f"  Avg terms/chunk: {checkpoint['metadata'].get('average_terms_per_chunk', 0):.1f}")
    print(f"  Est. cost: ${checkpoint['metadata']['total_cost_usd']:.2f}")
    print(f"  Output: {output_path}")
    
    return checkpoint


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate GT for scale testing")
    parser.add_argument("--num-chunks", type=int, required=True, choices=[30, 50, 100],
                      help="Number of chunks to generate GT for")
    
    args = parser.parse_args()
    
    num_chunks = args.num_chunks
    
    print("=" * 60)
    print(f"GT Generation for {num_chunks} Chunks")
    print("=" * 60)
    
    # Load all available chunks
    all_chunks = load_all_chunks()
    
    if len(all_chunks) < num_chunks:
        print(f"ERROR: Only {len(all_chunks)} chunks available, need {num_chunks}")
        return 1
    
    # Random sampling
    print(f"\nRandom sampling {num_chunks} chunks (seed={RANDOM_SEED})...")
    sampled_chunks = sample_chunks(all_chunks, num_chunks)
    print(f"Sampled chunk IDs: {[c['chunk_id'] for c in sampled_chunks[:5]]}...")
    
    # Setup paths
    output_path = ARTIFACTS_DIR / f"gt_{num_chunks}_chunks.json"
    checkpoint_path = CHECKPOINT_DIR / f"gt_{num_chunks}_checkpoint.json"
    
    # Generate GT with checkpointing
    result = generate_gt_with_checkpointing(
        sampled_chunks,
        output_path,
        checkpoint_path,
        num_chunks
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
