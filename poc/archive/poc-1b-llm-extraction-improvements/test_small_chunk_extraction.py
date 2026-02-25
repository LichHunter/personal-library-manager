#!/usr/bin/env python3
"""Test term extraction on small semantic chunks.

Key insight: We should extract terms from small paragraphs/sections,
not full documents. This matches how we'll store chunks in the database.

Approach:
1. Use MarkdownSemanticStrategy to create realistic chunks
2. Generate ground truth with Opus for each small chunk
3. Test multiple extraction strategies
4. Measure precision/recall/hallucination on 10 chunks each

Target: 90%+ recall, <10% hallucination
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

# Add paths
sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
sys.path.insert(0, str(Path(__file__).parent.parent / "chunking_benchmark_v2"))

from utils.llm_provider import call_llm

print("POC-1b: Small Chunk Term Extraction", flush=True)
print("=" * 70, flush=True)

# Paths
CORPUS_DIR = (
    Path(__file__).parent.parent / "chunking_benchmark_v2" / "corpus" / "kubernetes"
)
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"


# ============================================================================
# CHUNK CREATION (Simplified - no external deps)
# ============================================================================


@dataclass
class SimpleChunk:
    id: str
    doc_id: str
    content: str
    heading: str
    source_file: str


def extract_sections_simple(content: str, max_section_words: int = 300) -> list[dict]:
    """Simple section extraction from markdown content."""
    sections = []

    # Split by headings
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)

    headings = list(heading_pattern.finditer(content))

    if not headings:
        # No headings - return full content as one section
        return [{"heading": "Content", "content": content.strip(), "level": 0}]

    # Add intro section if content before first heading
    if headings[0].start() > 50:
        intro = content[: headings[0].start()].strip()
        if intro and len(intro.split()) >= 20:
            sections.append({"heading": "Introduction", "content": intro, "level": 0})

    # Extract each section
    for i, match in enumerate(headings):
        level = len(match.group(1))
        heading = match.group(2).strip()

        # Get content until next heading or end
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(content)

        section_content = content[start:end].strip()

        # Include heading in content
        full_content = f"{'#' * level} {heading}\n\n{section_content}"

        if section_content and len(section_content.split()) >= 15:
            sections.append(
                {"heading": heading, "content": full_content, "level": level}
            )

    return sections


def load_and_chunk_documents(
    max_docs: int = 20, max_chunks_per_doc: int = 3
) -> list[SimpleChunk]:
    """Load K8s documents and create chunks."""
    chunks = []

    if not CORPUS_DIR.exists():
        print(f"Warning: Corpus dir not found: {CORPUS_DIR}", flush=True)
        return chunks

    md_files = list(CORPUS_DIR.glob("**/*.md"))[:max_docs]

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            doc_id = md_file.stem

            sections = extract_sections_simple(content)

            for i, section in enumerate(sections[:max_chunks_per_doc]):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunks.append(
                    SimpleChunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        content=section["content"],
                        heading=section["heading"],
                        source_file=str(md_file.relative_to(CORPUS_DIR)),
                    )
                )
        except Exception as e:
            print(f"  Warning: Failed to process {md_file}: {e}", flush=True)

    return chunks


# ============================================================================
# GROUND TRUTH GENERATION WITH OPUS
# ============================================================================

OPUS_EXTRACTION_PROMPT = """You are annotating a small chunk of Kubernetes documentation for term extraction training.

TASK: Extract ALL technical terms from this chunk that should be indexed in a documentation search system.

CHUNK:
{chunk_content}

EXTRACT:
- Kubernetes resources (Pod, Service, Deployment, ConfigMap, etc.)
- Components (kubelet, kubectl, etcd, kube-proxy, etc.)
- Concepts (namespace, label, selector, annotation, controller, etc.)
- Feature gates (CamelCase names like ServiceAppProtocol)
- Lifecycle stages (alpha, beta, stable, GA, deprecated)
- CLI flags and options (--flag-name)
- API terms (spec, status, metadata, apiVersion, kind)
- Error states and conditions

DO NOT EXTRACT:
- Generic English words unless they're K8s-specific in this context
- YAML structure keywords (title, stages, defaultValue, etc.)
- Version numbers alone (1.18, 1.19)
- File paths or URLs

For each term, provide:
1. term: The exact term as it appears
2. tier: 1 (essential), 2 (important), 3 (nice-to-have)

Output JSON:
{{"terms": [
  {{"term": "Pod", "tier": 1}},
  {{"term": "namespace", "tier": 2}}
]}}"""


def generate_ground_truth_opus(chunks: list[SimpleChunk]) -> list[dict]:
    """Generate ground truth using Opus for each chunk."""
    print(
        f"\nGenerating ground truth for {len(chunks)} chunks with Opus...", flush=True
    )

    ground_truth = []

    for i, chunk in enumerate(chunks):
        print(f"  [{i + 1}/{len(chunks)}] {chunk.id}...", end=" ", flush=True)

        prompt = OPUS_EXTRACTION_PROMPT.format(chunk_content=chunk.content[:3000])

        try:
            response = call_llm(
                prompt, model="claude-opus", temperature=0, max_tokens=1500
            )

            # Parse response
            response = response.strip()
            response = re.sub(r"^```(?:json)?\s*", "", response)
            response = re.sub(r"\s*```$", "", response)

            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                data = json.loads(json_match.group())
                terms = data.get("terms", [])

                # Validate terms exist in chunk
                chunk_lower = chunk.content.lower()
                validated_terms = []
                for t in terms:
                    term = t.get("term", "")
                    if term and term.lower() in chunk_lower:
                        validated_terms.append(t)

                ground_truth.append(
                    {
                        "chunk_id": chunk.id,
                        "doc_id": chunk.doc_id,
                        "heading": chunk.heading,
                        "source_file": chunk.source_file,
                        "content": chunk.content,
                        "terms": validated_terms,
                        "term_count": len(validated_terms),
                    }
                )

                print(f"{len(validated_terms)} terms", flush=True)
            else:
                print("PARSE ERROR", flush=True)
                ground_truth.append(
                    {
                        "chunk_id": chunk.id,
                        "doc_id": chunk.doc_id,
                        "heading": chunk.heading,
                        "source_file": chunk.source_file,
                        "content": chunk.content,
                        "terms": [],
                        "term_count": 0,
                    }
                )

        except Exception as e:
            print(f"ERROR: {e}", flush=True)
            ground_truth.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "heading": chunk.heading,
                    "source_file": chunk.source_file,
                    "content": chunk.content,
                    "terms": [],
                    "term_count": 0,
                }
            )

    return ground_truth


# ============================================================================
# EXTRACTION STRATEGIES
# ============================================================================

# Strategy 1: Simple extraction
SIMPLE_EXTRACT_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.

CHUNK:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}"""


# Strategy 2: Quote-based extraction
QUOTE_EXTRACT_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.

For EACH term, provide the exact quote where it appears.

CHUNK:
{content}

Output JSON: {{"terms": [
  {{"quote": "exact text from chunk", "term": "TermName"}}
]}}"""


# Strategy 3: Exhaustive with categories
EXHAUSTIVE_PROMPT = """Extract ALL technical terms from this Kubernetes documentation chunk.

Be EXHAUSTIVE. Include:
- Resource types (Pod, Service, Deployment, etc.)
- Components (kubelet, kubectl, etcd, etc.)  
- Concepts (namespace, label, selector, etc.)
- Feature gates (CamelCase names)
- Lifecycle stages (alpha, beta, stable)
- CLI flags (--flag-name)
- API terms (spec, status, metadata)

CHUNK:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}"""


# Strategy 4: Chain-of-thought
COT_PROMPT = """Extract technical terms from this Kubernetes documentation chunk.

Think step by step:
1. First, identify the main topic of this chunk
2. Then, find all resources mentioned
3. Then, find all components mentioned
4. Then, find all concepts mentioned
5. Finally, list all unique terms

CHUNK:
{content}

After your analysis, output JSON: {{"terms": ["term1", "term2", ...]}}"""


def parse_extraction_response(response: str, require_quotes: bool = False) -> list[str]:
    """Parse extraction response to get term list."""
    try:
        response = response.strip()
        response = re.sub(r"^```(?:json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            terms_data = data.get("terms", [])

            if require_quotes:
                return [t.get("term", "") for t in terms_data if isinstance(t, dict)]
            elif isinstance(terms_data, list):
                if terms_data and isinstance(terms_data[0], dict):
                    return [t.get("term", "") for t in terms_data]
                else:
                    return [str(t) for t in terms_data]
    except (json.JSONDecodeError, KeyError):
        pass
    return []


def strict_span_verify(term: str, content: str) -> bool:
    """Verify term exists in content (strict)."""
    if not term or len(term) < 2:
        return False

    content_lower = content.lower()
    term_lower = term.lower().strip()

    # Exact match
    if term_lower in content_lower:
        return True

    # Handle underscores/hyphens
    normalized = term_lower.replace("_", " ").replace("-", " ")
    if normalized in content_lower.replace("_", " ").replace("-", " "):
        return True

    # CamelCase split
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True

    return False


def extract_simple(content: str, model: str = "claude-haiku") -> list[str]:
    """Simple extraction strategy."""
    prompt = SIMPLE_EXTRACT_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_extraction_response(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_quote(content: str, model: str = "claude-haiku") -> list[str]:
    """Quote-based extraction strategy."""
    prompt = QUOTE_EXTRACT_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1500)
    terms = parse_extraction_response(response, require_quotes=True)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_exhaustive(content: str, model: str = "claude-haiku") -> list[str]:
    """Exhaustive extraction strategy."""
    prompt = EXHAUSTIVE_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1000)
    terms = parse_extraction_response(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_cot(content: str, model: str = "claude-haiku") -> list[str]:
    """Chain-of-thought extraction strategy."""
    prompt = COT_PROMPT.format(content=content[:2500])
    response = call_llm(prompt, model=model, temperature=0, max_tokens=1500)
    terms = parse_extraction_response(response)
    return [t for t in terms if strict_span_verify(t, content)]


def extract_ensemble(content: str, model: str = "claude-haiku") -> list[str]:
    """Ensemble: union of multiple strategies."""
    all_terms = set()

    # Run multiple strategies
    simple = extract_simple(content, model)
    quote = extract_quote(content, model)
    exhaustive = extract_exhaustive(content, model)

    all_terms.update(simple)
    all_terms.update(quote)
    all_terms.update(exhaustive)

    # Final verification
    return [t for t in all_terms if strict_span_verify(t, content)]


# ============================================================================
# METRICS
# ============================================================================


def normalize_term(term: str) -> str:
    return term.lower().strip().replace("-", " ").replace("_", " ")


def match_terms(extracted: str, ground_truth: str) -> bool:
    ext_norm = normalize_term(extracted)
    gt_norm = normalize_term(ground_truth)

    if ext_norm == gt_norm:
        return True

    # Fuzzy match
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True

    # Token overlap for multi-word terms
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True

    return False


def calculate_metrics(extracted: list[str], ground_truth: list[dict]) -> dict:
    """Calculate precision, recall, hallucination."""
    gt_terms = [t.get("term", "") for t in ground_truth]

    matched_gt = set()
    matched_ext = set()
    tp = 0

    for i, ext in enumerate(extracted):
        for j, gt in enumerate(gt_terms):
            if j in matched_gt:
                continue
            if match_terms(ext, gt):
                matched_gt.add(j)
                matched_ext.add(i)
                tp += 1
                break

    fp = len(extracted) - tp
    fn = len(gt_terms) - tp

    precision = tp / len(extracted) if extracted else 0
    recall = tp / len(gt_terms) if gt_terms else 0
    hallucination = fp / len(extracted) if extracted else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    missed = [gt_terms[i] for i in range(len(gt_terms)) if i not in matched_gt]
    false_pos = [extracted[i] for i in range(len(extracted)) if i not in matched_ext]

    return {
        "precision": precision,
        "recall": recall,
        "hallucination": hallucination,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "extracted_count": len(extracted),
        "gt_count": len(gt_terms),
        "missed": missed,
        "false_positives": false_pos,
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment(num_chunks: int = 10):
    """Run extraction experiment on small chunks."""

    # Step 1: Load or create ground truth
    if GROUND_TRUTH_PATH.exists():
        print(f"\nLoading existing ground truth from {GROUND_TRUTH_PATH}", flush=True)
        with open(GROUND_TRUTH_PATH) as f:
            gt_data = json.load(f)
        ground_truth = gt_data["chunks"]
    else:
        print(f"\nCreating new ground truth...", flush=True)

        # Load and chunk documents
        chunks = load_and_chunk_documents(max_docs=30, max_chunks_per_doc=2)
        print(f"Created {len(chunks)} chunks from K8s docs", flush=True)

        if not chunks:
            print("ERROR: No chunks created. Check corpus path.", flush=True)
            return

        # Generate ground truth with Opus
        ground_truth = generate_ground_truth_opus(chunks[: num_chunks + 5])

        # Filter chunks with terms
        ground_truth = [g for g in ground_truth if g["term_count"] > 0]

        # Save ground truth
        gt_data = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_chunks": len(ground_truth),
            "total_terms": sum(g["term_count"] for g in ground_truth),
            "chunks": ground_truth,
        }

        with open(GROUND_TRUTH_PATH, "w") as f:
            json.dump(gt_data, f, indent=2)

        print(f"\nSaved ground truth to {GROUND_TRUTH_PATH}", flush=True)

    # Step 2: Select test chunks
    test_chunks = ground_truth[:num_chunks]
    print(f"\nTesting on {len(test_chunks)} chunks", flush=True)

    # Show chunk sizes
    avg_words = sum(len(c["content"].split()) for c in test_chunks) / len(test_chunks)
    avg_terms = sum(c["term_count"] for c in test_chunks) / len(test_chunks)
    print(f"  Avg chunk size: {avg_words:.0f} words", flush=True)
    print(f"  Avg terms/chunk: {avg_terms:.1f}", flush=True)

    # Step 3: Define strategies
    strategies = {
        "simple_haiku": lambda c: extract_simple(c, "claude-haiku"),
        "quote_haiku": lambda c: extract_quote(c, "claude-haiku"),
        "exhaustive_haiku": lambda c: extract_exhaustive(c, "claude-haiku"),
        "cot_haiku": lambda c: extract_cot(c, "claude-haiku"),
        "ensemble_haiku": lambda c: extract_ensemble(c, "claude-haiku"),
        "simple_sonnet": lambda c: extract_simple(c, "claude-sonnet"),
        "exhaustive_sonnet": lambda c: extract_exhaustive(c, "claude-sonnet"),
    }

    # Step 4: Run experiments
    results = {name: [] for name in strategies}

    print(f"\n{'=' * 70}", flush=True)
    print("RUNNING EXTRACTION EXPERIMENTS", flush=True)
    print("=" * 70, flush=True)

    for chunk in test_chunks:
        print(
            f"\n{chunk['chunk_id']} (GT: {chunk['term_count']} terms, {len(chunk['content'].split())} words):",
            flush=True,
        )

        for name, extractor in strategies.items():
            try:
                start = time.time()
                extracted = extractor(chunk["content"])
                elapsed = time.time() - start

                metrics = calculate_metrics(extracted, chunk["terms"])
                metrics["elapsed"] = elapsed
                results[name].append(metrics)

                r_mark = (
                    "OK"
                    if metrics["recall"] >= 0.90
                    else ("~" if metrics["recall"] >= 0.70 else "")
                )
                h_mark = (
                    "OK"
                    if metrics["hallucination"] < 0.10
                    else ("~" if metrics["hallucination"] < 0.20 else "")
                )

                print(
                    f"  {name:<20}: R={metrics['recall']:.0%}{r_mark} H={metrics['hallucination']:.0%}{h_mark} ({metrics['extracted_count']} terms, {elapsed:.1f}s)",
                    flush=True,
                )

            except Exception as e:
                print(f"  {name:<20}: ERROR - {e}", flush=True)

    # Step 5: Aggregate results
    print(f"\n{'=' * 70}", flush=True)
    print("AGGREGATE RESULTS", flush=True)
    print("=" * 70, flush=True)

    print(
        f"{'Strategy':<22} {'Precision':>10} {'Recall':>10} {'Halluc':>10} {'F1':>8}",
        flush=True,
    )
    print("-" * 65, flush=True)

    summary = {}
    for name, metrics_list in results.items():
        if not metrics_list:
            continue

        avg_p = sum(m["precision"] for m in metrics_list) / len(metrics_list)
        avg_r = sum(m["recall"] for m in metrics_list) / len(metrics_list)
        avg_h = sum(m["hallucination"] for m in metrics_list) / len(metrics_list)
        avg_f1 = sum(m["f1"] for m in metrics_list) / len(metrics_list)

        r_mark = "OK" if avg_r >= 0.90 else ("~" if avg_r >= 0.70 else "  ")
        h_mark = "OK" if avg_h < 0.10 else ("~" if avg_h < 0.20 else "  ")

        print(
            f"{name:<22} {avg_p:>10.1%} {avg_r:>8.1%} {r_mark} {avg_h:>8.1%} {h_mark} {avg_f1:>7.1%}",
            flush=True,
        )

        summary[name] = {
            "precision": avg_p,
            "recall": avg_r,
            "hallucination": avg_h,
            "f1": avg_f1,
        }

    print(f"\nTargets: Recall 90%+ [OK], Hallucination <10% [OK]", flush=True)
    print(f"Close:   Recall 70%+ [~],  Hallucination <20% [~]", flush=True)

    # Save results
    results_path = ARTIFACTS_DIR / "small_chunk_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_path}", flush=True)

    return summary


if __name__ == "__main__":
    run_experiment(num_chunks=10)
