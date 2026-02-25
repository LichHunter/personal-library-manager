#!/usr/bin/env python3
"""
Test extraction prompt variants to reduce generic term over-extraction.

Problem: Current prompts extract universally generic terms (string, name, field, etc.)
that would appear in ANY technical documentation.

This script tests multiple prompt variants to find one that:
1. Extracts domain-specific terms (high recall for meaningful terms)
2. Avoids universally generic terms (lower FP rate)
3. Works for any documentation domain (not just K8s)
"""

import json
import time
import re
from pathlib import Path
from collections import Counter

# Import from existing pipeline
from utils.llm_provider import AnthropicProvider

# ============================================================================
# CONFIGURATION
# ============================================================================

GT_FILE = Path("artifacts/gt_100_chunks.json")
RESULTS_FILE = Path("artifacts/prompt_variant_results.json")
NUM_CHUNKS = 20  # Test on subset for speed

# ============================================================================
# PROMPT VARIANTS
# ============================================================================

# V0: BASELINE - Current prompt (too broad)
PROMPT_V0_BASELINE = """Extract ALL technical terms from the following documentation chunk. Be EXHAUSTIVE — capture every technical term, concept, resource, component, tool, protocol, abbreviation, and domain-specific vocabulary.

DOCUMENTATION:
{content}

Extract every term that someone studying this documentation would need to understand. This includes:
- Domain-specific resources, components, and concepts
- Tools, CLI commands, API objects, and protocols
- Technical vocabulary (even if the term also exists in other domains)
- Abbreviations, acronyms, and proper nouns
- Infrastructure, security, and networking terms used in technical context
- Architecture and process terms (e.g., "high availability", "garbage collection")
- Compound terms AND their key individual components when independently meaningful

Rules:
- Extract terms EXACTLY as they appear in the text
- Be EXHAUSTIVE — missing a term is worse than including a borderline one
- DO include terms used across multiple domains IF they carry technical meaning here
- DO NOT include structural/formatting words (title, section, overview, content, weight)

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

# V1: DOMAIN-SPECIFIC FOCUS
# Explicitly asks for domain-specific terms, excludes generic programming vocabulary
PROMPT_V1_DOMAIN_SPECIFIC = """Extract DOMAIN-SPECIFIC technical terms from this documentation.

DOCUMENTATION:
{content}

Extract terms that are SPECIFIC to this documentation's subject matter - terms that help readers understand THIS particular technology or system.

INCLUDE:
- Named components, resources, and tools specific to this domain
- Domain-specific concepts and patterns
- API objects, resource types, and configuration options
- Commands and CLI tools specific to this technology
- Proper nouns and product names
- Technical abbreviations specific to this domain

DO NOT INCLUDE (these are too generic):
- Programming data types: string, int, boolean, byte, object, array, map, list
- Generic structural words: name, type, kind, field, key, value, spec, status, metadata
- Common actions: create, delete, get, set, update, list, run, start, stop
- Generic formats: yaml, json, xml, binary
- Shell commands: cat, rm, bash, echo, grep
- Generic network terms: port, host, url, http, endpoint
- Generic abbreviations: api, dns, tcp, ip (unless domain-specific usage)

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

# V2: GLOSSARY-WORTHY
# Framing: would this term appear in a glossary for this technology?
PROMPT_V2_GLOSSARY = """Extract terms that would appear in a GLOSSARY for this documentation's subject.

DOCUMENTATION:
{content}

A good glossary term is one that:
1. A newcomer would need to look up to understand this technology
2. Has specific meaning in this domain (not just general programming)
3. Would be defined in official documentation or a textbook

Extract terms that meet these criteria. Focus on:
- Concepts specific to this technology
- Named resources, components, and tools
- Domain-specific patterns and practices
- Technical terms with specialized meaning here

Skip terms that are:
- Standard programming vocabulary (string, int, object, array, map)
- Generic technical words (file, directory, port, host, server, client)
- Common actions (create, delete, update, configure, deploy)
- Formatting/structural (name, type, kind, value, field, status)

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

# V3: LEARNING-FOCUSED
# Framing: what terms would a learner need to understand?
PROMPT_V3_LEARNER = """You are helping someone learn this technology. Extract the KEY TERMS they need to know.

DOCUMENTATION:
{content}

Extract terms that a learner would need to understand this documentation. Focus on:
- Core concepts unique to this technology
- Named components, resources, and services
- Technical patterns and practices specific to this domain
- Important configuration options and features
- Tools and commands specific to this ecosystem

Do NOT extract:
- Basic programming types (string, int, boolean, list, map, object)
- Generic words that appear in all technical docs (name, field, key, value, type)
- Common operations (create, delete, update, get, set)
- Standard formats (yaml, json, xml)
- General computing terms (file, port, server, client, process)

A good test: Would explaining this term require domain-specific knowledge, or is it standard programming vocabulary?

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

# V4: NEGATIVE EXAMPLES
# Provides explicit negative examples of what NOT to extract
PROMPT_V4_NEGATIVE_EXAMPLES = """Extract technical terms from this documentation.

DOCUMENTATION:
{content}

Extract terms specific to this technology that a reader would need to understand.

EXTRACT these kinds of terms:
- Named resources: Pod, Deployment, ConfigMap, Service (Kubernetes examples)
- Tools: kubectl, helm, docker, terraform (technology-specific tools)
- Concepts: rolling update, horizontal scaling, service mesh (domain patterns)
- Configuration: replicas, selector, annotations (meaningful config options)

DO NOT EXTRACT these (they are too generic):
- Data types: string, int, boolean, byte, float, object, array, map, list, null
- Structural: name, kind, type, spec, status, metadata, field, key, value, items
- Actions: create, delete, update, get, set, list, run, apply, remove, start, stop
- Formats: yaml, json, xml, toml, binary, text
- Shell: cat, rm, ls, echo, bash, sh, grep, awk, sed, EOF
- Network generic: port, host, url, endpoint, address, socket, http, https
- Validation: required, optional, default, minimum, maximum, pattern

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

# V5: TWO-STAGE THINKING
# Ask the model to think about whether each term is generic
PROMPT_V5_REASONING = """Extract domain-specific technical terms from this documentation.

DOCUMENTATION:
{content}

For each potential term, ask yourself:
"Would this term appear in documentation for ANY technology, or is it specific to THIS domain?"

If a term would appear in any technical documentation (like "string", "name", "create", "port"), 
do NOT include it.

If a term is specific to this technology or has special meaning here, include it.

Examples of GENERIC terms to skip:
- Programming basics: string, int, boolean, object, array, list, map
- Structural words: name, type, kind, field, key, value, spec, status
- Common actions: create, delete, update, get, set, run, configure
- Standard tech: port, host, url, file, directory, server, client

Examples of SPECIFIC terms to include (hypothetical):
- Named components specific to this technology
- Configuration options with domain-specific meaning  
- Patterns and concepts unique to this ecosystem
- Tools and commands for this technology

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""

PROMPT_VARIANTS = {
    "V0_BASELINE": PROMPT_V0_BASELINE,
    "V1_DOMAIN_SPECIFIC": PROMPT_V1_DOMAIN_SPECIFIC,
    "V2_GLOSSARY": PROMPT_V2_GLOSSARY,
    "V3_LEARNER": PROMPT_V3_LEARNER,
    "V4_NEGATIVE_EXAMPLES": PROMPT_V4_NEGATIVE_EXAMPLES,
    "V5_REASONING": PROMPT_V5_REASONING,
}

# ============================================================================
# UNIVERSAL GENERIC TERMS (for analysis)
# ============================================================================

UNIVERSAL_GENERIC = {
    # Data types
    'string', 'int', 'integer', 'boolean', 'bool', 'byte', 'float', 'double',
    'object', 'array', 'map', 'list', 'set', 'null', 'void', 'type', 'types',
    '[]string', '[]byte', '[]int',
    
    # Structural/schema words
    'name', 'kind', 'spec', 'status', 'metadata', 'field', 'fields',
    'key', 'keys', 'value', 'values', 'item', 'items', 'data', 'property',
    'properties', 'attribute', 'attributes', 'element', 'elements',
    'config', 'configuration', 'settings', 'options', 'parameters',
    'args', 'arguments', 'param', 'params',
    
    # Actions/verbs
    'create', 'read', 'update', 'delete', 'get', 'set', 'list', 'add', 'remove',
    'start', 'stop', 'run', 'execute', 'apply', 'replace', 'patch', 'watch',
    'validate', 'validation', 'generate', 'generated', 'build',
    
    # File/IO
    'file', 'files', 'filename', 'directory', 'path', 'paths', 'folder',
    'input', 'output', 'read', 'write', 'load', 'save', 'export', 'import',
    
    # Network
    'port', 'ports', 'host', 'hostname', 'url', 'uri', 'endpoint', 'endpoints',
    'address', 'ip', 'socket', 'connection', 'protocol', 'localhost',
    
    # HTTP
    'request', 'response', 'header', 'headers', 'body', 'payload',
    'get', 'post', 'put', 'patch', 'delete', 'head', 'options',
    
    # Shell/CLI
    'command', 'commands', 'cli', 'shell', 'terminal', 'console',
    'cat', 'rm', 'ls', 'echo', 'bash', 'sh', 'vi', 'vim', 'tty',
    'stdin', 'stdout', 'stderr', 'eof',
    
    # Formats
    'yaml', 'json', 'xml', 'toml', 'csv', 'text', 'binary', 'format', 'encoding',
    
    # Validation/schema
    'required', 'optional', 'default', 'minimum', 'maximum', 'pattern',
    'minlength', 'maxlength', 'minitems', 'maxitems', 'nullable',
    'enum', 'oneof', 'anyof', 'allof', 'not', 'unique', 'uniqueitems',
    
    # Common abbreviations
    'api', 'apis', 'dns', 'tls', 'ssl', 'tcp', 'udp', 'http', 'https',
    'fqdn', 'cn', 'jwt', 'rpc', 'grpc', 'cpu', 'ram', 'io', 'os',
    
    # Auth/security
    'token', 'credential', 'credentials', 'secret', 'password', 'username',
    'certificate', 'key', 'keys', 'authentication', 'authorization',
    
    # Common nouns
    'client', 'server', 'cluster', 'node', 'instance', 'service', 'resource',
    'resources', 'namespace', 'scope', 'context', 'session', 'version',
    'label', 'labels', 'tag', 'tags', 'annotation', 'annotations',
    'event', 'events', 'handler', 'callback', 'listener', 'webhook',
    'admin', 'user', 'group', 'role', 'permission',
    'message', 'error', 'warning', 'info', 'debug', 'log', 'logger',
    'timeout', 'retry', 'interval', 'delay', 'duration',
    'state', 'status', 'condition', 'phase', 'stage',
    'source', 'target', 'destination', 'origin',
    'enable', 'disable', 'enabled', 'disabled',
    'true', 'false', 'yes', 'no', 'on', 'off',
    
    # Documentation meta
    'deprecated', 'note', 'warning', 'example', 'reference', 'see', 'also',
    'todo', 'fixme', 'version', 'since', 'description',
}

# ============================================================================
# HELPERS
# ============================================================================

provider = None

def get_provider():
    global provider
    if provider is None:
        provider = AnthropicProvider()
    return provider

def call_llm(prompt: str, model: str = "haiku", max_tokens: int = 2000, temperature: float = 0.0) -> str:
    """Call LLM with retry."""
    model_map = {
        "haiku": "claude-haiku-4-5",
        "sonnet": "claude-sonnet-4-20250514",
    }
    return get_provider().generate(prompt, model=model_map[model], max_tokens=max_tokens, temperature=temperature)

def parse_terms_response(response: str) -> list[str]:
    """Parse JSON terms response."""
    try:
        # Try direct JSON parse
        data = json.loads(response)
        if isinstance(data, dict) and "terms" in data:
            terms = data["terms"]
            if isinstance(terms, list):
                return [str(t) for t in terms if t]
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON in response
    match = re.search(r'\{[^{}]*"terms"\s*:\s*\[[^\]]*\][^{}]*\}', response, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return [str(t) for t in data.get("terms", []) if t]
        except json.JSONDecodeError:
            pass
    
    return []

def normalize_term(t: str) -> str:
    """Normalize term for comparison."""
    return t.lower().strip()

def verify_span(term: str, content: str) -> bool:
    """Check if term exists in content."""
    return term.lower() in content.lower()

# ============================================================================
# MAIN TEST
# ============================================================================

def test_prompt_variant(variant_name: str, prompt_template: str, chunks: list, gt_by_chunk: dict) -> dict:
    """Test a single prompt variant on all chunks."""
    print(f"\n{'='*60}")
    print(f"Testing: {variant_name}")
    print(f"{'='*60}")
    
    results = {
        "variant": variant_name,
        "chunks": [],
        "aggregate": {},
    }
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_extracted = 0
    total_gt = 0
    generic_fps = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["chunk_id"]
        content = chunk["content"]
        gt_terms = set(t["term"].lower() for t in gt_by_chunk[chunk_id]["terms"])
        
        print(f"  [{i+1}/{len(chunks)}] {chunk_id[:50]}...", end=" ", flush=True)
        
        # Extract terms
        prompt = prompt_template.format(content=content[:3000])
        response = call_llm(prompt, model="haiku", max_tokens=2000, temperature=0.0)
        extracted = parse_terms_response(response)
        
        # Filter to grounded terms only
        grounded = [t for t in extracted if verify_span(t, content)]
        extracted_set = set(normalize_term(t) for t in grounded)
        
        # Calculate metrics
        tp = len(extracted_set & gt_terms)
        fp = len(extracted_set - gt_terms)
        fn = len(gt_terms - extracted_set)
        
        # Track generic FPs
        fp_terms = extracted_set - gt_terms
        chunk_generic_fps = [t for t in fp_terms if t in UNIVERSAL_GENERIC]
        generic_fps.extend(chunk_generic_fps)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_extracted += len(extracted_set)
        total_gt += len(gt_terms)
        
        precision = tp / len(extracted_set) if extracted_set else 0
        recall = tp / len(gt_terms) if gt_terms else 0
        
        print(f"P={precision:.1%} R={recall:.1%} ({len(grounded)} terms, {len(chunk_generic_fps)} generic FPs)")
        
        results["chunks"].append({
            "chunk_id": chunk_id,
            "extracted": len(grounded),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "generic_fps": len(chunk_generic_fps),
        })
    
    # Aggregate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    hallucination = total_fp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    
    generic_fp_rate = len(generic_fps) / total_fp if total_fp else 0
    
    results["aggregate"] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "hallucination": hallucination,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_extracted": total_extracted,
        "total_gt": total_gt,
        "generic_fp_count": len(generic_fps),
        "generic_fp_rate": generic_fp_rate,
    }
    
    print(f"\n  AGGREGATE: P={precision:.1%} R={recall:.1%} F1={f1:.3f} H={hallucination:.1%}")
    print(f"  Generic FPs: {len(generic_fps)}/{total_fp} ({generic_fp_rate:.1%} of all FPs)")
    
    return results

def main():
    print("=" * 80)
    print("PROMPT VARIANT TESTING")
    print("=" * 80)
    
    # Load GT
    print(f"\nLoading GT from {GT_FILE}...")
    with open(GT_FILE) as f:
        gt_data = json.load(f)
    
    gt_by_chunk = {c["chunk_id"]: c for c in gt_data["chunks"]}
    chunks = gt_data["chunks"][:NUM_CHUNKS]
    print(f"Testing on {len(chunks)} chunks")
    
    # Test each variant
    all_results = {
        "metadata": {
            "num_chunks": len(chunks),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gt_file": str(GT_FILE),
        },
        "variants": {}
    }
    
    for variant_name, prompt_template in PROMPT_VARIANTS.items():
        results = test_prompt_variant(variant_name, prompt_template, chunks, gt_by_chunk)
        all_results["variants"][variant_name] = results
        time.sleep(1)  # Rate limiting
    
    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Variant':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Halluc':>10} {'GenericFP%':>12}")
    print("-" * 80)
    
    for variant_name, results in all_results["variants"].items():
        agg = results["aggregate"]
        print(f"{variant_name:<25} {agg['precision']:>10.1%} {agg['recall']:>10.1%} {agg['f1']:>10.3f} {agg['hallucination']:>10.1%} {agg['generic_fp_rate']:>12.1%}")
    
    # Find best variants
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    variants = list(all_results["variants"].items())
    
    # Best F1
    best_f1 = max(variants, key=lambda x: x[1]["aggregate"]["f1"])
    print(f"\nBest F1: {best_f1[0]} ({best_f1[1]['aggregate']['f1']:.3f})")
    
    # Lowest hallucination
    best_halluc = min(variants, key=lambda x: x[1]["aggregate"]["hallucination"])
    print(f"Lowest Hallucination: {best_halluc[0]} ({best_halluc[1]['aggregate']['hallucination']:.1%})")
    
    # Lowest generic FP rate
    best_generic = min(variants, key=lambda x: x[1]["aggregate"]["generic_fp_rate"])
    print(f"Lowest Generic FP%: {best_generic[0]} ({best_generic[1]['aggregate']['generic_fp_rate']:.1%})")
    
    # Best balanced (F1 * (1 - generic_fp_rate))
    best_balanced = max(variants, key=lambda x: x[1]["aggregate"]["f1"] * (1 - x[1]["aggregate"]["generic_fp_rate"]))
    print(f"Best Balanced: {best_balanced[0]}")

if __name__ == "__main__":
    main()
