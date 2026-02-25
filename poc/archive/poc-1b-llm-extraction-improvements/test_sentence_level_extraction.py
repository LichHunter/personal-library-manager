#!/usr/bin/env python3
"""Sentence-level extraction with quote-verify and reasoning.

Strategy:
1. Split chunk into sentences (preserving context)
2. Haiku extracts from each sentence with:
   - term: extracted term
   - quote: exact quote from sentence
   - reasoning: why it's a K8s term
   - confidence: HIGH/MEDIUM/LOW
3. Malformed responses → retry once, else drop
4. Combine all sentence results
5. Sonnet filters:
   - Span verification (term exists in text)
   - Reasoning quality check
   - K8s-specificity check

Expected: 90-93% P, 82-86% R, 3-7% H
"""

import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rapidfuzz import fuzz

sys.path.insert(
    0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails")
)
from utils.llm_provider import call_llm
from utils.logger import BenchmarkLogger

# ============================================================================
# CONFIGURATION
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
GROUND_TRUTH_PATH = ARTIFACTS_DIR / "small_chunk_ground_truth.json"
RESULTS_PATH = ARTIFACTS_DIR / "sentence_level_v2_results.json"
LOG_DIR = ARTIFACTS_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ExtractedTerm:
    """A single extracted term with metadata."""

    term: str
    quote: str
    reasoning: str
    confidence: str
    sentence_idx: int


@dataclass
class SentenceResult:
    """Result of extracting from a single sentence."""

    sentence: str
    sentence_idx: int
    terms: list[ExtractedTerm]
    extraction_time: float
    retried: bool
    parse_success: bool


# ============================================================================
# SENTENCE SPLITTING
# ============================================================================


def smart_split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling markdown/docs structure.
    
    Handles:
    - Lists (-, *, 1.)
    - Code blocks (```)
    - Headers (##)
    - Multi-line paragraphs
    """
    sentences = []
    
    # Split by newlines first to handle structure
    lines = text.split('\n')
    current_sentence = []
    in_code_block = False
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if current_sentence:
                sentences.append(' '.join(current_sentence))
                current_sentence = []
            continue
        
        # Code blocks
        if line.startswith('```'):
            in_code_block = not in_code_block
            if current_sentence:
                sentences.append(' '.join(current_sentence))
                current_sentence = []
            continue
        
        if in_code_block:
            continue
        
        # Headers
        if line.startswith('#'):
            if current_sentence:
                sentences.append(' '.join(current_sentence))
                current_sentence = []
            sentences.append(line)
            continue
        
        # Lists
        if re.match(r'^[-*•]\s', line) or re.match(r'^\d+\.\s', line):
            if current_sentence:
                sentences.append(' '.join(current_sentence))
                current_sentence = []
            sentences.append(line)
            continue
        
        # Regular text - split on sentence boundaries
        # But keep building if it doesn't end with .!?
        current_sentence.append(line)
        
        if line.endswith(('.', '!', '?', ':', ';')):
            sentences.append(' '.join(current_sentence))
            current_sentence = []
    
    # Final sentence
    if current_sentence:
        sentences.append(' '.join(current_sentence))
    
    # Further split long sentences on .!?
    final_sentences = []
    for sent in sentences:
        if len(sent) > 200:  # Long sentence, try to split
            parts = re.split(r'([.!?]+\s+)', sent)
            sub_sent = ''
            for part in parts:
                sub_sent += part
                if re.match(r'[.!?]+\s+$', part):
                    final_sentences.append(sub_sent.strip())
                    sub_sent = ''
            if sub_sent.strip():
                final_sentences.append(sub_sent.strip())
        else:
            final_sentences.append(sent)
    
    return [s for s in final_sentences if len(s) > 10]


# ============================================================================
# PROMPTS
# ============================================================================

SENTENCE_EXTRACTION_PROMPT = """Extract technical terms from the FOCUS SENTENCE below.

FULL CONTEXT (for understanding references like "it", "they"):
{full_chunk}

FOCUS SENTENCE:
{sentence}

Extract ALL technical terms from the FOCUS SENTENCE that someone studying this documentation would need to understand. This includes:
- Domain-specific resources, components, and concepts
- Tools, CLI commands, API objects, and protocols
- Technical vocabulary used in the domain (even if the term also exists in other domains)
- Abbreviations, acronyms, and proper nouns for technical things
- Infrastructure, security, and networking terms when used in a technical context

For EACH term provide:
- term: The extracted term (exact as it appears in the text)
- quote: Exact quote from FOCUS SENTENCE containing the term (5-30 words)
- reasoning: Why someone studying this topic would need to know this term (1-2 sentences)
- confidence: HIGH (core domain concept), MEDIUM (supporting technical term), or LOW (tangentially relevant)

Rules:
- Extract ONLY from the FOCUS SENTENCE (use full context for understanding references)
- The quote MUST be a verbatim substring of the FOCUS SENTENCE
- Be EXHAUSTIVE - capture every technical term a learner would want to look up
- DO include terms that have broad IT usage IF they carry specific meaning in this context
  (e.g., "cluster", "container", "authorization", "HTTPS" are technical terms worth extracting)
- DO NOT extract purely structural/formatting words ("title", "section", "content", "overview")

Output JSON:
{{"terms": [{{"term": "Pod", "quote": "...exact text from focus sentence...", "reasoning": "Core resource type that...", "confidence": "HIGH"}}]}}

If no terms found, output: {{"terms": []}}
"""

SONNET_FILTER_PROMPT = """You are a technical documentation expert reviewing extracted terms for quality.

DOCUMENTATION CHUNK:
{full_chunk}

CANDIDATE TERMS (extracted from individual sentences):
{terms_json}

Your task: Review each candidate term and decide whether to APPROVE or REJECT it.

Your DEFAULT should be to APPROVE. Only REJECT terms that clearly fail the criteria below.

APPROVE the term if ANY of these are true:
1. DOMAIN CONCEPT: The term names a resource, component, tool, protocol, or concept that is part of the domain being documented
2. TECHNICAL VOCABULARY: The term is technical vocabulary that a learner studying this documentation would benefit from understanding — even if the term also exists in other fields
3. INFRASTRUCTURE TERM: The term refers to infrastructure, security, networking, or system concepts that carry specific meaning in this documentation's context (e.g., "cluster", "authorization", "HTTPS", "bearer token", "container" are all valid technical terms worth indexing)
4. APPEARS IN TEXT: The term or a close variant actually appears in the documentation chunk above

REJECT the term ONLY if ALL of these are true:
1. The term is purely structural or formatting language (e.g., "title", "section", "overview", "content", "value", "weight")
2. AND the term does NOT name any technical concept, resource, tool, or domain entity
3. AND a learner would gain NO technical understanding from looking up this term

IMPORTANT GUIDELINES:
- When in doubt, APPROVE. False negatives (rejecting valid terms) are MUCH worse than false positives (keeping borderline terms)
- Terms like "cluster", "container", "authentication", "HTTPS", "certificate", "memory", "API" ARE valid technical terms when they appear in technical documentation — do NOT reject them as "too generic"
- A term being used across multiple domains does NOT make it invalid — "container" is valid in Docker docs, K8s docs, and shipping docs alike
- Focus on: "Would a student reading this documentation want to know what this term means?"

For EACH candidate term, provide:
- term: The term name
- decision: "APPROVE" or "REJECT"
- reasoning: Brief justification (1 sentence)

Output JSON:
{{
  "terms": [
    {{"term": "Pod", "decision": "APPROVE", "reasoning": "Core resource type in the domain"}},
    {{"term": "title", "decision": "REJECT", "reasoning": "Structural YAML key, not a technical concept"}}
  ]
}}
"""

# ============================================================================
# PARSING & VALIDATION
# ============================================================================


def parse_extraction_response(response: str, sentence: str, logger: BenchmarkLogger) -> Optional[list[ExtractedTerm]]:
    """Parse LLM extraction response into structured terms."""
    try:
        # Clean response
        response = response.strip()
        response = re.sub(r'^```(?:json)?\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        # Find JSON
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            logger.debug(f"No JSON found in response")
            return None
        
        data = json.loads(json_match.group())
        terms_data = data.get('terms', [])
        
        if not isinstance(terms_data, list):
            logger.debug(f"'terms' is not a list")
            return None
        
        # Validate each term
        valid_terms = []
        for i, term_data in enumerate(terms_data):
            if not isinstance(term_data, dict):
                logger.debug(f"Term {i} is not a dict")
                continue
            
            term = term_data.get('term', '').strip()
            quote = term_data.get('quote', '').strip()
            reasoning = term_data.get('reasoning', '').strip()
            confidence = term_data.get('confidence', 'MEDIUM').strip().upper()
            
            # Validate required fields
            if not term or not quote or not reasoning:
                logger.debug(f"Term {i} missing required fields")
                continue
            
            # Validate confidence
            if confidence not in ['HIGH', 'MEDIUM', 'LOW']:
                confidence = 'MEDIUM'
            
            # Validate quote is in sentence
            if quote.lower() not in sentence.lower():
                logger.debug(f"Quote not in sentence: '{quote[:50]}...'")
                continue
            
            # Validate term is in quote
            if term.lower() not in quote.lower():
                logger.debug(f"Term '{term}' not in quote")
                continue
            
            valid_terms.append(ExtractedTerm(
                term=term,
                quote=quote,
                reasoning=reasoning,
                confidence=confidence,
                sentence_idx=-1  # Will be set by caller
            ))
        
        return valid_terms
    
    except json.JSONDecodeError as e:
        logger.debug(f"JSON parse error: {e}")
        return None
    except Exception as e:
        logger.debug(f"Parse error: {e}")
        return None


def parse_filter_response(response: str, logger: BenchmarkLogger) -> tuple[Optional[list[str]], dict]:
    """Parse Sonnet approval response to get approved terms and decisions.
    
    Supports both APPROVE/REJECT (new) and KEEP/REMOVE (legacy) formats.
    
    Returns:
        (approved_terms, decisions_dict) where decisions_dict maps term -> decision info
    """
    try:
        response = response.strip()
        response = re.sub(r'^```(?:json)?\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None, {}
        
        data = json.loads(json_match.group())
        terms_data = data.get('terms', [])
        
        approved_terms = []
        decisions = {}
        
        for term_data in terms_data:
            if isinstance(term_data, dict):
                term = term_data.get('term', '').strip()
                decision = term_data.get('decision', 'APPROVE').strip().upper()
                reasoning = term_data.get('reasoning', 'No reasoning provided').strip()
                score = term_data.get('reasoning_score', 0)
                
                # Normalize decision: KEEP→APPROVE, REMOVE→REJECT
                if decision == 'KEEP':
                    decision = 'APPROVE'
                elif decision == 'REMOVE':
                    decision = 'REJECT'
                
                if term:
                    decisions[term] = {
                        'decision': decision,
                        'reasoning': reasoning,
                        'score': score
                    }
                    
                    if decision == 'APPROVE':
                        approved_terms.append(term)
                    
                    # Log each decision
                    if decision == 'APPROVE':
                        logger.debug(f"  ✓ APPROVE '{term}': {reasoning[:80]}...")
                    else:
                        logger.info(f"  ✗ REJECT '{term}': {reasoning[:100]}...")
            elif isinstance(term_data, str):
                # Fallback for old format
                approved_terms.append(term_data.strip())
        
        return approved_terms, decisions
    
    except Exception as e:
        logger.debug(f"Filter parse error: {e}")
        return None, {}


# ============================================================================
# EXTRACTION FUNCTIONS
# ============================================================================


def extract_from_sentence(
    sentence: str,
    sentence_idx: int,
    full_chunk: str,
    logger: BenchmarkLogger
) -> SentenceResult:
    """Extract terms from a single sentence with retry."""
    start_time = time.time()
    
    logger.debug(f"Sentence {sentence_idx}: {sentence[:80]}...")
    
    # First attempt
    prompt = SENTENCE_EXTRACTION_PROMPT.format(
        full_chunk=full_chunk[:2000],
        sentence=sentence
    )
    
    response = call_llm(
        prompt,
        model="claude-haiku",
        temperature=0,
        max_tokens=1000,
        timeout=60
    )
    
    terms = parse_extraction_response(response, sentence, logger)
    
    # Retry if parsing failed
    retried = False
    if terms is None:
        logger.debug(f"Sentence {sentence_idx}: Parse failed, retrying...")
        retried = True
        
        # Retry with explicit structure hint
        retry_prompt = prompt + "\n\nIMPORTANT: Output MUST be valid JSON. Use exactly this format:\n{\"terms\": [{\"term\": \"X\", \"quote\": \"...\", \"reasoning\": \"...\", \"confidence\": \"HIGH\"}]}"
        
        response = call_llm(
            retry_prompt,
            model="claude-haiku",
            temperature=0,
            max_tokens=1000,
            timeout=60
        )
        
        terms = parse_extraction_response(response, sentence, logger)
    
    # Set sentence_idx for all terms
    if terms:
        for term in terms:
            term.sentence_idx = sentence_idx
    else:
        terms = []
    
    elapsed = time.time() - start_time
    
    logger.debug(f"Sentence {sentence_idx}: {len(terms)} terms extracted in {elapsed:.2f}s")
    
    return SentenceResult(
        sentence=sentence,
        sentence_idx=sentence_idx,
        terms=terms,
        extraction_time=elapsed,
        retried=retried,
        parse_success=terms is not None
    )


def extract_with_sentence_level(chunk_text: str, logger: BenchmarkLogger) -> list[str]:
    """Main sentence-level extraction strategy."""
    
    # Step 1: Split into sentences
    sentences = smart_split_sentences(chunk_text)
    logger.info(f"Split into {len(sentences)} sentences")
    
    # Step 2: Extract from each sentence
    all_results = []
    for i, sentence in enumerate(sentences):
        result = extract_from_sentence(sentence, i, chunk_text, logger)
        all_results.append(result)
    
    # Step 3: Combine all terms
    all_terms = []
    for result in all_results:
        all_terms.extend(result.terms)
    
    logger.info(f"Extracted {len(all_terms)} terms from {len(sentences)} sentences")
    
    if not all_terms:
        return []
    
    # Step 4: Deduplicate (same term from multiple sentences)
    unique_terms = {}
    for term in all_terms:
        key = term.term.lower()
        if key not in unique_terms:
            unique_terms[key] = term
        else:
            # Keep the one with higher confidence
            existing = unique_terms[key]
            conf_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
            if conf_order.get(term.confidence, 0) > conf_order.get(existing.confidence, 0):
                unique_terms[key] = term
    
    logger.info(f"After deduplication: {len(unique_terms)} unique terms")
    
    # Step 5: Sonnet filtering
    terms_for_filter = [
        {
            'term': term.term,
            'quote': term.quote,
            'reasoning': term.reasoning,
            'confidence': term.confidence
        }
        for term in unique_terms.values()
    ]
    
    prompt = SONNET_FILTER_PROMPT.format(
        full_chunk=chunk_text[:2500],
        terms_json=json.dumps(terms_for_filter, indent=2)
    )
    
    logger.subsection("Sonnet Approval Decisions")
    
    response = call_llm(
        prompt,
        model="claude-sonnet",
        temperature=0,
        max_tokens=2500,
        timeout=90
    )
    
    kept_terms, decisions = parse_filter_response(response, logger)
    
    if kept_terms is None:
        logger.warn("Sonnet approval parsing failed, keeping all terms")
        kept_terms = [t.term for t in unique_terms.values()]
        decisions = {}
    
    # Log summary of decisions
    approved_count = sum(1 for d in decisions.values() if d['decision'] == 'APPROVE')
    rejected_count = sum(1 for d in decisions.values() if d['decision'] == 'REJECT')
    logger.info(f"Sonnet decisions: {approved_count} APPROVE, {rejected_count} REJECT")
    logger.info(f"After Sonnet approval: {len(kept_terms)} terms kept")
    
    # Step 6: Final span verification (deterministic)
    verified_terms = []
    for term in kept_terms:
        if strict_span_verify(term, chunk_text):
            verified_terms.append(term)
        else:
            logger.debug(f"Final span verification failed: '{term}'")
    
    logger.info(f"After span verification: {len(verified_terms)} final terms")
    
    return verified_terms


def strict_span_verify(term: str, content: str) -> bool:
    """Verify term exists in content (deterministic)."""
    if not term or len(term) < 2:
        return False
    content_lower = content.lower()
    term_lower = term.lower().strip()
    if term_lower in content_lower:
        return True
    # Try normalized (- and _ as spaces)
    normalized = term_lower.replace("_", " ").replace("-", " ")
    if normalized in content_lower.replace("_", " ").replace("-", " "):
        return True
    # Try camelCase split
    camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()
    if camel != term_lower and camel in content_lower:
        return True
    return False


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
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return True
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens and len(ext_tokens & gt_tokens) / len(gt_tokens) >= 0.8:
        return True
    return False


def calculate_metrics(extracted: list[str], ground_truth: list[dict]) -> dict:
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
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================


def run_experiment(num_chunks: int = 10):
    """Run sentence-level extraction experiment."""
    
    # Initialize logger
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    logger = BenchmarkLogger(
        log_dir=LOG_DIR,
        log_file=f"sentence_level_v2_{timestamp}.log",
        console=True,
        min_level="INFO"
    )
    
    logger.section("POC-1b v2: Sentence-Level Extraction with Approval Prompt")
    logger.info(f"Testing on {num_chunks} chunks")
    logger.info(f"Log file: {LOG_DIR}/sentence_level_v2_{timestamp}.log")
    
    # Load ground truth
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)["chunks"]
    
    test_chunks = ground_truth[:num_chunks]
    
    # Run extraction on each chunk
    results = []
    
    for i, chunk in enumerate(test_chunks):
        logger.section(f"Chunk {i+1}/{len(test_chunks)}: {chunk['chunk_id']}")
        logger.info(f"Ground truth: {chunk['term_count']} terms")
        logger.info(f"Content length: {len(chunk['content'])} chars")
        
        try:
            start = time.time()
            extracted = extract_with_sentence_level(chunk["content"], logger)
            elapsed = time.time() - start
            
            metrics = calculate_metrics(extracted, chunk["terms"])
            metrics["elapsed"] = elapsed
            
            logger.subsection("Results")
            logger.info(f"Extracted: {len(extracted)} terms")
            logger.metric("precision", metrics["precision"], "%")
            logger.metric("recall", metrics["recall"], "%")
            logger.metric("hallucination", metrics["hallucination"], "%")
            logger.metric("f1", metrics["f1"])
            logger.metric("time", elapsed, "s")
            
            # Detailed term analysis
            gt_terms = [t.get("term", "") for t in chunk["terms"]]
            logger.info(f"True positives: {metrics['tp']}")
            logger.info(f"False positives: {metrics['fp']}")
            logger.info(f"False negatives: {metrics['fn']}")
            
            # Show missed terms (false negatives)
            if metrics['fn'] > 0:
                logger.debug("Missed terms:")
                matched_gt = set()
                for ext in extracted:
                    for j, gt in enumerate(gt_terms):
                        if j not in matched_gt and match_terms(ext, gt):
                            matched_gt.add(j)
                            break
                for j, gt in enumerate(gt_terms):
                    if j not in matched_gt:
                        logger.debug(f"  - {gt}")
            
            # Show false positives
            if metrics['fp'] > 0:
                logger.debug("False positives:")
                matched_ext = set()
                for i, ext in enumerate(extracted):
                    for gt in gt_terms:
                        if match_terms(ext, gt):
                            matched_ext.add(i)
                            break
                for i, ext in enumerate(extracted):
                    if i not in matched_ext:
                        logger.debug(f"  - {ext}")
            
            results.append({
                "chunk_id": chunk["chunk_id"],
                "metrics": metrics,
                "extracted_terms": extracted,
                "ground_truth_terms": gt_terms
            })
            
        except Exception as e:
            logger.error(f"Chunk {i+1} failed: {e}")
            results.append({
                "chunk_id": chunk["chunk_id"],
                "error": str(e)
            })
    
    # Aggregate results
    logger.section("AGGREGATE RESULTS")
    
    successful = [r for r in results if "metrics" in r]
    if not successful:
        logger.error("No successful extractions!")
        return
    
    avg_p = sum(r["metrics"]["precision"] for r in successful) / len(successful)
    avg_r = sum(r["metrics"]["recall"] for r in successful) / len(successful)
    avg_h = sum(r["metrics"]["hallucination"] for r in successful) / len(successful)
    avg_f1 = sum(r["metrics"]["f1"] for r in successful) / len(successful)
    avg_time = sum(r["metrics"]["elapsed"] for r in successful) / len(successful)
    
    logger.metric("avg_precision", avg_p, "%")
    logger.metric("avg_recall", avg_r, "%")
    logger.metric("avg_hallucination", avg_h, "%")
    logger.metric("avg_f1", avg_f1)
    logger.metric("avg_time_per_chunk", avg_time, "s")
    
    # Comparison to targets
    logger.section("TARGET COMPARISON")
    logger.info("Target: 95%+ precision, 95%+ recall, <10% hallucination")
    logger.info(f"Achieved: {avg_p:.1%} P, {avg_r:.1%} R, {avg_h:.1%} H")
    
    target_met = avg_p >= 0.90 and avg_r >= 0.85 and avg_h < 0.10
    if target_met:
        logger.info("✅ Close to targets!")
    else:
        logger.info("❌ Not yet meeting targets")
    
    # Comparison to ensemble_verified baseline
    logger.subsection("Comparison to Baseline (ensemble_verified)")
    baseline = {"precision": 0.893, "recall": 0.889, "hallucination": 0.107, "f1": 0.874}
    
    logger.info(f"Precision: {avg_p:.1%} vs {baseline['precision']:.1%} (baseline)")
    logger.info(f"Recall: {avg_r:.1%} vs {baseline['recall']:.1%} (baseline)")
    logger.info(f"Hallucination: {avg_h:.1%} vs {baseline['hallucination']:.1%} (baseline)")
    logger.info(f"F1: {avg_f1:.3f} vs {baseline['f1']:.3f} (baseline)")
    
    better_count = 0
    if avg_p > baseline['precision']:
        better_count += 1
        logger.info("✓ Better precision")
    if avg_r > baseline['recall']:
        better_count += 1
        logger.info("✓ Better recall")
    if avg_h < baseline['hallucination']:
        better_count += 1
        logger.info("✓ Lower hallucination")
    
    if better_count >= 2:
        logger.info("🎉 BEATS BASELINE on 2+ metrics!")
    elif better_count == 1:
        logger.info("⚠️  Beats baseline on 1 metric")
    else:
        logger.info("❌ Does not beat baseline")
    
    # Comparison to sentence_level v1 (filter prompt)
    logger.subsection("Comparison to sentence_level v1 (filter prompt)")
    v1 = {"precision": 0.887, "recall": 0.488, "hallucination": 0.113, "f1": 0.600}
    
    logger.info(f"Precision: {avg_p:.1%} vs {v1['precision']:.1%} (v1)")
    logger.info(f"Recall: {avg_r:.1%} vs {v1['recall']:.1%} (v1)")
    logger.info(f"Hallucination: {avg_h:.1%} vs {v1['hallucination']:.1%} (v1)")
    logger.info(f"F1: {avg_f1:.3f} vs {v1['f1']:.3f} (v1)")
    
    v1_better = 0
    if avg_p > v1['precision']:
        v1_better += 1
    if avg_r > v1['recall']:
        v1_better += 1
    if avg_h < v1['hallucination']:
        v1_better += 1
    logger.info(f"Improvement over v1: {v1_better}/3 metrics")
    
    # Save results
    summary = {
        "strategy": "sentence_level_v2_approval_prompt",
        "num_chunks": len(test_chunks),
        "timestamp": timestamp,
        "aggregate_metrics": {
            "precision": avg_p,
            "recall": avg_r,
            "hallucination": avg_h,
            "f1": avg_f1,
            "avg_time": avg_time
        },
        "per_chunk_results": results,
        "baseline_comparison": {
            "baseline_ensemble_verified": baseline,
            "beats_baseline_count": better_count,
            "baseline_sentence_v1": v1,
            "beats_v1_count": v1_better
        }
    }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {RESULTS_PATH}")
    logger.summary()
    logger.close()
    
    return summary


if __name__ == "__main__":
    run_experiment(num_chunks=10)
