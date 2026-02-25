#!/usr/bin/env python3
"""Vocabulary-enhanced validator test: prefilter + vocab signals + thinking Sonnet → score.

Adds three domain-independent vocabulary signals to the existing pipeline:
1. Domain-independent negatives — generic programming words almost never named entities
2. SO tags lookup — is this an established technical term on StackOverflow?
3. Registry absence — if term not in any known source, soft negative signal

These signals are injected into the Sonnet prompt alongside training evidence.

Usage:
    python run_vocab_validator_test.py
    python run_vocab_validator_test.py --batch-size 25 --thinking-budget 5000
    python run_vocab_validator_test.py --no-training-evidence  # vocab signals only, no train data
"""

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

from scoring import v3_match, many_to_many_score
from utils.llm_provider import call_llm

RAW_PATH = Path("artifacts/results/extraction_raw_10docs.json")
TRAIN_PATH = Path("artifacts/train_documents.json")
SO_TAGS_PATH = Path("artifacts/so_tags.json")

# ---------------------------------------------------------------------------
# Domain-independent negatives
# Generic programming/CS words that are almost NEVER named software entities
# regardless of domain. These are NOT derived from SO NER training data.
# ---------------------------------------------------------------------------

DOMAIN_INDEPENDENT_NEGATIVES = {
    # Generic programming concepts — never a product/tool name in context
    "name", "value", "error", "line", "model", "handler", "service",
    "layout", "resources", "configuration", "selector", "instance",
    "interface", "module", "variable", "variables", "method", "function",
    "class", "object", "property", "type", "element", "tag", "event",
    "controller", "template", "plugin", "library", "framework", "package",
    "protocol", "database", "field", "fields", "target", "level",
    "namespace", "action", "index", "section", "endpoint", "header",
    "footer", "toolbar", "dialog", "node", "child", "parent", "root",
    "listener", "callback", "promise", "response", "body", "path",
    "query", "token", "hash", "flag", "option", "state", "context",
    "provider", "consumer", "adapter", "wrapper", "factory", "proxy",
    "observer", "iterator", "stream", "buffer", "socket", "pipe",
    "channel", "extension", "driver", "engine", "runtime", "preview",
    "code", "file", "print", "meta",
    # Structural/action words
    "main", "src", "bin", "lib", "config", "test", "build",
    # Too-short / fragment
    "h", "m", "x", "id",
    # Generic nouns commonly over-extracted
    "tall", "gain", "thumb", "handle", "seed", "seeds",
    "cascade", "specificity", "siblings",
    # CS vocabulary
    "classpath", "constant", "distribution", "filesystem",
    "repository", "command",
}

# Words that are in the negatives list but ARE sometimes legitimate entities
# when they match a known SO tag or package name exactly — these get upgraded
NEGATIVE_OVERRIDE_IF_KNOWN = {
    "config",  # pip: config package
    "error",   # sometimes a named type
    "model",   # sometimes a specific framework component
    "service", # sometimes a specific named service
}


# ---------------------------------------------------------------------------
# SO Tags index
# ---------------------------------------------------------------------------

def load_so_tags() -> set[str]:
    """Load StackOverflow tags as a set of lowercase strings."""
    if not SO_TAGS_PATH.exists():
        print(f"WARNING: {SO_TAGS_PATH} not found. Run with SO tags for best results.")
        return set()
    with open(SO_TAGS_PATH) as f:
        tags = json.load(f)
    return {t.lower().strip() for t in tags}


# ---------------------------------------------------------------------------
# Pre-filter (safe rules only — same as original)
# ---------------------------------------------------------------------------

DETERMINER_RE = re.compile(
    r"^(a|an|the|some|any|my|this|that|these|those)\s+(.+)$", re.I
)
NUMBER_PERCENT_RE = re.compile(r"^\d+\.?\d*\s+%$")


def prefilter(raw_union: list[str]) -> list[str]:
    union_lower = {t.lower().strip() for t in raw_union}
    kept = []
    for term in raw_union:
        if NUMBER_PERCENT_RE.match(term):
            continue
        m = DETERMINER_RE.match(term)
        if m:
            remainder = m.group(2)
            if remainder.lower().strip() in union_lower:
                continue
        kept.append(term)
    return kept


# ---------------------------------------------------------------------------
# Vocab-enhanced pre-rejection (hard filter for obvious cases)
# ---------------------------------------------------------------------------

def vocab_hard_reject(term: str, so_tags: set[str]) -> bool:
    """Hard reject terms that are clearly not entities.
    
    Only rejects terms with very high confidence — single-char tokens,
    terms in negatives AND not in any known registry.
    
    Returns True if term should be rejected outright (no LLM needed).
    """
    tl = term.lower().strip()
    
    # Single char (h, m, x) — never entities
    if len(tl) <= 1:
        return True
    
    return False  # Everything else goes to Sonnet with vocab evidence


# ---------------------------------------------------------------------------
# Vocab signal computation
# ---------------------------------------------------------------------------

def compute_vocab_signal(term: str, so_tags: set[str]) -> dict:
    """Compute domain-independent vocabulary signals for a candidate term."""
    tl = term.lower().strip()
    
    signals = {
        "is_negative": tl in DOMAIN_INDEPENDENT_NEGATIVES,
        "in_so_tags": tl in so_tags or tl.replace(" ", "-") in so_tags,
        "so_tag_match": None,
    }
    
    # Check if it matches an SO tag (possibly with normalization)
    if signals["in_so_tags"]:
        # Find the actual matching tag
        if tl in so_tags:
            signals["so_tag_match"] = tl
        elif tl.replace(" ", "-") in so_tags:
            signals["so_tag_match"] = tl.replace(" ", "-")
    
    # Also check for partial SO tag matches (e.g., "irb" in "irb" tag)
    # and multi-word matches (e.g., "command line" -> "command-line")
    if not signals["in_so_tags"]:
        # Try with dots: "asp.net" style
        for variant in [tl, tl.replace(" ", "-"), tl.replace(" ", ".")]:
            if variant in so_tags:
                signals["in_so_tags"] = True
                signals["so_tag_match"] = variant
                break
    
    # Override: if a negative is also a known SO tag, it might be an entity
    if signals["is_negative"] and signals["in_so_tags"]:
        if tl in NEGATIVE_OVERRIDE_IF_KNOWN:
            signals["is_negative"] = False  # Give it a chance
    
    return signals


# ---------------------------------------------------------------------------
# Training evidence index (same as original)
# ---------------------------------------------------------------------------

def build_term_index(train_docs: list[dict]) -> dict[str, dict]:
    term_entity_docs: dict[str, list[dict]] = defaultdict(list)
    term_generic_docs: dict[str, list[dict]] = defaultdict(list)

    for doc in train_docs:
        text_lower = doc["text"].lower()
        gt_lower = set(t.lower().strip() for t in doc["gt_terms"])

        seen: set[str] = set()
        for t in doc["gt_terms"]:
            tl = t.lower().strip()
            if tl in seen or len(tl) < 2:
                continue
            seen.add(tl)
            snippet = _extract_snippet(text_lower, tl, doc["text"])
            term_entity_docs[tl].append({"doc_id": doc["doc_id"], "snippet": snippet})

        words_in_text = set(re.findall(r"\b[\w#+.]+\b", text_lower))
        for w in words_in_text:
            if w not in gt_lower and len(w) >= 2 and w not in seen:
                if len(term_generic_docs[w]) < 5:
                    snippet = _extract_snippet(text_lower, w, doc["text"])
                    term_generic_docs[w].append({"doc_id": doc["doc_id"], "snippet": snippet})

    index: dict[str, dict] = {}
    all_terms = set(term_entity_docs.keys()) | set(term_generic_docs.keys())
    for t in all_terms:
        pos = term_entity_docs.get(t, [])
        neg = term_generic_docs.get(t, [])
        total = len(pos) + len(neg)
        index[t] = {
            "entity_ratio": len(pos) / total if total > 0 else 0.5,
            "entity_count": len(pos),
            "generic_count": len(neg),
            "positive_examples": pos[:3],
            "negative_examples": neg[:3],
        }

    return index


def _extract_snippet(text_lower: str, term_lower: str, original_text: str) -> str:
    idx = text_lower.find(term_lower)
    if idx == -1:
        return original_text[:150]
    start = max(0, idx - 60)
    end = min(len(original_text), idx + len(term_lower) + 60)
    snippet = original_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(original_text):
        snippet = snippet + "..."
    return snippet


# ---------------------------------------------------------------------------
# Evidence formatting with vocab signals
# ---------------------------------------------------------------------------

def format_candidate_evidence_with_vocab(
    term: str,
    term_index: dict[str, dict],
    vocab_signal: dict,
    include_training_evidence: bool = True,
) -> str:
    tl = term.lower().strip()
    
    parts = [f'- "{term}"']
    
    # Vocab signals
    vocab_tags = []
    if vocab_signal["is_negative"]:
        vocab_tags.append("GENERIC_WORD (rarely a named entity)")
    if vocab_signal["in_so_tags"]:
        vocab_tags.append(f"SO_TAG={vocab_signal['so_tag_match']}")
    
    if vocab_tags:
        parts.append(f" | vocab: {', '.join(vocab_tags)}")
    
    # Training evidence (if enabled)
    if include_training_evidence:
        info = term_index.get(tl)
        if not info or (info["entity_count"] == 0 and info["generic_count"] == 0):
            parts.append(" | training: UNSEEN")
        else:
            ratio = info["entity_ratio"]
            parts.append(
                f' | entity_ratio={ratio:.0%} '
                f'(entity in {info["entity_count"]} docs, generic in {info["generic_count"]} docs)'
            )
            if 0.15 < ratio < 0.85 and info["positive_examples"]:
                ex = info["positive_examples"][0]
                parts.append(f'\n  [ENTITY usage]: "{ex["snippet"]}"')
    else:
        # No training evidence — vocab signals only
        if not vocab_tags:
            parts.append(" | no external signals")
    
    return "".join(parts)


# ---------------------------------------------------------------------------
# Validator prompt with vocab awareness
# ---------------------------------------------------------------------------

VALIDATOR_PROMPT_VOCAB = """\
You are filtering NER candidates from a StackOverflow post. Your DEFAULT is ENTITY. \
Only REJECT a term if you have clear contextual evidence it is NOT a named entity.

ENTITY TYPES (from StackOverflow NER annotation guidelines, Tabassum et al. ACL 2020):

Application — Software names: Visual Studio, Homebrew, Docker, Chrome, browser, server, IDE
Library — Code frameworks/libraries: jQuery, React, NumPy, .NET, boost, Prism, libc++
Class — Class names: ArrayList, HttpClient, Session, WebClient, ListView
Function — Function/method names: recv(), querySelector(), map(), send()
Language — Programming languages: Java, Python, C#, JavaScript, HTML, CSS, SQL
Data_Type — Type descriptions: string, char, double, integer, boolean, float, long, Byte[]
Data_Structure — Data structures: array, linked list, hash table, table, column, image
UI_Element — Interface elements: button, checkbox, slider, screen, page, form, trackbar
Device — Hardware: phone, GPU, iPhone, camera, keyboard, microphone, CPU
Operating_System — OS names: Linux, iOS, Windows, Android, macOS, unix
Version — Version identifiers: 2.7, XP, v3.2, ES6, 1.9.3, Silverlight 4
File_Type — File formats: json, jar, exe, dll, CSV, WSDL, xaml, jpg, pom.xml
File_Name — File names/paths: /usr/bin/ruby, config/modules.config.php
Website — Website NAMES: MSDN, Google, GitHub, codeplex, codepen.io
Organization — Org names: Apache, Microsoft
Error_Name — Errors: NullPointerException, exception
HTML_XML_Tag — Tags: <div>, <input>, li, <span>
Keyboard_Key — Keys: CTRL+X, ALT, fn, Left, Right, PgUp, Tab
Algorithm — Algorithms/protocols: UDP, DFS, A*-search
Variable — Code variables/constants: user_id, math.inf

VOCABULARY SIGNALS (use these to calibrate your decisions):
- "GENERIC_WORD" = This term is a common programming concept (like "name", "value", \
"handler") that is almost NEVER a named entity. REJECT unless the current document \
clearly uses it as a specific named tool/library/class (e.g., "import handler" or \
"the Handler class").
- "SO_TAG=xxx" = This term has a StackOverflow tag, meaning it's a recognized technical \
concept. This is a POSITIVE signal — StackOverflow tags often correspond to named \
technologies. Consider this evidence FOR entity status.
- Terms with NEITHER signal and no training data are truly unknown — judge from context.

CALIBRATION RULES:
1. entity_ratio shows how often human annotators marked this term as entity in \
741 training documents. Treat entity_ratio > 20% as PRESUMPTION OF ENTITY.
2. GENERIC_WORD terms need STRONG contextual evidence to be ENTITY — just appearing \
in technical text is not enough. "the server crashed" = REJECT. "deploy to Server v2" = ENTITY.
3. SO_TAG terms in technical context are likely entities — they name specific technologies.
4. When multiple similar terms appear (height/width/padding), decide consistently.

ONLY REJECT when:
- The term is descriptive phrasing: "vertical orientation", "entropy pool"
- The term is a process/action: "loading", "serializing"  
- The term is generic CS meta-vocabulary used descriptively: "the code", "my method"
- The term is a GENERIC_WORD used generically (not naming a specific tool/library)
- The term is a bare number that is NOT a version
- The term is a sentence fragment

DOCUMENT TEXT:
{text}

CANDIDATES WITH EVIDENCE:
{candidates}

For each candidate, decide ENTITY or REJECT.
Output JSON: {{"results": [{{"term": "...", "decision": "ENTITY|REJECT"}}]}}
"""


def parse_validator_response(response: str) -> dict[str, str]:
    results = {}
    obj_match = re.search(r"\{.*\}", response, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict) and "results" in parsed:
                for item in parsed["results"]:
                    if isinstance(item, dict) and "term" in item and "decision" in item:
                        results[item["term"]] = item["decision"]
                return results
        except json.JSONDecodeError:
            pass

    arr_match = re.search(r"\[.*\]", response, re.DOTALL)
    if arr_match:
        try:
            items = json.loads(arr_match.group())
            for item in items:
                if isinstance(item, dict) and "term" in item and "decision" in item:
                    results[item["term"]] = item["decision"]
            return results
        except json.JSONDecodeError:
            pass

    return results


def validate_batch(
    text: str,
    candidates: list[str],
    term_index: dict[str, dict],
    so_tags: set[str],
    batch_size: int = 25,
    thinking_budget: int = 5000,
    include_training_evidence: bool = True,
) -> list[str]:
    kept = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]

        evidence_lines = []
        for term in batch:
            vocab_signal = compute_vocab_signal(term, so_tags)
            evidence_lines.append(
                format_candidate_evidence_with_vocab(
                    term, term_index, vocab_signal, include_training_evidence
                )
            )
        candidates_block = "\n".join(evidence_lines)

        prompt = VALIDATOR_PROMPT_VOCAB.format(text=text, candidates=candidates_block)

        response = call_llm(
            prompt,
            model="sonnet",
            max_tokens=4000,
            thinking_budget=thinking_budget,
        )
        results = parse_validator_response(response)

        for term in batch:
            decision = results.get(term, "ENTITY")
            if decision != "REJECT":
                kept.append(term)

    return kept


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--thinking-budget", type=int, default=5000)
    parser.add_argument(
        "--no-training-evidence", action="store_true",
        help="Disable training evidence (vocab signals only)",
    )
    args = parser.parse_args()

    include_training = not args.no_training_evidence

    with open(RAW_PATH) as f:
        docs = json.load(f)

    train_docs = []
    term_index = {}
    if include_training:
        with open(TRAIN_PATH) as f:
            train_docs = json.load(f)
        print(f"Loaded {len(docs)} test docs, {len(train_docs)} train docs")
        print("Building term index...")
        term_index = build_term_index(train_docs)
        print(f"Term index: {len(term_index)} terms")
    else:
        print(f"Loaded {len(docs)} test docs (NO training evidence)")

    so_tags = load_so_tags()
    print(f"SO tags: {len(so_tags)} tags")
    print(f"Domain-independent negatives: {len(DOMAIN_INDEPENDENT_NEGATIVES)} terms")
    print(f"Batch size: {args.batch_size}, thinking budget: {args.thinking_budget}")
    print(f"Training evidence: {'YES' if include_training else 'NO'}")

    # Show vocab signal stats for candidates
    total_candidates = 0
    total_negative = 0
    total_so_tag = 0
    total_hard_reject = 0

    for doc in docs:
        for term in prefilter(doc["raw_union"]):
            total_candidates += 1
            sig = compute_vocab_signal(term, so_tags)
            if sig["is_negative"]:
                total_negative += 1
            if sig["in_so_tags"]:
                total_so_tag += 1
            if vocab_hard_reject(term, so_tags):
                total_hard_reject += 1

    print(f"\nVocab signal coverage on {total_candidates} candidates:")
    print(f"  GENERIC_WORD: {total_negative} ({total_negative/total_candidates*100:.1f}%)")
    print(f"  SO_TAG match: {total_so_tag} ({total_so_tag/total_candidates*100:.1f}%)")
    print(f"  Hard reject:  {total_hard_reject} ({total_hard_reject/total_candidates*100:.1f}%)")

    all_validated = []
    total_time = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        gt = doc["gt_terms"]
        raw = doc["raw_union"]

        # Pre-filter
        filtered = prefilter(raw)

        # Hard vocab reject
        after_vocab = [t for t in filtered if not vocab_hard_reject(t, so_tags)]

        print(f"\n{'='*70}")
        print(f"{doc_id}: raw={len(raw)} → prefiltered={len(filtered)} → after_vocab={len(after_vocab)} (GT={len(gt)})")

        t0 = time.time()
        validated = validate_batch(
            doc["text"], after_vocab, term_index, so_tags,
            batch_size=args.batch_size,
            thinking_budget=args.thinking_budget,
            include_training_evidence=include_training,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        scores = many_to_many_score(validated, gt)
        p = scores["precision"] * 100
        r = scores["recall"] * 100
        f1 = scores["f1"]

        print(f"  → validated={len(validated)} | P={p:.1f}% R={r:.1f}% F1={f1:.3f} ({elapsed:.1f}s)")

        fps = [t for t in validated if not any(v3_match(t, g) for g in gt)]
        fns = [g for g in gt if not any(v3_match(t, g) for t in validated)]

        if fps:
            fp_details = []
            for fp in fps:
                sig = compute_vocab_signal(fp, so_tags)
                tags = []
                if sig["is_negative"]:
                    tags.append("NEG")
                if sig["in_so_tags"]:
                    tags.append(f"SO:{sig['so_tag_match']}")
                tag_str = f" [{','.join(tags)}]" if tags else ""
                fp_details.append(f"'{fp}'{tag_str}")
            print(f"  FPs ({len(fps)}): {fp_details}")
        if fns:
            print(f"  FNs ({len(fns)}): {fns}")

        all_validated.append({
            "doc_id": doc_id,
            "gt_terms": gt,
            "validated_terms": validated,
            "precision": scores["precision"],
            "recall": scores["recall"],
            "f1": f1,
            "fps": fps,
            "fns": fns,
        })

    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS (vocab-enhanced)")
    print(f"{'='*70}")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for r in all_validated:
        gt_count = len(r["gt_terms"])
        fp_count = len(r["fps"])
        fn_count = len(r["fns"])
        tp_count = gt_count - fn_count
        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Precision: {precision*100:.1f}% ({total_tp}/{total_tp + total_fp})")
    print(f"  Recall:    {recall*100:.1f}% ({total_tp}/{total_tp + total_fn})")
    print(f"  F1:        {f1:.3f}")
    print(f"  Total time: {total_time:.1f}s")

    # Compare with baseline
    print(f"\n  COMPARISON:")
    print(f"  Baseline (no vocab):  P=71.8% R=95.7% F1=0.821")
    print(f"  Strategy_v6 (train):  P=90.7% R=95.8% F1=0.932")
    print(f"  This run (vocab):     P={precision*100:.1f}% R={recall*100:.1f}% F1={f1:.3f}")

    # FP analysis by vocab signal
    print(f"\n  FP ANALYSIS BY VOCAB SIGNAL:")
    all_fps_with_signal = []
    for r_doc in all_validated:
        for fp in r_doc["fps"]:
            sig = compute_vocab_signal(fp, so_tags)
            all_fps_with_signal.append((r_doc["doc_id"], fp, sig))

    neg_fps = [(d, t) for d, t, s in all_fps_with_signal if s["is_negative"]]
    so_fps = [(d, t) for d, t, s in all_fps_with_signal if s["in_so_tags"]]
    no_signal_fps = [(d, t) for d, t, s in all_fps_with_signal if not s["is_negative"] and not s["in_so_tags"]]

    print(f"  FPs marked GENERIC_WORD (should have been rejected): {len(neg_fps)}")
    for doc_id, fp in sorted(neg_fps, key=lambda x: x[1].lower()):
        print(f"    [{doc_id}] '{fp}'")

    print(f"  FPs with SO_TAG (Sonnet kept despite being known): {len(so_fps)}")
    for doc_id, fp in sorted(so_fps, key=lambda x: x[1].lower()):
        sig = compute_vocab_signal(fp, so_tags)
        print(f"    [{doc_id}] '{fp}' (tag={sig['so_tag_match']})")

    print(f"  FPs with NO signal (unknown terms): {len(no_signal_fps)}")
    for doc_id, fp in sorted(no_signal_fps, key=lambda x: x[1].lower()):
        print(f"    [{doc_id}] '{fp}'")

    # Save results
    config_name = "vocab_enhanced" + ("_no_train" if not include_training else "")
    out_path = Path(f"artifacts/results/validator_{config_name}_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "batch_size": args.batch_size,
                "thinking_budget": args.thinking_budget,
                "prompt": "vocab_enhanced_thinking_v1",
                "include_training_evidence": include_training,
                "domain_independent_negatives": len(DOMAIN_INDEPENDENT_NEGATIVES),
                "so_tags": len(so_tags),
            },
            "aggregate": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "total_tp": total_tp,
                "total_fp": total_fp,
                "total_fn": total_fn,
            },
            "docs": all_validated,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
