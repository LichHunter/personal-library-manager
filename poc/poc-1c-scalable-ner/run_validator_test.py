#!/usr/bin/env python3
"""Offline validator test: prefilter → thinking Sonnet → score.

Uses inclusion-biased prompt with extended thinking and training evidence.

Usage:
    python run_validator_test.py
    python run_validator_test.py --batch-size 25 --thinking-budget 5000
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

# ---------------------------------------------------------------------------
# Pre-filter (safe rules only)
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
# Training evidence index
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


def format_candidate_evidence(
    term: str,
    term_index: dict[str, dict],
) -> str:
    tl = term.lower().strip()
    info = term_index.get(tl)

    if not info or (info["entity_count"] == 0 and info["generic_count"] == 0):
        return f'- "{term}" | training: UNSEEN TERM (no data)'

    ratio = info["entity_ratio"]
    line = (
        f'- "{term}" | entity_ratio={ratio:.0%} '
        f'(entity in {info["entity_count"]} docs, generic in {info["generic_count"]} docs)'
    )

    if 0.15 < ratio < 0.85 and info["positive_examples"]:
        ex = info["positive_examples"][0]
        line += f'\n  [ENTITY usage]: "{ex["snippet"]}"'

    return line


# ---------------------------------------------------------------------------
# Thinking Sonnet inclusion-biased validator
# ---------------------------------------------------------------------------

VALIDATOR_PROMPT = """\
You are filtering NER candidates from a StackOverflow post. Your DEFAULT is ENTITY. \
Only REJECT a term if you have clear contextual evidence it is NOT a named entity.

ENTITY TYPES (from StackOverflow NER annotation guidelines, Tabassum et al. ACL 2020):

Application — Software names (not code libraries): Visual Studio, Weka, Homebrew, \
FFmpeg, Docker, Chrome, browser, server, console, IDE
Library — Code frameworks/libraries (what you #include/import): jQuery, React, \
NumPy, .NET, boost, Prism, libc++, SOAP, OAuth
Class — Class names (library or user-defined, incl. C structs): ArrayList, \
HttpClient, Session, WebClient, ListView, IEnumerable
Function — Function/method names: recv(), querySelector(), map(), send(), post()
Language — Programming languages: Java, Python, C#, JavaScript, HTML, CSS, SQL, C++11
Data_Type — Type descriptions: string, char, double, integer, boolean, float, long, \
private, var, Byte[], bytearrays
Data_Structure — Data structures: array, linked list, hash table, heap, tree, stack, \
table, row, column, image, HashMap, graph, container
UI_Element — Interface elements: button, checkbox, slider, screen, page, form, \
trackbar, scrollbar, pad, text box, wizard
Device — Hardware: phone, GPU, iPhone, camera, keyboard, microphone, CPU
Operating_System — OS names: Linux, iOS, Windows, Android, macOS, unix
Version — Version identifiers: 2.7, XP, v3.2, ES6, 1.9.3, Silverlight 4
File_Type — File formats: json, jar, exe, dll, CSV, WSDL, xaml, jpg, pom.xml
File_Name — File names/paths: /usr/bin/ruby, config/modules.config.php
Website — Website NAMES (not URLs): MSDN, Google, GitHub, codeplex, codepen.io
Organization — Org names: Apache, Microsoft, Microsoft Research
Error_Name — Errors: Overflow, NullPointerException, exception
HTML_XML_Tag — Tags: <div>, <input>, li, <span>
Keyboard_Key — Keys: CTRL+X, ALT, fn, Left, Right, PgUp, Tab
Algorithm — Algorithms/protocols: UDP, DFS, A*-search, bubblesort
Variable — Code variables/constants: user_id, math.inf, swing.color

CALIBRATION RULES:
1. entity_ratio shows how often human annotators marked this term as entity in \
741 training documents. Treat entity_ratio > 20% as PRESUMPTION OF ENTITY — \
only reject if the current document clearly uses it generically.
2. COMMON ENGLISH WORDS ARE OFTEN ENTITIES: "height" in CSS = Data_Type/UI property, \
"column" in SQL/layout = Data_Structure, "container" in Docker/CSS = entity, \
"configuration" = Class, "kernel" = Application, "global" = Variable/Data_Type, \
"borders" = UI_Element/CSS property, "image" = Data_Structure, "key" = Data_Structure
3. When multiple similar terms appear (height/width/padding, or create/update/delete), \
they likely form a concept family — decide consistently. If one is entity, all are.
4. Unseen terms (no training data): judge purely from context. Technical terms in \
technical context → ENTITY.

ONLY REJECT when:
- The term is descriptive phrasing: "vertical orientation", "entropy pool"
- The term is a process/action: "loading", "serializing", "reprocess"
- The term is generic CS meta-vocabulary used descriptively: "the code", "my method"
- The term is a bare number that is NOT a version: counts, IDs, percentages
- The term is a sentence fragment: "hardware level", "code examples", "code into"

DOCUMENT TEXT:
{text}

CANDIDATES WITH TRAINING EVIDENCE:
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
    batch_size: int = 25,
    thinking_budget: int = 5000,
) -> list[str]:
    kept = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]

        evidence_lines = []
        for term in batch:
            evidence_lines.append(format_candidate_evidence(term, term_index))
        candidates_block = "\n".join(evidence_lines)

        prompt = VALIDATOR_PROMPT.format(text=text, candidates=candidates_block)

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
    args = parser.parse_args()

    with open(RAW_PATH) as f:
        docs = json.load(f)
    with open(TRAIN_PATH) as f:
        train_docs = json.load(f)

    print(f"Loaded {len(docs)} test docs, {len(train_docs)} train docs")
    print(f"Batch size: {args.batch_size}, thinking budget: {args.thinking_budget}")
    print("Building term index...")
    term_index = build_term_index(train_docs)
    print(f"Term index: {len(term_index)} terms")

    all_validated = []
    total_time = 0

    for doc in docs:
        doc_id = doc["doc_id"]
        gt = doc["gt_terms"]
        raw = doc["raw_union"]

        filtered = prefilter(raw)

        print(f"\n{'='*70}")
        print(f"{doc_id}: raw={len(raw)} → prefiltered={len(filtered)} (GT={len(gt)})")

        t0 = time.time()
        validated = validate_batch(
            doc["text"], filtered, term_index,
            batch_size=args.batch_size,
            thinking_budget=args.thinking_budget,
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
            print(f"  FPs ({len(fps)}): {fps}")
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
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0

    for r in all_validated:
        gt_count = len(r["gt_terms"])
        fp_count = len(r["fps"])
        fn_count = len(r["fns"])
        tp_count = gt_count - fn_count
        total_tp += tp_count
        total_fp += fp_count
        total_fn += fn_count
        total_gt += gt_count

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Precision: {precision*100:.1f}% ({total_tp}/{total_tp + total_fp})")
    print(f"  Recall:    {recall*100:.1f}% ({total_tp}/{total_tp + total_fn})")
    print(f"  F1:        {f1:.3f}")
    print(f"  Total time: {total_time:.1f}s")

    all_fps = []
    all_fns = []
    for r in all_validated:
        for fp in r["fps"]:
            all_fps.append((r["doc_id"], fp))
        for fn in r["fns"]:
            all_fns.append((r["doc_id"], fn))

    print(f"\n  ALL FPs ({len(all_fps)}):")
    for doc_id, fp in sorted(all_fps, key=lambda x: x[1].lower()):
        print(f"    [{doc_id}] '{fp}'")

    print(f"\n  ALL FNs ({len(all_fns)}):")
    for doc_id, fn in sorted(all_fns, key=lambda x: x[1].lower()):
        print(f"    [{doc_id}] '{fn}'")

    out_path = Path("artifacts/results/validator_test_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "config": {
                "batch_size": args.batch_size,
                "thinking_budget": args.thinking_budget,
                "prompt": "inclusion_biased_thinking_v1",
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
