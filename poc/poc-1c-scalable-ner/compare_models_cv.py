#!/usr/bin/env python3
"""Compare CV extraction quality across LLM models.

Runs the candidate_verify classification step with different models
on the same documents + candidates, then scores against ground truth.

Baseline: Sonnet (current production model in cv_v1).

Usage:
    # Set Gemini env vars (see poc/shared/README.md)
    export GEMINI_CLIENT_ID="..."
    export GEMINI_CLIENT_SECRET="..."

    # Run comparison (default 5 docs)
    python compare_models_cv.py

    # More docs
    python compare_models_cv.py --n-docs 10

    # Specific models only
    python compare_models_cv.py --models sonnet gemini-2.5-flash gemini-3-flash-preview
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Imports from the existing POC-1c codebase
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent.parent))  # add poc/ for shared

from parse_so_ner import select_documents
from scoring import many_to_many_score
from hybrid_ner import _parse_terms_json

# Use the shared LLM module (supports both Anthropic + Gemini)
from shared.llm import call_llm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

# Models to compare — order matters for display
DEFAULT_MODELS = [
    "sonnet",                  # baseline (Anthropic Sonnet 4.5)
    "gemini-2.0-flash",        # Gemini 2.0 Flash
    "gemini-2.5-flash",        # Gemini 2.5 Flash
    "gemini-2.5-flash-lite",   # Gemini 2.5 Flash Lite
    "gemini-2.5-pro",          # Gemini 2.5 Pro
    "gemini-3-flash-preview",  # Gemini 3 Flash
    "gemini-3-pro-preview",    # Gemini 3 Pro
]

# Thinking-enabled variants (appended if --thinking is passed)
THINKING_MODELS = [
    ("gemini-2.5-flash+think", "gemini-2.5-flash", 4096),
    ("gemini-2.5-pro+think",   "gemini-2.5-pro",   4096),
]


# ---------------------------------------------------------------------------
# Candidate generation (pure heuristic — shared across all models)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
    "during", "before", "after", "above", "below", "between", "out", "off",
    "over", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "because", "but",
    "and", "or", "if", "while", "that", "this", "these", "those", "it",
    "its", "he", "she", "they", "them", "we", "you", "me", "my", "your",
    "his", "her", "our", "their", "what", "which", "who", "whom",
    "am", "about", "up", "down", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "shouldn't", "couldn't", "can't", "I'm", "I've", "I'll",
}


def generate_candidates(text: str) -> list[str]:
    """Heuristic candidate generation — identical to cv_v1."""
    candidates: set[str] = set()

    words = text.split()
    for w in words:
        clean = w.strip(".,;:!?()[]{}\"'")
        if clean and len(clean) >= 1:
            candidates.add(clean)

    # Bigrams
    for i in range(len(words) - 1):
        w1 = words[i].strip(".,;:!?()[]{}\"'")
        w2 = words[i + 1].strip(".,;:!?()[]{}\"'")
        if w1 and w2:
            candidates.add(f"{w1} {w2}")

    # Hyphenated compounds
    for m in re.finditer(r"\b(\w+(?:-\w+)+)\b", text):
        candidates.add(m.group(1))
    # CamelCase
    for m in re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b", text):
        candidates.add(m.group(1))
    for m in re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text):
        candidates.add(m.group(1))
    # Lowercase words
    for m in re.finditer(r"\b([a-z]{3,30})\b", text):
        candidates.add(m.group(1))
    # Dotted identifiers
    for m in re.finditer(r"\b([\w]+(?:\.[\w]+){1,5})\b", text):
        term = m.group(1)
        if "." in term:
            candidates.add(term)
    # Function calls
    for m in re.finditer(r"\b([\w.]+\(\))", text):
        candidates.add(m.group(1))
    # ALL_CAPS
    for m in re.finditer(r"\b([A-Z][A-Z0-9_]{1,15})\b", text):
        candidates.add(m.group(1))
    # Backtick-quoted
    for m in re.finditer(r"`([^`]{2,50})`", text):
        candidates.add(m.group(1).strip())
    # Dot-prefixed
    for m in re.finditer(r"(?<!\w)(\.[a-zA-Z][a-zA-Z0-9_-]*)\b", text):
        if len(m.group(1)) >= 3:
            candidates.add(m.group(1))
    # Path patterns
    for m in re.finditer(r"(/[^\s]*\{[^}]+\}[^\s]*)", text):
        candidates.add(m.group(1).strip())
    for m in re.finditer(r"(/(?:[\w.+-]+/)+[\w.+-]*)", text):
        candidates.add(m.group(1))
    for m in re.finditer(r"([A-Z]:[\\/][^\s]+)", text):
        candidates.add(m.group(1))
    # Numbers / versions
    for m in re.finditer(r"\b(\d+)\b", text):
        candidates.add(m.group(1))
    for m in re.finditer(r"\b(\d+\.\d+(?:\.\d+)*)\b", text):
        candidates.add(m.group(1))
    # Pipe-delimited
    for m in re.finditer(r"(\w+\|[\w|]*)", text):
        candidates.add(m.group(1))
    # Function calls with args
    for m in re.finditer(r"\b(\w+\([^)]{1,30}\))\s*;?", text):
        full = m.group(0).strip()
        candidates.add(m.group(1))
        if full.endswith(";"):
            candidates.add(full)
    # Quoted single chars
    for m in re.finditer(r'"(\s*[A-Za-z]\s*)"', text):
        candidates.add(m.group(0))

    candidates = {c for c in candidates if c not in _STOPWORDS and len(c) >= 1}
    return sorted(candidates)


# ---------------------------------------------------------------------------
# Classification prompt (same as cv_v1)
# ---------------------------------------------------------------------------

CV_PROMPT_TEMPLATE = """\
Given this StackOverflow text, classify which of the candidate terms below are \
software named entities per the StackOverflow NER annotation guidelines.

ENTITY TYPES (28 categories): Algorithm, Application, Class, Code_Block, \
Data_Structure, Data_Type, Device, Error_Name, File_Name, File_Type, Function, \
HTML_XML_Tag, Keyboard_IP, Language, Library, License, Operating_System, \
Organization, Output_Block, User_Interface_Element, User_Name, Value, Variable, \
Version, Website

CRITICAL — COMMON ENGLISH WORDS ARE ENTITIES in this dataset:
- "page", "list", "set", "tree", "cursor", "global", "setup", "arrow" → YES, entities
- "string", "table", "button", "server", "keyboard", "exception" → YES, entities
- "session", "phone", "image", "row", "column", "private", "long" → YES, entities
- Bare version numbers like "14", "2010", "3.0" → YES, entities (Version)
- Hyphenated compounds: "jQuery-generated", "cross-browser" → YES, entities
- Multi-word names: "visual editor", "command line" → YES, entities
- Example/demo content shown in posts: "ABCDEF|", sample data → YES, entities (Output_Block/Value)
- Terms with special chars: "this[int]", "keydown/keyup/keypress" → YES, entities
- Function calls with args: "free(pt) ;", "size(y, 1)", "buttons(new/edit)" → YES, entities
- Quoted drive letters or single-char refs: '" C "', '" D "' → YES, entities

When a candidate appears in technical context, INCLUDE it. The downstream pipeline \
will handle false positives. Missing entities is MUCH WORSE than including extras.

TEXT:
{text}

CANDIDATE TERMS:
{numbered}

Return ALL terms that could be entities. Be AGGRESSIVE — include borderline cases. \
Output JSON: {{"entities": ["term1", "term2", ...]}}
"""


# ---------------------------------------------------------------------------
# Model runner
# ---------------------------------------------------------------------------

BATCH_SIZE = 80


def classify_candidates(
    text: str,
    candidates: list[str],
    model: str,
    thinking_budget: int | None = None,
) -> tuple[list[str], float]:
    """Run CV classification on candidates using the given model."""
    all_entities: list[str] = []
    total_time = 0.0

    for i in range(0, len(candidates), BATCH_SIZE):
        batch = candidates[i : i + BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {term}" for j, term in enumerate(batch))

        prompt = CV_PROMPT_TEMPLATE.format(text=text, numbered=numbered)

        t0 = time.time()
        response = call_llm(
            prompt,
            model=model,
            max_tokens=3000,
            temperature=0.0,
            thinking_budget=thinking_budget,
            timeout=180 if thinking_budget else 90,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        batch_terms = _parse_terms_json(response)
        all_entities.extend(batch_terms)

    return all_entities, total_time


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CV extraction across LLM models"
    )
    parser.add_argument("--n-docs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models to test (default: all). Use 'model+think' for thinking variants.",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Include thinking-enabled model variants",
    )
    parser.add_argument("--save", action="store_true", help="Save results JSON")
    args = parser.parse_args()

    # Build model list: [(display_name, api_model, thinking_budget | None)]
    model_configs: list[tuple[str, str, int | None]] = []

    if args.models:
        for m in args.models:
            if "+think" in m:
                base = m.replace("+think", "")
                model_configs.append((m, base, 4096))
            else:
                model_configs.append((m, m, None))
    else:
        for m in DEFAULT_MODELS:
            model_configs.append((m, m, None))
        if args.thinking:
            model_configs.extend(THINKING_MODELS)

    # Load data
    test_docs = json.loads((ARTIFACTS_DIR / "test_documents.json").read_text())
    selected = select_documents(test_docs, args.n_docs, seed=args.seed)
    print(f"Selected {len(selected)} test docs (seed={args.seed})")
    print(f"Models:  {', '.join(mc[0] for mc in model_configs)}")
    print()

    # Pre-generate candidates for all docs (model-independent)
    doc_candidates: list[tuple[dict, list[str]]] = []
    for doc in selected:
        text = doc["text"][:5000]
        candidates = generate_candidates(text)
        doc_candidates.append((doc, candidates))
        print(f"  {doc['doc_id']}: {len(doc['gt_terms'])} GT terms, {len(candidates)} candidates")
    print()

    # Run each model
    results: dict[str, dict] = {}

    for display_name, api_model, thinking_budget in model_configs:
        thinking_label = f" (thinking={thinking_budget})" if thinking_budget else ""
        print(f"{'='*70}")
        print(f"MODEL: {display_name}{thinking_label}")
        print(f"{'='*70}")

        totals = {"tp": 0, "fp": 0, "fn": 0, "time": 0.0, "docs": []}

        for doc, candidates in doc_candidates:
            doc_id = doc["doc_id"]
            gt = doc["gt_terms"]
            text = doc["text"][:5000]

            entities, elapsed = classify_candidates(
                text, candidates, api_model, thinking_budget,
            )
            scores = many_to_many_score(entities, gt)

            totals["tp"] += scores["tp"]
            totals["fp"] += scores["fp"]
            totals["fn"] += scores["fn"]
            totals["time"] += elapsed
            totals["docs"].append({
                "doc_id": doc_id,
                "extracted": len(entities),
                "tp": scores["tp"],
                "fp": scores["fp"],
                "fn": scores["fn"],
                "precision": round(scores["precision"], 4),
                "recall": round(scores["recall"], 4),
                "f1": round(scores["f1"], 4),
                "time": round(elapsed, 1),
                "fp_terms": scores["fp_terms"][:15],
                "fn_terms": scores["fn_terms"][:15],
            })

            print(
                f"  {doc_id}: P={scores['precision']*100:5.1f}%  "
                f"R={scores['recall']*100:5.1f}%  "
                f"F1={scores['f1']:.3f}  "
                f"({len(entities)} ext, {elapsed:.1f}s)"
            )

        # Aggregate
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        results[display_name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn,
            "time": round(totals["time"], 1),
            "thinking_budget": thinking_budget,
            "docs": totals["docs"],
        }

        print(f"  --- TOTAL: P={p*100:.1f}% R={r*100:.1f}% F1={f1:.3f} "
              f"TP={tp} FP={fp} FN={fn} ({totals['time']:.1f}s)")
        print()

    # Summary table
    print(f"\n{'='*90}")
    print("SUMMARY — CV Extraction Model Comparison")
    print(f"{'='*90}")
    print(f"{'Model':<26s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Time':>7s}")
    print("-" * 90)

    baseline_f1 = results.get(model_configs[0][0], {}).get("f1", 0)
    for display_name, _, _ in model_configs:
        r = results[display_name]
        delta = r["f1"] - baseline_f1
        delta_str = f" ({delta:+.3f})" if display_name != model_configs[0][0] else " (base)"
        print(
            f"{display_name:<26s} {r['precision']*100:6.1f}% {r['recall']*100:6.1f}% "
            f"{r['f1']:6.3f}{delta_str:>9s} {r['tp']:5d} {r['fp']:5d} {r['fn']:5d} "
            f"{r['time']:6.1f}s"
        )

    # Save
    if args.save:
        out_path = ARTIFACTS_DIR / "results" / "model_comparison_cv.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
