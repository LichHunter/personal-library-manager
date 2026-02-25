#!/usr/bin/env python3
"""Cross-validate candidate negative words across multiple large NER datasets.

For each word, checks: does it EVER appear as an entity annotation across
~454K sentences from 5 independent NER datasets + our 741 SO NER docs?

Words that are 0% entity rate across ALL datasets with 1K+ total appearances
are confirmed safe negatives.

Datasets:
1. Few-NERD (188K sentences, 66 fine-grained types incl. product-software)
2. OntoNotes 5 (76K sentences, 18 types incl. PRODUCT)
3. CoNLL-2003 (20K sentences, PER/LOC/ORG/MISC)
4. MultiNERD (164K sentences, 15 types)
5. WNUT 2017 (5.7K sentences, 6 types incl. product)
6. SO NER (741 docs, 20 software entity types) — our training data
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

TRAIN_PATH = Path("artifacts/train_documents.json")

# Words to check — our current 0%-in-SO-NER list plus extras from FP analysis
# These are all technical-sounding words that an NER extractor might plausibly extract
CANDIDATE_WORDS = {
    # Current confirmed 0% in SO NER (high doc count)
    "code", "method", "function", "class", "object", "line", "type",
    "database", "variable", "test", "element", "build", "property",
    "library", "option", "instance", "variables", "handle", "framework",
    "path", "action", "level", "print", "child", "package",
    "configuration", "context", "module", "state", "index", "parent",
    "extension", "runtime", "repository", "engine", "target", "callback",
    "interface", "protocol", "namespace", "constant", "iterator", "meta",
    "proxy", "selector", "wrapper", "consumer", "distribution", "listener",
    "promise", "provider", "socket", "factory", "filesystem", "pipe",

    # Words we KNOW are sometimes entities in SO NER — checking cross-dataset
    "event", "node", "header", "controller", "service", "handler",
    "query", "model", "name", "value", "error", "file", "command",
    "response", "template", "field", "tag", "layout", "plugin",
    "main", "config", "buffer", "dialog", "driver", "endpoint",
    "token", "body", "hash", "flag", "root", "stream", "section",
    "src", "bin", "lib", "id", "preview", "adapter", "footer",
    "toolbar", "observer",

    # Additional programming meta-vocabulary to test
    "parameter", "argument", "attribute", "method", "procedure",
    "exception", "component", "resource", "resources", "view",
    "data", "container", "collection", "request", "session",
    "connection", "thread", "process", "message", "output",
    "input", "format", "string", "table", "column", "row",
    "key", "list", "array", "map", "set", "stack", "queue",
    "pattern", "schema", "api", "server", "client", "browser",
    "window", "screen", "page", "form", "button", "image",
    "text", "link", "menu", "panel", "tab", "icon",
}


def analyze_hf_dataset(dataset_name: str, config: str | None, splits: list[str],
                       tokens_col: str, tags_col: str,
                       words: set[str]) -> dict[str, dict]:
    """Analyze a HuggingFace NER dataset for word entity rates."""
    print(f"\n  Loading {dataset_name} (config={config})...")

    try:
        if config:
            ds = load_dataset(dataset_name, config, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_name, trust_remote_code=True)
    except Exception as e:
        print(f"  ERROR loading {dataset_name}: {e}")
        return {}

    word_stats: dict[str, dict] = defaultdict(lambda: {"appearances": 0, "entity": 0})

    total_sentences = 0
    for split in splits:
        if split not in ds:
            print(f"  WARNING: split '{split}' not found in {dataset_name}")
            continue

        data = ds[split]
        total_sentences += len(data)

        for example in data:
            tokens = example[tokens_col]
            tags = example[tags_col]

            # Build set of lowercased tokens that are entities (tag != 0 = O)
            entity_tokens_lower = set()
            non_entity_tokens_lower = set()

            for tok, tag in zip(tokens, tags):
                tl = tok.lower().strip()
                if len(tl) < 2:
                    continue
                if tag != 0:  # Any non-O tag = entity
                    entity_tokens_lower.add(tl)
                else:
                    non_entity_tokens_lower.add(tl)

            # Check each candidate word
            for w in words:
                wl = w.lower()
                if wl in entity_tokens_lower:
                    word_stats[wl]["appearances"] += 1
                    word_stats[wl]["entity"] += 1
                elif wl in non_entity_tokens_lower:
                    word_stats[wl]["appearances"] += 1

    print(f"  Processed {total_sentences} sentences from {dataset_name}")
    return dict(word_stats)


def analyze_so_ner(words: set[str]) -> dict[str, dict]:
    """Analyze our SO NER training data."""
    print(f"\n  Loading SO NER training data...")
    with open(TRAIN_PATH) as f:
        docs = json.load(f)

    word_stats: dict[str, dict] = defaultdict(lambda: {"appearances": 0, "entity": 0})

    for doc in docs:
        text_lower = doc["text"].lower()
        gt_lower = set(t.lower().strip() for t in doc["gt_terms"])
        words_in_text = set(re.findall(r"\b[\w#+.]+\b", text_lower))

        for w in words:
            wl = w.lower()
            if wl in words_in_text:
                word_stats[wl]["appearances"] += 1
                if wl in gt_lower:
                    word_stats[wl]["entity"] += 1

    print(f"  Processed {len(docs)} docs from SO NER")
    return dict(word_stats)


def main() -> None:
    words = {w.lower() for w in CANDIDATE_WORDS}
    print(f"Checking {len(words)} candidate words across 6 NER datasets\n")

    # ── Dataset configs ──────────────────────────────────────────────
    datasets_config = [
        {
            "name": "Few-NERD",
            "hf_id": "DFKI-SLT/few-nerd",
            "config": "supervised",
            "splits": ["train", "validation", "test"],
            "tokens_col": "tokens",
            "tags_col": "fine_ner_tags",
        },
        {
            "name": "OntoNotes5",
            "hf_id": "tner/ontonotes5",
            "config": "ontonotes5",
            "splits": ["train", "validation", "test"],
            "tokens_col": "tokens",
            "tags_col": "tags",
        },
        {
            "name": "CoNLL-2003",
            "hf_id": "eriktks/conll2003",
            "config": None,
            "splits": ["train", "validation", "test"],
            "tokens_col": "tokens",
            "tags_col": "ner_tags",
        },
        {
            "name": "MultiNERD",
            "hf_id": "Babelscape/multinerd",
            "config": None,
            "splits": ["train", "validation", "test"],
            "tokens_col": "tokens",
            "tags_col": "ner_tags",
            "lang_filter": "en",
        },
        {
            "name": "WNUT-2017",
            "hf_id": "leondz/wnut_17",
            "config": None,
            "splits": ["train", "validation", "test"],
            "tokens_col": "tokens",
            "tags_col": "ner_tags",
        },
    ]

    all_results: dict[str, dict[str, dict]] = {}

    # ── HuggingFace datasets ────────────────────────────────────────
    for cfg in datasets_config:
        ds_name = cfg["name"]
        print(f"{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        if "lang_filter" in cfg:
            # MultiNERD: need to filter by language
            print(f"  Loading {cfg['hf_id']} (filtering lang={cfg['lang_filter']})...")
            try:
                ds = load_dataset(cfg["hf_id"], trust_remote_code=True)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            word_stats: dict[str, dict] = defaultdict(lambda: {"appearances": 0, "entity": 0})
            total = 0

            for split in cfg["splits"]:
                if split not in ds:
                    continue
                data = ds[split]
                # Filter to English
                for example in data:
                    if example.get("lang") != cfg["lang_filter"]:
                        continue
                    total += 1
                    tokens = example[cfg["tokens_col"]]
                    tags = example[cfg["tags_col"]]

                    entity_tokens_lower = set()
                    non_entity_tokens_lower = set()
                    for tok, tag in zip(tokens, tags):
                        tl = tok.lower().strip()
                        if len(tl) < 2:
                            continue
                        if tag != 0:
                            entity_tokens_lower.add(tl)
                        else:
                            non_entity_tokens_lower.add(tl)

                    for w in words:
                        if w in entity_tokens_lower:
                            word_stats[w]["appearances"] += 1
                            word_stats[w]["entity"] += 1
                        elif w in non_entity_tokens_lower:
                            word_stats[w]["appearances"] += 1

            print(f"  Processed {total} English sentences from {ds_name}")
            all_results[ds_name] = dict(word_stats)
        else:
            result = analyze_hf_dataset(
                cfg["hf_id"], cfg["config"], cfg["splits"],
                cfg["tokens_col"], cfg["tags_col"], words,
            )
            all_results[ds_name] = result

    # ── SO NER ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Dataset: SO-NER")
    print(f"{'='*60}")
    all_results["SO-NER"] = analyze_so_ner(words)

    # ── Compile results ─────────────────────────────────────────────
    ds_names = ["Few-NERD", "OntoNotes5", "CoNLL-2003", "MultiNERD", "WNUT-2017", "SO-NER"]

    print(f"\n\n{'='*120}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*120}")

    # Header
    header = f"{'Word':20s}"
    for ds in ds_names:
        header += f" | {ds:>14s}"
    header += f" | {'TOTAL':>8s} | {'VERDICT':>10s}"
    print(header)
    print("-" * len(header))

    # Sort words: first confirmed safe (0% everywhere), then unsafe
    word_verdicts = []

    for w in sorted(words):
        total_app = 0
        total_ent = 0
        any_entity = False
        cells = []

        for ds in ds_names:
            stats = all_results.get(ds, {}).get(w, {"appearances": 0, "entity": 0})
            app = stats["appearances"]
            ent = stats["entity"]
            total_app += app
            total_ent += ent
            if ent > 0:
                any_entity = True

            if app == 0:
                cells.append(f"{'---':>14s}")
            elif ent == 0:
                cells.append(f"{'0/'+str(app):>14s}")
            else:
                pct = ent / app * 100
                cells.append(f"{f'{ent}/{app} ({pct:.0f}%)':>14s}")

        if total_app == 0:
            verdict = "NO_DATA"
        elif total_ent == 0 and total_app >= 1000:
            verdict = "SAFE_1K+"
        elif total_ent == 0 and total_app >= 100:
            verdict = "SAFE_100+"
        elif total_ent == 0:
            verdict = "safe_low"
        else:
            pct = total_ent / total_app * 100
            if pct <= 1.0 and total_app >= 100:
                verdict = f"BORDERLINE"
            else:
                verdict = f"UNSAFE"

        word_verdicts.append((w, total_app, total_ent, any_entity, verdict, cells))

    # Print safe first, then unsafe
    print("\n── CONFIRMED SAFE (0% across ALL datasets, 1K+ appearances) ──\n")
    header = f"{'Word':20s}"
    for ds in ds_names:
        header += f" | {ds:>14s}"
    header += f" | {'TOTAL':>8s}"
    print(header)
    print("-" * len(header))

    safe_1k = [(w, ta, te, ae, v, c) for w, ta, te, ae, v, c in word_verdicts if v == "SAFE_1K+"]
    safe_1k.sort(key=lambda x: -x[1])
    for w, ta, te, ae, v, cells in safe_1k:
        row = f"{w:20s}"
        for cell in cells:
            row += f" | {cell}"
        row += f" | {f'0/{ta}':>8s}"
        print(row)

    print(f"\n── CONFIRMED SAFE (0%, 100-999 appearances) ──\n")
    print(header)
    print("-" * len(header))
    safe_100 = [(w, ta, te, ae, v, c) for w, ta, te, ae, v, c in word_verdicts if v == "SAFE_100+"]
    safe_100.sort(key=lambda x: -x[1])
    for w, ta, te, ae, v, cells in safe_100:
        row = f"{w:20s}"
        for cell in cells:
            row += f" | {cell}"
        row += f" | {f'0/{ta}':>8s}"
        print(row)

    print(f"\n── UNSAFE (entity in at least one dataset) ──\n")
    unsafe_header = f"{'Word':20s}"
    for ds in ds_names:
        unsafe_header += f" | {ds:>14s}"
    unsafe_header += f" | {'TOTAL':>8s} | {'RATE':>7s}"
    print(unsafe_header)
    print("-" * len(unsafe_header))

    unsafe = [(w, ta, te, ae, v, c) for w, ta, te, ae, v, c in word_verdicts if v in ("UNSAFE", "BORDERLINE")]
    unsafe.sort(key=lambda x: -x[2] / max(x[1], 1))
    for w, ta, te, ae, v, cells in unsafe:
        row = f"{w:20s}"
        for cell in cells:
            row += f" | {cell}"
        pct = te / ta * 100 if ta > 0 else 0
        row += f" | {f'{te}/{ta}':>8s} | {pct:>6.1f}%"
        print(row)

    # ── Low data ────────────────────────────────────────────────────
    low_data = [(w, ta, te, ae, v, c) for w, ta, te, ae, v, c in word_verdicts if v in ("safe_low", "NO_DATA")]
    if low_data:
        print(f"\n── INSUFFICIENT DATA (<100 appearances) ──\n")
        for w, ta, te, ae, v, cells in sorted(low_data, key=lambda x: -x[1]):
            print(f"  {w:25s}  {ta:4d} appearances, {te} entities")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    print(f"  Words checked:        {len(words)}")
    print(f"  SAFE (0%, 1K+ apps):  {len(safe_1k)}")
    print(f"  SAFE (0%, 100+ apps): {len(safe_100)}")
    print(f"  UNSAFE (>0% in any):  {len(unsafe)}")
    print(f"  Insufficient data:    {len(low_data)}")

    # Save results as JSON
    save_data = {
        "safe_1k": [w for w, *_ in safe_1k],
        "safe_100": [w for w, *_ in safe_100],
        "unsafe": [w for w, *_ in unsafe],
        "low_data": [w for w, *_ in low_data],
        "details": {},
    }
    for w, ta, te, ae, v, cells in word_verdicts:
        per_ds = {}
        for i, ds in enumerate(ds_names):
            stats = all_results.get(ds, {}).get(w, {"appearances": 0, "entity": 0})
            per_ds[ds] = stats
        save_data["details"][w] = {
            "total_appearances": ta,
            "total_entity": te,
            "verdict": v,
            "per_dataset": per_ds,
        }

    out_path = Path("artifacts/cross_validated_negatives.json")
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
