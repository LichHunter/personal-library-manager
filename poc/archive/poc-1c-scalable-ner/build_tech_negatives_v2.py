#!/usr/bin/env python3
"""Build tech-domain negatives using ALL available labeled data filtered to tech.

Data sources:
1. SO NER (1,237 docs) — all tech, count per-sentence
2. Few-NERD (188K sentences) — filter to tech-domain sentences only

For Few-NERD tech filtering:
- Sentences containing product-software / product-other entities → definitely tech
- Sentences containing known tech keywords (python, java, server, database, etc.)
  in any position → likely tech

Unit of analysis: SENTENCE (not document). Each sentence where a word appears
as a non-entity token = one labeled negative observation.

Threshold: word must have ≥1000 labeled chunks with 0% entity rate.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset

ARTIFACTS = Path("artifacts")

# Tech keywords for filtering Few-NERD sentences to tech domain
# These are unambiguous tech indicators — if a sentence contains one of these
# as a token, the sentence is about technology
TECH_KEYWORDS = {
    # Programming languages
    "python", "java", "javascript", "typescript", "ruby", "php", "golang",
    "rust", "swift", "kotlin", "scala", "perl", "haskell", "erlang",
    "clojure", "lua", "matlab", "fortran", "cobol", "assembly",
    "c++", "c#", "objective-c",
    # Platforms / OS
    "linux", "windows", "macos", "android", "ios", "unix", "ubuntu",
    "debian", "fedora", "centos", "freebsd",
    # Frameworks / tools
    "django", "flask", "rails", "react", "angular", "vue", "nodejs",
    "tensorflow", "pytorch", "kubernetes", "docker", "nginx", "apache",
    "jenkins", "gradle", "maven", "webpack", "npm", "pip", "conda",
    "git", "github", "gitlab", "bitbucket",
    # Databases
    "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "sqlite", "oracle", "cassandra", "dynamodb",
    # Cloud / infra
    "aws", "azure", "gcp", "heroku", "cloudflare",
    # Concepts that strongly signal tech context (less ambiguous ones)
    "api", "sql", "html", "css", "xml", "json", "yaml", "http",
    "https", "tcp", "udp", "ssh", "ssl", "tls", "dns",
    "regex", "oauth", "jwt", "crud", "orm", "mvc", "sdk",
    "ide", "cli", "gui",
    "algorithm", "compiler", "debugger", "runtime", "bytecode",
    "microservice", "middleware", "backend", "frontend",
    "repository", "codebase", "sourcecode", "README",
    "stackoverflow", "github.com",
    # Software products
    "chrome", "firefox", "safari", "vscode", "intellij", "eclipse",
    "photoshop", "illustrator", "slack", "jira", "confluence",
}

# Few-NERD fine_ner_tags that indicate tech/product content
# From the tag schema: product-software=58, product-other=59
TECH_ENTITY_TAG_IDS = {58, 59}

# Also: organization-company (29), product-car (56), product-airplane (55),
# product-ship (57), product-train (60) — less relevant but org-company
# often appears in tech context
TECH_ADJACENT_TAG_IDS = {29}  # organization-company


def split_into_sentences(text: str) -> list[str]:
    """Simple sentence splitter for SO posts."""
    # Split on common sentence boundaries but keep code blocks together
    sentences = re.split(r'(?<=[.!?])\s+|\n\n+|\n(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def main() -> None:
    words_to_check: set[str] = set()

    # ── Phase 1: Collect all candidate words from SO NER ──
    # Start with words that are 0% in SO NER at doc level
    all_docs = []
    for split_name in ["train_documents", "dev_documents", "test_documents"]:
        path = ARTIFACTS / f"{split_name}.json"
        with open(path) as f:
            docs = json.load(f)
        all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} SO NER documents")

    # Find all words with 0% entity rate in SO NER (doc level, >=5 docs)
    doc_word_count: dict[str, int] = defaultdict(int)
    doc_entity_count: dict[str, int] = defaultdict(int)

    for doc in all_docs:
        text_lower = doc["text"].lower()
        gt_lower = set(t.lower().strip() for t in doc["gt_terms"])
        words_in_text = set(re.findall(r"\b[\w#+.]+\b", text_lower))
        for w in words_in_text:
            if len(w) < 2:
                continue
            doc_word_count[w] += 1
            if w in gt_lower:
                doc_entity_count[w] += 1

    # Candidate words: 0% entity rate in SO NER, >=5 docs
    for w, dc in doc_word_count.items():
        if doc_entity_count.get(w, 0) == 0 and dc >= 5:
            words_to_check.add(w)

    # Also add words with low (1-5%) entity rate — to verify they're truly borderline
    for w, dc in doc_word_count.items():
        ec = doc_entity_count.get(w, 0)
        if dc >= 10 and 0 < ec and ec / dc <= 0.05:
            words_to_check.add(w)

    print(f"Checking {len(words_to_check)} candidate words\n")

    # ── Phase 2: Count at SENTENCE level across SO NER ──
    print("=" * 80)
    print("Processing SO NER at sentence level...")
    print("=" * 80)

    # Per-word stats: {word: {"appearances": N, "entity": N}}
    so_stats: dict[str, dict] = defaultdict(lambda: {"appearances": 0, "entity": 0})
    so_sentence_count = 0

    for doc in all_docs:
        text = doc["text"]
        gt_lower = set(t.lower().strip() for t in doc["gt_terms"])

        # Split doc into sentences/chunks
        sentences = split_into_sentences(text)
        for sent in sentences:
            so_sentence_count += 1
            sent_lower = sent.lower()
            words_in_sent = set(re.findall(r"\b[\w#+.]+\b", sent_lower))

            for w in words_to_check:
                if w in words_in_sent:
                    so_stats[w]["appearances"] += 1
                    if w in gt_lower:
                        so_stats[w]["entity"] += 1

    print(f"  SO NER: {so_sentence_count} sentences from {len(all_docs)} docs\n")

    # ── Phase 3: Few-NERD filtered to tech domain ──
    print("=" * 80)
    print("Processing Few-NERD (tech-filtered)...")
    print("=" * 80)

    ds = load_dataset("DFKI-SLT/few-nerd", "supervised")

    fewnerd_stats: dict[str, dict] = defaultdict(lambda: {"appearances": 0, "entity": 0})
    fewnerd_total = 0
    fewnerd_tech = 0

    for split in ["train", "validation", "test"]:
        data = ds[split]
        for example in data:
            fewnerd_total += 1
            tokens = example["tokens"]
            fine_tags = example["fine_ner_tags"]

            tokens_lower = [t.lower() for t in tokens]

            # Check if this sentence is tech-domain
            is_tech = False

            # Method 1: contains product-software or product-other entity
            if any(t in TECH_ENTITY_TAG_IDS for t in fine_tags):
                is_tech = True

            # Method 2: contains tech keywords as tokens
            if not is_tech:
                for tl in tokens_lower:
                    if tl in TECH_KEYWORDS:
                        is_tech = True
                        break

            if not is_tech:
                continue

            fewnerd_tech += 1

            # Now count word appearances in this tech sentence
            entity_tokens = set()
            non_entity_tokens = set()
            for tok_lower, tag in zip(tokens_lower, fine_tags):
                if len(tok_lower) < 2:
                    continue
                if tag != 0:
                    entity_tokens.add(tok_lower)
                else:
                    non_entity_tokens.add(tok_lower)

            for w in words_to_check:
                if w in entity_tokens:
                    fewnerd_stats[w]["appearances"] += 1
                    fewnerd_stats[w]["entity"] += 1
                elif w in non_entity_tokens:
                    fewnerd_stats[w]["appearances"] += 1

    print(f"  Few-NERD total: {fewnerd_total} sentences")
    print(f"  Few-NERD tech-filtered: {fewnerd_tech} sentences ({fewnerd_tech/fewnerd_total*100:.1f}%)\n")

    # ── Phase 4: Combine and classify ──
    print("=" * 80)
    print("COMBINED RESULTS (SO NER sentences + Few-NERD tech sentences)")
    print("=" * 80)

    combined: list[tuple] = []
    for w in words_to_check:
        so_app = so_stats[w]["appearances"]
        so_ent = so_stats[w]["entity"]
        fn_app = fewnerd_stats[w]["appearances"]
        fn_ent = fewnerd_stats[w]["entity"]
        total_app = so_app + fn_app
        total_ent = so_ent + fn_ent

        if total_app < 5:
            continue

        combined.append((w, so_app, so_ent, fn_app, fn_ent, total_app, total_ent))

    # ── Print results by tier ──
    safe_1k = [(w, sa, se, fa, fe, ta, te)
               for w, sa, se, fa, fe, ta, te in combined
               if te == 0 and ta >= 1000]
    safe_1k.sort(key=lambda x: -x[5])

    safe_500 = [(w, sa, se, fa, fe, ta, te)
                for w, sa, se, fa, fe, ta, te in combined
                if te == 0 and 500 <= ta < 1000]
    safe_500.sort(key=lambda x: -x[5])

    safe_200 = [(w, sa, se, fa, fe, ta, te)
                for w, sa, se, fa, fe, ta, te in combined
                if te == 0 and 200 <= ta < 500]
    safe_200.sort(key=lambda x: -x[5])

    safe_100 = [(w, sa, se, fa, fe, ta, te)
                for w, sa, se, fa, fe, ta, te in combined
                if te == 0 and 100 <= ta < 200]
    safe_100.sort(key=lambda x: -x[5])

    unsafe = [(w, sa, se, fa, fe, ta, te)
              for w, sa, se, fa, fe, ta, te in combined
              if te > 0]
    unsafe.sort(key=lambda x: -x[6] / max(x[5], 1))

    def print_tier(label: str, items: list, show_unsafe: bool = False):
        print(f"\n── {label} ──\n")
        header = f"{'Word':25s} | {'SO NER':>12s} | {'FewNERD-tech':>14s} | {'COMBINED':>12s}"
        if show_unsafe:
            header += f" | {'Rate':>7s}"
        print(header)
        print("-" * len(header))

        for w, sa, se, fa, fe, ta, te in items:
            so_cell = f"0/{sa}" if se == 0 else f"{se}/{sa}"
            fn_cell = f"0/{fa}" if fe == 0 else f"{fe}/{fa}"
            tot_cell = f"0/{ta}" if te == 0 else f"{te}/{ta}"
            row = f"{w:25s} | {so_cell:>12s} | {fn_cell:>14s} | {tot_cell:>12s}"
            if show_unsafe:
                rate = te / ta * 100
                row += f" | {rate:>6.1f}%"
            print(row)

    print_tier(f"SAFE — ≥1000 chunks, 0% entity rate ({len(safe_1k)} words)", safe_1k)
    print_tier(f"SAFE — 500-999 chunks, 0% entity rate ({len(safe_500)} words)", safe_500)
    print_tier(f"SAFE — 200-499 chunks, 0% entity rate ({len(safe_200)} words)", safe_200)
    print_tier(f"SAFE — 100-199 chunks, 0% entity rate ({len(safe_100)} words)", safe_100)
    print_tier(f"UNSAFE — entity in at least one chunk ({len(unsafe)} words, top 80)", unsafe[:80], show_unsafe=True)

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Data: {so_sentence_count} SO NER sentences + {fewnerd_tech} Few-NERD tech sentences = {so_sentence_count + fewnerd_tech} total")
    print(f"  Words checked:         {len(combined)}")
    print(f"  SAFE ≥1000 chunks:     {len(safe_1k)}")
    print(f"  SAFE 500-999 chunks:   {len(safe_500)}")
    print(f"  SAFE 200-499 chunks:   {len(safe_200)}")
    print(f"  SAFE 100-199 chunks:   {len(safe_100)}")
    print(f"  UNSAFE (>0% anywhere): {len(unsafe)}")

    # ── Save ──
    result = {
        "data_sources": {
            "so_ner_sentences": so_sentence_count,
            "so_ner_docs": len(all_docs),
            "fewnerd_total_sentences": fewnerd_total,
            "fewnerd_tech_sentences": fewnerd_tech,
            "total_tech_chunks": so_sentence_count + fewnerd_tech,
        },
        "safe_1000": sorted(safe_1k, key=lambda x: -x[5]),
        "safe_500": sorted(safe_500, key=lambda x: -x[5]),
        "safe_200": sorted(safe_200, key=lambda x: -x[5]),
        "safe_100": sorted(safe_100, key=lambda x: -x[5]),
    }

    # Convert tuples to dicts for JSON
    def to_dict(items):
        return [{"word": w, "so_app": sa, "so_ent": se, "fewnerd_app": fa,
                 "fewnerd_ent": fe, "total_app": ta, "total_ent": te}
                for w, sa, se, fa, fe, ta, te in items]

    save = {
        "data_sources": result["data_sources"],
        "safe_1000": to_dict(result["safe_1000"]),
        "safe_500": to_dict(result["safe_500"]),
        "safe_200": to_dict(result["safe_200"]),
        "safe_100": to_dict(result["safe_100"]),
    }

    out_path = ARTIFACTS / "tech_domain_negatives_v2.json"
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
