#!/usr/bin/env python3
"""V2: Find the REAL negatives list — 0% entity rate, filtered to only
words that could plausibly be NER candidates (not stop words).

A word qualifies for the negatives list ONLY if:
1. entity_ratio = 0.00 across ALL 741 training documents
2. Appears in >= N documents (high confidence)
3. Is a "technical-sounding" word that an NER extractor might plausibly extract
   (not a stop word, not a common English word that no extractor would ever output)
"""

import json
import re
from collections import defaultdict
from pathlib import Path

TRAIN_PATH = Path("artifacts/train_documents.json")

# Stop words and common English that no NER extractor would ever extract
# These don't need to be on a negatives list because they'd never be candidates
STOP_WORDS = {
    # Function words
    "the", "to", "is", "and", "in", "this", "it", "of", "that", "you",
    "have", "for", "but", "can", "with", "be", "not", "on", "do", "if",
    "my", "an", "as", "from", "like", "are", "your", "or", "there",
    "when", "want", "how", "what", "am", "will", "which", "need", "then",
    "any", "by", "all", "here", "would", "also", "one", "some", "should",
    "just", "only", "at", "does", "me", "has", "was", "been", "were",
    "being", "had", "did", "its", "they", "them", "their", "we", "our",
    "he", "she", "his", "her", "who", "whom", "whose", "where", "why",
    "each", "every", "both", "few", "more", "most", "other", "than",
    "too", "very", "much", "many", "so", "no", "nor", "yet", "about",
    "after", "before", "during", "between", "through", "into", "out",
    "up", "down", "over", "under", "again", "further", "above", "below",

    # Common verbs / verbal forms
    "using", "use", "used", "trying", "try", "tried", "get", "got",
    "getting", "set", "setting", "make", "made", "making", "work",
    "working", "worked", "works", "know", "think", "see", "look",
    "looking", "find", "found", "help", "run", "running", "create",
    "creating", "created", "add", "added", "adding", "change", "changed",
    "call", "called", "calling", "calls", "take", "takes", "give",
    "gives", "given", "seem", "seems", "going", "come", "comes",
    "pass", "passed", "passing", "keep", "write", "writing", "read",
    "reading", "show", "shows", "start", "end", "tell", "says",
    "define", "defined", "check", "checked", "return", "returns",
    "returned", "remove", "specify", "update", "move",

    # Common nouns unlikely to be NER candidates
    "example", "thing", "things", "problem", "issue", "issues",
    "question", "answer", "solution", "idea", "case", "cases",
    "time", "times", "way", "ways", "part", "parts", "point",
    "result", "results", "reason", "order", "number", "place",
    "fact", "something", "anything", "nothing", "everything",
    "everyone", "someone", "lot", "bit", "kind", "stuff",
    "step", "steps", "piece", "side", "top", "bottom", "left", "right",

    # Common adjectives/adverbs
    "new", "first", "last", "different", "same", "other", "possible",
    "sure", "able", "good", "better", "best", "bad", "worse", "worst",
    "simple", "easy", "hard", "right", "wrong", "true", "false",
    "actually", "basically", "probably", "maybe", "really", "already",
    "still", "even", "however", "though", "always", "never", "often",
    "well", "instead", "else", "since", "unless",

    # Misc non-technical
    "thanks", "please", "sorry", "hope", "edit", "note", "update",
    "yes", "ok", "okay", "hi", "hello",
}


def main() -> None:
    with open(TRAIN_PATH) as f:
        docs = json.load(f)
    print(f"Loaded {len(docs)} training documents\n")

    word_doc_count: dict[str, int] = defaultdict(int)
    word_entity_count: dict[str, int] = defaultdict(int)

    for doc in docs:
        text_lower = doc["text"].lower()
        gt_lower = set(t.lower().strip() for t in doc["gt_terms"])
        words_in_text = set(re.findall(r"\b[\w#+.]+\b", text_lower))

        for w in words_in_text:
            if len(w) < 2:
                continue
            word_doc_count[w] += 1
            if w in gt_lower:
                word_entity_count[w] += 1

    # Find true negatives that are NOT stop words — these are "technical-sounding"
    # words that an extractor might plausibly output, but are never entities
    print("=" * 80)
    print("PRINCIPLED NEGATIVES: 0% entity rate, NOT stop words")
    print("These are words an NER extractor MIGHT extract but SHOULDN'T")
    print("=" * 80)

    for min_docs in [20, 10, 5]:
        candidates = []
        for word, doc_count in word_doc_count.items():
            if doc_count < min_docs:
                continue
            if word_entity_count.get(word, 0) > 0:
                continue
            if word.lower() in STOP_WORDS:
                continue
            if len(word) <= 2:  # skip 2-char like "it", "is", etc.
                continue
            candidates.append((word, doc_count))

        candidates.sort(key=lambda x: -x[1])

        print(f"\n--- >= {min_docs} docs, 0% entity rate, not stop word ---")
        print(f"Total candidates: {len(candidates)}")
        for word, count in candidates[:80]:
            print(f"  {word:30s}  {count:4d} docs")

    # Now let's also check: among our 43 FPs from the benchmark,
    # which ones have 0% entity rate with high doc count?
    print("\n" + "=" * 80)
    print("CHECK: Our 43 benchmark FPs against training data")
    print("=" * 80)

    benchmark_fps = [
        "getInputSizes", "height", "LEVEL-3", "Preview", "ZSL",
        "boost", "cryptographic APIs", "data", "fopen", "microsoft", "MSVC", "std::fopen",
        "DSL", "endpoint", "main", "server", "service", "service class", "SOAP", "Weblogic", "xml",
        "aspx", "compositewpf.codeplex.com", "View.aspx",
        "command line", "gem", "irb", "rdoc", "ri", "Ruby.framework", "testrb",
        "64-bit", "config", "latex", "notebook",
        "/codepen.io/anon/pen/VjOKGX", "https://codepen.io/anon/pen/VjOKGX", "VjOKGX",
        "ASCII", "ASCII capable", "controls", "Date",
        "Second Edition", "data",
    ]

    for fp in sorted(set(benchmark_fps), key=str.lower):
        fl = fp.lower().strip()
        dc = word_doc_count.get(fl, 0)
        ec = word_entity_count.get(fl, 0)
        if dc == 0:
            status = "UNSEEN in training"
        elif ec == 0:
            status = f"0% entity rate ({dc} docs) → SAFE for negatives"
        else:
            ratio = ec / dc * 100
            status = f"{ratio:.1f}% entity rate ({ec}/{dc} docs) → NOT safe"
        print(f"  {fp:40s}  {status}")


if __name__ == "__main__":
    main()
