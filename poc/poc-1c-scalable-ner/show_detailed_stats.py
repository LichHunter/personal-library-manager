#!/usr/bin/env python3
"""Show detailed stats for the 200-499 tier words + borderline words.

For each word shows:
- Total tech chunks in our data
- Chunks where word appears
- Coverage (% of all tech chunks containing this word)
- Labeled as entity
- Not labeled as entity
- Entity rate
"""

import json
from pathlib import Path

ARTIFACTS = Path("artifacts")
TOTAL_CHUNKS = 16931  # 10860 SO + 6071 Few-NERD tech


def main() -> None:
    with open(ARTIFACTS / "tech_domain_negatives_v2.json") as f:
        data = json.load(f)

    total = data["data_sources"]["total_tech_chunks"]

    # Combine safe_200 and safe_100 for a fuller picture
    all_safe = []
    for tier_key in ["safe_1000", "safe_500", "safe_200", "safe_100"]:
        for item in data[tier_key]:
            all_safe.append(item)

    # Filter to 200-499 only for main table
    tier_200 = [w for w in all_safe if 200 <= w["total_app"] < 500]
    tier_200.sort(key=lambda x: -x["total_app"])

    print(f"Total tech-domain labeled chunks: {total}")
    print(f"  SO NER sentences: {data['data_sources']['so_ner_sentences']}")
    print(f"  Few-NERD tech:    {data['data_sources']['fewnerd_tech_sentences']}")
    print()

    # ── Main table: 200-499 tier ──
    print("=" * 110)
    print("WORDS IN 200-499 CHUNKS (0% entity rate)")
    print("=" * 110)
    print(f"{'Word':20s} | {'Present in':>10s} | {'Coverage':>8s} | {'As entity':>10s} | {'Not entity':>10s} | {'Entity %':>8s} | {'SO NER':>8s} | {'FewNERD':>8s}")
    print("-" * 110)

    for w in tier_200:
        word = w["word"]
        present = w["total_app"]
        entity = w["total_ent"]
        not_entity = present - entity
        coverage = present / total * 100
        entity_pct = entity / present * 100 if present > 0 else 0
        so = w["so_app"]
        fn = w["fewnerd_app"]

        print(f"{word:20s} | {present:>10d} | {coverage:>7.1f}% | {entity:>10d} | {not_entity:>10d} | {entity_pct:>7.1f}% | {so:>8d} | {fn:>8d}")

    print(f"\n  Total words in this tier: {len(tier_200)}")

    # ── Also show 100-199 tier but only technical words (filter out English) ──
    tier_100 = [w for w in all_safe if 100 <= w["total_app"] < 200]
    tier_100.sort(key=lambda x: -x["total_app"])

    # Filter to words that look technical (not pure English)
    tech_words_100 = []
    english_skip = {
        "later", "called", "why", "possible", "built", "able", "did", "still",
        "without", "below", "most", "found", "many", "made", "even", "please",
        "above", "much", "think", "doing", "every", "edit", "sure", "result",
        "several", "put", "having", "seems", "last", "always", "inside", "better",
        "again", "single", "those", "known", "replaced", "designed", "currently",
        "around", "wrong", "got", "really", "answer", "anyone", "already", "once",
        "actually", "included", "etc", "within", "added", "given", "side",
        "released", "original", "take", "note", "simply", "either", "issue",
        "thanks", "created",
    }
    for w in tier_100:
        if w["word"] not in english_skip:
            tech_words_100.append(w)

    print(f"\n{'='*110}")
    print("TECHNICAL WORDS IN 100-199 CHUNKS (0% entity rate, English filtered out)")
    print("=" * 110)
    print(f"{'Word':20s} | {'Present in':>10s} | {'Coverage':>8s} | {'As entity':>10s} | {'Not entity':>10s} | {'Entity %':>8s} | {'SO NER':>8s} | {'FewNERD':>8s}")
    print("-" * 110)

    for w in tech_words_100:
        word = w["word"]
        present = w["total_app"]
        entity = w["total_ent"]
        not_entity = present - entity
        coverage = present / total * 100
        entity_pct = entity / present * 100 if present > 0 else 0
        so = w["so_app"]
        fn = w["fewnerd_app"]

        print(f"{word:20s} | {present:>10d} | {coverage:>7.1f}% | {entity:>10d} | {not_entity:>10d} | {entity_pct:>7.1f}% | {so:>8d} | {fn:>8d}")

    print(f"\n  Technical words in this tier: {len(tech_words_100)}")


if __name__ == "__main__":
    main()
