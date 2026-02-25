#!/usr/bin/env python3
"""Parse StackOverflow NER BIO-tagged data into structured JSON documents.

Reads train.txt/test.txt from the SO NER dataset (Tabassum et al., ACL 2020)
and produces JSON files suitable for retrieval and benchmarking.

Usage:
    python parse_so_ner.py
    python parse_so_ner.py --so-ner-dir /path/to/StackOverflowNER
"""

import argparse
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Entity type taxonomy
# ---------------------------------------------------------------------------

RELEVANT_ENTITY_TYPES = {
    "Application",
    "Library",
    "Library_Class",
    "Library_Function",
    "Library_Variable",
    "Language",
    "Data_Structure",
    "Data_Type",
    "Algorithm",
    "Operating_System",
    "Device",
    "File_Type",
    "HTML_XML_Tag",
    "Error_Name",
    "Version",
    "Website",
    "Class_Name",
    "Function_Name",
    "User_Interface_Element",
    "Keyboard_IP",
    "Organization",
    "File_Name",
}

EXCLUDED_ENTITY_TYPES = {
    "Code_Block",
    "Output_Block",
    "Variable_Name",
    "Value",
    "User_Name",
}

# ---------------------------------------------------------------------------
# BIO parser (ported from poc-1b so_ner_benchmark.py lines 90-263)
# ---------------------------------------------------------------------------


def parse_so_ner_file(filepath: str) -> list[dict]:
    """Parse SO NER BIO-tagged file into documents grouped by Question_ID.

    Returns list of documents, each with:
    - question_id: str
    - question_url: str
    - text: str (reconstructed text)
    - entities: list of {text, type}
    """
    documents: list[dict] = []
    current_doc: dict | None = None
    current_text_tokens: list[str] = []
    current_entity_tokens: list[str] = []
    current_entity_type: str | None = None
    metadata_key: str | None = None

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            # Empty line = sentence boundary
            if not line.strip():
                if current_entity_tokens and current_entity_type:
                    entity_text = " ".join(current_entity_tokens)
                    if current_doc is not None:
                        current_doc["entities"].append({
                            "text": entity_text,
                            "type": current_entity_type,
                        })
                    current_entity_tokens = []
                    current_entity_type = None
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            token = parts[0].strip()
            tag = parts[1].strip()

            # Detect Question_ID markers (starts a new document)
            if token == "Question_ID":
                if current_doc is not None and current_text_tokens:
                    current_doc["text"] = " ".join(current_text_tokens)
                    if len(current_doc["text"]) > 20:
                        documents.append(current_doc)

                current_doc = {
                    "question_id": "",
                    "question_url": "",
                    "text": "",
                    "entities": [],
                }
                current_text_tokens = []
                current_entity_tokens = []
                current_entity_type = None
                metadata_key = "question_id"
                continue

            # Answer_to_Question_ID is a structural marker separating
            # answers from the question within the same SO thread.
            # Skip it and its following ": XXXXXXX" metadata — the answer
            # text belongs to the same document, not a new one.
            if token == "Answer_to_Question_ID":
                # Flush any pending entity
                if current_entity_tokens and current_entity_type:
                    if current_doc is not None:
                        current_doc["entities"].append({
                            "text": " ".join(current_entity_tokens),
                            "type": current_entity_type,
                        })
                    current_entity_tokens = []
                    current_entity_type = None
                metadata_key = "answer_id"
                continue

            if token == "Question_URL":
                metadata_key = "question_url"
                continue

            # Skip Answer_to_Question_ID's number (": XXXXXXX")
            if metadata_key == "answer_id" and re.match(r"^\d+$", token):
                metadata_key = None
                continue

            if token == ":" and metadata_key:
                continue

            if metadata_key == "question_id" and re.match(r"^\d+$", token):
                if current_doc:
                    current_doc["question_id"] = token
                metadata_key = None
                continue

            if metadata_key == "question_url" and token.startswith("http"):
                if current_doc:
                    current_doc["question_url"] = token
                metadata_key = None
                continue

            if current_doc is None:
                continue

            metadata_key = None

            # Skip Code_Block / Output_Block content
            if tag.startswith("B-Code_Block") or tag.startswith("I-Code_Block"):
                if current_entity_tokens and current_entity_type and current_entity_type != "Code_Block":
                    current_doc["entities"].append({
                        "text": " ".join(current_entity_tokens),
                        "type": current_entity_type,
                    })
                    current_entity_tokens = []
                    current_entity_type = None
                if tag.startswith("B-Code_Block"):
                    current_text_tokens.append("[CODE]")
                continue

            if tag.startswith("B-Output_Block") or tag.startswith("I-Output_Block"):
                if current_entity_tokens and current_entity_type and current_entity_type != "Output_Block":
                    current_doc["entities"].append({
                        "text": " ".join(current_entity_tokens),
                        "type": current_entity_type,
                    })
                    current_entity_tokens = []
                    current_entity_type = None
                if tag.startswith("B-Output_Block"):
                    current_text_tokens.append("[OUTPUT]")
                continue

            # Regular token
            current_text_tokens.append(token)

            # BIO entity extraction
            if tag.startswith("B-"):
                if current_entity_tokens and current_entity_type:
                    current_doc["entities"].append({
                        "text": " ".join(current_entity_tokens),
                        "type": current_entity_type,
                    })
                current_entity_type = tag[2:]
                current_entity_tokens = [token]

            elif tag.startswith("I-"):
                entity_type = tag[2:]
                if current_entity_type == entity_type:
                    current_entity_tokens.append(token)
                else:
                    if current_entity_tokens and current_entity_type:
                        current_doc["entities"].append({
                            "text": " ".join(current_entity_tokens),
                            "type": current_entity_type,
                        })
                    current_entity_tokens = []
                    current_entity_type = None

            else:  # O tag
                if current_entity_tokens and current_entity_type:
                    current_doc["entities"].append({
                        "text": " ".join(current_entity_tokens),
                        "type": current_entity_type,
                    })
                    current_entity_tokens = []
                    current_entity_type = None

    # Flush last document
    if current_doc is not None and current_text_tokens:
        current_doc["text"] = " ".join(current_text_tokens)
        if len(current_doc["text"]) > 20:
            documents.append(current_doc)

    return documents


def normalize_term(term: str) -> str:
    """Normalize a term for deduplication."""
    return term.lower().strip().replace("-", " ").replace("_", " ")


def extract_gt_terms(doc: dict, include_types: set[str] | None = None) -> list[str]:
    """Extract unique GT terms from a parsed document."""
    types_to_use = include_types or RELEVANT_ENTITY_TYPES

    seen: set[str] = set()
    terms: list[str] = []
    for entity in doc["entities"]:
        if entity["type"] not in types_to_use:
            continue
        term = entity["text"].strip()
        if not term or len(term) < 2:
            continue
        if re.match(r"^[^\w]+$", term):
            continue
        key = normalize_term(term)
        if key not in seen:
            seen.add(key)
            terms.append(term)

    return terms


def select_documents(
    documents: list[dict], n: int = 10, min_entities: int = 5, seed: int = 42,
) -> list[dict]:
    """Select n documents suitable for benchmarking (same logic as iter 26).

    Works with both raw parsed docs (having 'entities') and pre-computed
    JSON records (having 'gt_terms' and 'entity_types').
    """
    import random
    rng = random.Random(seed)

    candidates = []
    for doc in documents:
        if "gt_terms" in doc:
            gt_count = len(doc["gt_terms"])
            type_count = len(doc.get("entity_types", []))
            text_len = len(doc["text"])
        else:
            gt_terms = extract_gt_terms(doc)
            gt_count = len(gt_terms)
            type_count = len({
                e["type"] for e in doc["entities"]
                if e["type"] in RELEVANT_ENTITY_TYPES
            })
            text_len = len(doc["text"])

        if gt_count >= min_entities and text_len >= 100:
            candidates.append({
                "doc": doc,
                "gt_count": gt_count,
                "type_count": type_count,
            })

    rng.shuffle(candidates)
    candidates.sort(key=lambda c: (c["type_count"], c["gt_count"]), reverse=True)

    selected = candidates[:n]
    return [c["doc"] for c in selected]


def build_document_record(doc: dict) -> dict:
    """Build a structured record for JSON export."""
    gt_terms = extract_gt_terms(doc)
    entity_types = list({
        e["type"] for e in doc["entities"]
        if e["type"] in RELEVANT_ENTITY_TYPES
    })
    return {
        "doc_id": f"Q{doc['question_id']}",
        "question_url": doc.get("question_url", ""),
        "text": doc["text"],
        "gt_terms": gt_terms,
        "entity_types": sorted(entity_types),
        "num_entities": len(gt_terms),
        "text_length": len(doc["text"]),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SO NER BIO data")
    parser.add_argument(
        "--so-ner-dir",
        default="/tmp/StackOverflowNER/resources/annotated_ner_data/StackOverflow",
        help="Path to SO NER data directory",
    )
    args = parser.parse_args()

    so_dir = Path(args.so_ner_dir)
    artifacts = Path(__file__).parent / "artifacts"
    artifacts.mkdir(exist_ok=True)

    for split in ("train", "test", "dev"):
        filepath = so_dir / f"{split}.txt"
        if not filepath.exists():
            print(f"  Skipping {split}.txt (not found)")
            continue

        print(f"Parsing {split}.txt ...")
        raw_docs = parse_so_ner_file(str(filepath))
        records = [build_document_record(d) for d in raw_docs]

        out_path = artifacts / f"{split}_documents.json"
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)

        total_entities = sum(r["num_entities"] for r in records)
        print(f"  {split}: {len(records)} documents, {total_entities} total entities")
        print(f"  Saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
