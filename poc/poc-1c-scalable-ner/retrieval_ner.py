#!/usr/bin/env python3
"""Approach A: Retrieval-augmented few-shot NER extraction.

Embeds train.txt documents, retrieves K similar ones for each test doc,
and uses their GT annotations as few-shot examples in the extraction prompt.

Zero vocabulary lists. Scales with more training data.
"""

import json
import re
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from scoring import normalize_term, verify_span
from utils.llm_provider import call_llm


EMBEDDINGS_PATH = Path(__file__).parent / "artifacts" / "train_embeddings.npy"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def build_retrieval_index(
    train_docs: list[dict],
    cache_path: Path = EMBEDDINGS_PATH,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> tuple[faiss.Index, np.ndarray, SentenceTransformer]:
    model = SentenceTransformer(model_name)

    if cache_path.exists():
        embeddings = np.load(str(cache_path))
        if embeddings.shape[0] == len(train_docs):
            index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            return index, embeddings, model

    texts = [doc["text"][:2000] for doc in train_docs]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), embeddings.copy())

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings, model


def compute_jaccard(text1: str, text2: str, n: int = 13) -> float:
    """Character n-gram Jaccard similarity for near-duplicate detection."""
    if len(text1) < n or len(text2) < n:
        return 0.0
    grams1 = set(text1[i : i + n] for i in range(len(text1) - n + 1))
    grams2 = set(text2[i : i + n] for i in range(len(text2) - n + 1))
    if not grams1 or not grams2:
        return 0.0
    return len(grams1 & grams2) / len(grams1 | grams2)


def safe_retrieve(
    test_doc: dict,
    train_docs: list[dict],
    index: faiss.Index,
    model: SentenceTransformer,
    k: int = 5,
    jaccard_threshold: float = 0.8,
) -> list[dict]:
    """Retrieve K similar train docs with near-duplicate filtering."""
    test_embedding = model.encode([test_doc["text"][:2000]], convert_to_numpy=True)
    faiss.normalize_L2(test_embedding)

    distances, indices = index.search(test_embedding, k * 3)

    filtered: list[dict] = []
    for idx_val in indices[0]:
        idx = int(idx_val)
        if idx < 0 or idx >= len(train_docs):
            continue
        candidate = train_docs[idx]
        jaccard = compute_jaccard(test_doc["text"], candidate["text"])
        if jaccard < jaccard_threshold:
            filtered.append(candidate)
        if len(filtered) >= k:
            break

    return filtered


def build_fewshot_prompt(test_doc: dict, retrieved_docs: list[dict]) -> str:
    examples_parts: list[str] = []
    for i, doc in enumerate(retrieved_docs, 1):
        text_preview = doc["text"][:800]
        terms = json.dumps(doc["gt_terms"], ensure_ascii=False)
        examples_parts.append(
            f"--- Example {i} ---\n"
            f"TEXT: {text_preview}\n"
            f"ENTITIES: {terms}"
        )

    examples_block = "\n\n".join(examples_parts)

    return f"""You are extracting technical named entities from StackOverflow posts.

Here are examples of correct entity extraction from similar posts:

{examples_block}

---

Now extract ALL technical entities from this text, following the same annotation style as the examples above.

Rules:
- Extract specific named technologies, libraries, frameworks, languages, APIs, functions, classes, tools, platforms, data types, UI components, file formats, error types, and websites
- Do NOT extract generic programming vocabulary (e.g. "function", "method", "server", "client", "request")
- Extract the minimal span (e.g. "React" not "the React library")
- Include version numbers when present (e.g. "Python 3.9", "C++11")
- Context matters: the same word can be an entity in one context and not in another

TEXT: {test_doc["text"]}

Return ONLY a JSON array of entity strings. No explanations.
ENTITIES:"""


def parse_entity_response(response: str) -> list[str]:
    response = response.strip()
    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if json_match:
        try:
            entities = json.loads(json_match.group())
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if isinstance(e, str) and e.strip()]
        except json.JSONDecodeError:
            pass

    entities: list[str] = []
    for line in response.splitlines():
        line = line.strip().strip("-*•").strip()
        line = line.strip('"').strip("'").strip(",").strip()
        if line and len(line) >= 2 and not line.startswith("{") and not line.startswith("["):
            entities.append(line)
    return entities


def extract_with_retrieval(
    test_doc: dict,
    train_docs: list[dict],
    index: faiss.Index,
    model: SentenceTransformer,
    k: int = 5,
) -> list[str]:
    similar_docs = safe_retrieve(test_doc, train_docs, index, model, k)
    prompt = build_fewshot_prompt(test_doc, similar_docs)
    response = call_llm(prompt, model="sonnet", max_tokens=2000, temperature=0.0)
    entities = parse_entity_response(response)

    verified: list[str] = []
    seen: set[str] = set()
    for e in entities:
        ok, _ = verify_span(e, test_doc["text"])
        if ok:
            key = normalize_term(e)
            if key not in seen:
                seen.add(key)
                verified.append(e)

    return verified
