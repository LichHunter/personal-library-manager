"""Seed extraction and vocabulary loading for V6 pipeline.

Provides high-recall extraction by matching known technical terms from
auto-vocabulary against document text using word-boundary regex.
"""

import json
import re
from pathlib import Path
from typing import TypedDict


class AutoVocab(TypedDict):
    bypass: list[str]
    seeds: list[str]
    negatives: list[str]
    contextual_seeds: list[str]
    low_precision: list[str]


def load_auto_vocab(path: Path) -> AutoVocab:
    """Load auto-vocabulary from JSON file.
    
    Args:
        path: Path to auto_vocab.json
        
    Returns:
        Dictionary with bypass, seeds, negatives, contextual_seeds, low_precision
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    return AutoVocab(
        bypass=data.get("bypass", []),
        seeds=data.get("seeds", []),
        negatives=data.get("negatives", []),
        contextual_seeds=data.get("contextual_seeds", []),
        low_precision=data.get("low_precision", []),
    )


def extract_seeds(text: str, seeds: list[str]) -> list[str]:
    """Extract seed terms from text using word-boundary regex matching.
    
    For each seed term, checks if it appears in the text using word boundary
    matching. Returns the actual form found in text (preserving case).
    
    Args:
        text: Document text to search
        seeds: List of seed terms to match
        
    Returns:
        List of matched terms (with original case from text)
    """
    text_lower = text.lower()
    found: list[str] = []
    
    for seed in seeds:
        seed_lower = seed.lower()
        pattern = rf"\b{re.escape(seed_lower)}\b"
        
        if re.search(pattern, text_lower):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                found.append(match.group())
            else:
                found.append(seed)
    
    return found


def get_bypass_set(vocab: AutoVocab) -> set[str]:
    """Get the bypass set (terms to auto-keep without validation).
    
    Args:
        vocab: Loaded auto-vocabulary
        
    Returns:
        Set of lowercase bypass terms
    """
    return {t.lower() for t in vocab["bypass"]}


def get_seeds_set(vocab: AutoVocab) -> set[str]:
    """Get the full seeds set (all terms for seed extraction).
    
    Args:
        vocab: Loaded auto-vocabulary
        
    Returns:
        Set of lowercase seed terms
    """
    return {t.lower() for t in vocab["seeds"]}


def get_contextual_seeds_set(vocab: AutoVocab) -> set[str]:
    """Get contextual seeds set (medium-confidence tier terms).
    
    Args:
        vocab: Loaded auto-vocabulary
        
    Returns:
        Set of lowercase contextual seed terms
    """
    return {t.lower() for t in vocab["contextual_seeds"]}


class TermInfo(TypedDict):
    """Information about a term from training data."""
    entity_ratio: float
    entity_count: int
    generic_count: int


def load_term_index(path: Path) -> dict[str, TermInfo]:
    """Load term index with entity_ratio data from training.
    
    Args:
        path: Path to term_index.json
        
    Returns:
        Dictionary mapping term (lowercase) to TermInfo
        
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is invalid JSON
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Convert to TermInfo format (strip examples to save memory)
    result: dict[str, TermInfo] = {}
    for term, info in data.items():
        result[term] = TermInfo(
            entity_ratio=info.get("entity_ratio", 0.5),
            entity_count=info.get("entity_count", 0),
            generic_count=info.get("generic_count", 0),
        )
    return result
