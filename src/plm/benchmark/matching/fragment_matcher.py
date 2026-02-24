"""Fragment and reciprocal matching utilities for benchmark framework.

Handles anchor normalization and reciprocal containment detection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


@dataclass
class ReciprocalMatch:
    """Represents a reciprocal containment match between answer and chunk."""

    matched_words: list[str]
    word_count: int
    chunk_id: str
    direction: Literal["chunk_in_answer", "answer_in_chunk"]


MIN_RECIPROCAL_WORDS = 20


def normalize_anchor(anchor: str) -> str:
    """Normalize fragment anchor to canonical form.
    
    Handles variations like:
    - #pod-lifecycle -> pod-lifecycle
    - #pod_lifecycle -> pod-lifecycle
    - #PodLifecycle -> pod-lifecycle
    - #pod--lifecycle -> pod-lifecycle
    - #pod lifecycle -> pod-lifecycle
    """
    if anchor.startswith('#'):
        anchor = anchor[1:]
    
    anchor = anchor.lower()
    anchor = anchor.replace('_', '-')
    anchor = anchor.replace(' ', '-')
    anchor = re.sub(r'-+', '-', anchor)
    anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
    anchor = anchor.strip('-')
    
    return anchor


def extract_words(text: str) -> list[str]:
    """Extract words from text, normalized for comparison."""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text)
    return words


def find_reciprocal_matches(
    answer_html: str,
    chunk_content: str,
    chunk_id: str,
) -> list[ReciprocalMatch]:
    """Find reciprocal containment matches between answer and chunk.
    
    Detects when >= 20 contiguous words from one text appear in the other.
    
    Args:
        answer_html: HTML body of SO answer (will extract text)
        chunk_content: Text content of documentation chunk
        chunk_id: ID of the chunk for reference
    
    Returns:
        List of ReciprocalMatch objects for matches >= 20 words
    """
    from bs4 import BeautifulSoup
    
    matches = []
    
    soup = BeautifulSoup(answer_html, 'html.parser')
    answer_text = soup.get_text()
    
    answer_words = extract_words(answer_text)
    chunk_words = extract_words(chunk_content)
    
    answer_str = ' '.join(answer_words)
    chunk_str = ' '.join(chunk_words)
    
    if len(chunk_words) >= MIN_RECIPROCAL_WORDS:
        for i in range(len(chunk_words) - MIN_RECIPROCAL_WORDS + 1):
            window = chunk_words[i:i + MIN_RECIPROCAL_WORDS]
            window_str = ' '.join(window)
            
            if window_str in answer_str:
                matches.append(
                    ReciprocalMatch(
                        matched_words=window,
                        word_count=len(window),
                        chunk_id=chunk_id,
                        direction="chunk_in_answer",
                    )
                )
                break
    
    if len(answer_words) >= MIN_RECIPROCAL_WORDS:
        for i in range(len(answer_words) - MIN_RECIPROCAL_WORDS + 1):
            window = answer_words[i:i + MIN_RECIPROCAL_WORDS]
            window_str = ' '.join(window)
            
            if window_str in chunk_str:
                matches.append(
                    ReciprocalMatch(
                        matched_words=window,
                        word_count=len(window),
                        chunk_id=chunk_id,
                        direction="answer_in_chunk",
                    )
                )
                break
    
    return matches
