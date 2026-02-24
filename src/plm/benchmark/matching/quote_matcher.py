"""Quote matching utilities for benchmark framework.

Finds exact text matches between StackOverflow answers and documentation chunks.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Literal, cast

from bs4 import BeautifulSoup


@dataclass
class QuoteMatch:
    """Represents an exact text match between answer and chunk."""

    matched_text: str
    match_length: int
    source_type: Literal["code", "blockquote", "prose"]
    chunk_id: str
    chunk_offset: int
    answer_offset: int


GENERIC_BLACKLIST = {
    "run the following command",
    "for more information",
    "see the documentation",
    "as shown below",
    "for example",
}

MIN_QUOTE_LENGTH = 30


def normalize_text(text: str) -> str:
    """Normalize text for matching.
    
    Steps:
    1. HTML decode
    2. Collapse whitespace
    3. Strip outer whitespace
    4. Lowercase for comparison
    """
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.lower()
    return text


def extract_text_from_html(html_content: str) -> dict[str, list[str]]:
    """Extract text from SO answer HTML by tag type.
    
    Returns dict with keys: 'code', 'blockquote', 'prose'
    Each value is a list of extracted text strings.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    extracted = {
        'code': [],
        'blockquote': [],
        'prose': [],
    }
    
    for code_tag in soup.find_all('code'):
        text = code_tag.get_text()
        if text.strip():
            extracted['code'].append(text)
    
    for pre_tag in soup.find_all('pre'):
        text = pre_tag.get_text()
        if text.strip():
            extracted['code'].append(text)
    
    for blockquote_tag in soup.find_all('blockquote'):
        text = blockquote_tag.get_text()
        if text.strip():
            extracted['blockquote'].append(text)
    
    for p_tag in soup.find_all(['p', 'div']):
        if not p_tag.find_parent(['code', 'pre', 'blockquote']):
            text = p_tag.get_text()
            if text.strip():
                extracted['prose'].append(text)
    
    return extracted


def is_generic_text(text: str) -> bool:
    """Check if text is in the generic blacklist."""
    normalized = normalize_text(text)
    return normalized in GENERIC_BLACKLIST


def find_quote_matches(
    answer_html: str,
    chunk_content: str,
    chunk_id: str,
) -> list[QuoteMatch]:
    """Find exact quote matches between SO answer and chunk.
    
    Args:
        answer_html: HTML body of SO answer
        chunk_content: Text content of documentation chunk
        chunk_id: ID of the chunk for reference
    
    Returns:
        List of QuoteMatch objects for matches >= 30 characters
    """
    matches = []
    
    extracted = extract_text_from_html(answer_html)
    normalized_chunk = normalize_text(chunk_content)
    
    for source_type, text_list in extracted.items():
        for answer_text in text_list:
            if not answer_text.strip():
                continue
            
            normalized_answer = normalize_text(answer_text)
            
            if is_generic_text(normalized_answer):
                continue
            
            if normalized_answer in normalized_chunk:
                chunk_offset = normalized_chunk.find(normalized_answer)
                answer_offset = normalized_answer.find(normalized_answer)
                match_length = len(normalized_answer)
                
                if match_length >= MIN_QUOTE_LENGTH:
                    matched_text = answer_text.strip()
                    
                    matches.append(
                        QuoteMatch(
                            matched_text=matched_text,
                            match_length=match_length,
                            source_type=cast(Literal["code", "blockquote", "prose"], source_type),
                            chunk_id=chunk_id,
                            chunk_offset=chunk_offset,
                            answer_offset=answer_offset,
                        )
                    )
    
    return matches
