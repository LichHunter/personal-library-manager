"""Format transformer for slow extraction output.

Transforms slow extraction output format to match fast extraction format
for consistent queue message structure.

Slow extraction format:
    {
        "file": "doc.txt",
        "processed_at": "2026-01-01T00:00:00Z",
        "chunks": [{
            "text": "...",
            "chunk_index": 0,
            "heading": "Section",
            "terms": [{"term": "Kubernetes", "confidence": 0.9, "level": "HIGH", "sources": ["v6"]}]
        }],
        "stats": {...}
    }

Fast extraction format (target):
    {
        "source_file": "/path/to/doc.md",
        "headings": [{
            "heading": "Section",
            "level": 1,
            "chunks": [{
                "text": "...",
                "terms": ["Kubernetes"],
                "entities": [{"text": "Kubernetes", "label": "technology", "score": 0.9, "start": 0, "end": 10}],
                "keywords": [],
                "start_char": 0,
                "end_char": 100
            }]
        }],
        "avg_confidence": 0.9,
        "total_entities": 1,
        "is_low_confidence": false,
        "error": null
    }
"""

from __future__ import annotations


def _confidence_level_to_score(level: str, confidence: float) -> float:
    """Convert confidence level + value to numeric score.
    
    Slow extraction uses level (HIGH/MEDIUM/LOW) as primary indicator.
    """
    if level == "HIGH":
        return max(confidence, 0.8)
    elif level == "MEDIUM":
        return max(min(confidence, 0.79), 0.5)
    else:  # LOW
        return min(confidence, 0.49)


def _find_term_position(text: str, term: str) -> tuple[int, int]:
    """Find position of term in text.
    
    Returns (start, end) or (0, len(term)) if not found.
    """
    # Case-insensitive search
    text_lower = text.lower()
    term_lower = term.lower()
    pos = text_lower.find(term_lower)
    if pos >= 0:
        return pos, pos + len(term)
    return 0, len(term)


def transform_to_fast_format(slow_result: dict, source_file: str | None = None) -> dict:
    """Transform slow extraction output to fast extraction format.

    Args:
        slow_result: Dictionary in slow extraction format.
        source_file: Override source file path (uses slow_result["file"] if None).

    Returns:
        Dictionary in fast extraction format.
    """
    # Get source file
    file_name = slow_result.get("file", "unknown.txt")
    final_source = source_file or file_name
    
    # Group chunks by heading
    headings_map: dict[str | None, list[dict]] = {}
    
    for chunk in slow_result.get("chunks", []):
        heading = chunk.get("heading")
        if heading not in headings_map:
            headings_map[heading] = []
        headings_map[heading].append(chunk)
    
    # Build headings list
    headings = []
    total_entities = 0
    all_confidences = []
    
    for heading_text, chunks in headings_map.items():
        converted_chunks = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            terms = chunk.get("terms", [])
            
            # Convert terms to entities
            entities = []
            term_texts = []
            
            for term_data in terms:
                term_text = term_data.get("term", "")
                confidence = term_data.get("confidence", 0.5)
                level = term_data.get("level", "MEDIUM")
                
                score = _confidence_level_to_score(level, confidence)
                start, end = _find_term_position(text, term_text)
                
                entities.append({
                    "text": term_text,
                    "label": "technology",  # Default label for slow extraction
                    "score": score,
                    "start": start,
                    "end": end,
                })
                term_texts.append(term_text)
                all_confidences.append(score)
            
            total_entities += len(entities)
            
            converted_chunks.append({
                "text": text,
                "terms": term_texts,
                "entities": entities,
                "keywords": [],  # Slow extraction doesn't produce keywords
                "start_char": 0,  # Not tracked in slow extraction
                "end_char": len(text),
            })
        
        headings.append({
            "heading": heading_text or "",
            "level": 1,  # Default level
            "chunks": converted_chunks,
        })
    
    # Calculate average confidence
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    return {
        "source_file": final_source,
        "headings": headings,
        "avg_confidence": avg_confidence,
        "total_entities": total_entities,
        "is_low_confidence": avg_confidence < 0.7,
        "error": None,
    }
