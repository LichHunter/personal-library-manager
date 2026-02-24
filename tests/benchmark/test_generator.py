import json
from plm.benchmark.generation.generator import (
    count_words,
    normalize_whitespace,
    validate_quote_in_chunks,
    parse_llm_response,
    validate_generated_case
)

def test_count_words():
    assert count_words("Hello world") == 2
    assert count_words("  Multiple   spaces  ") == 2
    assert count_words("") == 0

def test_normalize_whitespace():
    assert normalize_whitespace("  Hello   world  \n  ") == "Hello world"

def test_validate_quote_in_chunks():
    chunks = ["This is a test chunk.", "Another chunk with some text."]
    assert validate_quote_in_chunks("test chunk", chunks) is True
    assert validate_quote_in_chunks("Another chunk", chunks) is True
    assert validate_quote_in_chunks("not in here", chunks) is False
    assert validate_quote_in_chunks("", chunks) is True

def test_parse_llm_response():
    resp1 = '{"query": "test"}'
    assert parse_llm_response(resp1) == {"query": "test"}
    
    resp2 = '```json\n{"query": "test"}\n```'
    assert parse_llm_response(resp2) == {"query": "test"}
    
    resp3 = 'Here is the JSON:\n```json\n{"query": "test"}\n```\nHope this helps.'
    assert parse_llm_response(resp3) == {"query": "test"}
    
    assert parse_llm_response("not json") is None

def test_validate_generated_case():
    chunk_contents = ["This is a long enough chunk to contain a thirty character quote for gold tier."]
    
    valid_parsed = {
        "query": "How do I test this component with a long enough query?",
        "matched_quote": "thirty character quote for gold tier",
        "evidence_text": "Found in chunk",
        "reasoning": "Matches the query"
    }
    is_valid, error = validate_generated_case(valid_parsed, "gold", chunk_contents)
    assert is_valid is True
    
    short_query = valid_parsed.copy()
    short_query["query"] = "Too short"
    is_valid, error = validate_generated_case(short_query, "gold", chunk_contents)
    assert is_valid is False
    assert "too short" in error.lower()
    
    bad_quote = valid_parsed.copy()
    bad_quote["matched_quote"] = "not in chunks at all anywhere"
    is_valid, error = validate_generated_case(bad_quote, "gold", chunk_contents)
    assert is_valid is False
    assert "not found verbatim" in error.lower()
    
    short_gold_quote = valid_parsed.copy()
    short_gold_quote["matched_quote"] = "long enough chunk"
    is_valid, error = validate_generated_case(short_gold_quote, "gold", chunk_contents)
    assert is_valid is False
    assert "gold quote too short" in error.lower()
