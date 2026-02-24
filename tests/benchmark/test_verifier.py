from plm.benchmark.verification.verifier import verify
from plm.benchmark.generation.generator import GeneratedCase

def test_verify_success():
    case = GeneratedCase(
        case_id="case1",
        bundle_id="bundle1",
        query="This is a valid query with more than five words.",
        matched_quote="this is a verbatim quote that is definitely long enough for gold tier",
        evidence_text="Evidence",
        reasoning="Reasoning",
        chunk_ids=["c1"],
        tier_from_signals="gold",
        generation_timestamp="2024-01-01T00:00:00Z",
        generator_model="test-model",
        internal_retries=0
    )
    corpus = {"c1": "This is a verbatim quote that is definitely long enough for gold tier."}
    
    result = verify(case, corpus)
    assert result.passed is True
    assert len(result.failures) == 0

def test_verify_missing_chunk():
    case = GeneratedCase(
        case_id="case1",
        bundle_id="bundle1",
        query="This is a valid query with more than five words.",
        matched_quote=None,
        evidence_text="Evidence",
        reasoning="Reasoning",
        chunk_ids=["missing"],
        tier_from_signals="silver",
        generation_timestamp="2024-01-01T00:00:00Z",
        generator_model="test-model",
        internal_retries=0
    )
    corpus = {"c1": "content"}
    
    result = verify(case, corpus)
    assert result.passed is False
    assert any(f.check_name == "chunk_exists" for f in result.failures)

def test_verify_missing_quote():
    case = GeneratedCase(
        case_id="case1",
        bundle_id="bundle1",
        query="This is a valid query with more than five words.",
        matched_quote="not in corpus",
        evidence_text="Evidence",
        reasoning="Reasoning",
        chunk_ids=["c1"],
        tier_from_signals="silver",
        generation_timestamp="2024-01-01T00:00:00Z",
        generator_model="test-model",
        internal_retries=0
    )
    corpus = {"c1": "some other content"}
    
    result = verify(case, corpus)
    assert result.passed is False
    assert any(f.check_name == "quote_exists" for f in result.failures)

def test_verify_short_gold_quote():
    case = GeneratedCase(
        case_id="case1",
        bundle_id="bundle1",
        query="This is a valid query with more than five words.",
        matched_quote="short",
        evidence_text="Evidence",
        reasoning="Reasoning",
        chunk_ids=["c1"],
        tier_from_signals="gold",
        generation_timestamp="2024-01-01T00:00:00Z",
        generator_model="test-model",
        internal_retries=0
    )
    corpus = {"c1": "short"}
    
    result = verify(case, corpus)
    assert result.passed is False
    assert any(f.check_name == "quote_length" for f in result.failures)

def test_verify_invalid_query_length():
    case = GeneratedCase(
        case_id="case1",
        bundle_id="bundle1",
        query="Short.",
        matched_quote=None,
        evidence_text="Evidence",
        reasoning="Reasoning",
        chunk_ids=["c1"],
        tier_from_signals="silver",
        generation_timestamp="2024-01-01T00:00:00Z",
        generator_model="test-model",
        internal_retries=0
    )
    corpus = {"c1": "content"}
    
    result = verify(case, corpus)
    assert result.passed is False
    assert any(f.check_name == "query_length" for f in result.failures)
