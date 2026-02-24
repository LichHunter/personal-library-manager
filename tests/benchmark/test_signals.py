import hashlib
from plm.benchmark.extraction.signals import compute_corpus_version_hash, extract_url_path

def test_compute_corpus_version_hash():
    chunk_ids = ["c1", "c2"]
    chunks = {"c1": "content1", "c2": "content2"}
    
    hash1 = compute_corpus_version_hash(chunk_ids, chunks)
    hash2 = compute_corpus_version_hash(chunk_ids, chunks)
    
    assert hash1 == hash2
    assert len(hash1) == 64
    
    hash3 = compute_corpus_version_hash(["c2", "c1"], chunks)
    assert hash1 == hash3
    
    hash4 = compute_corpus_version_hash(chunk_ids, {"c1": "changed", "c2": "content2"})
    assert hash1 != hash4

def test_extract_url_path():
    assert extract_url_path("https://kubernetes.io/docs/home/") == "/docs/home"
    assert extract_url_path("https://kubernetes.io/docs/home#section") == "/docs/home"
    assert extract_url_path("https://kubernetes.io/") == "/"
