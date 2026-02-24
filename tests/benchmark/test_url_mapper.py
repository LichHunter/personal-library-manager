from pathlib import Path
from plm.benchmark.mapping.url_mapper import normalize_url, slugify_heading, build_mappings, ChunkInfo

def test_normalize_url():
    assert normalize_url("https://kubernetes.io/docs/home/") == ("/docs/home", None)
    assert normalize_url("http://kubernetes.io/docs/home") == ("/docs/home", None)
    assert normalize_url("https://kubernetes.io/docs/home#section") == ("/docs/home", "section")
    assert normalize_url("https://kubernetes.io/") == ("/", None)
    assert normalize_url("https://kubernetes.io") == ("", None)

def test_slugify_heading():
    assert slugify_heading("## Introduction") == "introduction"
    assert slugify_heading("### Getting Started with K8s") == "getting-started-with-k8s"
    assert slugify_heading("Heading with !@#$%^&*() symbols") == "heading-with-symbols"
    assert slugify_heading("  Multiple   Spaces  ") == "multiple-spaces"
    assert slugify_heading("---Leading and Trailing---") == "leading-and-trailing"

def test_build_mappings(tmp_path):
    chunks = [
        ChunkInfo(
            chunk_id="c1",
            doc_id="d1",
            source_file="https://kubernetes.io/docs/concepts/workloads/pods/",
            heading="## Pods",
            heading_id="pods",
            start_char=0,
            end_char=100
        ),
        ChunkInfo(
            chunk_id="c2",
            doc_id="d1",
            source_file="https://kubernetes.io/docs/concepts/workloads/pods/",
            heading="### Pod Lifecycle",
            heading_id="pod-lifecycle",
            start_char=101,
            end_char=200
        ),
        ChunkInfo(
            chunk_id="c3",
            doc_id="d2",
            source_file="not-a-url",
            heading=None,
            heading_id=None,
            start_char=None,
            end_char=None
        )
    ]
    
    unmapped_log = tmp_path / "unmapped.log"
    url_to_docid, url_to_chunks, anchor_to_heading = build_mappings(chunks, unmapped_log)
    
    assert "/docs/concepts/workloads/pods" in url_to_docid
    assert url_to_docid["/docs/concepts/workloads/pods"] == ["d1"]
    assert url_to_chunks["/docs/concepts/workloads/pods"] == ["c1", "c2"]
    
    assert "pods" in anchor_to_heading
    assert anchor_to_heading["pods"][0]["doc_id"] == "d1"
    assert anchor_to_heading["pods"][0]["heading_text"] == "## Pods"
    
    assert "pod-lifecycle" in anchor_to_heading
    
    assert unmapped_log.exists()
    assert "not-a-url" in unmapped_log.read_text()
