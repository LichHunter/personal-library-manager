"""Tests for GLiNER fast extraction pipeline.

Covers: chunker, extraction wrapper, document processor, and CLI.
Uses real GLiNER model (no mocking) with StackOverflow test data.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from plm.extraction.chunking import get_chunker
from plm.extraction.chunking.gliner_chunker import (
    GLiNERChunker,
    count_gliner_tokens,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# GLiNER token counting
# ---------------------------------------------------------------------------

class TestGLiNERTokenCounting:
    def test_simple_words(self):
        assert count_gliner_tokens("hello world") == 2

    def test_punctuation_counted_separately(self):
        # "Hello," → "Hello" + ","
        assert count_gliner_tokens("Hello,") == 2

    def test_hyphenated_words_stay_together(self):
        # "hello-world" → 1 token (regex: \w+(?:[-_]\w+)*)
        assert count_gliner_tokens("hello-world") == 1

    def test_technical_text_expansion(self):
        text = 'We use React.js and PostgreSQL.'
        tokens = count_gliner_tokens(text)
        python_words = len(text.split())
        assert tokens > python_words, "Technical text should expand beyond Python word count"

    def test_empty_string(self):
        assert count_gliner_tokens("") == 0

    def test_code_heavy_text(self):
        text = 'config.get("key", default=None)'
        tokens = count_gliner_tokens(text)
        assert tokens > len(text.split())


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

class TestSentenceSplitting:
    def test_basic_sentences(self):
        sentences = _split_sentences("First sentence. Second sentence.")
        assert len(sentences) == 2

    def test_abbreviations_not_split(self):
        sentences = _split_sentences("Dr. Smith said hello.")
        joined = "".join(sentences)
        assert "Dr." in joined

    def test_empty_text(self):
        sentences = _split_sentences("")
        assert sentences == [""]  or len(sentences) == 0 or sentences == ['']


# ---------------------------------------------------------------------------
# GLiNER Chunker
# ---------------------------------------------------------------------------

class TestGLiNERChunker:
    @pytest.fixture
    def chunker(self):
        return GLiNERChunker(max_tokens=200)

    @pytest.fixture
    def small_chunker(self):
        return GLiNERChunker(max_tokens=50)

    def test_name_property(self, chunker):
        assert chunker.name == "gliner"

    def test_registry_lookup(self):
        c = get_chunker("gliner")
        assert isinstance(c, GLiNERChunker)

    def test_empty_input(self, chunker):
        assert chunker.chunk("") == []
        assert chunker.chunk("   \n  ") == []

    def test_all_chunks_within_token_limit(self, chunker):
        text = "This is a test sentence. " * 100
        chunks = chunker.chunk(text, filename="test.txt")
        for c in chunks:
            tokens = count_gliner_tokens(c.text)
            assert tokens <= 200, f"Chunk {c.index} has {tokens} tokens (limit 200)"

    def test_sentence_boundaries_respected(self, small_chunker):
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        chunks = small_chunker.chunk(text, filename="test.txt")
        for c in chunks:
            stripped = c.text.strip()
            if stripped:
                # Each chunk should end at a sentence boundary (period, !, ?)
                assert stripped[-1] in ".!?", f"Chunk does not end at sentence boundary: ...{stripped[-30:]}"

    def test_greedy_packing(self, chunker):
        # 4 short sentences ~40 tokens each = ~160 total, should fit in 1 chunk
        text = ("Word " * 9 + "end. ") * 4
        chunks = chunker.chunk(text, filename="test.txt")
        assert len(chunks) <= 2, f"Expected 1-2 chunks for ~160 tokens, got {len(chunks)}"

    def test_markdown_headings_detected(self, chunker):
        md = "# Intro\n\nSome content here.\n\n## Setup\n\nMore content here."
        chunks = chunker.chunk(md, filename="doc.md")
        headings = {c.heading for c in chunks}
        assert any("Intro" in (h or "") for h in headings)
        assert any("Setup" in (h or "") for h in headings)

    def test_plain_text_no_headings(self, chunker):
        text = "Just a plain text document about software engineering."
        chunks = chunker.chunk(text, filename="doc.txt")
        assert len(chunks) >= 1
        assert all(c.heading is None for c in chunks)

    def test_char_positions_valid(self, small_chunker):
        text = "First sentence. Second sentence. Third sentence."
        chunks = small_chunker.chunk(text, filename="test.txt")
        for c in chunks:
            assert c.start_char >= 0
            assert c.end_char >= c.start_char

    def test_sequential_indices(self, small_chunker):
        text = "Sentence one. " * 30
        chunks = small_chunker.chunk(text, filename="test.txt")
        for i, c in enumerate(chunks):
            assert c.index == i

    def test_oversized_sentence_word_split(self):
        chunker = GLiNERChunker(max_tokens=20)
        # Single sentence with way more than 20 tokens
        text = "word " * 50 + "end."
        chunks = chunker.chunk(text, filename="test.txt")
        assert len(chunks) >= 2
        for c in chunks:
            assert count_gliner_tokens(c.text) <= 20


# ---------------------------------------------------------------------------
# GLiNER extraction wrapper
# ---------------------------------------------------------------------------

class TestGLiNERExtraction:
    def test_extract_known_entities(self):
        from plm.extraction.fast.gliner import extract_entities
        text = "We use React and PostgreSQL for our TypeScript application."
        entities = extract_entities(text)
        names = {e.text for e in entities}
        # Should find at least some of these well-known technologies
        assert names & {"React", "PostgreSQL", "TypeScript"}, f"Expected tech entities, got: {names}"

    def test_entity_fields(self):
        from plm.extraction.fast.gliner import extract_entities
        text = "Python is a programming language."
        entities = extract_entities(text)
        if entities:
            e = entities[0]
            assert isinstance(e.text, str)
            assert isinstance(e.label, str)
            assert isinstance(e.score, float)
            assert 0.0 <= e.score <= 1.0
            assert isinstance(e.start, int)
            assert isinstance(e.end, int)

    def test_empty_input(self):
        from plm.extraction.fast.gliner import extract_entities
        assert extract_entities("") == []
        assert extract_entities("   ") == []

    def test_deduplication(self):
        from plm.extraction.fast.gliner import extract_entities
        text = "Use Python with Python libraries."
        entities = extract_entities(text)
        # Each (text, start, end) should be unique
        keys = [(e.text, e.start, e.end) for e in entities]
        assert len(keys) == len(set(keys))

    def test_sorted_by_position(self):
        from plm.extraction.fast.gliner import extract_entities
        text = "React works well with PostgreSQL and TypeScript."
        entities = extract_entities(text)
        positions = [e.start for e in entities]
        assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Document processor
# ---------------------------------------------------------------------------

class TestDocumentProcessor:
    def test_process_markdown_file(self, tmp_path):
        from plm.extraction.fast.document_processor import process_document
        doc = tmp_path / "test.md"
        doc.write_text("# Tech Stack\n\nWe use React and Node.js.\n\n## DB\n\nPostgreSQL for storage.\n")
        result = process_document(doc)
        assert result.source_file == str(doc)
        assert len(result.headings) >= 2
        assert result.total_entities >= 1
        assert isinstance(result.avg_confidence, float)
        assert result.error is None

    def test_process_plain_text(self, tmp_path):
        from plm.extraction.fast.document_processor import process_document
        doc = tmp_path / "test.txt"
        doc.write_text("Python and Docker are popular technologies.")
        result = process_document(doc)
        assert len(result.headings) == 1
        assert result.headings[0].heading == "(root)"

    def test_missing_file(self, tmp_path):
        from plm.extraction.fast.document_processor import process_document
        result = process_document(tmp_path / "nonexistent.txt")
        assert result.error is not None
        assert result.is_low_confidence is True

    def test_empty_file(self, tmp_path):
        from plm.extraction.fast.document_processor import process_document
        doc = tmp_path / "empty.txt"
        doc.write_text("")
        result = process_document(doc)
        assert result.total_entities == 0
        assert result.avg_confidence == 0.0

    def test_zero_entity_confidence(self, tmp_path):
        from plm.extraction.fast.document_processor import process_document
        doc = tmp_path / "gibberish.txt"
        doc.write_text("asdf qwer zxcv jklm poiu mnbv")
        result = process_document(doc)
        assert result.avg_confidence == 0.0
        assert result.is_low_confidence is True

    def test_chunk_results_have_terms(self, tmp_path):
        from plm.extraction.fast.document_processor import process_document
        doc = tmp_path / "tech.md"
        doc.write_text("# Tools\n\nWe use React, PostgreSQL, and TypeScript daily.")
        result = process_document(doc)
        all_terms = []
        for section in result.headings:
            for chunk in section.chunks:
                all_terms.extend(chunk.terms)
        assert len(all_terms) >= 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCLI:
    def test_help_flag(self):
        from plm.extraction.fast.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_missing_input_dir(self, tmp_path):
        from plm.extraction.fast.cli import main
        ret = main(["--input", str(tmp_path / "nope"), "--output", str(tmp_path / "out")])
        assert ret == 1

    def test_no_matching_files(self, tmp_path):
        from plm.extraction.fast.cli import main
        input_dir = tmp_path / "empty_in"
        input_dir.mkdir()
        output_dir = tmp_path / "out"
        ret = main(["--input", str(input_dir), "--output", str(output_dir)])
        assert ret == 0

    def test_processes_files_and_produces_json(self, tmp_path):
        from plm.extraction.fast.cli import main
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "doc1.txt").write_text("React and TypeScript are great frameworks.")
        output_dir = tmp_path / "out"

        ret = main(["--input", str(input_dir), "--output", str(output_dir), "--pattern", "**/*.txt"])
        assert ret == 0
        json_file = output_dir / "doc1.json"
        assert json_file.exists()
        data = json.loads(json_file.read_text())
        assert "headings" in data
        assert "avg_confidence" in data

    def test_low_confidence_dir(self, tmp_path):
        from plm.extraction.fast.cli import main
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        (input_dir / "gibberish.txt").write_text("asdf qwer zxcv jklm random words nothing")
        output_dir = tmp_path / "out"
        low_dir = tmp_path / "low"

        ret = main([
            "--input", str(input_dir),
            "--output", str(output_dir),
            "--low-confidence-dir", str(low_dir),
            "--confidence-threshold", "0.7",
            "--pattern", "**/*.txt",
        ])
        assert ret == 0
        # Gibberish should be flagged (0 entities → avg_confidence=0.0 < 0.7)
        manifest = low_dir / "manifest.json"
        assert manifest.exists()
        manifest_data = json.loads(manifest.read_text())
        assert manifest_data["total_flagged"] >= 1


# ---------------------------------------------------------------------------
# Integration: StackOverflow data
# ---------------------------------------------------------------------------

class TestStackOverflowIntegration:
    """Integration tests using real StackOverflow Q&A data."""

    @pytest.fixture
    def so_data(self):
        path = Path("data/vocabularies/train_documents.json")
        if not path.exists():
            pytest.skip("StackOverflow data not available")
        return json.loads(path.read_text())

    def test_extract_from_so_docs(self, so_data, tmp_path):
        """Process several SO docs and verify entities are found."""
        from plm.extraction.fast.document_processor import process_document

        # Pick docs with known tech terms
        tech_docs = [d for d in so_data if any(
            kw.lower() in d["text"].lower()
            for kw in ["python", "javascript", "react", "java", "docker"]
        )][:5]

        for doc in tech_docs:
            filepath = tmp_path / f"{doc['doc_id']}.txt"
            filepath.write_text(doc["text"])
            result = process_document(filepath)

            assert result.error is None, f"Error processing {doc['doc_id']}: {result.error}"
            # Tech-mentioning docs should find at least 1 entity
            assert result.total_entities >= 1, (
                f"{doc['doc_id']} should have entities. Text: {doc['text'][:100]}"
            )

    def test_so_doc_gt_overlap(self, so_data, tmp_path):
        """Check that extracted entities overlap with ground truth for a known doc."""
        from plm.extraction.fast.document_processor import process_document

        # Q26585170 mentions Haskell, C++, Java, Python, Julia, Lisp etc.
        doc = next((d for d in so_data if d["doc_id"] == "Q26585170"), None)
        if doc is None:
            pytest.skip("Q26585170 not in dataset")

        filepath = tmp_path / "Q26585170.txt"
        filepath.write_text(doc["text"])
        result = process_document(filepath)

        extracted = set()
        for section in result.headings:
            for chunk in section.chunks:
                extracted.update(t.lower() for t in chunk.terms)

        gt_lower = {t.lower() for t in doc["gt_terms"]}
        overlap = extracted & gt_lower
        # This doc has many programming languages — should find several
        assert len(overlap) >= 3, (
            f"Expected ≥3 GT overlaps, got {len(overlap)}. "
            f"Extracted: {sorted(extracted)}, GT: {sorted(gt_lower)}"
        )

    def test_no_chunks_exceed_token_limit(self, so_data, tmp_path):
        """Verify no chunk exceeds 200 GLiNER tokens across multiple docs."""
        chunker = GLiNERChunker(max_tokens=200)

        for doc in so_data[:20]:
            filepath = tmp_path / f"{doc['doc_id']}.txt"
            filepath.write_text(doc["text"])
            chunks = chunker.chunk(doc["text"], filename=filepath.name)
            for c in chunks:
                tokens = count_gliner_tokens(c.text)
                assert tokens <= 200, (
                    f"{doc['doc_id']} chunk {c.index}: {tokens} tokens > 200"
                )
