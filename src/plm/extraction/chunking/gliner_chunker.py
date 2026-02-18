"""GLiNER-aware sentence-packing chunker.

Splits text into sentences using pysbd, then greedily packs sentences into
chunks that stay under the GLiNER token limit. Uses GLiNER's WordsSplitter
for accurate token counting. Never splits mid-sentence unless a single
sentence exceeds the limit.
"""

from __future__ import annotations

import re
import warnings

from pysbd import Segmenter

from .base import Chunk, Chunker, register_chunker

# GLiNER's WordsSplitter uses this regex internally for word splitting.
# We replicate it directly to avoid importing GLiNER at module level
# (which pulls in torch and is slow). This is the exact regex from
# gliner.data_processing.tokenizer.WordsSplitter(splitter_type="whitespace").
_GLINER_WORD_RE = re.compile(r"\w+(?:[-_]\w+)*|\S")

# pysbd emits SyntaxWarnings on Python 3.12+ due to unescaped sequences
# in its regex patterns. Suppress them once at import time.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    _SEGMENTER = Segmenter(language="en", clean=False)

# Default token limit — 200 GLiNER tokens leaves safe headroom under the
# model's max_len=384 even for punctuation-heavy technical text.
DEFAULT_MAX_TOKENS = 200


def count_gliner_tokens(text: str) -> int:
    """Count tokens the way GLiNER counts them.

    Uses the same regex as GLiNER's WordsSplitter so our budget
    matches the model's actual token consumption.
    """
    return len(_GLINER_WORD_RE.findall(text))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using pysbd.

    Handles abbreviations (Dr., e.g.), URLs, code paths,
    and other edge cases that break naive regex splitting.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SyntaxWarning)
        result = _SEGMENTER.segment(text)
    # pysbd returns list[str] at runtime but type stubs are imprecise
    return list(result)  # type: ignore[arg-type]


def _word_based_split(text: str, max_tokens: int) -> list[str]:
    """Fallback: split an oversized sentence by GLiNER tokens.

    Only used when a single sentence exceeds the token limit (rare).
    """
    words = _GLINER_WORD_RE.findall(text)
    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        # Re-join tokens — this is lossy for whitespace but the best
        # we can do when forced to split a single sentence.
        chunk_text = " ".join(words[start:end])
        chunks.append(chunk_text)
        start = end

    return chunks


@register_chunker
class GLiNERChunker(Chunker):
    """Sentence-aware chunker designed for GLiNER's token limits.

    Algorithm:
    1. For markdown: detect headings, split into heading sections.
    2. Within each section, split text into sentences using pysbd.
    3. Greedily pack sentences into chunks ≤ max_tokens GLiNER tokens.
    4. If a heading is set, subtract its token cost from the budget.
    5. Include sentence-level overlap: last sentence of previous chunk
       starts the next chunk, catching cross-boundary entities.
    """

    def __init__(self, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self.max_tokens = max_tokens

    @property
    def name(self) -> str:
        return "gliner"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str, filename: str | None = None) -> list[Chunk]:
        """Chunk a document into GLiNER-safe pieces.

        For markdown files (.md extension or detected headings), splits by
        heading sections first. For plain text, treats the entire document
        as a single section.
        """
        if not text or not text.strip():
            return []

        is_markdown = (filename and filename.endswith(".md")) or bool(
            re.search(r"^#{1,6}\s+.+$", text, re.MULTILINE)
        )

        if is_markdown:
            return self._chunk_markdown(text)
        return self._chunk_plain(text)

    # ------------------------------------------------------------------
    # Internal: markdown chunking
    # ------------------------------------------------------------------

    def _chunk_markdown(self, text: str) -> list[Chunk]:
        """Split markdown by headings, then sentence-chunk each section."""
        sections = self._split_by_headings(text)
        chunks: list[Chunk] = []
        chunk_idx = 0

        for heading, level, section_text, section_start in sections:
            section_chunks = self._sentence_pack(
                section_text,
                heading=heading,
                base_offset=section_start,
            )
            for sc in section_chunks:
                chunks.append(
                    Chunk(
                        text=sc["text"],
                        index=chunk_idx,
                        heading=f"{'#' * level} {heading}" if heading else None,
                        start_char=sc["start"],
                        end_char=sc["end"],
                    )
                )
                chunk_idx += 1

        return chunks

    def _split_by_headings(
        self, text: str
    ) -> list[tuple[str | None, int, str, int]]:
        """Split markdown into (heading, level, body_text, body_start) tuples."""
        heading_re = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)
        matches = list(heading_re.finditer(text))

        if not matches:
            # No headings — treat as plain text section
            return [(None, 0, text, 0)]

        sections: list[tuple[str | None, int, str, int]] = []

        # Content before first heading (if any)
        first_heading_pos = matches[0].start()
        if first_heading_pos > 0:
            pre_text = text[:first_heading_pos].strip()
            if pre_text:
                sections.append((None, 0, pre_text, 0))

        for i, m in enumerate(matches):
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            body_start = m.end()

            # Body extends until next heading or end of text
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[body_start:body_end].strip()

            if body:
                sections.append((heading_text, level, body, body_start))

        return sections

    # ------------------------------------------------------------------
    # Internal: plain text chunking
    # ------------------------------------------------------------------

    def _chunk_plain(self, text: str) -> list[Chunk]:
        """Chunk plain text (no headings) using sentence packing."""
        packed = self._sentence_pack(text, heading=None, base_offset=0)
        return [
            Chunk(
                text=p["text"],
                index=i,
                heading=None,
                start_char=p["start"],
                end_char=p["end"],
            )
            for i, p in enumerate(packed)
        ]

    # ------------------------------------------------------------------
    # Core algorithm: sentence packing with overlap
    # ------------------------------------------------------------------

    def _sentence_pack(
        self,
        text: str,
        heading: str | None,
        base_offset: int,
    ) -> list[dict]:
        """Pack sentences into chunks respecting GLiNER token limits.

        Returns list of dicts with keys: text, start, end.
        """
        if not text.strip():
            return []

        # Compute effective limit after heading budget
        effective_limit = self.max_tokens
        if heading:
            heading_tokens = count_gliner_tokens(heading)
            effective_limit = self.max_tokens - heading_tokens
            if effective_limit < 20:
                # Heading is absurdly long — use minimum budget
                effective_limit = 20

        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[dict] = []
        current_sentences: list[str] = []
        current_token_count = 0
        last_sentence_of_prev: str | None = None

        for sentence in sentences:
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                continue

            sentence_tokens = count_gliner_tokens(sentence_stripped)

            # Case 1: single sentence exceeds limit → word-based fallback
            if sentence_tokens > effective_limit:
                # Flush current buffer first
                if current_sentences:
                    chunk_text = " ".join(current_sentences)
                    start = self._find_offset(text, chunk_text, base_offset)
                    chunks.append({
                        "text": chunk_text,
                        "start": start,
                        "end": start + len(chunk_text),
                    })
                    last_sentence_of_prev = current_sentences[-1]
                    current_sentences = []
                    current_token_count = 0

                # Word-split the oversized sentence
                sub_chunks = _word_based_split(sentence_stripped, effective_limit)
                for sc in sub_chunks:
                    start = self._find_offset(text, sc, base_offset)
                    chunks.append({
                        "text": sc,
                        "start": start,
                        "end": start + len(sc),
                    })
                last_sentence_of_prev = None  # Can't overlap word-split
                continue

            # Case 2: adding this sentence exceeds limit → flush & start new
            if current_token_count + sentence_tokens > effective_limit:
                chunk_text = " ".join(current_sentences)
                start = self._find_offset(text, chunk_text, base_offset)
                chunks.append({
                    "text": chunk_text,
                    "start": start,
                    "end": start + len(chunk_text),
                })
                last_sentence_of_prev = current_sentences[-1]

                # Overlap: start new chunk with last sentence of previous,
                # but only if overlap + new sentence fits within the limit
                if last_sentence_of_prev:
                    overlap_tokens = count_gliner_tokens(last_sentence_of_prev)
                    if overlap_tokens + sentence_tokens <= effective_limit:
                        current_sentences = [last_sentence_of_prev, sentence_stripped]
                        current_token_count = overlap_tokens + sentence_tokens
                    else:
                        current_sentences = [sentence_stripped]
                        current_token_count = sentence_tokens
                else:
                    current_sentences = [sentence_stripped]
                    current_token_count = sentence_tokens
                continue

            # Case 3: sentence fits → add to current chunk
            current_sentences.append(sentence_stripped)
            current_token_count += sentence_tokens

        # Flush remaining
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start = self._find_offset(text, chunk_text, base_offset)
            chunks.append({
                "text": chunk_text,
                "start": start,
                "end": start + len(chunk_text),
            })

        return chunks

    @staticmethod
    def _find_offset(text: str, chunk_text: str, base_offset: int) -> int:
        """Find character offset of chunk_text within the source text.

        Falls back to base_offset if exact match not found (can happen
        after word-based splitting or whitespace normalization).
        """
        # Try finding the first few words as anchor
        anchor = chunk_text[:60]
        idx = text.find(anchor)
        if idx >= 0:
            return base_offset + idx
        return base_offset
