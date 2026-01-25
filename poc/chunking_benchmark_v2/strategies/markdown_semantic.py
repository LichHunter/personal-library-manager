"""Markdown-aware semantic chunking strategy.

Two-stage splitting approach:
1. Split by markdown headings (preserve semantic structure)
2. Split large sections at natural boundaries (paragraphs, not mid-sentence)

Features:
- Code block preservation (never splits inside code blocks)
- Minimum chunk size (merges tiny chunks with adjacent content)
- Heading context preserved in each chunk
- Code-heavy detection for downstream processing
"""

import re
from dataclasses import dataclass
from .base import ChunkingStrategy, Chunk, Document


CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Section:
    heading: str
    heading_level: int
    content: str
    start_char: int
    end_char: int


def extract_code_blocks(text: str) -> list[tuple[int, int, str]]:
    """Extract code block positions and content.

    Returns list of (start, end, code_content) tuples.
    """
    blocks = []
    for match in CODE_BLOCK_PATTERN.finditer(text):
        blocks.append((match.start(), match.end(), match.group()))
    return blocks


def is_inside_code_block(pos: int, code_blocks: list[tuple[int, int, str]]) -> bool:
    """Check if position is inside any code block."""
    for start, end, _ in code_blocks:
        if start <= pos < end:
            return True
    return False


def calculate_code_ratio(text: str) -> float:
    """Calculate ratio of code to total text."""
    if not text:
        return 0.0
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    return code_chars / len(text)


def is_mostly_code(text: str, threshold: float = 0.5) -> bool:
    """Check if text is predominantly code (> threshold ratio)."""
    return calculate_code_ratio(text) > threshold


def word_count(text: str) -> int:
    """Count words in text, excluding code blocks for accurate count."""
    text_without_code = CODE_BLOCK_PATTERN.sub(" ", text)
    return len(text_without_code.split())


class MarkdownSemanticStrategy(ChunkingStrategy):
    """Markdown-aware semantic chunking with code block preservation.

    Two-stage approach:
    1. Split by headings to preserve semantic structure
    2. Split large sections at paragraph boundaries

    Code blocks are never split - they're kept as atomic units.
    Tiny sections are merged with adjacent content.
    """

    def __init__(
        self,
        max_heading_level: int = 4,
        target_chunk_size: int = 400,
        min_chunk_size: int = 50,
        max_chunk_size: int = 800,
        overlap_sentences: int = 1,
    ):
        """
        Args:
            max_heading_level: Deepest heading level to split on (1-6)
            target_chunk_size: Target chunk size in words
            min_chunk_size: Minimum chunk size in words (smaller chunks get merged)
            max_chunk_size: Maximum chunk size before forced split
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.max_heading_level = max_heading_level
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences

    @property
    def name(self) -> str:
        return f"markdown_semantic_{self.target_chunk_size}"

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document using markdown-aware semantic chunking."""
        content = document.content

        if not content.strip():
            return []

        sections = self._extract_sections(content)

        if not sections:
            return self._create_single_chunk(document, content)

        sections = self._merge_tiny_sections(sections)

        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(document.id, section)
            chunks.extend(section_chunks)

        chunks = self._merge_tiny_chunks(chunks)

        for i, chunk in enumerate(chunks):
            chunk.id = f"{document.id}_mdsem_{i}"
            chunk.metadata["chunk_idx"] = i

        return chunks

    def _extract_sections(self, content: str) -> list[Section]:
        """Extract sections based on markdown headings."""
        heading_pattern = re.compile(
            r"^(#{1," + str(self.max_heading_level) + r"})\s+(.+)$", re.MULTILINE
        )

        headings = []
        for match in heading_pattern.finditer(content):
            headings.append(
                {
                    "level": len(match.group(1)),
                    "title": match.group(2).strip(),
                    "start": match.start(),
                    "end": match.end(),
                }
            )

        if not headings:
            return []

        sections = []

        if headings[0]["start"] > 0:
            intro_content = content[: headings[0]["start"]].strip()
            if intro_content and word_count(intro_content) >= self.min_chunk_size // 2:
                sections.append(
                    Section(
                        heading="Introduction",
                        heading_level=0,
                        content=intro_content,
                        start_char=0,
                        end_char=headings[0]["start"],
                    )
                )

        for i, heading in enumerate(headings):
            section_end = (
                headings[i + 1]["start"] if i + 1 < len(headings) else len(content)
            )
            section_content = content[heading["end"] : section_end].strip()

            full_content = (
                f"{'#' * heading['level']} {heading['title']}\n\n{section_content}"
            )

            sections.append(
                Section(
                    heading=heading["title"],
                    heading_level=heading["level"],
                    content=full_content,
                    start_char=heading["start"],
                    end_char=section_end,
                )
            )

        return sections

    def _merge_tiny_sections(self, sections: list[Section]) -> list[Section]:
        """Merge sections smaller than min_chunk_size with adjacent sections."""
        if len(sections) <= 1:
            return sections

        merged = []
        i = 0

        while i < len(sections):
            section = sections[i]
            wc = word_count(section.content)

            if wc < self.min_chunk_size and i + 1 < len(sections):
                next_section = sections[i + 1]
                combined_content = f"{section.content}\n\n{next_section.content}"
                merged_section = Section(
                    heading=section.heading,
                    heading_level=section.heading_level,
                    content=combined_content,
                    start_char=section.start_char,
                    end_char=next_section.end_char,
                )
                merged.append(merged_section)
                i += 2
            else:
                merged.append(section)
                i += 1

        return merged

    def _chunk_section(self, doc_id: str, section: Section) -> list[Chunk]:
        """Chunk a single section, respecting code blocks and size limits."""
        content = section.content
        wc = word_count(content)

        if wc <= self.max_chunk_size:
            return [self._create_chunk(doc_id, section, content, 0, is_split=False)]

        return self._split_large_section(doc_id, section)

    def _split_large_section(self, doc_id: str, section: Section) -> list[Chunk]:
        """Split a large section at natural boundaries, preserving code blocks."""
        content = section.content
        code_blocks = extract_code_blocks(content)

        paragraphs = self._split_into_paragraphs(content, code_blocks)

        chunks = []
        current_paragraphs = []
        current_word_count = 0
        chunk_idx = 0

        for para in paragraphs:
            para_wc = word_count(para)

            if (
                current_word_count + para_wc > self.target_chunk_size
                and current_paragraphs
            ):
                chunk_content = "\n\n".join(current_paragraphs)

                if chunk_idx == 0:
                    pass
                else:
                    chunk_content = f"{'#' * section.heading_level} {section.heading} (continued)\n\n{chunk_content}"

                chunks.append(
                    self._create_chunk(
                        doc_id, section, chunk_content, chunk_idx, is_split=True
                    )
                )
                chunk_idx += 1

                if self.overlap_sentences > 0 and current_paragraphs:
                    overlap_para = current_paragraphs[-1]
                    sentences = re.split(r"(?<=[.!?])\s+", overlap_para)
                    overlap = (
                        " ".join(sentences[-self.overlap_sentences :])
                        if len(sentences) > self.overlap_sentences
                        else overlap_para
                    )
                    current_paragraphs = [overlap]
                    current_word_count = word_count(overlap)
                else:
                    current_paragraphs = []
                    current_word_count = 0

            current_paragraphs.append(para)
            current_word_count += para_wc

        if current_paragraphs:
            chunk_content = "\n\n".join(current_paragraphs)
            if chunk_idx > 0:
                chunk_content = f"{'#' * section.heading_level} {section.heading} (continued)\n\n{chunk_content}"
            chunks.append(
                self._create_chunk(
                    doc_id, section, chunk_content, chunk_idx, is_split=True
                )
            )

        return chunks

    def _split_into_paragraphs(
        self, content: str, code_blocks: list[tuple[int, int, str]]
    ) -> list[str]:
        """Split content into paragraphs, keeping code blocks intact."""
        if not code_blocks:
            return [p.strip() for p in content.split("\n\n") if p.strip()]

        paragraphs = []
        last_end = 0

        for start, end, code in code_blocks:
            if start > last_end:
                before_code = content[last_end:start]
                for p in before_code.split("\n\n"):
                    if p.strip():
                        paragraphs.append(p.strip())

            paragraphs.append(code)
            last_end = end

        if last_end < len(content):
            after_code = content[last_end:]
            for p in after_code.split("\n\n"):
                if p.strip():
                    paragraphs.append(p.strip())

        return paragraphs

    def _create_chunk(
        self, doc_id: str, section: Section, content: str, part_idx: int, is_split: bool
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        code_ratio = calculate_code_ratio(content)

        return Chunk(
            id=f"{doc_id}_mdsem_{part_idx}",
            doc_id=doc_id,
            content=content,
            start_char=section.start_char,
            end_char=section.end_char,
            heading=section.heading,
            heading_path=[section.heading],
            level=section.heading_level,
            metadata={
                "strategy": self.name,
                "heading_level": section.heading_level,
                "is_split": is_split,
                "part_idx": part_idx,
                "code_ratio": round(code_ratio, 2),
                "is_mostly_code": code_ratio > 0.5,
                "word_count": word_count(content),
            },
        )

    def _create_single_chunk(self, document: Document, content: str) -> list[Chunk]:
        """Create a single chunk for documents without headings."""
        code_ratio = calculate_code_ratio(content)

        return [
            Chunk(
                id=f"{document.id}_mdsem_0",
                doc_id=document.id,
                content=content,
                start_char=0,
                end_char=len(content),
                heading=document.title,
                heading_path=[document.title],
                level=0,
                metadata={
                    "strategy": self.name,
                    "heading_level": 0,
                    "is_split": False,
                    "part_idx": 0,
                    "code_ratio": round(code_ratio, 2),
                    "is_mostly_code": code_ratio > 0.5,
                    "word_count": word_count(content),
                },
            )
        ]

    def _merge_tiny_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Final pass: merge any remaining tiny chunks."""
        if len(chunks) <= 1:
            return chunks

        merged = []
        i = 0

        while i < len(chunks):
            chunk = chunks[i]
            wc = chunk.metadata.get("word_count", word_count(chunk.content))

            if wc < self.min_chunk_size and merged:
                prev_chunk = merged[-1]
                combined_content = f"{prev_chunk.content}\n\n{chunk.content}"
                prev_chunk.content = combined_content
                prev_chunk.end_char = chunk.end_char
                prev_chunk.metadata["word_count"] = word_count(combined_content)
                prev_chunk.metadata["code_ratio"] = round(
                    calculate_code_ratio(combined_content), 2
                )
                prev_chunk.metadata["is_mostly_code"] = (
                    prev_chunk.metadata["code_ratio"] > 0.5
                )
            else:
                merged.append(chunk)

            i += 1

        return merged
