"""HeadingChunker - paragraph-based chunking with heading context."""

import re
from .base import Chunk, Chunker, register_chunker


@register_chunker
class HeadingChunker(Chunker):
    def __init__(
        self,
        min_tokens: int = 50,
        max_tokens: int = 256,
        prepend_heading: bool = True,
        heading_separator: str = "\n\n",
    ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.prepend_heading = prepend_heading
        self.heading_separator = heading_separator
        self.min_words = int(min_tokens * 0.75)
        self.max_words = int(max_tokens * 0.75)
    
    @property
    def name(self) -> str:
        return "heading"
    
    def chunk(self, text: str, filename: str | None = None) -> list[Chunk]:
        heading_map = self._build_heading_map(text)
        paragraphs = self._split_paragraphs(text)
        
        if not paragraphs:
            return []
        
        chunks: list[Chunk] = []
        chunk_idx = 0
        i = 0
        
        while i < len(paragraphs):
            para = paragraphs[i]
            words = para["text"].split()
            current_heading = self._get_heading_at(heading_map, para["start"])
            
            if len(words) > self.max_words:
                sub_chunks = self._split_large_paragraph(
                    para, current_heading, chunk_idx
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                i += 1
                continue
            
            merged_text = para["text"]
            merged_start = para["start"]
            merged_end = para["end"]
            
            while len(merged_text.split()) < self.min_words and i + 1 < len(paragraphs):
                next_para = paragraphs[i + 1]
                next_words = next_para["text"].split()
                
                if len(merged_text.split()) + len(next_words) > self.max_words:
                    break
                
                next_heading = self._get_heading_at(heading_map, next_para["start"])
                if next_heading != current_heading:
                    break
                
                merged_text = merged_text + "\n\n" + next_para["text"]
                merged_end = next_para["end"]
                i += 1
            
            if self.prepend_heading and current_heading:
                display_text = f"{current_heading}{self.heading_separator}{merged_text}"
            else:
                display_text = merged_text
            
            chunks.append(Chunk(
                text=display_text,
                index=chunk_idx,
                heading=current_heading,
                start_char=merged_start,
                end_char=merged_end,
            ))
            chunk_idx += 1
            i += 1
        
        return chunks
    
    def _build_heading_map(self, content: str) -> list[dict]:
        headings = []
        for match in re.finditer(r"^(#{1,6})\s+(.+?)$", content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({
                "pos": match.start(),
                "level": level,
                "text": text,
            })
        return sorted(headings, key=lambda h: h["pos"])
    
    def _get_heading_at(self, heading_map: list[dict], pos: int) -> str | None:
        if not heading_map:
            return None
        
        active_headings: list[dict] = []
        for h in heading_map:
            if h["pos"] > pos:
                break
            while active_headings and active_headings[-1]["level"] >= h["level"]:
                active_headings.pop()
            active_headings.append(h)
        
        if not active_headings:
            return None
        
        return "#" * active_headings[-1]["level"] + " " + active_headings[-1]["text"]
    
    def _split_paragraphs(self, content: str) -> list[dict]:
        raw_paragraphs = re.split(r"\n\s*\n", content)
        paragraphs = []
        current_pos = 0
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                current_pos = content.find("\n\n", current_pos)
                if current_pos >= 0:
                    current_pos += 2
                continue
            
            if re.match(r"^#{1,6}\s+.+$", para) and "\n" not in para:
                current_pos = content.find(para, current_pos)
                if current_pos >= 0:
                    current_pos += len(para)
                continue
            
            start = content.find(para, current_pos)
            if start < 0:
                start = current_pos
            
            paragraphs.append({
                "text": para,
                "start": start,
                "end": start + len(para),
            })
            current_pos = start + len(para)
        
        return paragraphs
    
    def _split_large_paragraph(
        self,
        para: dict,
        heading: str | None,
        base_idx: int,
    ) -> list[Chunk]:
        chunks = []
        words = para["text"].split()
        sub_idx = 0
        start_word = 0
        
        while start_word < len(words):
            end_word = min(start_word + self.max_words, len(words))
            
            if end_word < len(words):
                for i in range(end_word, start_word + int(self.max_words * 0.7), -1):
                    if words[i - 1].endswith((".", "!", "?")):
                        end_word = i
                        break
            
            chunk_text = " ".join(words[start_word:end_word])
            
            if self.prepend_heading and heading:
                display_text = f"{heading}{self.heading_separator}{chunk_text}"
            else:
                display_text = chunk_text
            
            chars_before = len(" ".join(words[:start_word])) + (1 if start_word > 0 else 0)
            
            chunks.append(Chunk(
                text=display_text,
                index=base_idx + sub_idx,
                heading=heading,
                start_char=para["start"] + chars_before,
                end_char=para["start"] + chars_before + len(chunk_text),
            ))
            
            sub_idx += 1
            start_word = end_word
        
        return chunks
