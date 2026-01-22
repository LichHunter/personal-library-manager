"""ClusterSemanticChunker - Embedding-based boundary detection.

Based on Chroma Research's ClusterSemanticChunker:
https://github.com/brandonstarxel/chunking_evaluation

Uses embedding similarity to find natural semantic boundaries.
More expensive (requires embeddings for each sentence) but may
produce more semantically coherent chunks.
"""

import re
from typing import Optional
import numpy as np

from .base import ChunkingStrategy, Chunk, Document


class ClusterSemanticStrategy(ChunkingStrategy):
    """Semantic chunking using embedding similarity.
    
    Algorithm:
    1. Split text into sentences
    2. Embed each sentence
    3. Find boundaries where adjacent sentences are least similar
    4. Group sentences into chunks around these boundaries
    """
    
    def __init__(
        self,
        target_chunk_size: int = 200,  # Target tokens per chunk
        embedder=None,  # SentenceTransformer model
        similarity_threshold: float = 0.5,  # Min similarity to merge
    ):
        """
        Args:
            target_chunk_size: Target chunk size in tokens
            embedder: SentenceTransformer model for embeddings
            similarity_threshold: Similarity threshold for merging
        """
        self.target_chunk_size = target_chunk_size
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.target_words = int(target_chunk_size * 0.75)
    
    @property
    def name(self) -> str:
        return f"semantic_{self.target_chunk_size}"
    
    def set_embedder(self, embedder):
        """Set the embedding model (called before chunking)."""
        self.embedder = embedder
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document using semantic similarity."""
        if self.embedder is None:
            raise ValueError("Embedder must be set before chunking. Call set_embedder() first.")
        
        content = document.content
        
        # Split into sentences with positions
        sentences = self._split_sentences(content)
        
        if not sentences:
            return []
        
        if len(sentences) == 1:
            return [Chunk(
                id=f"{document.id}_sem_0",
                doc_id=document.id,
                content=sentences[0]["text"],
                start_char=sentences[0]["start"],
                end_char=sentences[0]["end"],
                level=0,
                metadata={"strategy": self.name, "chunk_idx": 0}
            )]
        
        # Get embeddings for all sentences
        texts = [s["text"] for s in sentences]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        
        # Calculate similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Find chunk boundaries using greedy approach
        boundaries = self._find_boundaries(sentences, similarities)
        
        # Create chunks from boundaries
        chunks = self._create_chunks(document.id, sentences, boundaries)
        
        return chunks
    
    def _split_sentences(self, content: str) -> list[dict]:
        """Split text into sentences with character positions."""
        # Simple sentence splitting on .!? followed by space or newline
        pattern = r'(?<=[.!?])\s+'
        
        sentences = []
        current_pos = 0
        
        parts = re.split(pattern, content)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Find position in original content
            start = content.find(part, current_pos)
            if start < 0:
                start = current_pos
            
            sentences.append({
                "text": part,
                "start": start,
                "end": start + len(part),
                "word_count": len(part.split()),
            })
            current_pos = start + len(part)
        
        return sentences
    
    def _find_boundaries(
        self,
        sentences: list[dict],
        similarities: list[float],
    ) -> list[int]:
        """Find chunk boundaries using greedy approach.
        
        Greedily group sentences until target size is reached,
        then find the best split point (lowest similarity).
        
        Returns list of boundary indices (exclusive end indices).
        """
        boundaries = []
        current_start = 0
        current_words = 0
        
        for i, sent in enumerate(sentences):
            current_words += sent["word_count"]
            
            # If we've reached target size, find best split point
            if current_words >= self.target_words:
                if i == current_start:
                    # Single sentence exceeds target, just include it
                    boundaries.append(i + 1)
                    current_start = i + 1
                    current_words = 0
                else:
                    # Find lowest similarity point in current range
                    range_sims = similarities[current_start:i]
                    if range_sims:
                        min_idx = current_start + np.argmin(range_sims) + 1
                        boundaries.append(min_idx)
                        current_start = min_idx
                        # Recalculate words from new start
                        current_words = sum(
                            s["word_count"] for s in sentences[min_idx:i+1]
                        )
                    else:
                        boundaries.append(i + 1)
                        current_start = i + 1
                        current_words = 0
        
        # Add final boundary if needed
        if current_start < len(sentences):
            boundaries.append(len(sentences))
        
        return boundaries
    
    def _create_chunks(
        self,
        doc_id: str,
        sentences: list[dict],
        boundaries: list[int],
    ) -> list[Chunk]:
        """Create chunks from sentence boundaries."""
        chunks = []
        prev_boundary = 0
        
        for chunk_idx, boundary in enumerate(boundaries):
            chunk_sentences = sentences[prev_boundary:boundary]
            
            if not chunk_sentences:
                prev_boundary = boundary
                continue
            
            chunk_text = " ".join(s["text"] for s in chunk_sentences)
            start_char = chunk_sentences[0]["start"]
            end_char = chunk_sentences[-1]["end"]
            
            chunks.append(Chunk(
                id=f"{doc_id}_sem_{chunk_idx}",
                doc_id=doc_id,
                content=chunk_text,
                start_char=start_char,
                end_char=end_char,
                level=0,
                metadata={
                    "strategy": self.name,
                    "chunk_idx": chunk_idx,
                    "sentence_count": len(chunk_sentences),
                }
            ))
            
            prev_boundary = boundary
        
        return chunks
