"""Level-of-Detail (LOD) retrieval with LLM-guided routing."""

import subprocess
from typing import Callable, Optional

import numpy as np

from strategies import Chunk, Document, FixedSizeStrategy
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument


def create_ollama_router(model: str = "llama3.2:3b") -> Callable[[str, list[str]], list[int]]:
    """Create an LLM-based router using Ollama.
    
    The router takes a query and list of options, returns indices of relevant options.
    """
    
    def route(query: str, options: list[str], max_select: int = 3) -> list[int]:
        if not options:
            return []
        
        # Build prompt
        options_text = "\n".join(f"{i}. {opt[:200]}" for i, opt in enumerate(options))
        prompt = f"""Given the query, select the {max_select} most relevant options by number.
Return ONLY comma-separated numbers (e.g., "0,2,5"). No explanation.

Query: {query}

Options:
{options_text}

Most relevant options (numbers only):"""

        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                # Parse response
                response = result.stdout.strip()
                indices = []
                for part in response.replace(" ", "").split(","):
                    try:
                        idx = int(part.strip())
                        if 0 <= idx < len(options):
                            indices.append(idx)
                    except ValueError:
                        continue
                return indices[:max_select] if indices else list(range(min(max_select, len(options))))
        except Exception:
            pass
        
        # Fallback: return first max_select options
        return list(range(min(max_select, len(options))))
    
    return route


class LODLLMRetrieval(RetrievalStrategy, EmbedderMixin):
    """Level-of-Detail retrieval with LLM-guided routing at each level.
    
    Similar to LODRetrieval but uses an LLM to select documents and sections
    instead of pure embedding similarity. This can be more accurate for
    complex queries but is slower due to LLM calls.
    
    Hierarchy:
    - Level 2: LLM selects relevant documents
    - Level 1: LLM selects relevant sections within selected docs
    - Level 0: Embedding search for chunks within selected sections
    
    Args:
        llm_model: Ollama model name for routing (default "llama3.2:3b").
        chunk_size: Size of chunks at level 0.
        doc_top_k: Number of documents to select via LLM.
        section_top_k: Number of sections to select via LLM.
    """

    def __init__(
        self,
        name: str = "lod_llm",
        llm_model: str = "llama3.2:3b",
        chunk_size: int = 512,
        doc_top_k: int = 3,
        section_top_k: int = 5,
        summary_max_chars: int = 300,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.doc_top_k = doc_top_k
        self.section_top_k = section_top_k
        self.summary_max_chars = summary_max_chars
        
        self.router: Optional[Callable] = None
        
        # Level 2: Documents (id, title, summary)
        self.doc_entries: list[tuple[str, str, str]] = []
        
        # Level 1: Sections (doc_id, section_id, heading, summary)
        self.section_entries: list[tuple[str, str, str, str]] = []
        
        # Level 0: Chunks with embeddings
        self.chunk_entries: list[tuple[str, str, Chunk, np.ndarray]] = []

    def set_llm_model(self, model: str):
        """Set the LLM model for routing."""
        self.llm_model = model
        self.router = create_ollama_router(model)

    def index(
        self,
        chunks: Optional[list[Chunk]] = None,
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Build 3-level index from structured documents."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")
        
        if structured_docs is None:
            raise ValueError("LOD_LLM requires structured_docs.")

        # Initialize router
        self.router = create_ollama_router(self.llm_model)

        # Level 2: Document metadata (no embeddings needed - LLM routes)
        for doc in structured_docs:
            summary = doc.summary[:self.summary_max_chars]
            self.doc_entries.append((doc.id, doc.title, summary))

        # Level 1: Section metadata
        for doc in structured_docs:
            for section in doc.sections:
                summary = section.content[:self.summary_max_chars]
                self.section_entries.append((doc.id, section.id, section.heading, summary))

        # Level 0: Chunks with embeddings (embedding search at leaf level)
        chunker = FixedSizeStrategy(chunk_size=self.chunk_size, overlap=0)
        chunk_texts = []

        for doc in structured_docs:
            for section in doc.sections:
                temp_doc = Document(
                    id=f"{doc.id}_{section.id}",
                    title=section.heading,
                    content=section.content,
                )
                section_chunks = chunker.chunk(temp_doc)
                
                for chunk in section_chunks:
                    chunk.doc_id = doc.id
                    self.chunk_entries.append((doc.id, section.id, chunk, np.array([])))
                    chunk_texts.append(chunk.content)

        if chunk_texts:
            chunk_embeddings = self.encode_texts(chunk_texts)
            self.chunk_entries = [
                (entry[0], entry[1], entry[2], emb)
                for entry, emb in zip(self.chunk_entries, chunk_embeddings)
            ]

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve using LLM-guided hierarchical search."""
        if not self.chunk_entries or self.router is None:
            return []

        # Level 2: LLM selects documents
        doc_options = [f"{title}: {summary}" for _, title, summary in self.doc_entries]
        selected_doc_indices = self.router(query, doc_options, self.doc_top_k)
        selected_doc_ids = {self.doc_entries[i][0] for i in selected_doc_indices}

        if not selected_doc_ids:
            return []

        # Level 1: LLM selects sections within selected docs
        filtered_sections = [
            (i, entry) for i, entry in enumerate(self.section_entries)
            if entry[0] in selected_doc_ids
        ]
        
        if not filtered_sections:
            return []

        section_options = [f"{entry[2]}: {entry[3]}" for _, entry in filtered_sections]
        selected_section_indices = self.router(query, section_options, self.section_top_k)
        selected_section_ids = {
            (filtered_sections[i][1][0], filtered_sections[i][1][1])  # (doc_id, section_id)
            for i in selected_section_indices
        }

        if not selected_section_ids:
            return []

        # Level 0: Embedding search within selected sections
        filtered_chunks = [
            (i, entry) for i, entry in enumerate(self.chunk_entries)
            if (entry[0], entry[1]) in selected_section_ids
        ]

        if not filtered_chunks:
            return []

        q_emb = self.encode_query(query)
        chunk_sims = [(i, np.dot(entry[3], q_emb)) for i, entry in filtered_chunks]
        chunk_sims.sort(key=lambda x: x[1], reverse=True)
        top_chunk_indices = [i for i, _ in chunk_sims[:k]]

        return [self.chunk_entries[i][2] for i in top_chunk_indices]

    def get_index_stats(self) -> dict:
        return {
            "num_documents": len(self.doc_entries),
            "num_sections": len(self.section_entries),
            "num_chunks": len(self.chunk_entries),
            "llm_model": self.llm_model,
            "doc_top_k": self.doc_top_k,
            "section_top_k": self.section_top_k,
        }
