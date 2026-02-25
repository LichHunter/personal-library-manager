"""Reverse HyDE - Generate hypothetical questions at index time.

Instead of transforming queries at search time, this strategy generates
hypothetical questions that each chunk could answer during indexing.
These questions are embedded alongside the content, improving matching
with natural language queries.
"""

import subprocess
from typing import Optional

import numpy as np

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, RerankerMixin, StructuredDocument


def call_ollama(prompt: str, model: str = "llama3.2:3b", timeout: int = 90) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


class ReverseHyDERetrieval(RetrievalStrategy, EmbedderMixin, RerankerMixin):
    """Reverse HyDE: Generate questions at index time for better query matching."""

    QUESTION_PROMPT = """Read this documentation text and generate 3 questions that it answers. Keep questions concise and natural, like how a user would ask.

Text:
{content}

Questions (one per line):"""

    def __init__(
        self,
        name: str = "reverse_hyde",
        llm_model: str = "llama3.2:3b",
        questions_per_chunk: int = 3,
        use_reranker: bool = False,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.llm_model = llm_model
        self.questions_per_chunk = questions_per_chunk
        self.use_reranker = use_reranker
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.question_embeddings: Optional[np.ndarray] = None
        self.question_to_chunk: dict[int, int] = {}

    def generate_questions(self, content: str) -> list[str]:
        truncated = content[:1500]
        prompt = self.QUESTION_PROMPT.format(content=truncated)
        response = call_ollama(prompt, self.llm_model)

        questions = []
        for line in response.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line[0].isdigit():
                parts = line.split(". ", 1)
                if len(parts) > 1:
                    questions.append(parts[1])
            elif "?" in line:
                questions.append(line)

        return questions[: self.questions_per_chunk]

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks
        self.embeddings = self.encode_texts([c.content for c in chunks])

        all_questions = []
        self.question_to_chunk = {}

        print(f"Generating questions for {len(chunks)} chunks...")
        for chunk_idx, chunk in enumerate(chunks):
            questions = self.generate_questions(chunk.content)
            for q in questions:
                self.question_to_chunk[len(all_questions)] = chunk_idx
                all_questions.append(q)
            if (chunk_idx + 1) % 10 == 0:
                print(
                    f"  Processed {chunk_idx + 1}/{len(chunks)} chunks ({len(all_questions)} questions)"
                )

        if all_questions:
            self.question_embeddings = self.encode_texts(all_questions)
            print(f"Generated {len(all_questions)} questions total")
        else:
            self.question_embeddings = None
            print("Warning: No questions generated")

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if self.chunks is None or self.embeddings is None:
            return []

        q_emb = self.encode_query(query)

        content_sims = np.dot(self.embeddings, q_emb)

        chunk_scores = content_sims.copy()

        if self.question_embeddings is not None and len(self.question_embeddings) > 0:
            question_sims = np.dot(self.question_embeddings, q_emb)
            for q_idx, sim in enumerate(question_sims):
                chunk_idx = self.question_to_chunk[q_idx]
                chunk_scores[chunk_idx] = max(chunk_scores[chunk_idx], sim)

        if self.use_reranker and self.reranker is not None:
            top_idx = np.argsort(chunk_scores)[::-1][: k * 4]
            candidates = [self.chunks[i] for i in top_idx]
            return self.rerank(query, candidates, k)
        else:
            top_idx = np.argsort(chunk_scores)[::-1][:k]
            return [self.chunks[i] for i in top_idx]

    def get_index_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "num_questions": len(self.question_to_chunk),
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "llm_model": self.llm_model,
            "questions_per_chunk": self.questions_per_chunk,
        }
