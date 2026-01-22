"""
Local model implementations for RAPTOR testing.
Uses Ollama for LLM tasks and SentenceTransformers for embeddings.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import time

import ollama
from sentence_transformers import SentenceTransformer

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


# ============================================================================
# Base Classes (matching RAPTOR's interface)
# ============================================================================

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text: str) -> List[float]:
        pass


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        pass


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context: str, question: str) -> str:
        pass


# ============================================================================
# Local Implementations
# ============================================================================

class LocalEmbeddingModel(BaseEmbeddingModel):
    """Uses SentenceTransformers for local embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize local embedding model.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "sentence-transformers/all-MiniLM-L6-v2" (fast, 384 dims)
                - "sentence-transformers/all-mpnet-base-v2" (better quality, 768 dims)
                - "sentence-transformers/multi-qa-mpnet-base-cos-v1" (optimized for QA)
        """
        logging.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logging.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
    
    def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for multiple texts (more efficient)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings.tolist()


class OllamaSummarizationModel(BaseSummarizationModel):
    """Uses Ollama for local summarization."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize Ollama summarization model.
        
        Args:
            model: Ollama model name. Options:
                - "llama3.2:3b" (fast, good quality)
                - "llama3.2:1b" (faster, lower quality)
                - "mistral:7b" (better quality, slower)
                - "phi3:mini" (fast, decent quality)
        """
        self.model = model
        logging.info(f"Using Ollama model for summarization: {model}")
    
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        """Summarize the given context."""
        prompt = f"""Write a concise summary of the following text, including key details:

{context}

Summary:"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": max_tokens,
                    "temperature": 0.3,  # Lower temperature for more focused summaries
                }
            )
            return response['response'].strip()
        except Exception as e:
            logging.error(f"Ollama summarization failed: {e}")
            return f"[Summarization failed: {e}]"


class OllamaQAModel(BaseQAModel):
    """Uses Ollama for local question answering."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Initialize Ollama QA model.
        
        Args:
            model: Ollama model name.
        """
        self.model = model
        logging.info(f"Using Ollama model for QA: {model}")
    
    def answer_question(self, context: str, question: str) -> str:
        """Answer a question based on the given context."""
        prompt = f"""Based on the following context, answer the question. Only use information from the context. If the answer is not in the context, say "I cannot find this information in the provided context."

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "num_predict": 500,
                    "temperature": 0.1,  # Very low temperature for factual answers
                }
            )
            return response['response'].strip()
        except Exception as e:
            logging.error(f"Ollama QA failed: {e}")
            return f"[QA failed: {e}]"


# ============================================================================
# Utilities
# ============================================================================

def check_ollama_available(model: str = "llama3.2:3b") -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        response = ollama.list()
        available_models = [m.model for m in response.models]
        
        if model in available_models or any(model in m for m in available_models):
            logging.info(f"Ollama model '{model}' is available")
            return True
        else:
            logging.warning(f"Model '{model}' not found. Available: {available_models}")
            logging.info(f"Run 'ollama pull {model}' to download it")
            return False
    except Exception as e:
        logging.error(f"Ollama not available: {e}")
        logging.info("Make sure Ollama is running: 'ollama serve'")
        return False


def benchmark_embedding_model(model: LocalEmbeddingModel, texts: List[str]) -> dict:
    """Benchmark embedding model performance."""
    start = time.perf_counter()
    embeddings = model.create_embeddings_batch(texts)
    elapsed = time.perf_counter() - start
    
    return {
        "num_texts": len(texts),
        "total_time_seconds": elapsed,
        "texts_per_second": len(texts) / elapsed,
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
    }


if __name__ == "__main__":
    # Quick test
    print("Testing local models...")
    
    # Test embeddings
    print("\n1. Testing embedding model...")
    embed_model = LocalEmbeddingModel()
    test_text = "Machine learning is a subset of artificial intelligence."
    embedding = embed_model.create_embedding(test_text)
    print(f"   Embedding dimension: {len(embedding)}")
    print(f"   First 5 values: {embedding[:5]}")
    
    # Test Ollama
    print("\n2. Testing Ollama availability...")
    if check_ollama_available():
        print("\n3. Testing summarization...")
        summarizer = OllamaSummarizationModel()
        summary = summarizer.summarize(
            "Machine learning is a method of data analysis that automates analytical "
            "model building. It is a branch of artificial intelligence based on the idea "
            "that systems can learn from data, identify patterns and make decisions with "
            "minimal human intervention."
        )
        print(f"   Summary: {summary}")
        
        print("\n4. Testing QA...")
        qa = OllamaQAModel()
        answer = qa.answer_question(
            context="Python was created by Guido van Rossum and first released in 1991.",
            question="Who created Python?"
        )
        print(f"   Answer: {answer}")
    else:
        print("   Skipping LLM tests - Ollama not available")
