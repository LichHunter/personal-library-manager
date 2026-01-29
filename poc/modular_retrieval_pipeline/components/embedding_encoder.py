"""Embedding encoder component for modular retrieval pipeline.

This module provides the EmbeddingEncoder component that wraps sentence-transformers
for converting text to dense vector embeddings. It implements the Component protocol
and supports batch encoding for efficiency.

The component uses lazy loading to avoid loading the model until the first process()
call, enabling efficient initialization and testing.

Example:
    >>> from embedding_encoder import EmbeddingEncoder
    >>> encoder = EmbeddingEncoder(model='BAAI/bge-base-en-v1.5')
    >>> result = encoder.process({'text': 'hello world'})
    >>> print(result['embedding'][:3])  # First 3 dimensions
    (0.123, -0.456, 0.789)
    >>> print(len(result['embedding']))  # 768 dimensions
    768

Batch encoding:
    >>> texts = ['hello world', 'goodbye world']
    >>> embeddings = encoder.encode_batch(texts, batch_size=32)
    >>> print(len(embeddings))  # 2 embeddings
    2
    >>> print(len(embeddings[0]))  # 768 dimensions each
    768
"""

from typing import Any, Optional
import numpy as np

from ..base import Component
from ..utils.logger import get_logger


class EmbeddingEncoder(Component):
    """Component that encodes text to dense vector embeddings.

    Wraps sentence-transformers library to convert text strings to embedding vectors.
    Uses lazy loading to defer model initialization until first use.

    The component accepts either:
    - A string (text to encode)
    - A dict with 'text' field (text to encode)

    And returns a dict with:
    - 'text': Original text
    - 'embedding': Tuple of floats (immutable, 768-dimensional for BGE model)
    - 'model': Model name used for encoding
    - 'dimension': Vector dimension (768 for BGE)

    Attributes:
        model: Model name (e.g., 'BAAI/bge-base-en-v1.5')
        batch_size: Default batch size for encoding (default: 32)
        _embedder: Lazy-loaded sentence-transformers model (None until first use)
    """

    def __init__(
        self,
        model: str = "BAAI/bge-base-en-v1.5",
        batch_size: int = 32,
    ):
        """Initialize EmbeddingEncoder.

        Args:
            model: Model name from HuggingFace (default: BAAI/bge-base-en-v1.5)
            batch_size: Default batch size for encoding (default: 32)

        Note:
            The model is NOT loaded in __init__. It's loaded lazily on first
            process() call to avoid unnecessary initialization overhead.
        """
        self._log = get_logger()
        self.model = model
        self.batch_size = batch_size
        self._embedder = None  # Lazy-loaded model
        self._log.debug(
            f"[EmbeddingEncoder] Initialized with model={model}, batch_size={batch_size}"
        )

    def _load_model(self) -> None:
        """Load the sentence-transformers model (lazy loading).

        This method is called on first process() call to defer model loading
        until it's actually needed. This enables fast initialization and testing.

        Raises:
            ImportError: If sentence-transformers is not installed
            OSError: If model cannot be downloaded from HuggingFace
        """
        if self._embedder is not None:
            return  # Already loaded

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for EmbeddingEncoder. "
                "Install with: pip install sentence-transformers"
            )

        # Load model from HuggingFace
        self._embedder = SentenceTransformer(self.model)

    def process(self, data: Any) -> dict[str, Any]:
        """Encode text to embedding vector.

        Accepts either a string or a dict with 'text' field.
        Returns a dict with 'embedding' field containing the vector as a tuple.

        Args:
            data: Either a string or dict with 'text' field

        Returns:
            Dict with keys:
            - 'text': Original text
            - 'embedding': Tuple of floats (immutable)
            - 'model': Model name used
            - 'dimension': Vector dimension

        Raises:
            TypeError: If data is not a string or dict
            KeyError: If data is dict but missing 'text' field
            ValueError: If text is empty
        """
        # Extract text from input
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict):
            if "text" not in data:
                raise KeyError("Input dict must have 'text' field")
            text = data["text"]
        else:
            raise TypeError(
                f"Input must be string or dict with 'text' field, got {type(data).__name__}"
            )

        # Validate text
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")

        self._log.debug(f"[EmbeddingEncoder] Encoding text of length {len(text)}")

        # Load model on first use (lazy loading)
        self._load_model()

        # Encode text to embedding
        embedding_array = self._embedder.encode(
            [text],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )[0]

        # Convert numpy array to immutable tuple
        embedding_tuple = tuple(float(x) for x in embedding_array)

        self._log.trace(
            f"[EmbeddingEncoder] Generated embedding with dimension {len(embedding_tuple)}"
        )
        self._log.debug(f"[EmbeddingEncoder] Completed encoding")

        # Return result dict
        return {
            "text": text,
            "embedding": embedding_tuple,
            "model": self.model,
            "dimension": len(embedding_tuple),
        }

    def encode_batch(
        self,
        texts: list[str],
        batch_size: Optional[int] = None,
    ) -> list[tuple[float, ...]]:
        """Encode multiple texts to embedding vectors (batch mode).

        Efficiently encodes multiple texts in batches for better performance.

        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (default: self.batch_size)

        Returns:
            List of embedding tuples (one per text)

        Raises:
            TypeError: If texts is not a list
            ValueError: If any text is empty
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be a list, got {type(texts).__name__}")

        if not texts:
            return []

        # Validate all texts
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text:
                raise ValueError(f"Text at index {i} must be a non-empty string")

        self._log.debug(f"[EmbeddingEncoder] Batch encoding {len(texts)} texts")

        # Load model on first use (lazy loading)
        self._load_model()

        # Use provided batch_size or default
        batch_size = batch_size or self.batch_size

        # Encode all texts in batch
        embeddings_array = self._embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        # Convert numpy arrays to immutable tuples
        embeddings_tuples = [
            tuple(float(x) for x in embedding) for embedding in embeddings_array
        ]

        self._log.debug(
            f"[EmbeddingEncoder] Completed batch encoding {len(embeddings_tuples)} texts"
        )

        return embeddings_tuples
