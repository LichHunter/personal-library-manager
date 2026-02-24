"""SPLADE implementation of SparseRetriever interface.

SPLADE (SParse Lexical AnD Expansion) uses BERT's MLM head to produce
learned sparse term weights, combining exact match with semantic expansion.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from plm.search.components.sparse.base import SparseRetriever

if TYPE_CHECKING:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"


@dataclass
class SPLADEConfig:
    model_name: str = DEFAULT_MODEL
    device: str | None = None
    max_length: int = 512
    batch_size: int = 32


class SPLADERetriever(SparseRetriever):
    """SPLADE sparse retriever using learned term weights.

    Uses a BERT-based model to produce sparse term vectors for both
    documents and queries. Documents are encoded at index time, queries
    at search time.

    The encoder is lazy-loaded on first search to avoid loading the model
    when only using a pre-built index.
    """

    def __init__(self, config: SPLADEConfig | None = None) -> None:
        self._config = config or SPLADEConfig()
        self._sparse_matrix: sparse.csr_matrix | None = None
        self._documents: list[str] = []
        self._vocab_size: int = 0

        self._model: AutoModelForMaskedLM | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._id_to_token: dict[int, str] = {}

    def _ensure_encoder(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        device = self._config.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading SPLADE model: {self._config.model_name} on {device}")
        load_start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
        self._model = AutoModelForMaskedLM.from_pretrained(self._config.model_name)
        self._model.to(device)
        self._model.eval()

        self._vocab_size = self._tokenizer.vocab_size
        self._id_to_token = {v: k for k, v in self._tokenizer.vocab.items()}
        self._device = device

        logger.info(f"SPLADE model loaded in {time.time() - load_start:.2f}s")

    def _encode_text(self, text: str) -> dict[int, float]:
        import torch

        self._ensure_encoder()
        assert self._model is not None
        assert self._tokenizer is not None

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=self._config.max_length,
            truncation=True,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        # SPLADE aggregation: max pooling over sequence, then log(1 + ReLU(x))
        relu_log = torch.log1p(torch.relu(logits))
        sparse_vec, _ = torch.max(relu_log, dim=1)
        sparse_vec = sparse_vec.squeeze(0).cpu().numpy()

        nonzero_indices = np.nonzero(sparse_vec)[0]
        return {int(idx): float(sparse_vec[idx]) for idx in nonzero_indices}

    def _encode_batch(self, texts: list[str]) -> list[dict[int, float]]:
        import torch
        from tqdm import tqdm

        self._ensure_encoder()
        assert self._model is not None
        assert self._tokenizer is not None

        results: list[dict[int, float]] = []
        batch_size = self._config.batch_size

        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch_texts = texts[i : i + batch_size]

            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=self._config.max_length,
                truncation=True,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits

            relu_log = torch.log1p(torch.relu(logits))
            sparse_vecs, _ = torch.max(relu_log, dim=1)
            sparse_vecs = sparse_vecs.cpu().numpy()

            for vec in sparse_vecs:
                nonzero_indices = np.nonzero(vec)[0]
                results.append(
                    {int(idx): float(vec[idx]) for idx in nonzero_indices}
                )

        return results

    def index(self, documents: list[str]) -> None:
        self._documents = list(documents)
        logger.info(f"Indexing {len(documents)} documents with SPLADE...")

        encode_start = time.time()
        sparse_vecs = self._encode_batch(documents)
        encode_time = time.time() - encode_start

        self._ensure_encoder()
        self._vocab_size = self._tokenizer.vocab_size  # type: ignore[union-attr]

        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for doc_idx, sparse_vec in enumerate(sparse_vecs):
            for term_id, weight in sparse_vec.items():
                rows.append(doc_idx)
                cols.append(term_id)
                data.append(weight)

        self._sparse_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(documents), self._vocab_size),
            dtype=np.float32,
        )

        nnz = self._sparse_matrix.nnz
        size_mb = (
            self._sparse_matrix.data.nbytes
            + self._sparse_matrix.indices.nbytes
            + self._sparse_matrix.indptr.nbytes
        ) / (1024 * 1024)

        logger.info(
            f"SPLADE index built: {len(documents)} docs, {nnz:,} terms, "
            f"{size_mb:.2f}MB, {encode_time:.2f}s"
        )

    def search(self, query: str, k: int) -> list[dict]:
        if self._sparse_matrix is None:
            raise RuntimeError("Index has not been built. Call index() or load() first.")

        query_vec = self._encode_text(query)

        query_sparse = sparse.csr_matrix(
            ([query_vec[term_id] for term_id in query_vec.keys()],
             ([0] * len(query_vec), list(query_vec.keys()))),
            shape=(1, self._vocab_size),
            dtype=np.float32,
        )

        scores = self._sparse_matrix.dot(query_sparse.T).toarray().flatten()
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[dict] = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "content": self._documents[idx][:500] if idx < len(self._documents) else "",
            })

        return results

    def save(self, path: str | Path) -> None:
        if self._sparse_matrix is None:
            raise RuntimeError("Index has not been built. Call index() first.")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        sparse.save_npz(save_path / "matrix.npz", self._sparse_matrix)

        metadata = {
            "vocab_size": self._vocab_size,
            "config": asdict(self._config),
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        with open(save_path / "documents.pkl", "wb") as f:
            pickle.dump(self._documents, f)

        logger.info(f"SPLADE index saved to {save_path}")

    @classmethod
    def load(cls, path: str | Path) -> "SPLADERetriever":
        load_path = Path(path)

        with open(load_path / "metadata.json") as f:
            metadata = json.load(f)

        if "config" in metadata:
            config = SPLADEConfig(**metadata["config"])
        else:
            config = SPLADEConfig()

        instance = cls(config=config)
        instance._sparse_matrix = sparse.load_npz(load_path / "matrix.npz")
        instance._vocab_size = metadata["vocab_size"]

        documents_path = load_path / "documents.pkl"
        contents_path = load_path / "contents.pkl"

        if documents_path.exists():
            with open(documents_path, "rb") as f:
                instance._documents = pickle.load(f)
        elif contents_path.exists():
            with open(contents_path, "rb") as f:
                instance._documents = pickle.load(f)
        elif "doc_ids" in metadata:
            instance._documents = metadata["doc_ids"]
        else:
            instance._documents = []

        logger.info(
            f"SPLADE index loaded: {len(instance._documents)} docs, "
            f"{instance._sparse_matrix.nnz:,} terms"
        )

        return instance

    @property
    def is_ready(self) -> bool:
        return self._sparse_matrix is not None

    @property
    def document_count(self) -> int:
        return len(self._documents)
