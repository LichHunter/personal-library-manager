"""RAPTOR retrieval strategy - Recursive Abstractive Processing for Tree-Organized Retrieval."""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.mixture import GaussianMixture

from ..core.protocols import Embedder, VectorStore, LLM, RetrievalStrategy
from ..core.types import (
    Document,
    Chunk,
    IndexStats,
    SearchResponse,
    SearchHit,
    SearchStats,
)

logger = logging.getLogger(__name__)


@dataclass
class RaptorNode:
    """A node in the RAPTOR tree."""
    id: str
    text: str
    embedding: Optional[list[float]] = None
    layer: int = 0
    children: set[str] = field(default_factory=set)
    document_id: Optional[str] = None
    section_id: Optional[str] = None


class RaptorStrategy(RetrievalStrategy):
    """
    RAPTOR strategy - builds hierarchical tree with clustering and summarization.
    
    Index:
      1. Chunk documents into leaf nodes
      2. Embed all chunks
      3. Cluster similar chunks using GMM
      4. Summarize each cluster using LLM
      5. Repeat clustering/summarization for higher layers
      6. Store all nodes (leaves + summaries) in vector store
    
    Search:
      - Collapsed: Search all nodes, return top-k
      - Tree traversal: Start from roots, descend to leaves (not implemented yet)
    """
    
    COLLECTION_NAME = "raptor_nodes"
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_layers: int = 3,
        cluster_threshold: float = 0.1,
        reduction_dim: int = 10,
        summary_max_tokens: int = 150,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_layers = max_layers
        self._cluster_threshold = cluster_threshold
        self._reduction_dim = reduction_dim
        self._summary_max_tokens = summary_max_tokens
        
        self._embedder: Optional[Embedder] = None
        self._store: Optional[VectorStore] = None
        self._llm: Optional[LLM] = None
        
        self._is_indexed = False
        self._index_stats: Optional[IndexStats] = None
        self._all_nodes: dict[str, RaptorNode] = {}
    
    @property
    def name(self) -> str:
        return "raptor"
    
    @property
    def requires_llm(self) -> bool:
        return True
    
    def configure(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: Optional[LLM] = None,
    ) -> None:
        if llm is None:
            raise ValueError("RAPTOR strategy requires an LLM for summarization")
        
        self._embedder = embedder
        self._store = store
        self._llm = llm
        
        self._store.create_collection(
            self.COLLECTION_NAME,
            dimension=self._embedder.dimension,
        )
    
    def index(self, documents: list[Document]) -> IndexStats:
        if self._embedder is None or self._store is None or self._llm is None:
            raise RuntimeError("Strategy not configured. Call configure() first.")
        
        start_time = time.perf_counter()
        total_llm_calls = 0
        total_embed_calls = 0
        
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
            logger.debug(f"Document {doc.id}: {len(chunks)} chunks")
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        
        if not all_chunks:
            return self._empty_stats(documents, start_time)
        
        # Layer 0: Create leaf nodes with embeddings
        logger.info("Layer 0: Embedding leaf chunks...")
        texts = [chunk.content for chunk in all_chunks]
        embeddings = self._embedder.embed_batch(texts)
        total_embed_calls += 1
        
        current_layer_nodes: dict[str, RaptorNode] = {}
        for chunk, embedding in zip(all_chunks, embeddings):
            node = RaptorNode(
                id=chunk.id,
                text=chunk.content,
                embedding=embedding,
                layer=0,
                document_id=chunk.document_id,
                section_id=chunk.section_id,
            )
            current_layer_nodes[node.id] = node
            self._all_nodes[node.id] = node
        
        logger.info(f"Layer 0: {len(current_layer_nodes)} leaf nodes")
        
        # Build higher layers
        node_counter = len(all_chunks)
        for layer in range(1, self._max_layers + 1):
            if len(current_layer_nodes) <= self._reduction_dim + 1:
                logger.info(f"Stopping at layer {layer-1}: only {len(current_layer_nodes)} nodes")
                break
            
            logger.info(f"Layer {layer}: Clustering {len(current_layer_nodes)} nodes...")
            
            # Cluster
            node_list = list(current_layer_nodes.values())
            embeddings_array = np.array([n.embedding for n in node_list])
            
            clusters = self._cluster_embeddings(embeddings_array)
            logger.info(f"Layer {layer}: Found {len(clusters)} clusters")
            
            if len(clusters) <= 1:
                logger.info(f"Stopping at layer {layer}: only {len(clusters)} cluster(s)")
                break
            
            # Summarize each cluster
            new_layer_nodes: dict[str, RaptorNode] = {}
            for cluster_idx, cluster_indices in enumerate(clusters):
                cluster_nodes = [node_list[i] for i in cluster_indices]
                combined_text = "\n\n".join(n.text for n in cluster_nodes)
                
                # Truncate if too long
                if len(combined_text) > 4000:
                    combined_text = combined_text[:4000]
                
                # Summarize
                summary = self._summarize(combined_text)
                total_llm_calls += 1
                
                # Embed summary
                summary_embedding = self._embedder.embed(summary)
                total_embed_calls += 1
                
                # Create parent node
                node_id = f"raptor_L{layer}_{node_counter}"
                node_counter += 1
                
                parent_node = RaptorNode(
                    id=node_id,
                    text=summary,
                    embedding=summary_embedding,
                    layer=layer,
                    children={n.id for n in cluster_nodes},
                )
                
                new_layer_nodes[node_id] = parent_node
                self._all_nodes[node_id] = parent_node
            
            logger.info(f"Layer {layer}: Created {len(new_layer_nodes)} summary nodes")
            current_layer_nodes = new_layer_nodes
        
        # Store all nodes in vector store
        logger.info(f"Storing {len(self._all_nodes)} total nodes...")
        ids = []
        store_embeddings = []
        metadata = []
        
        for node in self._all_nodes.values():
            ids.append(node.id)
            store_embeddings.append(node.embedding)
            metadata.append({
                "content": node.text,
                "layer": node.layer,
                "document_id": node.document_id or "",
                "section_id": node.section_id or "",
            })
        
        self._store.insert(
            collection=self.COLLECTION_NAME,
            ids=ids,
            embeddings=store_embeddings,
            metadata=metadata,
        )
        
        duration = time.perf_counter() - start_time
        num_layers = max(n.layer for n in self._all_nodes.values()) + 1
        
        self._index_stats = IndexStats(
            strategy=self.name,
            backend=self._store.backend_name,
            embedding_model=self._embedder.model_name,
            llm_model=self._llm.model_name,
            duration_sec=duration,
            num_documents=len(documents),
            num_chunks=len(all_chunks),
            num_vectors=len(self._all_nodes),
            llm_calls=total_llm_calls,
            embed_calls=total_embed_calls,
        )
        
        self._is_indexed = True
        logger.info(
            f"RAPTOR indexed: {len(self._all_nodes)} nodes, {num_layers} layers, "
            f"{total_llm_calls} LLM calls, {duration:.2f}s"
        )
        
        return self._index_stats
    
    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        if self._embedder is None or self._store is None:
            raise RuntimeError("Strategy not configured. Call configure() first.")
        
        if not self._is_indexed:
            return SearchResponse(
                hits=[],
                stats=SearchStats(duration_ms=0.0, embed_calls=0, llm_calls=0, vectors_searched=0),
            )
        
        start_time = time.perf_counter()
        
        query_embedding = self._embedder.embed(query)
        
        # Collapsed search: search all nodes
        hits = self._store.search(
            collection=self.COLLECTION_NAME,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        vectors_searched = self._store.count(self.COLLECTION_NAME)
        
        return SearchResponse(
            hits=hits,
            stats=SearchStats(
                duration_ms=duration_ms,
                embed_calls=1,
                llm_calls=0,
                vectors_searched=vectors_searched,
            ),
        )
    
    def clear(self) -> None:
        if self._store is not None:
            self._store.delete_collection(self.COLLECTION_NAME)
            if self._embedder is not None:
                self._store.create_collection(
                    self.COLLECTION_NAME,
                    dimension=self._embedder.dimension,
                )
        
        self._all_nodes.clear()
        self._is_indexed = False
        self._index_stats = None
    
    def _chunk_document(self, doc: Document) -> list[Chunk]:
        """Chunk document into leaf nodes."""
        chunks: list[Chunk] = []
        
        for section in doc.sections:
            section_chunks = self._chunk_text(
                text=section.content,
                document_id=doc.id,
                section_id=section.id,
            )
            chunks.extend(section_chunks)
        
        if not chunks and doc.content:
            chunks = self._chunk_text(
                text=doc.content,
                document_id=doc.id,
                section_id=None,
            )
        
        return chunks
    
    def _chunk_text(
        self,
        text: str,
        document_id: str,
        section_id: Optional[str],
    ) -> list[Chunk]:
        """Split text into overlapping chunks."""
        if not text.strip():
            return []
        
        chunks: list[Chunk] = []
        text = text.strip()
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self._chunk_size
            
            if end < len(text):
                for delim in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_delim = text[start:end].rfind(delim)
                    if last_delim > self._chunk_size // 2:
                        end = start + last_delim + len(delim)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = f"{document_id}_{section_id or 'full'}_{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document_id,
                        section_id=section_id,
                        content=chunk_text,
                        level=0,
                        metadata={"chunk_idx": chunk_idx},
                    )
                )
                chunk_idx += 1
            
            start = end - self._chunk_overlap
            if start >= end:
                break
        
        return chunks
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[list[int]]:
        """Cluster embeddings using GMM with soft assignment."""
        n_samples = len(embeddings)
        
        if n_samples <= 1:
            return [[0]] if n_samples == 1 else []
        
        # Reduce dimensions if needed
        if n_samples > self._reduction_dim + 1:
            embeddings = self._reduce_dimensions(embeddings)
        
        # Find optimal k using BIC
        max_k = min(50, n_samples // 2)
        best_bic = float('inf')
        best_k = 1
        
        for k in range(1, max(2, max_k)):
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, max_iter=100)
                gmm.fit(embeddings)
                bic = gmm.bic(embeddings)
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception:
                break
        
        # Fit final model
        gmm = GaussianMixture(n_components=best_k, random_state=42)
        gmm.fit(embeddings)
        probs = gmm.predict_proba(embeddings)
        
        # Soft assignment
        clusters: dict[int, list[int]] = {}
        for i, prob_row in enumerate(probs):
            for cluster_idx in np.where(prob_row > self._cluster_threshold)[0]:
                if cluster_idx not in clusters:
                    clusters[cluster_idx] = []
                clusters[cluster_idx].append(i)
        
        return list(clusters.values())
    
    def _reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """Reduce embedding dimensions using PCA."""
        from sklearn.decomposition import PCA
        
        n_samples = len(embeddings)
        target_dim = min(self._reduction_dim, n_samples - 2)
        
        if target_dim < 2:
            return embeddings
        
        pca = PCA(n_components=target_dim)
        return pca.fit_transform(embeddings)
    
    def _summarize(self, text: str) -> str:
        """Generate summary using LLM."""
        prompt = f"""Summarize the following text in {self._summary_max_tokens} words or less.
Focus on key facts and main ideas. Be concise and factual.

Text:
{text}

Summary:"""
        
        return self._llm.generate(
            prompt=prompt,
            max_tokens=self._summary_max_tokens * 2,
            temperature=0.1,
        )
    
    def _empty_stats(self, documents: list[Document], start_time: float) -> IndexStats:
        return IndexStats(
            strategy=self.name,
            backend=self._store.backend_name if self._store else "unknown",
            embedding_model=self._embedder.model_name if self._embedder else "unknown",
            llm_model=self._llm.model_name if self._llm else None,
            duration_sec=time.perf_counter() - start_time,
            num_documents=len(documents),
            num_chunks=0,
            num_vectors=0,
            llm_calls=0,
            embed_calls=0,
        )
