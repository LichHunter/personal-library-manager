"""RAPTOR retrieval - Recursive Abstractive Processing for Tree-Organized Retrieval."""

import subprocess
from typing import Callable, Optional

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument


def create_ollama_summarizer(model: str = "llama3.2:3b") -> Callable[[str], str]:
    """Create a summarization function using Ollama."""
    
    def summarize(text: str) -> str:
        prompt = f"""Summarize the following text in 100 words or less.
Focus on key facts and main ideas. Be concise and factual.

Text:
{text[:2000]}

Summary:"""

        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        # Fallback: extractive summary (first 150 words)
        words = text.split()[:150]
        return " ".join(words)
    
    return summarize


class RAPTORRetrieval(RetrievalStrategy, EmbedderMixin):
    """RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.
    
    Builds a tree structure where:
    - Leaf nodes are chunks
    - Higher levels are LLM-generated summaries of clusters
    - Search can use "collapsed" (all nodes) or "tree traversal" mode
    
    Args:
        llm_model: Ollama model for summarization.
        max_layers: Maximum tree depth beyond leaves.
        cluster_threshold: GMM probability threshold for soft clustering.
        search_mode: "collapsed" (search all nodes) or "tree" (traverse).
    """

    def __init__(
        self,
        name: str = "raptor",
        llm_model: str = "llama3.2:3b",
        max_layers: int = 2,
        cluster_threshold: float = 0.1,
        search_mode: str = "collapsed",  # "collapsed" or "tree"
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.llm_model = llm_model
        self.max_layers = max_layers
        self.cluster_threshold = cluster_threshold
        self.search_mode = search_mode
        
        self.summarizer: Optional[Callable] = None
        
        # All nodes in the tree
        self.all_nodes: list[dict] = []
        self.all_embeddings: Optional[np.ndarray] = None
        
        # Tree structure for traversal mode
        self.layer_nodes: dict[int, list[int]] = {}  # layer -> node indices

    def set_llm_model(self, model: str):
        """Set the LLM model for summarization."""
        self.llm_model = model
        self.summarizer = create_ollama_summarizer(model)

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Build RAPTOR tree from chunks."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        if not chunks:
            return

        # Initialize summarizer
        self.summarizer = create_ollama_summarizer(self.llm_model)

        # Layer 0: Leaf nodes (chunks)
        texts = [c.content for c in chunks]
        embeddings = self.encode_texts(texts)

        self.layer_nodes[0] = []
        current_layer = []
        
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            node_idx = len(self.all_nodes)
            self.all_nodes.append({
                "text": chunk.content,
                "embedding": emb,
                "layer": 0,
                "doc_id": chunk.doc_id,
                "chunk": chunk,
                "children": [],
            })
            self.layer_nodes[0].append(node_idx)
            current_layer.append((chunk.content, emb, node_idx))

        # Build higher layers
        for layer in range(1, self.max_layers + 1):
            if len(current_layer) <= 3:
                break

            # Cluster current layer
            layer_embeddings = np.array([emb for _, emb, _ in current_layer])
            clusters = self._cluster_embeddings(layer_embeddings)

            if len(clusters) <= 1:
                break

            self.layer_nodes[layer] = []
            new_layer = []

            for cluster_indices in clusters:
                # Get texts from cluster
                cluster_texts = [current_layer[i][0] for i in cluster_indices]
                cluster_node_indices = [current_layer[i][2] for i in cluster_indices]
                combined = "\n\n".join(cluster_texts)[:4000]

                # Generate summary
                summary = self.summarizer(combined)

                # Embed summary
                summary_emb = self.embedder.encode([summary], normalize_embeddings=True)[0]

                # Create summary node
                node_idx = len(self.all_nodes)
                self.all_nodes.append({
                    "text": summary,
                    "embedding": summary_emb,
                    "layer": layer,
                    "doc_id": None,
                    "chunk": None,
                    "children": cluster_node_indices,
                })
                self.layer_nodes[layer].append(node_idx)
                new_layer.append((summary, summary_emb, node_idx))

            current_layer = new_layer

        # Build combined embedding matrix
        self.all_embeddings = np.array([n["embedding"] for n in self.all_nodes])

    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[list[int]]:
        """Cluster embeddings using GMM with soft assignment."""
        n_samples = len(embeddings)
        if n_samples <= 1:
            return [[0]] if n_samples == 1 else []

        # Reduce dimensions if needed
        work_embeddings = embeddings
        if n_samples > 12:
            target_dim = min(10, n_samples - 2)
            if target_dim >= 2:
                pca = PCA(n_components=target_dim)
                work_embeddings = pca.fit_transform(embeddings)

        # Find optimal k using BIC
        max_k = min(50, n_samples // 2)
        best_bic = float('inf')
        best_k = 1

        for k in range(1, max(2, max_k)):
            try:
                gmm = GaussianMixture(n_components=k, random_state=42, max_iter=100)
                gmm.fit(work_embeddings)
                bic = gmm.bic(work_embeddings)
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception:
                break

        # Fit final model
        gmm = GaussianMixture(n_components=best_k, random_state=42)
        gmm.fit(work_embeddings)
        probs = gmm.predict_proba(work_embeddings)

        # Soft assignment
        clusters: dict[int, list[int]] = {}
        for i, prob_row in enumerate(probs):
            for cluster_idx in np.where(prob_row > self.cluster_threshold)[0]:
                if cluster_idx not in clusters:
                    clusters[cluster_idx] = []
                clusters[cluster_idx].append(i)

        return list(clusters.values())

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve chunks using collapsed or tree search."""
        if self.all_embeddings is None or len(self.all_nodes) == 0:
            return []

        if self.search_mode == "tree":
            return self._retrieve_tree(query, k)
        else:
            return self._retrieve_collapsed(query, k)

    def _retrieve_collapsed(self, query: str, k: int) -> list[Chunk]:
        """Collapsed search: search all nodes, return leaf chunks."""
        q_emb = self.encode_query(query)
        sims = np.dot(self.all_embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1]

        # Return top-k leaf nodes (those with actual chunks)
        results = []
        for idx in top_indices:
            node = self.all_nodes[idx]
            if node["chunk"] is not None:
                results.append(node["chunk"])
                if len(results) >= k:
                    break

        # If not enough leaf nodes found, create pseudo-chunks from summaries
        if len(results) < k:
            for idx in top_indices:
                node = self.all_nodes[idx]
                if node["chunk"] is None and len(results) < k:
                    pseudo_chunk = Chunk(
                        id=f"raptor_summary_{idx}",
                        doc_id=node["doc_id"] or "summary",
                        content=node["text"],
                        start_char=0,
                        end_char=len(node["text"]),
                    )
                    results.append(pseudo_chunk)

        return results[:k]

    def _retrieve_tree(self, query: str, k: int) -> list[Chunk]:
        """Tree traversal search: start from top, descend to leaves."""
        q_emb = self.encode_query(query)

        # Start from highest layer
        max_layer = max(self.layer_nodes.keys())
        
        # Get top nodes at highest layer
        current_candidates = self.layer_nodes.get(max_layer, self.layer_nodes[0])
        
        # Descend through layers
        for layer in range(max_layer, 0, -1):
            if not current_candidates:
                break
                
            # Score current candidates
            candidate_sims = [
                (idx, np.dot(self.all_nodes[idx]["embedding"], q_emb))
                for idx in current_candidates
            ]
            candidate_sims.sort(key=lambda x: x[1], reverse=True)
            
            # Select top candidates and get their children
            top_k_candidates = [idx for idx, _ in candidate_sims[:k * 2]]
            next_candidates = []
            for idx in top_k_candidates:
                next_candidates.extend(self.all_nodes[idx].get("children", []))
            
            current_candidates = next_candidates if next_candidates else current_candidates

        # Final scoring at leaf level
        if current_candidates:
            leaf_sims = [
                (idx, np.dot(self.all_nodes[idx]["embedding"], q_emb))
                for idx in current_candidates
                if self.all_nodes[idx]["chunk"] is not None
            ]
            leaf_sims.sort(key=lambda x: x[1], reverse=True)
            return [self.all_nodes[idx]["chunk"] for idx, _ in leaf_sims[:k]]

        return []

    def get_index_stats(self) -> dict:
        layer_counts = {layer: len(nodes) for layer, nodes in self.layer_nodes.items()}
        return {
            "total_nodes": len(self.all_nodes),
            "layer_counts": layer_counts,
            "llm_model": self.llm_model,
            "max_layers": self.max_layers,
            "search_mode": self.search_mode,
        }
