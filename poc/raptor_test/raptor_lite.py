"""
RAPTOR-Lite: Simplified RAPTOR implementation for testing.

This is a minimal implementation to benchmark:
1. Indexing time (chunking, embedding, clustering, summarization)
2. Retrieval accuracy
3. Local model performance
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import re
import copy

import numpy as np
from sklearn.mixture import GaussianMixture

# Conditional imports for UMAP (can be slow to import)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available, using PCA fallback")

import tiktoken

from local_models import (
    LocalEmbeddingModel, 
    OllamaSummarizationModel, 
    OllamaQAModel,
    BaseEmbeddingModel,
    BaseSummarizationModel,
    BaseQAModel,
)

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Node:
    """A node in the RAPTOR tree."""
    text: str
    index: int
    children: Set[int] = field(default_factory=set)
    embedding: Optional[List[float]] = None
    layer: int = 0
    
    def __hash__(self):
        return hash(self.index)


@dataclass
class Tree:
    """The RAPTOR tree structure."""
    all_nodes: Dict[int, Node]
    root_nodes: Dict[int, Node]
    leaf_nodes: Dict[int, Node]
    num_layers: int
    layer_to_nodes: Dict[int, List[Node]]


@dataclass
class TimingStats:
    """Timing statistics for benchmarking."""
    chunking_seconds: float = 0.0
    embedding_seconds: float = 0.0
    clustering_seconds: float = 0.0
    summarization_seconds: float = 0.0
    total_seconds: float = 0.0
    
    num_chunks: int = 0
    num_nodes: int = 0
    num_layers: int = 0
    num_summaries: int = 0


# ============================================================================
# Text Splitting
# ============================================================================

def split_text(
    text: str, 
    max_tokens: int = 100, 
    overlap: int = 0,
    tokenizer = None
) -> List[str]:
    """
    Split text into chunks respecting sentence boundaries.
    
    Args:
        text: Input text to split
        max_tokens: Maximum tokens per chunk
        overlap: Number of sentences to overlap between chunks
        tokenizer: Tiktoken tokenizer (defaults to cl100k_base)
    
    Returns:
        List of text chunks
    """
    if tokenizer is None:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Split by sentence delimiters
    delimiters = r'[.!?\n]+'
    sentences = re.split(delimiters, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_tokens = len(tokenizer.encode(" " + sentence))
        
        if sentence_tokens > max_tokens:
            # Sentence too long - split by clauses
            clauses = re.split(r'[,;:]', sentence)
            for clause in clauses:
                clause = clause.strip()
                if not clause:
                    continue
                clause_tokens = len(tokenizer.encode(" " + clause))
                if current_length + clause_tokens > max_tokens and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = current_chunk[-overlap:] if overlap else []
                    current_length = sum(len(tokenizer.encode(" " + s)) for s in current_chunk)
                current_chunk.append(clause)
                current_length += clause_tokens
        elif current_length + sentence_tokens > max_tokens and current_chunk:
            # Start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap else []
            current_length = sum(len(tokenizer.encode(" " + s)) for s in current_chunk)
            current_chunk.append(sentence)
            current_length += sentence_tokens
        else:
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # Don't forget last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


# ============================================================================
# Clustering
# ============================================================================

def reduce_dimensions(embeddings: np.ndarray, dim: int = 10) -> np.ndarray:
    """Reduce embedding dimensions using UMAP or PCA."""
    n_samples = len(embeddings)
    
    if n_samples <= dim + 1:
        return embeddings
    
    target_dim = min(dim, n_samples - 2)
    
    if UMAP_AVAILABLE:
        n_neighbors = int(np.sqrt(n_samples - 1))
        n_neighbors = max(2, min(n_neighbors, n_samples - 1))
        
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=target_dim,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    else:
        # PCA fallback
        from sklearn.decomposition import PCA
        pca = PCA(n_components=target_dim)
        return pca.fit_transform(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray, 
    threshold: float = 0.1,
    max_clusters: int = 50
) -> List[List[int]]:
    """
    Cluster embeddings using GMM with soft assignment.
    
    Returns list of clusters, where each cluster is a list of indices.
    A node can appear in multiple clusters (soft clustering).
    """
    n_samples = len(embeddings)
    
    if n_samples <= 1:
        return [[0]] if n_samples == 1 else []
    
    # Find optimal number of clusters using BIC
    max_k = min(max_clusters, n_samples)
    best_bic = float('inf')
    best_k = 1
    
    for k in range(1, max_k):
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
    
    # Soft assignment: node belongs to cluster if prob > threshold
    clusters = {}
    for i, prob_row in enumerate(probs):
        for cluster_idx in np.where(prob_row > threshold)[0]:
            if cluster_idx not in clusters:
                clusters[cluster_idx] = []
            clusters[cluster_idx].append(i)
    
    return list(clusters.values())


# ============================================================================
# RAPTOR Tree Builder
# ============================================================================

class RaptorLite:
    """Simplified RAPTOR implementation for testing."""
    
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        summarization_model: BaseSummarizationModel,
        qa_model: BaseQAModel,
        max_tokens_per_chunk: int = 100,
        max_layers: int = 5,
        clustering_threshold: float = 0.1,
        reduction_dim: int = 10,
        max_cluster_tokens: int = 3500,
        summarization_tokens: int = 100,
    ):
        self.embedding_model = embedding_model
        self.summarization_model = summarization_model
        self.qa_model = qa_model
        
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_layers = max_layers
        self.clustering_threshold = clustering_threshold
        self.reduction_dim = reduction_dim
        self.max_cluster_tokens = max_cluster_tokens
        self.summarization_tokens = summarization_tokens
        
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tree: Optional[Tree] = None
        self.stats = TimingStats()
    
    def build_tree(self, text: str) -> Tree:
        """Build RAPTOR tree from text."""
        total_start = time.perf_counter()
        
        # Step 1: Chunking
        logging.info("Step 1: Chunking text...")
        chunk_start = time.perf_counter()
        chunks = split_text(text, self.max_tokens_per_chunk, tokenizer=self.tokenizer)
        self.stats.chunking_seconds = time.perf_counter() - chunk_start
        self.stats.num_chunks = len(chunks)
        logging.info(f"  Created {len(chunks)} chunks in {self.stats.chunking_seconds:.2f}s")
        
        # Step 2: Create leaf nodes with embeddings
        logging.info("Step 2: Creating embeddings...")
        embed_start = time.perf_counter()
        leaf_nodes = {}
        for i, chunk in enumerate(chunks):
            embedding = self.embedding_model.create_embedding(chunk)
            node = Node(text=chunk, index=i, embedding=embedding, layer=0)
            leaf_nodes[i] = node
        self.stats.embedding_seconds = time.perf_counter() - embed_start
        logging.info(f"  Created {len(leaf_nodes)} embeddings in {self.stats.embedding_seconds:.2f}s")
        
        # Step 3: Build hierarchy
        logging.info("Step 3: Building hierarchy...")
        all_nodes = copy.deepcopy(leaf_nodes)
        layer_to_nodes = {0: list(leaf_nodes.values())}
        current_layer_nodes = leaf_nodes
        next_node_idx = len(leaf_nodes)
        
        cluster_time_total = 0.0
        summary_time_total = 0.0
        num_summaries = 0
        
        for layer in range(1, self.max_layers + 1):
            logging.info(f"  Building layer {layer}...")
            
            if len(current_layer_nodes) <= self.reduction_dim + 1:
                logging.info(f"    Stopping: only {len(current_layer_nodes)} nodes (need > {self.reduction_dim + 1})")
                break
            
            # Cluster
            cluster_start = time.perf_counter()
            node_list = list(current_layer_nodes.values())
            embeddings = np.array([n.embedding for n in node_list])
            
            reduced = reduce_dimensions(embeddings, self.reduction_dim)
            clusters = cluster_embeddings(reduced, self.clustering_threshold)
            cluster_time_total += time.perf_counter() - cluster_start
            
            logging.info(f"    Found {len(clusters)} clusters")
            
            # Summarize each cluster
            new_layer_nodes = {}
            for cluster_indices in clusters:
                cluster_nodes = [node_list[i] for i in cluster_indices]
                combined_text = "\n\n".join(n.text for n in cluster_nodes)
                
                # Check token limit
                if len(self.tokenizer.encode(combined_text)) > self.max_cluster_tokens:
                    # Split large clusters (simplified: just take first part)
                    combined_text = combined_text[:self.max_cluster_tokens * 4]  # rough char estimate
                
                # Summarize
                summary_start = time.perf_counter()
                summary = self.summarization_model.summarize(combined_text, self.summarization_tokens)
                summary_time_total += time.perf_counter() - summary_start
                num_summaries += 1
                
                # Create parent node
                embedding = self.embedding_model.create_embedding(summary)
                child_indices = {node_list[i].index for i in cluster_indices}
                
                parent_node = Node(
                    text=summary,
                    index=next_node_idx,
                    children=child_indices,
                    embedding=embedding,
                    layer=layer,
                )
                
                new_layer_nodes[next_node_idx] = parent_node
                all_nodes[next_node_idx] = parent_node
                next_node_idx += 1
            
            layer_to_nodes[layer] = list(new_layer_nodes.values())
            current_layer_nodes = new_layer_nodes
            
            logging.info(f"    Created {len(new_layer_nodes)} summary nodes")
        
        self.stats.clustering_seconds = cluster_time_total
        self.stats.summarization_seconds = summary_time_total
        self.stats.num_summaries = num_summaries
        self.stats.num_layers = len(layer_to_nodes) - 1
        self.stats.num_nodes = len(all_nodes)
        self.stats.total_seconds = time.perf_counter() - total_start
        
        logging.info(f"Tree built: {self.stats.num_nodes} nodes, {self.stats.num_layers} layers, {self.stats.total_seconds:.2f}s total")
        
        self.tree = Tree(
            all_nodes=all_nodes,
            root_nodes=current_layer_nodes,
            leaf_nodes=leaf_nodes,
            num_layers=self.stats.num_layers,
            layer_to_nodes=layer_to_nodes,
        )
        
        return self.tree
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        collapse_tree: bool = True
    ) -> List[Node]:
        """
        Retrieve relevant nodes for a query.
        
        Args:
            query: Search query
            top_k: Number of nodes to retrieve
            collapse_tree: If True, search all nodes. If False, use tree traversal.
        
        Returns:
            List of most relevant nodes
        """
        if self.tree is None:
            raise ValueError("No tree built. Call build_tree() first.")
        
        query_embedding = np.array(self.embedding_model.create_embedding(query))
        
        if collapse_tree:
            # Search all nodes
            nodes = list(self.tree.all_nodes.values())
        else:
            # Start from root and traverse
            nodes = list(self.tree.root_nodes.values())
            for _ in range(self.tree.num_layers):
                # Get children of current nodes
                child_indices = set()
                for node in nodes:
                    child_indices.update(node.children)
                if child_indices:
                    child_nodes = [self.tree.all_nodes[i] for i in child_indices]
                    nodes.extend(child_nodes)
        
        # Calculate distances
        distances = []
        for node in nodes:
            node_embedding = np.array(node.embedding)
            # Cosine distance
            dist = 1 - np.dot(query_embedding, node_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding)
            )
            distances.append((node, dist))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        return [node for node, _ in distances[:top_k]]
    
    def answer_question(self, question: str, top_k: int = 5) -> Tuple[str, List[Node]]:
        """
        Answer a question using the RAPTOR tree.
        
        Returns:
            Tuple of (answer, source_nodes)
        """
        # Retrieve relevant nodes
        nodes = self.retrieve(question, top_k=top_k)
        
        # Build context
        context = "\n\n---\n\n".join(node.text for node in nodes)
        
        # Generate answer
        answer = self.qa_model.answer_question(context, question)
        
        return answer, nodes
    
    def get_stats(self) -> TimingStats:
        """Get timing statistics."""
        return self.stats
    
    def print_stats(self):
        """Print timing statistics."""
        s = self.stats
        print("\n" + "="*60)
        print("RAPTOR-Lite Performance Statistics")
        print("="*60)
        print(f"Document Processing:")
        print(f"  Chunks created:     {s.num_chunks}")
        print(f"  Total nodes:        {s.num_nodes}")
        print(f"  Tree layers:        {s.num_layers}")
        print(f"  Summaries created:  {s.num_summaries}")
        print()
        print(f"Timing Breakdown:")
        print(f"  Chunking:           {s.chunking_seconds:6.2f}s")
        print(f"  Embedding:          {s.embedding_seconds:6.2f}s")
        print(f"  Clustering:         {s.clustering_seconds:6.2f}s")
        print(f"  Summarization:      {s.summarization_seconds:6.2f}s")
        print(f"  ─────────────────────────────")
        print(f"  TOTAL:              {s.total_seconds:6.2f}s")
        print()
        if s.num_chunks > 0:
            print(f"Performance Metrics:")
            print(f"  Chunks/second:      {s.num_chunks / s.total_seconds:.2f}")
            print(f"  Embed time/chunk:   {s.embedding_seconds / s.num_chunks * 1000:.1f}ms")
            if s.num_summaries > 0:
                print(f"  Summary time/node:  {s.summarization_seconds / s.num_summaries:.2f}s")
        print("="*60)


if __name__ == "__main__":
    # Quick test with a small document
    test_text = """
    Machine learning is a method of data analysis that automates analytical model building. 
    It is a branch of artificial intelligence based on the idea that systems can learn from data, 
    identify patterns and make decisions with minimal human intervention.
    
    Deep learning is a subset of machine learning that uses neural networks with many layers.
    These deep neural networks can learn representations of data with multiple levels of abstraction.
    Deep learning has been particularly successful in areas like image recognition and natural language processing.
    
    Natural language processing (NLP) is a field of artificial intelligence that focuses on the interaction
    between computers and humans through natural language. The ultimate objective of NLP is to read, 
    decipher, understand, and make sense of human languages in a valuable way.
    
    Computer vision is a field of artificial intelligence that trains computers to interpret and understand
    the visual world. Using digital images from cameras and videos and deep learning models, machines can
    accurately identify and classify objects.
    """
    
    print("Testing RAPTOR-Lite...")
    print("="*60)
    
    # Initialize models
    print("\nInitializing models...")
    embed_model = LocalEmbeddingModel()
    
    # Check if Ollama is available
    from local_models import check_ollama_available
    if check_ollama_available():
        summarizer = OllamaSummarizationModel()
        qa_model = OllamaQAModel()
    else:
        print("WARNING: Ollama not available. Using dummy models.")
        
        class DummySummarizer(BaseSummarizationModel):
            def summarize(self, context, max_tokens=150):
                return context[:200] + "..."
        
        class DummyQA(BaseQAModel):
            def answer_question(self, context, question):
                return f"[Dummy answer for: {question}]"
        
        summarizer = DummySummarizer()
        qa_model = DummyQA()
    
    # Build tree
    raptor = RaptorLite(
        embedding_model=embed_model,
        summarization_model=summarizer,
        qa_model=qa_model,
        max_tokens_per_chunk=50,  # Small chunks for testing
        max_layers=3,
    )
    
    tree = raptor.build_tree(test_text)
    raptor.print_stats()
    
    # Test retrieval
    print("\nTesting retrieval...")
    query = "What is deep learning?"
    nodes = raptor.retrieve(query, top_k=3)
    print(f"\nQuery: {query}")
    print(f"Retrieved {len(nodes)} nodes:")
    for i, node in enumerate(nodes):
        print(f"  {i+1}. [Layer {node.layer}] {node.text[:100]}...")
    
    # Test QA
    print("\nTesting Q&A...")
    answer, sources = raptor.answer_question("What is deep learning and what is it used for?")
    print(f"\nQuestion: What is deep learning and what is it used for?")
    print(f"Answer: {answer}")
    print(f"\nSources used: {len(sources)} nodes from layers {set(n.layer for n in sources)}")
