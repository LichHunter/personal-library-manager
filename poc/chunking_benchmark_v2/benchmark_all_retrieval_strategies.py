#!/usr/bin/env python3
"""Comprehensive Retrieval Strategy Benchmark.

Tests ALL retrieval strategies on the expanded corpus (52 docs, 53 queries, 180 facts):

Embedding-based strategies:
- Semantic retrieval (various embedding models)
- Hybrid BM25 + semantic with RRF fusion
- Hybrid with cross-encoder reranking

Hierarchical strategies:
- LOD (Level-of-Detail): 3-level hierarchy (doc -> section -> chunk)
- RAPTOR: Tree with clustering + LLM summarization

Configurable model lists for comprehensive benchmarking.
"""

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import sys

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from strategies import Document, Chunk, FixedSizeStrategy


# ==============================================================================
# MODEL CONFIGURATIONS
# ==============================================================================

# Embedding models to test (name, use_prefix_for_query)
EMBEDDING_MODELS = [
    ("all-MiniLM-L6-v2", False),
    ("all-MiniLM-L12-v2", False),
    ("BAAI/bge-small-en-v1.5", True),
    ("BAAI/bge-base-en-v1.5", True),
    ("thenlper/gte-small", False),
]

# Cross-encoder rerankers to test
RERANKER_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]

# Ollama LLM models for RAPTOR summarization
RAPTOR_LLM_MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b",
]


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class Section:
    """A section within a document."""
    id: str
    heading: str
    content: str
    level: int = 1


@dataclass 
class StructuredDocument:
    """Document with section structure for LOD strategies."""
    id: str
    title: str
    content: str
    summary: str
    sections: list[Section] = field(default_factory=list)


def parse_document_sections(doc: Document) -> StructuredDocument:
    """Parse markdown document into structured sections."""
    import re
    
    content = doc.content
    lines = content.split('\n')
    
    # Extract title from first H1 or use doc title
    title = doc.title
    for line in lines:
        if line.startswith('# '):
            title = line[2:].strip()
            break
    
    # Find all headings
    heading_pattern = r'^(#{1,4})\s+(.+)$'
    sections = []
    current_section = None
    current_content = []
    
    for line in lines:
        match = re.match(heading_pattern, line)
        if match:
            # Save previous section
            if current_section:
                current_section['content'] = '\n'.join(current_content).strip()
                if current_section['content']:
                    sections.append(Section(
                        id=f"sec_{len(sections)}",
                        heading=current_section['heading'],
                        content=current_section['content'],
                        level=current_section['level'],
                    ))
            
            # Start new section
            level = len(match.group(1))
            heading = match.group(2).strip()
            current_section = {'heading': heading, 'level': level}
            current_content = []
        else:
            current_content.append(line)
    
    # Save final section
    if current_section:
        current_section['content'] = '\n'.join(current_content).strip()
        if current_section['content']:
            sections.append(Section(
                id=f"sec_{len(sections)}",
                heading=current_section['heading'],
                content=current_section['content'],
                level=current_section['level'],
            ))
    
    # If no sections found, treat whole content as one section
    if not sections:
        sections.append(Section(
            id="sec_0",
            heading=title,
            content=content,
            level=1,
        ))
    
    # Create summary from first 500 chars
    summary = content[:500].strip()
    if len(content) > 500:
        summary += "..."
    
    return StructuredDocument(
        id=doc.id,
        title=title,
        content=content,
        summary=summary,
        sections=sections,
    )


# ==============================================================================
# OLLAMA SUMMARIZER FOR RAPTOR
# ==============================================================================

def create_ollama_summarizer(model: str = "llama3.2:3b") -> Callable[[str], str]:
    """Create a summarization function using Ollama (runs on GPU)."""
    
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
            else:
                # Fallback: extractive summary
                words = text.split()[:150]
                return " ".join(words)
        except Exception as e:
            # Fallback: extractive summary
            words = text.split()[:150]
            return " ".join(words)
    
    return summarize


def check_ollama_available() -> bool:
    """Check if Ollama is available and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# ==============================================================================
# RETRIEVAL STRATEGIES
# ==============================================================================

class SemanticRetrieval:
    """Pure semantic embedding retrieval."""
    
    def __init__(self, embedder, name: str, use_prefix: bool = False):
        self.embedder = embedder
        self.name = name
        self.use_prefix = use_prefix
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
    
    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
    
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        q = f"Represent this sentence for searching relevant passages: {query}" if self.use_prefix else query
        q_emb = self.embedder.encode([q], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.chunks[i] for i in top_idx]


class HybridRetrieval:
    """Hybrid BM25 + semantic retrieval with RRF fusion."""
    
    def __init__(self, embedder, name: str, use_prefix: bool = False, rrf_k: int = 60):
        self.embedder = embedder
        self.name = name
        self.use_prefix = use_prefix
        self.rrf_k = rrf_k
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
    
    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        self.bm25 = BM25Okapi([c.content.lower().split() for c in chunks])
    
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        # Semantic scores
        q = f"Represent this sentence for searching relevant passages: {query}" if self.use_prefix else query
        q_emb = self.embedder.encode([q], normalize_embeddings=True)[0]
        sem_scores = np.dot(self.embeddings, q_emb)
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        # RRF fusion
        sem_ranks = np.argsort(sem_scores)[::-1]
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        rrf = {}
        for rank, idx in enumerate(sem_ranks[:50]):
            rrf[idx] = rrf.get(idx, 0) + 1 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranks[:50]):
            rrf[idx] = rrf.get(idx, 0) + 1 / (self.rrf_k + rank)
        
        top_idx = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:k]
        return [self.chunks[i] for i in top_idx]


class HybridRerankRetrieval:
    """Hybrid retrieval with cross-encoder reranking."""
    
    def __init__(
        self, 
        embedder, 
        reranker, 
        name: str, 
        use_prefix: bool = False, 
        rrf_k: int = 60, 
        initial_k: int = 20
    ):
        self.embedder = embedder
        self.reranker = reranker
        self.name = name
        self.use_prefix = use_prefix
        self.rrf_k = rrf_k
        self.initial_k = initial_k
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
    
    def index(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        self.bm25 = BM25Okapi([c.content.lower().split() for c in chunks])
    
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        # Get candidates via hybrid
        q = f"Represent this sentence for searching relevant passages: {query}" if self.use_prefix else query
        q_emb = self.embedder.encode([q], normalize_embeddings=True)[0]
        sem_scores = np.dot(self.embeddings, q_emb)
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        sem_ranks = np.argsort(sem_scores)[::-1]
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        rrf = {}
        for rank, idx in enumerate(sem_ranks[:50]):
            rrf[idx] = rrf.get(idx, 0) + 1 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranks[:50]):
            rrf[idx] = rrf.get(idx, 0) + 1 / (self.rrf_k + rank)
        
        candidates = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:self.initial_k]
        
        # Rerank
        pairs = [[query, self.chunks[i].content] for i in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_idx = [idx for idx, _ in reranked[:k]]
        return [self.chunks[i] for i in top_idx]


class LODEmbedRetrieval:
    """Level-of-Detail retrieval with 3-level hierarchy."""
    
    def __init__(
        self, 
        embedder, 
        name: str = "lod_embed",
        chunk_size: int = 512,
        doc_top_k: int = 3,
        section_top_k: int = 5,
    ):
        self.embedder = embedder
        self.name = name
        self.chunk_size = chunk_size
        self.doc_top_k = doc_top_k
        self.section_top_k = section_top_k
        
        self.doc_entries: list = []
        self.section_entries: list = []
        self.chunk_entries: list = []
        self.chunk_embeddings: Optional[np.ndarray] = None
    
    def index(self, documents: list[StructuredDocument]):
        """Build 3-level index."""
        # Level 2: Document summaries
        doc_texts = []
        for doc in documents:
            text = f"{doc.title}\n\n{doc.summary}"
            self.doc_entries.append((doc.id, text))
            doc_texts.append(text)
        
        doc_embeddings = self.embedder.encode(
            doc_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        for i, emb in enumerate(doc_embeddings):
            self.doc_entries[i] = (*self.doc_entries[i], emb)
        
        # Level 1: Section summaries
        section_texts = []
        for doc in documents:
            for section in doc.sections:
                text = f"{section.heading}\n\n{section.content[:500]}"
                self.section_entries.append((doc.id, section.id, text))
                section_texts.append(text)
        
        section_embeddings = self.embedder.encode(
            section_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        for i, emb in enumerate(section_embeddings):
            self.section_entries[i] = (*self.section_entries[i], emb)
        
        # Level 0: Chunks
        chunker = FixedSizeStrategy(chunk_size=self.chunk_size, overlap=0)
        chunk_texts = []
        
        for doc in documents:
            for section in doc.sections:
                temp_doc = Document(
                    id=f"{doc.id}_{section.id}",
                    title=section.heading,
                    content=section.content,
                )
                chunks = chunker.chunk(temp_doc)
                for chunk in chunks:
                    self.chunk_entries.append((doc.id, section.id, chunk))
                    chunk_texts.append(chunk.content)
        
        if chunk_texts:
            self.chunk_embeddings = self.embedder.encode(
                chunk_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,
            )
    
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if not self.chunk_entries:
            return []
        
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        
        # Level 2: Find top documents
        doc_sims = [np.dot(entry[2], q_emb) for entry in self.doc_entries]
        top_doc_indices = np.argsort(doc_sims)[::-1][:self.doc_top_k]
        selected_doc_ids = {self.doc_entries[i][0] for i in top_doc_indices}
        
        # Level 1: Find top sections within selected docs
        filtered_sections = [
            (i, entry) for i, entry in enumerate(self.section_entries)
            if entry[0] in selected_doc_ids
        ]
        
        if not filtered_sections:
            return []
        
        section_sims = [(i, np.dot(entry[3], q_emb)) for i, entry in filtered_sections]
        section_sims.sort(key=lambda x: x[1], reverse=True)
        top_section_indices = [i for i, _ in section_sims[:self.section_top_k]]
        selected_section_ids = {
            (self.section_entries[i][0], self.section_entries[i][1]) 
            for i in top_section_indices
        }
        
        # Level 0: Find top chunks within selected sections
        filtered_chunks = [
            (i, entry) for i, entry in enumerate(self.chunk_entries)
            if (entry[0], entry[1]) in selected_section_ids
        ]
        
        if not filtered_chunks:
            return []
        
        chunk_sims = [(i, np.dot(self.chunk_embeddings[i], q_emb)) for i, _ in filtered_chunks]
        chunk_sims.sort(key=lambda x: x[1], reverse=True)
        top_chunk_indices = [i for i, _ in chunk_sims[:k]]
        
        return [self.chunk_entries[i][2] for i in top_chunk_indices]


class RAPTORRetrieval:
    """RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval."""
    
    def __init__(
        self,
        embedder,
        summarizer: Callable[[str], str],
        name: str = "raptor",
        chunk_size: int = 512,
        max_layers: int = 2,
        cluster_threshold: float = 0.1,
    ):
        self.embedder = embedder
        self.summarizer = summarizer
        self.name = name
        self.chunk_size = chunk_size
        self.max_layers = max_layers
        self.cluster_threshold = cluster_threshold
        
        self.all_nodes: list[dict] = []
        self.all_embeddings: Optional[np.ndarray] = None
    
    def index(self, chunks: list[Chunk]):
        """Build RAPTOR tree from chunks."""
        from sklearn.mixture import GaussianMixture
        from sklearn.decomposition import PCA
        
        if not chunks:
            return
        
        # Layer 0: Leaf nodes
        texts = [c.content for c in chunks]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        
        current_layer = []
        for chunk, emb in zip(chunks, embeddings):
            self.all_nodes.append({
                "text": chunk.content,
                "embedding": emb,
                "layer": 0,
                "doc_id": chunk.doc_id,
                "chunk": chunk,
            })
            current_layer.append((chunk.content, emb))
        
        log(f"    Layer 0: {len(current_layer)} leaf nodes")
        
        # Build higher layers
        for layer in range(1, self.max_layers + 1):
            if len(current_layer) <= 3:
                log(f"    Stopping at layer {layer-1}: only {len(current_layer)} nodes")
                break
            
            # Cluster current layer
            layer_embeddings = np.array([emb for _, emb in current_layer])
            clusters = self._cluster_embeddings(layer_embeddings)
            
            if len(clusters) <= 1:
                log(f"    Stopping at layer {layer}: only {len(clusters)} cluster(s)")
                break
            
            log(f"    Layer {layer}: clustering {len(current_layer)} nodes into {len(clusters)} clusters")
            
            # Summarize each cluster
            new_layer = []
            for cluster_indices in clusters:
                cluster_texts = [current_layer[i][0] for i in cluster_indices]
                combined = "\n\n".join(cluster_texts)[:4000]
                
                # Generate summary
                summary = self.summarizer(combined)
                
                # Embed summary
                summary_emb = self.embedder.encode([summary], normalize_embeddings=True)[0]
                
                self.all_nodes.append({
                    "text": summary,
                    "embedding": summary_emb,
                    "layer": layer,
                    "doc_id": None,
                    "chunk": None,
                })
                new_layer.append((summary, summary_emb))
            
            current_layer = new_layer
            log(f"    Layer {layer}: created {len(new_layer)} summary nodes")
        
        # Build combined embedding matrix
        self.all_embeddings = np.array([n["embedding"] for n in self.all_nodes])
        log(f"    Total RAPTOR nodes: {len(self.all_nodes)}")
    
    def _cluster_embeddings(self, embeddings: np.ndarray) -> list[list[int]]:
        """Cluster embeddings using GMM with soft assignment."""
        from sklearn.mixture import GaussianMixture
        from sklearn.decomposition import PCA
        
        n_samples = len(embeddings)
        if n_samples <= 1:
            return [[0]] if n_samples == 1 else []
        
        # Reduce dimensions if needed
        reduction_dim = 10
        if n_samples > reduction_dim + 1:
            target_dim = min(reduction_dim, n_samples - 2)
            if target_dim >= 2:
                pca = PCA(n_components=target_dim)
                embeddings = pca.fit_transform(embeddings)
        
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
            for cluster_idx in np.where(prob_row > self.cluster_threshold)[0]:
                if cluster_idx not in clusters:
                    clusters[cluster_idx] = []
                clusters[cluster_idx].append(i)
        
        return list(clusters.values())
    
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Collapsed search: search all nodes."""
        if self.all_embeddings is None or len(self.all_nodes) == 0:
            return []
        
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.all_embeddings, q_emb)
        top_indices = np.argsort(sims)[::-1]
        
        # Return top-k nodes that have actual chunks
        results = []
        for idx in top_indices:
            node = self.all_nodes[idx]
            if node["chunk"] is not None:
                results.append(node["chunk"])
                if len(results) >= k:
                    break
        
        # If not enough leaf nodes, create pseudo-chunks from summaries
        if len(results) < k:
            for idx in top_indices:
                node = self.all_nodes[idx]
                if node["chunk"] is None and len(results) < k:
                    pseudo_chunk = Chunk(
                        id=f"raptor_summary_{idx}",
                        doc_id=node["doc_id"] or "unknown",
                        content=node["text"],
                        start_char=0,
                        end_char=len(node["text"]),
                    )
                    results.append(pseudo_chunk)
        
        return results[:k]


# ==============================================================================
# EVALUATION
# ==============================================================================

def exact_match(fact: str, text: str) -> bool:
    return fact.lower() in text.lower()


def fuzzy_match(fact: str, text: str) -> bool:
    text_lower = text.lower()
    fact_lower = fact.lower()
    
    if fact_lower in text_lower:
        return True
    
    words = fact_lower.split()
    if len(words) >= 2 and all(w in text_lower for w in words):
        return True
    
    return False


def evaluate_strategy(strategy, queries: list[dict], k: int = 5, match_fn=exact_match) -> dict:
    """Evaluate a retrieval strategy."""
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    found = 0
    times = []
    
    for q in queries:
        start = time.perf_counter()
        retrieved = strategy.retrieve(q["query"], k=k)
        times.append(time.perf_counter() - start)
        
        text = " ".join(c.content for c in retrieved)
        for fact in q.get("key_facts", []):
            if match_fn(fact, text):
                found += 1
    
    return {
        "coverage": found / total_facts if total_facts else 0,
        "found": found,
        "total": total_facts,
        "avg_ms": np.mean(times) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000 if times else 0,
    }


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_expanded_corpus() -> tuple[list[Document], list[StructuredDocument]]:
    """Load expanded corpus as both flat and structured documents."""
    corpus_dir = Path(__file__).parent / "corpus"
    metadata_path = corpus_dir / "corpus_metadata_expanded.json"
    docs_dir = corpus_dir / "expanded_documents"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    flat_docs = []
    structured_docs = []
    
    for doc_meta in metadata:
        doc_path = docs_dir / doc_meta["filename"]
        if doc_path.exists():
            doc = Document(
                id=doc_meta["id"],
                title=doc_meta["title"],
                content=doc_path.read_text(),
                path=str(doc_path),
            )
            flat_docs.append(doc)
            structured_docs.append(parse_document_sections(doc))
    
    return flat_docs, structured_docs


def load_queries() -> list[dict]:
    """Load expanded ground truth queries."""
    gt_path = Path(__file__).parent / "corpus" / "ground_truth_expanded.json"
    with open(gt_path) as f:
        return json.load(f)["queries"]


# ==============================================================================
# MAIN
# ==============================================================================

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 80)
    log("COMPREHENSIVE RETRIEVAL STRATEGY BENCHMARK")
    log("Expanded corpus: 52 docs, 53 queries, 180 key facts")
    log("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    
    # Check Ollama
    ollama_available = check_ollama_available()
    log(f"Ollama available: {ollama_available}")
    
    # Load data
    flat_docs, structured_docs = load_expanded_corpus()
    log(f"Loaded {len(flat_docs)} documents")
    
    queries = load_queries()
    log(f"Loaded {len(queries)} queries")
    
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    log(f"Total key facts: {total_facts}")
    
    # Create flat chunks
    chunker = FixedSizeStrategy(chunk_size=512, overlap=0)
    flat_chunks = chunker.chunk_many(flat_docs)
    log(f"Created {len(flat_chunks)} flat chunks")
    
    # Load first embedding model for initial strategies
    log("\nLoading embedding model...")
    emb_name, use_prefix = EMBEDDING_MODELS[0]
    embedder = SentenceTransformer(emb_name, device=device)
    log(f"Loaded: {emb_name}")
    
    # Load reranker
    log("Loading reranker...")
    reranker = CrossEncoder(RERANKER_MODELS[0], device=device)
    log(f"Loaded: {RERANKER_MODELS[0]}")
    
    # Build strategies
    strategies = []
    
    # Basic strategies with first embedder
    strategies.append(SemanticRetrieval(embedder, f"semantic_{emb_name.split('/')[-1]}", use_prefix))
    strategies.append(HybridRetrieval(embedder, f"hybrid_{emb_name.split('/')[-1]}", use_prefix))
    strategies.append(HybridRerankRetrieval(embedder, reranker, f"hybrid_rerank_{emb_name.split('/')[-1]}", use_prefix))
    
    # LOD strategy
    strategies.append(LODEmbedRetrieval(embedder, f"lod_{emb_name.split('/')[-1]}", doc_top_k=3, section_top_k=5))
    
    # RAPTOR with Ollama (if available)
    if ollama_available and RAPTOR_LLM_MODELS:
        llm_model = RAPTOR_LLM_MODELS[0]
        log(f"\nSetting up RAPTOR with Ollama ({llm_model})...")
        summarizer = create_ollama_summarizer(llm_model)
        strategies.append(RAPTORRetrieval(embedder, summarizer, f"raptor_{llm_model.replace(':', '_')}"))
    else:
        log("\nSkipping RAPTOR (Ollama not available)")
    
    # Index strategies
    log("\nIndexing strategies...")
    for s in strategies:
        log(f"  {s.name}")
        if isinstance(s, LODEmbedRetrieval):
            s.index(structured_docs)
        else:
            s.index(flat_chunks)
    
    # Evaluate
    k_values = [5, 10]
    results = {"exact": {}, "fuzzy": {}}
    
    for k in k_values:
        log(f"\n{'='*80}")
        log(f"RESULTS k={k}")
        log(f"{'='*80}")
        
        log(f"\n{'Strategy':<40} | {'Exact':>8} | {'Fuzzy':>8} | {'Avg ms':>8}")
        log("-" * 70)
        
        for s in strategies:
            exact_res = evaluate_strategy(s, queries, k=k, match_fn=exact_match)
            fuzzy_res = evaluate_strategy(s, queries, k=k, match_fn=fuzzy_match)
            
            results["exact"][f"{s.name}_k{k}"] = exact_res
            results["fuzzy"][f"{s.name}_k{k}"] = fuzzy_res
            
            log(f"{s.name:<40} | {exact_res['coverage']:>7.1%} | {fuzzy_res['coverage']:>7.1%} | {exact_res['avg_ms']:>7.1f}")
    
    # Summary
    log(f"\n{'='*80}")
    log("SUMMARY")
    log(f"{'='*80}")
    
    for k in k_values:
        exact_k = {name: v for name, v in results["exact"].items() if f"k{k}" in name}
        best = max(exact_k.items(), key=lambda x: x[1]["coverage"])
        log(f"\nBest at k={k}: {best[0]} with {best[1]['coverage']:.1%} ({best[1]['found']}/{best[1]['total']})")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    output_path = results_dir / f"{timestamp}_all_retrieval_strategies.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    log(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
