# RAPTOR Dependency Analysis

> Analysis of all external dependencies used by RAPTOR

## Overview

RAPTOR relies on a mix of ML/AI libraries, primarily for embeddings, clustering, and language model inference. The project is designed to work with OpenAI's APIs by default but supports extensibility for local models.

---

## Dependencies from requirements.txt

```
faiss-cpu
numpy==1.26.3
openai==1.3.3
scikit-learn
sentence-transformers==2.2.2
tenacity==8.2.3
tiktoken==0.5.1 
torch
transformers==4.38.1
umap-learn==0.5.5
urllib3==1.26.6
```

---

## Detailed Dependency Analysis

### Core ML/AI Dependencies

| Dependency | Version | Category | Purpose | Used In |
|------------|---------|----------|---------|---------|
| `openai` | 1.3.3 | Required | OpenAI API client | EmbeddingModels.py, SummarizationModels.py, QAModels.py |
| `torch` | (any) | Required | PyTorch deep learning | QAModels.py (UnifiedQAModel) |
| `transformers` | 4.38.1 | Required | Hugging Face models | QAModels.py (UnifiedQAModel) |
| `sentence-transformers` | 2.2.2 | Required | Sentence embeddings | EmbeddingModels.py (SBertEmbeddingModel) |

### Clustering & Dimensionality Reduction

| Dependency | Version | Category | Purpose | Used In |
|------------|---------|----------|---------|---------|
| `umap-learn` | 0.5.5 | Required | UMAP dimensionality reduction | cluster_utils.py |
| `scikit-learn` | (any) | Required | GaussianMixture clustering | cluster_utils.py |

### Vector Search

| Dependency | Version | Category | Purpose | Used In |
|------------|---------|----------|---------|---------|
| `faiss-cpu` | (any) | Optional | Fast similarity search | FaissRetriever.py |

### Tokenization

| Dependency | Version | Category | Purpose | Used In |
|------------|---------|----------|---------|---------|
| `tiktoken` | 0.5.1 | Required | OpenAI tokenizer | tree_builder.py, tree_retriever.py, cluster_utils.py, utils.py, FaissRetriever.py |

### Utilities

| Dependency | Version | Category | Purpose | Used In |
|------------|---------|----------|---------|---------|
| `numpy` | 1.26.3 | Required | Numerical operations | cluster_utils.py, utils.py, FaissRetriever.py |
| `scipy` | (implicit) | Required | Distance calculations | utils.py |
| `tenacity` | 8.2.3 | Required | Retry logic for API calls | tree_builder.py, tree_retriever.py, EmbeddingModels.py, SummarizationModels.py, QAModels.py |
| `tqdm` | (implicit) | Required | Progress bars | FaissRetriever.py |
| `urllib3` | 1.26.6 | Transitive | HTTP client | (indirect via openai) |

---

## Dependency Details

### openai (1.3.3)

**What it is:** Official Python client for OpenAI's API.

**Why it's used:** Primary interface for LLM capabilities:
- Text embeddings (`text-embedding-ada-002`)
- Summarization (GPT-3.5-turbo, GPT-4)
- Question answering (GPT-3.5-turbo, GPT-4, text-davinci-003)

**Files using it:**
| File | Usage |
|------|-------|
| `EmbeddingModels.py:4` | `from openai import OpenAI` |
| `SummarizationModels.py:5` | `from openai import OpenAI` |
| `QAModels.py:4` | `from openai import OpenAI` |
| `tree_builder.py:9` | `import openai` (unused direct import) |

**API Methods Used:**
- `client.embeddings.create()` - Generate embeddings
- `client.chat.completions.create()` - Chat-based generation
- `client.completions.create()` - Legacy completion API

**Environment Requirement:** `OPENAI_API_KEY` must be set

---

### torch (PyTorch)

**What it is:** Deep learning framework.

**Why it's used:** Backend for running local transformer models.

**Files using it:**
| File | Usage |
|------|-------|
| `QAModels.py:10` | `import torch` |

**Specific usage:**
```python
# Device selection for UnifiedQAModel
self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**Category:** Required only if using `UnifiedQAModel`

---

### transformers (4.38.1)

**What it is:** Hugging Face's transformer model library.

**Why it's used:** Load and run T5 models for local question answering.

**Files using it:**
| File | Usage |
|------|-------|
| `QAModels.py:12` | `from transformers import T5ForConditionalGeneration, T5Tokenizer` |

**Models loaded:**
- `allenai/unifiedqa-v2-t5-3b-1363200` (default for UnifiedQAModel)

**Category:** Required only if using `UnifiedQAModel`

---

### sentence-transformers (2.2.2)

**What it is:** Library for computing dense vector representations of sentences.

**Why it's used:** Local embedding generation alternative to OpenAI.

**Files using it:**
| File | Usage |
|------|-------|
| `EmbeddingModels.py:5` | `from sentence_transformers import SentenceTransformer` |

**Models loaded:**
- `sentence-transformers/multi-qa-mpnet-base-cos-v1` (default for SBertEmbeddingModel)

**Category:** Required only if using `SBertEmbeddingModel`

---

### umap-learn (0.5.5)

**What it is:** Uniform Manifold Approximation and Projection - dimensionality reduction technique.

**Why it's used:** Core component of RAPTOR's clustering algorithm. Reduces high-dimensional embeddings to lower dimensions before GMM clustering.

**Files using it:**
| File | Usage |
|------|-------|
| `cluster_utils.py:8` | `import umap` |

**Specific usage:**
```python
# Global clustering - n_neighbors based on sqrt of data size
reduced_embeddings = umap.UMAP(
    n_neighbors=n_neighbors, 
    n_components=dim, 
    metric="cosine"
).fit_transform(embeddings)

# Local clustering - fixed n_neighbors=10
reduced_embeddings = umap.UMAP(
    n_neighbors=num_neighbors, 
    n_components=dim, 
    metric=metric
).fit_transform(embeddings)
```

**Category:** Required (core algorithm)

---

### scikit-learn

**What it is:** Machine learning library with classical algorithms.

**Why it's used:** Gaussian Mixture Model (GMM) clustering for grouping similar nodes.

**Files using it:**
| File | Usage |
|------|-------|
| `cluster_utils.py:9` | `from sklearn.mixture import GaussianMixture` |

**Specific usage:**
```python
# Find optimal clusters using BIC
gm = GaussianMixture(n_components=n, random_state=random_state)
gm.fit(embeddings)
bics.append(gm.bic(embeddings))

# Soft clustering
gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
gm.fit(embeddings)
probs = gm.predict_proba(embeddings)
labels = [np.where(prob > threshold)[0] for prob in probs]
```

**Category:** Required (core algorithm)

---

### faiss-cpu

**What it is:** Facebook AI Similarity Search - efficient similarity search library.

**Why it's used:** Alternative retrieval mechanism for flat (non-hierarchical) search.

**Files using it:**
| File | Usage |
|------|-------|
| `FaissRetriever.py:4` | `import faiss` |

**Specific usage:**
```python
# Create index
self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
self.index.add(self.embeddings)

# Search
_, indices = self.index.search(query_embedding, self.top_k)
```

**Category:** Optional (only for FaissRetriever)

---

### tiktoken (0.5.1)

**What it is:** OpenAI's fast tokenizer library.

**Why it's used:** Token counting for text chunking and context window management.

**Files using it:**
| File | Usage |
|------|-------|
| `tree_builder.py:10` | `import tiktoken` |
| `tree_retriever.py:5` | `import tiktoken` |
| `cluster_utils.py:7` | `import tiktoken` |
| `utils.py:6` | `import tiktoken` |
| `FaissRetriever.py:6` | `import tiktoken` |

**Encoding used:**
```python
tiktoken.get_encoding("cl100k_base")  # GPT-4 / ChatGPT tokenizer
```

**Category:** Required

---

### numpy (1.26.3)

**What it is:** Numerical computing library.

**Why it's used:** Array operations, distance calculations, embedding manipulation.

**Files using it:**
| File | Usage |
|------|-------|
| `cluster_utils.py:6` | `import numpy as np` |
| `utils.py:5` | `import numpy as np` |
| `FaissRetriever.py:5` | `import numpy as np` |

**Key operations:**
- `np.array()` - Convert embeddings to arrays
- `np.argsort()` - Sort indices by distance
- `np.where()` - Find cluster assignments
- `np.concatenate()` - Combine cluster labels

**Category:** Required

---

### scipy (implicit)

**What it is:** Scientific computing library.

**Why it's used:** Distance metric calculations.

**Files using it:**
| File | Usage |
|------|-------|
| `utils.py:7` | `from scipy import spatial` |

**Specific usage:**
```python
distance_metrics = {
    "cosine": spatial.distance.cosine,
    "L1": spatial.distance.cityblock,
    "L2": spatial.distance.euclidean,
    "Linf": spatial.distance.chebyshev,
}
```

**Category:** Required

---

### tenacity (8.2.3)

**What it is:** Retry library for Python.

**Why it's used:** Handles transient API failures with exponential backoff.

**Files using it:**
| File | Usage |
|------|-------|
| `tree_builder.py:11` | `from tenacity import retry, stop_after_attempt, wait_random_exponential` |
| `tree_retriever.py:6` | `from tenacity import retry, stop_after_attempt, wait_random_exponential` |
| `EmbeddingModels.py:6` | `from tenacity import retry, stop_after_attempt, wait_random_exponential` |
| `SummarizationModels.py:6` | `from tenacity import retry, stop_after_attempt, wait_random_exponential` |
| `QAModels.py:11` | `from tenacity import retry, stop_after_attempt, wait_random_exponential` |

**Pattern used:**
```python
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def api_call():
    ...
```

**Category:** Required

---

### tqdm (implicit)

**What it is:** Progress bar library.

**Why it's used:** Visual feedback during embedding generation.

**Files using it:**
| File | Usage |
|------|-------|
| `FaissRetriever.py:7` | `from tqdm import tqdm` |

**Category:** Optional (only for FaissRetriever)

---

## Dependency Categories

### Required Dependencies

These must be installed for basic RAPTOR functionality:

| Dependency | Reason |
|------------|--------|
| `numpy` | Core array operations |
| `scipy` | Distance calculations |
| `tiktoken` | Token counting |
| `tenacity` | API retry logic |
| `umap-learn` | Clustering algorithm |
| `scikit-learn` | GMM clustering |
| `openai` | Default models |

### Optional Dependencies

These are only needed for specific features:

| Dependency | Feature |
|------------|---------|
| `faiss-cpu` | FaissRetriever (non-hierarchical search) |
| `sentence-transformers` | SBertEmbeddingModel (local embeddings) |
| `torch` | UnifiedQAModel (local QA) |
| `transformers` | UnifiedQAModel (local QA) |
| `tqdm` | Progress bars in FaissRetriever |

---

## Dependency Weight Analysis

### Heavy Dependencies (>100MB installed)

| Dependency | Approx Size | Notes |
|------------|-------------|-------|
| `torch` | ~2GB | CPU/GPU, includes CUDA bindings |
| `transformers` | ~500MB | Plus model downloads |
| `sentence-transformers` | ~200MB | Plus model downloads |
| `faiss-cpu` | ~50MB | |

### Lightweight Dependencies (<10MB)

| Dependency | Approx Size |
|------------|-------------|
| `openai` | ~1MB |
| `tiktoken` | ~2MB |
| `tenacity` | ~100KB |
| `umap-learn` | ~5MB |
| `scikit-learn` | ~30MB |
| `numpy` | ~20MB |
| `scipy` | ~40MB |

---

## Transitive Dependencies

Key transitive dependencies that RAPTOR inherits:

```
openai
├── httpx
├── pydantic
└── anyio

sentence-transformers
├── torch
├── transformers
├── huggingface-hub
└── tokenizers

umap-learn
├── numba
├── pynndescent
└── scipy

scikit-learn
├── numpy
├── scipy
└── joblib
```

---

## Environment Considerations

### Minimum Python Version
- Python 3.8+ (based on demo.ipynb kernel)

### Environment Variables Required
| Variable | Purpose | Required When |
|----------|---------|---------------|
| `OPENAI_API_KEY` | OpenAI API access | Using OpenAI models (default) |
| `HF_TOKEN` | Hugging Face access | Using gated models (Gemma, etc.) |

### Hardware Considerations

| Feature | CPU | GPU |
|---------|-----|-----|
| OpenAI models | Yes | N/A (API) |
| SBert embeddings | Yes | Recommended |
| UnifiedQA | Possible | Recommended |
| UMAP clustering | Yes | N/A |
| FAISS search | Yes | Optional (faiss-gpu) |

---

## Minimal Installation Profiles

### Profile 1: OpenAI-Only (Minimal)

```txt
numpy==1.26.3
scipy
openai==1.3.3
tiktoken==0.5.1
tenacity==8.2.3
umap-learn==0.5.5
scikit-learn
```

**Use case:** Cloud-only setup using OpenAI for all ML tasks.

### Profile 2: Local Embeddings

```txt
# Profile 1 plus:
sentence-transformers==2.2.2
torch
```

**Use case:** Local embedding generation with OpenAI for LLM tasks.

### Profile 3: Fully Local

```txt
# Profile 2 plus:
transformers==4.38.1
```

**Use case:** Complete local inference without OpenAI dependency.

### Profile 4: Full Installation

```txt
# Profile 3 plus:
faiss-cpu
tqdm
```

**Use case:** All features including FAISS retriever.

---

## Version Pinning Concerns

| Dependency | Pinned Version | Latest (approx) | Risk |
|------------|----------------|-----------------|------|
| `numpy` | 1.26.3 | 2.0+ | Breaking changes in 2.0 |
| `openai` | 1.3.3 | 1.50+ | API changes likely |
| `transformers` | 4.38.1 | 4.40+ | Model compatibility |
| `umap-learn` | 0.5.5 | 0.5.5 | Current |
| `tiktoken` | 0.5.1 | 0.7+ | Minor updates |
| `urllib3` | 1.26.6 | 2.0+ | Security updates |

**Recommendations:**
1. Update `openai` - significant improvements since 1.3.3
2. Consider `urllib3>=2.0` for security
3. Pin `numpy<2.0` to avoid breaking changes
4. Test with latest `transformers` for model support
