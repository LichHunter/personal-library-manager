# Vocabulary Extraction & Expansion Research

> Research Date: 2026-01-30
> Context: RAG pipeline for Kubernetes docs, adaptive vocabulary system

---

## Executive Summary

This research covers techniques for:
1. Extracting entities/keywords from documents
2. Building and expanding vocabulary dynamically when new documents/topics arrive
3. Small model (Haiku, Llama 3 8B) performance on metadata extraction

---

## Part 1: Keyword/Keyphrase Extraction Techniques

### Statistical Methods

| Method | Algorithm | GitHub | Stars | Can Discover NEW? |
|--------|-----------|--------|-------|-------------------|
| **RAKE** | Co-occurrence + degree/frequency | csurfer/rake-nltk | 1.1K | No |
| **YAKE** | 5 statistical features combined | LIAAD/yake | 1.5K | No |
| **TF-IDF** | Term frequency-inverse document frequency | sklearn | - | No |

**YAKE Example:**
```python
import yake
kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=10)
keywords = kw_extractor.extract_keywords(text)
```

### Graph-Based Methods

| Method | Algorithm | Library | Notes |
|--------|-----------|---------|-------|
| **TextRank** | PageRank on word co-occurrence graph | pytextrank | spaCy integration |
| **SingleRank** | Sliding window co-occurrence | pke | |
| **TopicRank** | Cluster candidates into topics first | pke | Better diversity |
| **PositionRank** | TextRank + position weighting | pytextrank | Good for news/papers |
| **MultipartiteRank** | Multipartite graph (no intra-topic edges) | pke | Best diversity |

**pytextrank Example:**
```python
import spacy
import pytextrank

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")
doc = nlp(text)
for phrase in doc._.phrases[:10]:
    print(phrase.text, phrase.rank)
```

### Embedding-Based Methods

| Method | Algorithm | GitHub | Stars |
|--------|-----------|--------|-------|
| **KeyBERT** | BERT embedding similarity + MMR | MaartenGr/KeyBERT | 4.1K |
| **EmbedRank** | Sent2Vec + MMR | swisscom/ai-research-keyphrase-extraction | 300 |
| **SIFRank** | ELMo + SIF weighting | sunyilgdx/SIFRank | 121 |

**KeyBERT Example:**
```python
from keybert import KeyBERT

kw_model = KeyBERT()
keywords = kw_model.extract_keywords(
    doc, 
    keyphrase_ngram_range=(1, 2),
    use_mmr=True,
    diversity=0.7
)
```

### LLM-Based Methods (Can Discover NEW Terms!)

| Method | Approach | Notes |
|--------|----------|-------|
| **KeyLLM** | KeyBERT + LLM | Part of KeyBERT library |
| **Zero-shot prompting** | Direct LLM extraction | Most flexible |
| **PromptRank** | T5/BART generation probability | State-of-art unsupervised |

**KeyLLM Example:**
```python
from keybert.llm import OpenAI
from keybert import KeyLLM

llm = OpenAI(client)
kw_model = KeyLLM(llm)
keywords = kw_model.extract_keywords(documents)
```

### All-in-One Library: PKE

```python
import pke

# Available: TfIdf, KPMiner, YAKE, TextRank, SingleRank, 
# TopicRank, TopicalPageRank, PositionRank, MultipartiteRank

extractor = pke.unsupervised.MultipartiteRank()
extractor.load_document(input=text, language='en')
extractor.candidate_selection()
extractor.candidate_weighting()
keyphrases = extractor.get_n_best(n=10)
```

---

## Part 2: Entity Extraction Techniques

### Zero-Shot NER (Can Extract ANY Entity Type!)

#### GLiNER (Recommended)
- **GitHub**: urchade/GLiNER (3.5K stars)
- **Key Feature**: Define entity types at runtime, no retraining

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# Define ANY custom labels!
labels = ["K8S_RESOURCE", "ERROR_TYPE", "COMMAND", "CONFIG_FIELD"]
entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(entity["text"], "=>", entity["label"])
```

#### UniversalNER
- **What**: Distilled from ChatGPT, 13K+ entity types
- **Model**: Universal-NER/UniNER-7B-type
- **Trade-off**: Larger (7B params), needs GPU

### Traditional NER

| Tool | Stars | Languages | Best For |
|------|-------|-----------|----------|
| **Flair** | 14.3K | Multi | High accuracy (94.36% F1) |
| **Stanza** | 7.7K | 23 langs | Multilingual |
| **HuggingFace** | - | Multi | Easy integration |

### Relation Extraction (Knowledge Triples)

#### REBEL
- **GitHub**: Babelscape/rebel (554 stars)
- **Output**: (subject, relation, object) triples

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text = "Punta Cana is a resort town in the Dominican Republic."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
# Output: [{'head': 'Punta Cana', 'type': 'country', 'tail': 'Dominican Republic'}]
```

---

## Part 3: Dynamic Dictionary/Vocabulary Expansion

### Automatic Term Recognition (ATR)

#### C-value/NC-value Method
- **Paper**: Frantzi, Ananiadou, Mima (700+ citations)
- **Algorithm**: Combines linguistic patterns + statistical frequency
- **Formula**: `C-value(a) = log2|a| * (f(a) - (1/P(Ta)) * sum(f(b)))`

**Python Implementation**: huanyannizu/C-Value-Term-Extraction

### Corpus-Based Expansion

#### FastText OOV Handling
- **Key Feature**: Handles never-seen words via character n-grams
- **Use Case**: Detect truly novel terms

```python
from gensim.models import FastText

model = FastText.load("domain_model.bin")

# Works even for OOV words!
similar = model.wv.most_similar("CrashLoopBackOff", topn=10)

# OOV detection via vector norm
vector = model.wv["new_term"]
norm = np.linalg.norm(vector)
# Low norm = poorly composed from subwords = truly novel
```

#### Word2Vec Neighbor Expansion
```python
# Find synonyms/related terms
similar_terms = model.wv.most_similar(
    positive=['kubernetes', 'container'], 
    topn=10
)
```

### Synonym Mining

#### WordNet
```python
from nltk.corpus import wordnet as wn

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemma_names():
            synonyms.add(lemma.replace('_', ' '))
    return synonyms
```

#### ConceptNet Numberbatch
- **GitHub**: commonsense/conceptnet-numberbatch (1.3K stars)
- **What**: Embeddings combining Word2Vec + knowledge graph relations

### Query Expansion (PRF/RM3)

```python
import pyterrier as pt

# Pseudo-relevance feedback pipeline
bm25 = pt.terrier.Retriever(index, wmodel="BM25")
qe = pt.terrier.QueryExpansion(index, fb_docs=3, fb_terms=10)
pipeline = bm25 >> qe >> bm25
```

### Production Systems Patterns

#### Elasticsearch Synonyms
```json
{
  "settings": {
    "analysis": {
      "filter": {
        "synonym_filter": {
          "type": "synonym_graph",
          "synonyms_path": "synonyms.txt",
          "updateable": true
        }
      }
    }
  }
}
```

#### Querqy (E-commerce Search)
- **GitHub**: querqy/querqy
- **Features**: SYNONYM, UP/DOWN boost, FILTER, DELETE

```
macbook pro =>
  SYNONYM: apple laptop
  FILTER: * brand:Apple
```

---

## Part 4: Small Model Performance (Haiku, Llama 3 8B)

### Benchmark Results

| Model | JSON Accuracy | NER F1 | Cost | Speed |
|-------|--------------|--------|------|-------|
| **Claude 3.5 Haiku** | ~85-90% | Good | $0.25/1M | ~50 tok/s |
| **Llama 3 8B** | ~84% | Moderate | Free | ~30 tok/s |
| **GPT-4o-mini** | ~90-95% | Excellent | $0.15/1M | Fast |
| **Mistral 7B** | ~80-85% | Best OSS | Free | ~35 tok/s |

### Key Findings

1. **Hermes 2 Pro Llama 3 8B**: 84% on structured JSON output
2. **Mistral 7B outperformed GPT-4o** on NER task (UBIAI research)
3. **Consistency**: Mistral returns consistent labels; GPT-4 varies between iterations

### Recommendations for Metadata Extraction

**Option 1: Claude 3.5 Haiku (Best Balance)**
```python
response = client.messages.create(
    model="claude-3-5-haiku-20241022",
    messages=[{
        "role": "user",
        "content": f"""Extract metadata:
- alternate_phrasings: 3-5 ways users might describe this
- user_scenarios: 2-3 situations when someone would search
- task_type: troubleshooting|configuration|concept|reference

Chunk: {chunk_text[:3000]}
JSON:"""
    }]
)
```

**Option 2: Local Llama 3 8B with Ollama (Free)**
```python
response = ollama.chat(
    model="llama3:8b-instruct-q8_0",
    messages=[{"role": "user", "content": prompt}],
    format="json"  # Guaranteed JSON
)
```

**Option 3: Outlines for Guaranteed JSON**
```python
from outlines import models, generate
from pydantic import BaseModel

class ChunkMetadata(BaseModel):
    alternate_phrasings: list[str]
    user_scenarios: list[str]
    task_type: str

model = models.transformers("meta-llama/Meta-Llama-3-8B-Instruct")
generator = generate.json(model, ChunkMetadata)
metadata = generator(prompt)  # 100% valid JSON
```

---

## Part 5: Adaptive Vocabulary Architecture

### Complete Pipeline

```
NEW DOCUMENT
     │
     ▼
┌─────────────┐
│ 1. EXTRACT  │  YAKE keywords + GLiNER entities (fast)
└─────────────┘
     │
     ▼
┌─────────────┐
│ 2. DETECT   │  Is this NEW? (FastText OOV + embedding distance)
│   NOVELTY   │
└─────────────┘
     │
     ▼
┌─────────────┐
│ 3. EXPAND   │  LLM alternate_phrasings + Word2Vec neighbors
└─────────────┘
     │
     ▼
┌─────────────┐
│ 4. INDEX    │  Add to searchable vocabulary
└─────────────┘
```

### Novelty Detection Code

```python
class NoveltyDetector:
    def is_novel(self, term: str) -> tuple[bool, float]:
        # Check 1: In known vocabulary?
        if term.lower() in self.known_vocabulary:
            return False, 0.0
        
        # Check 2: FastText OOV score
        vector = self.model.wv[term]
        norm = np.linalg.norm(vector)
        oov_score = 1.0 - min(norm / 10.0, 1.0)
        
        # Check 3: Distance from nearest known term
        similar = self.model.wv.most_similar(term, topn=1)
        distance_score = 1.0 - similar[0][1]
        
        novelty_score = (oov_score + distance_score) / 2
        return novelty_score > 0.5, novelty_score
```

### Vocabulary Expansion Code

```python
def expand_term(self, term: str, context: str) -> dict:
    # Method 1: Word2Vec neighbors (fast)
    synonyms = [w for w, s in self.word2vec.most_similar(term, topn=10) if s > 0.7]
    
    # Method 2: LLM for user phrasings (the key)
    llm_response = ollama.chat(
        model="llama3:8b-instruct-q8_0",
        messages=[{
            "role": "user",
            "content": f"""Term: {term}
Context: {context[:500]}

Generate JSON:
{{"phrasings": ["how users would Google this"], "related": ["technical terms"]}}"""
        }],
        format="json"
    )
    
    return {
        "synonyms": synonyms,
        "user_phrasings": llm_response["phrasings"],
        "related": llm_response["related"]
    }
```

---

## Key Libraries Summary

| Category | Tool | Stars | Install |
|----------|------|-------|---------|
| **Keywords** | YAKE | 1.5K | `pip install yake` |
| **Keywords** | KeyBERT | 4.1K | `pip install keybert` |
| **Keywords** | pke | 1.6K | `pip install git+https://github.com/boudinfl/pke` |
| **NER** | GLiNER | 3.5K | `pip install gliner` |
| **NER** | Flair | 14.3K | `pip install flair` |
| **Relations** | REBEL | 554 | HuggingFace |
| **Embeddings** | FastText | Gensim | `pip install gensim` |
| **Synonyms** | ConceptNet | 1.3K | API/Numberbatch |
| **Query Exp** | PyTerrier | 400+ | `pip install python-terrier` |
| **Structured** | Outlines | - | `pip install outlines` |

---

## Bottom Line Recommendations

### For Your K8s RAG Pipeline

1. **Extraction**: YAKE (fast keywords) + GLiNER (custom entity types)
2. **Novelty Detection**: FastText OOV scores + embedding distance
3. **Expansion**: Small LLM (Haiku/Llama 3 8B) for `alternate_phrasings`
4. **Cost**: ~$1-2 for 7K chunks with Haiku, or free with local Llama

### The Key Insight

> The vocabulary mismatch problem (36% Hit@5) is solved by storing **user-facing phrasings** alongside technical terms. When you see "CrashLoopBackOff", generate and index: "pod keeps restarting", "container crash loop", "app won't stay up".

This is what small models excel at - no GPT-4 needed.
