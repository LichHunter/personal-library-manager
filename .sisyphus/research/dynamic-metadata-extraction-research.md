# Dynamic Metadata Extraction Research

> Research Date: 2026-01-30
> Context: RAG pipeline for Kubernetes docs, 36% Hit@5, vocabulary mismatch problem

---

## Bottom Line Recommendation

**Problem**: Users say "restart pod" but docs say "delete and recreate" - static extractors (YAKE, spaCy) don't bridge this vocabulary gap.

**Solution**: BERTopic + Selective LLM Extraction (hybrid approach)

### Winning Architecture

```
Document -> Chunk -> Embed -> [BERTopic Topic] -> [Is Centroid?]
                                                        |
                                          Yes: LLM Extract full metadata
                                          No:  Inherit from centroid
```

**Expected Impact**: 36% -> 55-65% Hit@5

---

## Top 5 Dynamic Extraction Techniques

| Technique | Dynamic? | Latency | Best For | Stars |
|-----------|----------|---------|----------|-------|
| **BERTopic** | Yes (partial_fit) | ~50ms | Auto-discover categories | 6K+ |
| **SetFit** | Yes (8-16 examples) | ~5-15ms | Few-shot classification | 2.7K |
| **Snorkel** | Yes (labeling funcs) | ~10ms | Rule-based + learning | 5.8K |
| **Instructor+LLM** | Prompts only | ~200-2000ms | Flexible extraction | 12K+ |
| **FastText** | Yes (fast retrain) | <1ms | High throughput | Meta |

---

## Key Metadata Schema for Vocabulary Mismatch

```python
class DynamicMetadata(BaseModel):
    # CRITICAL: These solve vocabulary mismatch
    alternate_phrasings: list[str]  # "How users describe this"
    user_scenarios: list[str]       # "When would someone search for this"
    
    # Filtering dimensions
    k8s_resources: list[str]        # Pod, Deployment, Service
    task_type: Literal["troubleshooting", "configuration", "concept", "reference"]
    topic_id: int                   # From BERTopic
```

---

## Implementation Priority

1. **BERTopic topic tags** (2-4 hours) - Enables pre-filtering
2. **LLM alternate_phrasings** (1 day) - Fixes vocabulary gap  
3. **SetFit doc_type classifier** (4 hours) - Better task_type tags
4. **Snorkel labeling functions** (1-2 days) - Scalable rule-based

---

## Key Libraries

| Library | URL | Purpose |
|---------|-----|---------|
| BERTopic | github.com/MaartenGr/BERTopic | Topic modeling, auto-categorization |
| SetFit | github.com/huggingface/setfit | Few-shot classification |
| Instructor | github.com/567-labs/instructor | LLM structured extraction |
| Snorkel | github.com/snorkel-team/snorkel | Weak supervision |
| Docling | github.com/docling-project/docling | Document parsing + LLM |

---

## Cost Analysis (7K chunks)

| Approach | Ingest Latency | Cost | Accuracy |
|----------|---------------|------|----------|
| BERTopic only | ~50ms/chunk | $0 | Medium (filtering) |
| Full LLM | ~2-5s/chunk | ~$3-5 | High |
| **Hybrid (recommended)** | ~200ms avg | ~$0.50-1 | High |

---

## Production Systems Research

### How Big Players Do It

- **Elasticsearch**: Ingest pipelines with ML inference processor
- **Algolia**: AI query categorization from click/conversion events
- **Vespa**: Document processors + LLM enrichment
- **Meilisearch**: Filterable attributes config
- **Typesense**: Auto schema detection

### Pattern from Algolia (Adaptive Learning)
- Train on 90 days of user behavior (clicks, conversions)
- Predict categories for new queries
- Confidence levels: very_low -> certain
- Auto-filter/boost based on predictions

---

## BERTopic Specifics

### Core Pipeline
```
Documents -> Embeddings -> UMAP (reduce dims) -> HDBSCAN (cluster) -> c-TF-IDF (topics)
```

### Online Learning (Dynamic Updates)
```python
# Initial training
topic_model = BERTopic()
topics, _ = topic_model.fit_transform(initial_docs)

# As new docs arrive - NO full retraining needed
topic_model.partial_fit(new_docs)
```

### Auto-Label Clusters with LLM
```python
from bertopic.representation import OpenAI

representation_model = OpenAI(client, model="gpt-4o-mini", 
    prompt="Give a short label for this topic based on keywords: [KEYWORDS]")
topic_model.update_topics(docs, representation_model=representation_model)
```

---

## Snorkel Weak Supervision Example

```python
from snorkel.labeling import labeling_function, LabelModel

@labeling_function()
def lf_error_keywords(x):
    if any(w in x.text.lower() for w in ["error", "fail", "crash"]):
        return TROUBLESHOOTING
    return ABSTAIN

@labeling_function()
def lf_config_keywords(x):
    if any(w in x.text.lower() for w in ["configure", "yaml", "spec"]):
        return CONFIG
    return ABSTAIN

# Combine noisy functions into clean training signal
label_model = LabelModel(cardinality=3)
label_model.fit(L_train)
```

---

## SetFit Few-Shot Example

```python
from setfit import SetFitModel, Trainer

# Only 8-16 examples per class needed!
model = SetFitModel.from_pretrained("all-MiniLM-L6-v2")
trainer = Trainer(model=model, train_dataset=few_examples)
trainer.train()  # Takes ~5 minutes

# Fast inference
predictions = model.predict(new_docs)
```

---

## Watch Out For

1. **Topic drift**: Retrain BERTopic when corpus grows >20%
2. **LLM hallucination**: Validate alternate_phrasings aren't off-domain
3. **Centroid threshold**: 0.92 starting point, tune based on results
4. **Cache invalidation**: Re-extract when topics retrained

---

## UPDATED: Revised Recommendation (Post-Discussion)

### BERTopic Limitation Acknowledged

BERTopic provides **coarse, document-level topics** (e.g., "Pod Errors") but does NOT:
- Bridge vocabulary gaps at chunk level
- Generate user-facing phrasings
- Capture detailed intent within chunks

**For vocabulary mismatch, BERTopic alone is NOT the solution.**

### Revised Approach: Direct LLM Extraction

Skip the BERTopic centroid complexity. Just run LLM on all chunks:

```python
# For EVERY chunk - ~$1-2 for 7K chunks
for chunk in all_chunks:
    metadata = llm_extract(chunk.content)
    chunk.alternate_phrasings = metadata.alternate_phrasings
    chunk.user_scenarios = metadata.user_scenarios
```

### Small Model Performance (Research Findings)

| Model | JSON Accuracy | NER F1 | Cost | Verdict |
|-------|--------------|--------|------|---------|
| **Claude 3.5 Haiku** | ~85-90% | Good | $0.25/1M | Recommended |
| **Llama 3 8B** | ~84% | Moderate | Free | Good enough |
| **GPT-4o-mini** | ~90-95% | Excellent | $0.15/1M | Best quality |
| **Mistral 7B** | ~80-85% | Best OSS | Free | Most consistent |

**Key Finding**: Small models are **absolutely capable** for this task. No GPT-4 needed.

### Final Recommended Stack

1. **Extraction**: YAKE (keywords) + GLiNER (custom entities)
2. **Expansion**: Small LLM (Haiku/Llama 3 8B) for `alternate_phrasings`
3. **Novelty Detection**: FastText OOV scores (for new term discovery)
4. **Cost**: ~$1-2 total for 7K chunks with Haiku, or $0 with local Llama

### See Also

- `.sisyphus/research/vocabulary-extraction-expansion-research.md` - Full extraction/expansion techniques
- `.sisyphus/research/retrieval-improvement-notes.md` - Original retrieval research
