# Modular Retrieval Pipeline

## What

A modular RAG pipeline for Kubernetes documentation with swappable components and Redis caching. Breaks down the monolithic `enriched_hybrid_llm` into testable, independent modules.

## Why

**Questions answered:**
1. Can modular architecture match 90% baseline accuracy?
2. How much faster with intelligent caching?
3. Can components be swapped without breaking the pipeline?

**Result:** Yes to all three. Modular achieves 90% accuracy with 85-90% speedup on warm cache.

## Setup

```bash
# Enter Nix shell
cd /home/fujin/Code/personal-library-manager
direnv allow

# Start Redis cache
cd poc/modular_retrieval_pipeline
docker compose up -d

# Verify
docker exec plm-redis-cache redis-cli PING  # Should output: PONG
```

## Usage

```bash
# Quick test (5 questions, with cache)
python benchmark.py --strategy modular-no-llm --quick

# Full benchmark
python benchmark.py --strategy modular-no-llm

# Without cache (slower, for testing)
python benchmark.py --no-cache --strategy modular-no-llm

# Compare all strategies
python benchmark.py --strategy all
```

**Available strategies:**
- `modular-no-llm` - No LLM rewriting (recommended: fast, 90% accuracy)
- `modular-llm` - With Claude Haiku rewriting (slower, same accuracy)
- `baseline` - Original monolithic implementation
- `all` - Run all three for comparison

**CLI options:**
- `--strategy [baseline|modular|modular-no-llm|all]` - Which strategy to run
- `--questions [path]` - Questions JSON file (default: needle_questions.json)
- `--no-cache` - Disable Redis caching
- `--quick` - Test with first 5 questions only
- `--output [path]` - Results JSON file

## File Structure

```
poc/modular_retrieval_pipeline/
├── components/                     # Modular retrieval components
│   ├── keyword_extractor.py        # YAKE keyword extraction
│   ├── entity_extractor.py         # spaCy NER
│   ├── content_enricher.py         # Combines keywords + entities
│   ├── query_expander.py           # Domain-specific query expansion
│   ├── query_rewriter.py           # LLM-based query rewriting (optional)
│   ├── embedding_encoder.py        # SentenceTransformer wrapper
│   ├── bm25_scorer.py              # Okapi BM25 lexical search
│   ├── similarity_scorer.py        # Cosine similarity semantic search
│   ├── rrf_fuser.py                # Reciprocal Rank Fusion
│   └── reranker.py                 # Cross-encoder reranking
├── cache/                          # Redis caching wrappers
│   ├── redis_client.py             # RedisCacheClient (graceful fallback)
│   ├── cached_keyword_extractor.py # Wraps KeywordExtractor
│   ├── cached_entity_extractor.py  # Wraps EntityExtractor
│   └── caching_embedder.py         # Wraps SentenceTransformer
├── utils/
│   └── logger.py                   # BenchmarkLogger (structured logging)
├── benchmark.py                    # Main benchmark runner
├── modular_enriched_hybrid.py      # No-LLM orchestrator
├── modular_enriched_hybrid_llm.py  # With-LLM orchestrator
├── docker-compose.yml              # Redis 7 Alpine service
├── BENCHMARK_RESULTS.md            # Detailed benchmark analysis
└── README.md                       # This file
```

## Adding Components

### 1. Create Component

Create `components/my_component.py`:

```python
class MyComponent:
    def __init__(self, config: dict):
        self.config = config
    
    def process(self, input_data):
        # Your logic here
        return result
```

### 2. Add Caching (if expensive)

Create `cache/cached_my_component.py`:

```python
from cache.redis_client import RedisCacheClient
from components.my_component import MyComponent

class CachedMyComponent:
    def __init__(self, component: MyComponent, cache: RedisCacheClient):
        self.component = component
        self.cache = cache
    
    def process(self, input_data):
        cache_key = self._make_cache_key(input_data)
        cached = self.cache.get(cache_key)
        if cached:
            return cached
        
        result = self.component.process(input_data)
        self.cache.set(cache_key, result)
        return result
    
    def _make_cache_key(self, input_data):
        config_hash = hashlib.md5(json.dumps(self.component.config).encode()).hexdigest()
        content_hash = hashlib.md5(str(input_data).encode()).hexdigest()
        return f"v1:my_component:{config_hash}:{content_hash}"
```

### 3. Integrate into Orchestrator

Update `modular_enriched_hybrid.py`:

```python
from components.my_component import MyComponent
from cache.cached_my_component import CachedMyComponent

class ModularEnrichedHybrid:
    def __init__(self, cache: RedisCacheClient = None):
        # Initialize component
        my_component = MyComponent(config={...})
        
        # Wrap with cache if available
        if cache and cache.is_connected():
            self.my_component = CachedMyComponent(my_component, cache)
        else:
            self.my_component = my_component
    
    def index(self, documents):
        for doc in documents:
            result = self.my_component.process(doc)
            # Use result...
```

### 4. Test

```bash
# Run benchmark to verify accuracy maintained
python benchmark.py --strategy modular-no-llm --quick
```

## Redis Cache

**Cache key format:** `v1:{component}:{config_hash}:{content_hash}`

**Cached operations:**
- Keyword extraction (YAKE) - ~100-200ms per document
- Entity extraction (spaCy NER) - ~50-150ms per document
- Embeddings (SentenceTransformer) - ~20-50ms per chunk

**Performance:**
- Cold cache (first run): 5-8 minutes
- Warm cache (subsequent runs): 30-60 seconds
- Speedup: **85-90% faster indexing**

**Cache commands:**
```bash
# Check cache size
docker exec plm-redis-cache redis-cli DBSIZE

# Clear cache
docker exec plm-redis-cache redis-cli FLUSHDB

# Stop Redis (cache persists in volume)
docker compose down

# Stop and delete cache
docker compose down -v
```

**Graceful fallback:** Pipeline works without Redis, just slower.

## Troubleshooting

### Redis Connection Failed
```bash
# Start Redis
docker compose up -d

# Check status
docker ps | grep plm-redis-cache
```

### Cache Not Working
```bash
# Verify Redis has data
docker exec plm-redis-cache redis-cli DBSIZE

# Check for cache keys
docker exec plm-redis-cache redis-cli KEYS "v1:*"
```

### Slow Performance
- First run is always slow (populating cache)
- Check `--no-cache` flag isn't set
- Verify Redis is running: `docker ps`

## Results

**Accuracy: 90.0% (18/20 questions)** - All strategies achieve same accuracy

| Strategy | Accuracy | Avg Latency | Notes |
|----------|----------|-------------|-------|
| baseline | 90.0% | 150ms | Original monolithic |
| modular-llm | 90.0% | 155ms | With LLM rewriting |
| modular-no-llm | 90.0% | 45ms | **Recommended** |

**Failed queries:**
- q_014: "What is the purpose of the kube-scheduler component?" (vocabulary mismatch)
- q_018: "How do you configure a Pod to use a specific ServiceAccount?" (vocabulary mismatch)

See [BENCHMARK_RESULTS.md](./BENCHMARK_RESULTS.md) for detailed analysis.

## Conclusion

**What we learned:**
1. **Modularity works** - Component architecture matches baseline accuracy
2. **Caching is essential** - 85-90% speedup makes iteration practical
3. **LLM is optional** - Query rewriting adds latency without improving accuracy
4. **Graceful degradation** - System works without Redis, just slower

**Next steps:**
- Investigate failed queries (vocabulary mismatch)
- Add reranking for better top-1 accuracy
- Experiment with different embedding models
