# RAPTOR Error Handling

> Complete analysis of validation, exceptions, API error handling, and recovery strategies

## Overview

RAPTOR's error handling can be categorized into:

1. **Configuration Validation** - Errors raised during config instantiation
2. **Runtime Validation** - Errors raised during method execution
3. **API Error Handling** - How external API failures are managed
4. **Retry Logic** - Automatic retry mechanisms
5. **Recovery Strategies** - What happens when operations fail

---

## 1. Configuration Validation

### TreeBuilderConfig Validation

**Location:** `raptor/tree_builder.py:24-103`

| Parameter | Validation | Exception |
|-----------|------------|-----------|
| `max_tokens` | Must be int >= 1 | `ValueError("max_tokens must be an integer and at least 1")` |
| `num_layers` | Must be int >= 1 | `ValueError("num_layers must be an integer and at least 1")` |
| `threshold` | Must be int/float in [0, 1] | `ValueError("threshold must be a number between 0 and 1")` |
| `top_k` | Must be int >= 1 | `ValueError("top_k must be an integer and at least 1")` |
| `selection_mode` | Must be "top_k" or "threshold" | `ValueError("selection_mode must be either 'top_k' or 'threshold'")` |
| `summarization_model` | Must be BaseSummarizationModel instance | `ValueError("summarization_model must be an instance of BaseSummarizationModel")` |
| `embedding_models` | Must be dict | `ValueError("embedding_models must be a dictionary of model_name: instance pairs")` |
| `embedding_models.values()` | Must all be BaseEmbeddingModel | `ValueError("All embedding models must be an instance of BaseEmbeddingModel")` |
| `cluster_embedding_model` | Must be key in embedding_models | `ValueError("cluster_embedding_model must be a key in the embedding_models dictionary")` |

```python
# Example validation code from source
if not isinstance(max_tokens, int) or max_tokens < 1:
    raise ValueError("max_tokens must be an integer and at least 1")

if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
    raise ValueError("threshold must be a number between 0 and 1")
```

### ClusterTreeConfig Validation

**Location:** `raptor/cluster_tree_builder.py:17-38`

Inherits all TreeBuilderConfig validation. Additional parameters (`reduction_dimension`, `clustering_algorithm`, `clustering_params`) have **no explicit validation** - they are accepted as-is.

### TreeRetrieverConfig Validation

**Location:** `raptor/tree_retriever.py:19-103`

| Parameter | Validation | Exception |
|-----------|------------|-----------|
| `threshold` | Must be float in [0, 1] | `ValueError("threshold must be a float between 0 and 1")` |
| `top_k` | Must be int >= 1 | `ValueError("top_k must be an integer and at least 1")` |
| `selection_mode` | Must be string "top_k" or "threshold" | `ValueError("selection_mode must be a string and either 'top_k' or 'threshold'")` |
| `context_embedding_model` | Must be string | `ValueError("context_embedding_model must be a string")` |
| `embedding_model` | Must be BaseEmbeddingModel | `ValueError("embedding_model must be an instance of BaseEmbeddingModel")` |
| `num_layers` | If not None, must be int >= 0 | `ValueError("num_layers must be an integer and at least 0")` |
| `start_layer` | If not None, must be int >= 0 | `ValueError("start_layer must be an integer and at least 0")` |

**Note:** `threshold` validation is stricter here (must be `float`) vs TreeBuilderConfig (allows `int` or `float`).

### FaissRetrieverConfig Validation

**Location:** `raptor/FaissRetriever.py:14-56`

| Parameter | Validation | Exception |
|-----------|------------|-----------|
| `max_tokens` | Must be >= 1 | `ValueError("max_tokens must be at least 1")` |
| `top_k` | Must be >= 1 | `ValueError("top_k must be at least 1")` |
| `max_context_tokens` | If not None, must be >= 1 | `ValueError("max_context_tokens must be at least 1 or None")` |
| `embedding_model` | If not None, must be BaseEmbeddingModel | `ValueError("embedding_model must be an instance of BaseEmbeddingModel or None")` |
| `question_embedding_model` | If not None, must be BaseEmbeddingModel | `ValueError("question_embedding_model must be an instance of BaseEmbeddingModel or None")` |

### RetrievalAugmentationConfig Validation

**Location:** `raptor/RetrievalAugmentation.py:18-150`

| Parameter | Validation | Exception |
|-----------|------------|-----------|
| `tree_builder_type` | Must be in supported_tree_builders | `ValueError("tree_builder_type must be one of ['cluster']")` |
| `qa_model` | If not None, must be BaseQAModel | `ValueError("qa_model must be an instance of BaseQAModel")` |
| `embedding_model` | If not None, must be BaseEmbeddingModel | `ValueError("embedding_model must be an instance of BaseEmbeddingModel")` |
| `embedding_model` + `tb_embedding_models` | Can't specify both | `ValueError("Only one of 'tb_embedding_models' or 'embedding_model' should be provided, not both.")` |
| `summarization_model` | If not None, must be BaseSummarizationModel | `ValueError("summarization_model must be an instance of BaseSummarizationModel")` |
| `summarization_model` + `tb_summarization_model` | Can't specify both | `ValueError("Only one of 'tb_summarization_model' or 'summarization_model' should be provided, not both.")` |
| `tree_builder_config` | If provided, must match tree_builder_type | `ValueError("tree_builder_config must be a direct instance of {class} for tree_builder_type '{type}'"` |
| `tree_retriever_config` | If provided, must be TreeRetrieverConfig | `ValueError("tree_retriever_config must be an instance of TreeRetrieverConfig")` |

---

## 2. Runtime Validation

### TreeRetriever Initialization

**Location:** `raptor/tree_retriever.py:106-131`

```python
class TreeRetriever(BaseRetriever):
    def __init__(self, config, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        if config.num_layers is not None and config.num_layers > tree.num_layers + 1:
            raise ValueError(
                "num_layers in config must be less than or equal to tree.num_layers + 1"
            )

        if config.start_layer is not None and config.start_layer > tree.num_layers:
            raise ValueError(
                "start_layer in config must be less than or equal to tree.num_layers"
            )
        
        # After setting defaults:
        if self.num_layers > self.start_layer + 1:
            raise ValueError("num_layers must be less than or equal to start_layer + 1")
```

### ClusterTreeBuilder Initialization

**Location:** `raptor/cluster_tree_builder.py:42-53`

```python
class ClusterTreeBuilder(TreeBuilder):
    def __init__(self, config) -> None:
        super().__init__(config)

        if not isinstance(config, ClusterTreeConfig):
            raise ValueError("config must be an instance of ClusterTreeConfig")
```

### RetrievalAugmentation Initialization

**Location:** `raptor/RetrievalAugmentation.py:159-198`

```python
def __init__(self, config=None, tree=None):
    if config is None:
        config = RetrievalAugmentationConfig()
    if not isinstance(config, RetrievalAugmentationConfig):
        raise ValueError("config must be an instance of RetrievalAugmentationConfig")

    # Tree loading validation
    if isinstance(tree, str):
        try:
            with open(tree, "rb") as file:
                self.tree = pickle.load(file)
            if not isinstance(self.tree, Tree):
                raise ValueError("The loaded object is not an instance of Tree")
        except Exception as e:
            raise ValueError(f"Failed to load tree from {tree}: {e}")
    elif isinstance(tree, Tree) or tree is None:
        self.tree = tree
    else:
        raise ValueError(
            "tree must be an instance of Tree, a path to a pickled Tree, or None"
        )
```

### TreeRetriever.retrieve() Validation

**Location:** `raptor/tree_retriever.py:252-300`

```python
def retrieve(self, query, start_layer=None, num_layers=None, ...):
    if not isinstance(query, str):
        raise ValueError("query must be a string")

    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be an integer and at least 1")

    if not isinstance(collapse_tree, bool):
        raise ValueError("collapse_tree must be a boolean")

    if not isinstance(start_layer, int) or not (0 <= start_layer <= self.tree.num_layers):
        raise ValueError("start_layer must be an integer between 0 and tree.num_layers")

    if not isinstance(num_layers, int) or num_layers < 1:
        raise ValueError("num_layers must be an integer and at least 1")

    if num_layers > (start_layer + 1):
        raise ValueError("num_layers must be less than or equal to start_layer + 1")
```

### RetrievalAugmentation.retrieve() Validation

**Location:** `raptor/RetrievalAugmentation.py:248-251`

```python
def retrieve(self, question, ...):
    if self.retriever is None:
        raise ValueError(
            "The TreeRetriever instance has not been initialized. Call 'add_documents' first."
        )
```

### Tree Saving Validation

**Location:** `raptor/RetrievalAugmentation.py:301-306`

```python
def save(self, path):
    if self.tree is None:
        raise ValueError("There is no tree to save.")
```

---

## 3. API Error Handling

### OpenAI Embedding Model

**Location:** `raptor/EmbeddingModels.py:17-29`

```python
class OpenAIEmbeddingModel(BaseEmbeddingModel):
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )
```

**Error Handling:**
- Uses `tenacity` retry decorator
- **NO try/except** - exceptions propagate after 6 failed attempts
- Any OpenAI API error will raise after exhausting retries

### OpenAI Summarization Models

**Location:** `raptor/SummarizationModels.py:17-44, 47-74`

```python
class GPT3TurboSummarizationModel(BaseSummarizationModel):
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        try:
            client = OpenAI()
            response = client.chat.completions.create(...)
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return e  # WARNING: Returns exception object, not string!
```

**Error Handling:**
- Retry decorator applied
- **Returns exception object on failure** (problematic - see issues below)
- Prints error to stdout

### OpenAI QA Models

**Location:** `raptor/QAModels.py`

#### GPT3QAModel

```python
class GPT3QAModel(BaseQAModel):
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, ...):
        try:
            response = self.client.completions.create(...)
            return response.choices[0].text.strip()
        except Exception as e:
            print(e)
            return ""  # Returns empty string on failure
```

#### GPT3TurboQAModel / GPT4QAModel

```python
class GPT3TurboQAModel(BaseQAModel):
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context, question, ...):
        response = self.client.chat.completions.create(...)
        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question, ...):
        try:
            return self._attempt_answer_question(...)
        except Exception as e:
            print(e)
            return e  # WARNING: Returns exception object!
```

**Note:** Has **double retry decorators** - both on `_attempt_answer_question` and `answer_question`.

---

## 4. Retry Logic

### Tenacity Configuration

All OpenAI-based models use the same retry configuration:

```python
from tenacity import retry, stop_after_attempt, wait_random_exponential

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def api_call():
    ...
```

| Parameter | Value | Effect |
|-----------|-------|--------|
| `wait` | `wait_random_exponential(min=1, max=20)` | Random exponential backoff between 1-20 seconds |
| `stop` | `stop_after_attempt(6)` | Maximum 6 attempts total |

### Total Maximum Wait Time

With exponential backoff:
- Attempt 1: Immediate
- Attempt 2: Wait 1-20s (random)
- Attempt 3: Wait 1-20s
- Attempt 4: Wait 1-20s
- Attempt 5: Wait 1-20s
- Attempt 6: Wait 1-20s

**Maximum total wait:** ~100 seconds (5 waits * 20s max)
**Minimum total wait:** ~5 seconds (5 waits * 1s min)

### Which Methods Have Retry Logic

| Class | Method | Has Retry |
|-------|--------|-----------|
| `OpenAIEmbeddingModel` | `create_embedding` | Yes |
| `SBertEmbeddingModel` | `create_embedding` | No (local model) |
| `GPT3TurboSummarizationModel` | `summarize` | Yes |
| `GPT3SummarizationModel` | `summarize` | Yes |
| `GPT3QAModel` | `answer_question` | Yes |
| `GPT3TurboQAModel` | `answer_question` | Yes (double!) |
| `GPT3TurboQAModel` | `_attempt_answer_question` | Yes |
| `GPT4QAModel` | `answer_question` | Yes (double!) |
| `GPT4QAModel` | `_attempt_answer_question` | Yes |
| `UnifiedQAModel` | `answer_question` | No (local model) |

---

## 5. What Happens When Operations Fail

### Embedding Failures

**If `OpenAIEmbeddingModel.create_embedding()` fails:**

1. Retries 6 times with exponential backoff
2. After 6 failures, exception propagates up
3. **Impact on tree building:**
   - `create_node()` fails
   - `multithreaded_create_leaf_nodes()` raises exception
   - Tree building stops completely
   - No partial tree is created

```python
# In tree_builder.py
def create_node(self, index, text, children_indices=None):
    embeddings = {
        model_name: model.create_embedding(text)  # Can raise!
        for model_name, model in self.embedding_models.items()
    }
    return (index, Node(text, index, children_indices, embeddings))
```

### Summarization Failures

**If summarization fails:**

```python
# In SummarizationModels.py
def summarize(self, context, max_tokens=500, stop_sequence=None):
    try:
        # ... API call ...
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        return e  # Returns exception object!
```

**Impact on tree building:**

1. `summarize()` returns the exception object (not a string)
2. `create_node()` is called with exception as text
3. Embedding is created for the exception's string representation
4. **Tree building continues with corrupted node!**

```python
# In cluster_tree_builder.py
summarized_text = self.summarize(context=node_texts, max_tokens=summarization_length)
# summarized_text could be an Exception object!

__, new_parent_node = self.create_node(
    next_node_index, 
    summarized_text,  # Exception object passed as text
    {node.index for node in cluster}
)
```

### QA Model Failures

**If QA fails:**

```python
# GPT3QAModel returns empty string
return ""

# GPT3TurboQAModel/GPT4QAModel return exception object
return e
```

**Impact:**
- User receives empty string or exception object as "answer"
- No indication that failure occurred (besides printed error)

---

## 6. Known Issues and Gaps

### Issue 1: Exception Objects Returned as Values

**Problem:** Summarization and QA models return exception objects instead of raising them:

```python
except Exception as e:
    print(e)
    return e  # This is problematic!
```

**Impact:**
- Type inconsistency (returns Exception instead of str)
- Silent failures - caller doesn't know operation failed
- Corrupted tree nodes with exception text as content

**Recommendation:** Either raise the exception or return a consistent error value.

### Issue 2: No Input Validation for Text Content

**Problem:** No validation on input text to `add_documents()` or queries:

```python
def add_documents(self, docs):
    # No validation on docs content
    self.tree = self.tree_builder.build_from_text(text=docs)
```

**Missing validations:**
- Empty string check
- None check
- Type check (must be string)

### Issue 3: Double Retry Decorators

**Problem:** `GPT3TurboQAModel` and `GPT4QAModel` have retry on both `_attempt_answer_question` AND `answer_question`:

```python
@retry(...)
def _attempt_answer_question(self, ...):
    ...

@retry(...)  # Second retry decorator!
def answer_question(self, ...):
    return self._attempt_answer_question(...)
```

**Impact:** Potentially 36 total attempts (6 * 6) before final failure.

### Issue 4: No Graceful Degradation

**Problem:** Partial failures don't allow recovery:

- If 1 of 100 embeddings fails, entire tree build fails
- No checkpoint/resume capability
- No option to skip failed nodes

### Issue 5: Inconsistent Error Return Types

| Model | On Failure Returns |
|-------|-------------------|
| `GPT3QAModel` | `""` (empty string) |
| `GPT3TurboQAModel` | Exception object |
| `GPT4QAModel` | Exception object |
| `GPT3TurboSummarizationModel` | Exception object |
| `GPT3SummarizationModel` | Exception object |
| `OpenAIEmbeddingModel` | Raises exception |

### Issue 6: Best Indices Potentially Unbound

**Location:** `tree_retriever.py:234` and `tree_builder.py:234`

```python
if self.selection_mode == "threshold":
    best_indices = [...]
elif self.selection_mode == "top_k":
    best_indices = indices[: self.top_k]

# best_indices used here - but what if selection_mode is neither?
nodes_to_add = [list_nodes[idx] for idx in best_indices]  # Potentially unbound!
```

**Impact:** If `selection_mode` is corrupted, `best_indices` would be unbound.

---

## 7. Error Handling Matrix

| Operation | Validation | Retry | On Failure |
|-----------|------------|-------|------------|
| Config creation | Yes (ValueError) | No | Raises immediately |
| Load pickled tree | Yes | No | Raises ValueError |
| Create embedding (OpenAI) | No | Yes (6x) | Raises after retries |
| Create embedding (SBERT) | No | No | Raises immediately |
| Summarize (OpenAI) | No | Yes (6x) | Returns exception object |
| Answer question (GPT3) | No | Yes (6x) | Returns empty string |
| Answer question (GPT3Turbo/4) | No | Yes (12x) | Returns exception object |
| Build tree | No | No | Raises on first failure |
| Retrieve | Yes | No | Raises on validation failure |

---

## 8. Implementing Robust Error Handling

### Recommended Custom Embedding Model

```python
from raptor import BaseEmbeddingModel
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

class RobustEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.client = ...  # Your client
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True  # Important: reraise the exception
    )
    def create_embedding(self, text: str):
        if not text or not isinstance(text, str):
            logger.warning("Invalid input text, using zero vector")
            return [0.0] * self.embedding_dim
        
        try:
            return self._call_api(text)
        except Exception as e:
            logger.error(f"Embedding failed for text (first 100 chars): {text[:100]}")
            raise  # Reraise to trigger retry
```

### Recommended Custom Summarization Model

```python
from raptor import BaseSummarizationModel
import logging

logger = logging.getLogger(__name__)

class RobustSummarizationModel(BaseSummarizationModel):
    def __init__(self, fallback_prefix: str = "[SUMMARIZATION FAILED] "):
        self.fallback_prefix = fallback_prefix
    
    def summarize(self, context: str, max_tokens: int = 150) -> str:
        """Always returns a string, never an exception."""
        if not context:
            return ""
        
        try:
            return self._call_api(context, max_tokens)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Return truncated original instead of failing
            truncated = context[:max_tokens * 4]  # Rough token estimate
            return f"{self.fallback_prefix}{truncated}"
```

### Recommended Custom QA Model

```python
from raptor import BaseQAModel
import logging

logger = logging.getLogger(__name__)

class RobustQAModel(BaseQAModel):
    def __init__(self, error_response: str = "I was unable to answer this question."):
        self.error_response = error_response
    
    def answer_question(self, context: str, question: str) -> str:
        """Always returns a string response."""
        if not context or not question:
            return self.error_response
        
        try:
            return self._call_api(context, question)
        except Exception as e:
            logger.error(f"QA failed for question: {question[:100]}")
            return self.error_response
```

---

## 9. Summary

### Validation Coverage

| Component | Configuration Validation | Runtime Validation |
|-----------|-------------------------|-------------------|
| TreeBuilderConfig | Comprehensive | N/A |
| ClusterTreeConfig | Inherits + minimal | Checks config type |
| TreeRetrieverConfig | Comprehensive | Tree compatibility |
| FaissRetrieverConfig | Comprehensive | N/A |
| RetrievalAugmentationConfig | Comprehensive | Tree loading |
| TreeRetriever.retrieve() | N/A | All parameters |
| RetrievalAugmentation | N/A | Retriever initialized |

### Key Takeaways

1. **Configuration validation is thorough** - Most invalid configs raise `ValueError` immediately
2. **API error handling is inconsistent** - Some return exceptions, some empty strings
3. **Retry logic exists but is flawed** - Double decorators, inconsistent retry counts
4. **No graceful degradation** - Single failure stops entire operation
5. **Type safety issues** - Exception objects returned where strings expected

### Recommended Improvements for Production

1. Implement consistent error return types
2. Add input validation for text content
3. Remove double retry decorators
4. Add checkpoint/resume for large documents
5. Implement graceful degradation (skip failed nodes)
6. Add structured logging instead of print statements
7. Consider circuit breaker pattern for API calls
