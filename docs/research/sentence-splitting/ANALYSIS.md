# Sentence Splitting Analysis for Kubernetes Documentation

## Executive Summary

**Finding**: No existing sentence splitting utilities found in the codebase. The POC-1b project focuses on **term extraction** from documentation chunks, not sentence-level splitting.

**Recommendation**: Use **regex-based splitting with markdown-aware preprocessing** for Kubernetes documentation, with special handling for code blocks, lists, and technical syntax.

---

## 1. Current State of the Codebase

### Existing Libraries Already Imported

| Library | Purpose | Used In |
|---------|---------|---------|
| `spacy` | NLP/NER (not for sentence splitting) | `test_pattern_plus_llm.py`, `test_hybrid_ner.py` |
| `rapidfuzz` | Fuzzy string matching | Multiple test files |
| `instructor` | Structured LLM output validation | `run_experiment.py` |
| `pydantic` | Data validation | `run_experiment.py` |
| `re` (regex) | Text pattern matching | All files |
| `sentence-transformers` | Semantic embeddings (not splitting) | `pyproject.toml` |

**Key Finding**: `nltk` is **NOT** imported anywhere. `spacy` is imported but used for NER, not sentence tokenization.

### Text Processing Patterns Found

#### 1. **Markdown Parsing** (test_small_chunk_extraction.py)
```python
def extract_sections_simple(content: str, max_section_words: int = 300) -> list[dict]:
    """Simple section extraction from markdown content."""
    # Split by headings
    heading_pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    # ... extracts sections between headings
```

**Edge Cases Handled**:
- Heading levels (H1-H4)
- Content before first heading (intro section)
- Minimum word count (15+ words per section)
- Preserves heading in content

#### 2. **JSON Response Parsing** (run_experiment.py)
```python
def parse_json_response(response: str) -> list[dict]:
    """Parse JSON response, handling markdown code blocks."""
    # Remove markdown code blocks
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    # ... extracts JSON from markdown
```

**Edge Cases Handled**:
- Markdown code blocks (```json ... ```)
- JSON objects and arrays
- Malformed JSON recovery

#### 3. **Text Normalization** (multiple files)
```python
# CamelCase splitting
camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", term).lower()

# Token splitting
ext_tokens = set(ext_norm.split())
gt_tokens = set(gt_norm.split())
```

**Edge Cases Handled**:
- CamelCase terms (e.g., `PersistentVolume` → `persistent volume`)
- Whitespace normalization
- Token-level comparison

---

## 2. Kubernetes Documentation Specifics

### Special Cases to Handle

#### Code Blocks
```markdown
# Example
Here's how to create a Pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
  - name: app
    image: nginx:latest
```

This creates a Pod resource.
```

**Issue**: Sentences inside code blocks should NOT be split (they're YAML/JSON, not prose).

#### Lists
```markdown
## Features

- **Pod**: Basic compute unit
- **Service**: Network abstraction
- **Deployment**: Declarative updates

Each provides different capabilities.
```

**Issue**: List items may not be complete sentences. Should preserve list structure.

#### Technical Syntax
```markdown
The `kubectl apply -f deployment.yaml` command deploys resources.
The `spec.containers[0].image` field specifies the container image.
```

**Issue**: Inline code (backticks) and field paths contain periods/dots that aren't sentence boundaries.

#### Abbreviations
```markdown
Kubernetes (K8s) uses etcd for state management.
The kube-apiserver (API server) handles requests.
```

**Issue**: Abbreviations with periods (e.g., "K8s.") shouldn't trigger sentence splits.

---

## 3. Recommended Approach

### Option 1: **Regex-Based Splitting (RECOMMENDED)**

**Pros**:
- No external dependencies (uses `re` already in codebase)
- Fast and deterministic
- Easy to customize for Kubernetes domain
- Handles markdown preprocessing

**Cons**:
- Requires careful regex tuning
- May miss edge cases

**Implementation**:
```python
import re

def split_sentences_kubernetes(text: str) -> list[str]:
    """Split text into sentences, handling K8s documentation edge cases."""
    
    # 1. Preserve code blocks
    code_blocks = []
    def preserve_code(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"
    
    text = re.sub(r'```[\s\S]*?```', preserve_code, text)
    text = re.sub(r'`[^`]+`', preserve_code, text)
    
    # 2. Preserve inline code and field paths
    text = re.sub(r'(\w+\.\w+)+', lambda m: f"__FIELD_{len(code_blocks)}__", text)
    code_blocks.append(m.group(0))
    
    # 3. Split on sentence boundaries
    # Match: period/question/exclamation + space + capital letter
    # BUT NOT: abbreviations (K8s, etc.), decimal numbers, URLs
    sentences = re.split(
        r'(?<![A-Z])\.(?=\s+[A-Z])|'  # Period + space + capital
        r'\?(?=\s+[A-Z])|'              # Question mark
        r'!(?=\s+[A-Z])',               # Exclamation mark
        text
    )
    
    # 4. Restore code blocks
    result = []
    for sent in sentences:
        for i, block in enumerate(code_blocks):
            sent = sent.replace(f"__CODE_BLOCK_{i}__", block)
            sent = sent.replace(f"__FIELD_{i}__", block)
        result.append(sent.strip())
    
    return [s for s in result if s]
```

### Option 2: **NLTK Sentence Tokenizer**

**Pros**:
- Battle-tested, handles many edge cases
- Trained on English text
- Handles abbreviations well

**Cons**:
- Adds dependency (not currently in project)
- Slower than regex
- May not understand K8s-specific abbreviations

**Implementation**:
```python
import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def split_sentences_nltk(text: str) -> list[str]:
    """Split using NLTK (requires punkt model)."""
    return sent_tokenize(text)
```

### Option 3: **SpaCy Sentence Segmentation**

**Pros**:
- Already in dependencies (`spacy>=3.7.0`)
- Handles complex cases
- Can be customized with rules

**Cons**:
- Requires language model
- Slower than regex
- Overkill for simple splitting

**Implementation**:
```python
import spacy

nlp = spacy.load("en_core_web_sm")

def split_sentences_spacy(text: str) -> list[str]:
    """Split using SpaCy."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]
```

---

## 4. Comparison Matrix

| Approach | Speed | Accuracy | K8s-Aware | Dependencies | Maintenance |
|----------|-------|----------|-----------|--------------|-------------|
| **Regex** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | None | Medium |
| **NLTK** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | New dep | Low |
| **SpaCy** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Already have | Low |

---

## 5. Edge Cases to Handle

### Critical for Kubernetes Docs

| Case | Example | Solution |
|------|---------|----------|
| **Code blocks** | ```yaml ... ``` | Preserve before splitting |
| **Inline code** | `kubectl apply` | Preserve before splitting |
| **Field paths** | `spec.containers[0].image` | Don't split on dots |
| **Abbreviations** | K8s, CRD, HPA | Maintain abbreviation list |
| **URLs** | `https://example.com` | Don't split on dots |
| **Lists** | `- Item 1\n- Item 2` | Preserve list structure |
| **Decimal numbers** | `1.2.3` (version) | Don't split on dots |
| **Camel case** | `PersistentVolume` | Optional: split for indexing |

---

## 6. Existing Code Patterns in POC-1b

### How Text is Currently Processed

1. **Chunk-level extraction** (not sentence-level):
   - Documents split into sections by headings
   - Sections are 300+ words (not sentences)
   - Terms extracted from entire sections

2. **Markdown handling**:
   - Code blocks removed before JSON parsing
   - Headings preserved in content
   - No sentence-level processing

3. **Text normalization**:
   - CamelCase splitting for matching
   - Whitespace normalization
   - Token-level comparison

### Implication
The codebase is **section-aware but not sentence-aware**. Sentence splitting would be a new capability.

---

## 7. Recommendation

### Use Case Matters

**If you need sentence splitting for**:
- **Indexing/search**: Use **regex-based splitting** with K8s-specific rules
- **LLM context windows**: Use **SpaCy** (already available)
- **General NLP**: Use **NLTK** (add dependency)

### Recommended Implementation

```python
# utils/sentence_splitter.py

import re
from typing import List

class KubernetesSentenceSplitter:
    """Sentence splitter aware of K8s documentation conventions."""
    
    # K8s abbreviations that shouldn't trigger splits
    K8S_ABBREVIATIONS = {
        'K8s', 'CRD', 'HPA', 'PVC', 'PV', 'RBAC', 'YAML', 'JSON',
        'API', 'HTTP', 'HTTPS', 'DNS', 'IP', 'TCP', 'UDP', 'TLS',
        'SSL', 'ETCD', 'OIDC', 'SAML', 'LDAP', 'OAuth'
    }
    
    @staticmethod
    def split(text: str) -> List[str]:
        """Split text into sentences, preserving code blocks and K8s syntax."""
        
        # 1. Preserve code blocks
        code_blocks = []
        text = re.sub(
            r'```[\s\S]*?```',
            lambda m: f'__CODE_{len(code_blocks) - 1}__' 
                     if (code_blocks.append(m.group(0)) or True) else '',
            text
        )
        
        # 2. Preserve inline code
        text = re.sub(
            r'`[^`]+`',
            lambda m: f'__INLINE_{len(code_blocks) - 1}__'
                     if (code_blocks.append(m.group(0)) or True) else '',
            text
        )
        
        # 3. Split on sentence boundaries
        # Avoid splitting on: abbreviations, decimals, URLs
        sentences = re.split(
            r'(?<![A-Z])\.(?=\s+[A-Z])|'  # Period + space + capital
            r'\?(?=\s+[A-Z])|'              # Question mark
            r'!(?=\s+[A-Z])',               # Exclamation mark
            text
        )
        
        # 4. Restore preserved blocks
        result = []
        for sent in sentences:
            for i, block in enumerate(code_blocks):
                sent = sent.replace(f'__CODE_{i}__', block)
                sent = sent.replace(f'__INLINE_{i}__', block)
            result.append(sent.strip())
        
        return [s for s in result if s]
```

### Integration Point
Add to `utils/` directory alongside existing `llm_provider.py` and `logger.py`.

---

## 8. Testing Recommendations

```python
def test_sentence_splitter():
    """Test cases for K8s documentation."""
    
    test_cases = [
        # Code blocks
        ("Here's YAML:\n```yaml\nkind: Pod\n```\nThis is a Pod.",
         ["Here's YAML:\n```yaml\nkind: Pod\n```", "This is a Pod."]),
        
        # Inline code
        ("Use `kubectl apply -f file.yaml` to deploy.",
         ["Use `kubectl apply -f file.yaml` to deploy."]),
        
        # Field paths
        ("Set `spec.containers[0].image` to nginx.",
         ["Set `spec.containers[0].image` to nginx."]),
        
        # Abbreviations
        ("Kubernetes (K8s) uses etcd. It's distributed.",
         ["Kubernetes (K8s) uses etcd.", "It's distributed."]),
        
        # Lists
        ("Features:\n- Pod: compute unit\n- Service: network",
         ["Features:\n- Pod: compute unit\n- Service: network"]),
    ]
    
    splitter = KubernetesSentenceSplitter()
    for text, expected in test_cases:
        result = splitter.split(text)
        assert result == expected, f"Failed: {text}"
```

---

## 9. Summary Table

| Aspect | Finding | Recommendation |
|--------|---------|-----------------|
| **Existing utilities** | None found | Build custom regex-based splitter |
| **Libraries available** | spacy, rapidfuzz, re | Use `re` for simplicity |
| **NLTK/SpaCy usage** | Not for sentence splitting | SpaCy available if needed |
| **Markdown handling** | Partial (code blocks only) | Extend to preserve lists, inline code |
| **K8s-specific needs** | Abbreviations, field paths, code | Add K8s abbreviation list |
| **Integration point** | `utils/sentence_splitter.py` | New module alongside existing utils |
| **Complexity** | Low-Medium | ~100 lines of regex + tests |

---

## Next Steps

1. **Clarify use case**: Why do you need sentence splitting?
   - For indexing? → Regex approach
   - For LLM context? → SpaCy approach
   - For general NLP? → NLTK approach

2. **Implement chosen approach** in `utils/sentence_splitter.py`

3. **Add test cases** for K8s-specific edge cases

4. **Integrate** with existing term extraction pipeline if needed

5. **Benchmark** against manual annotations to validate accuracy
