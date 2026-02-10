# Sentence Splitting Quick Reference

## TL;DR

**No existing sentence splitting utilities found in codebase.**

### Recommendation: Use Regex-Based Splitting

```python
import re

def split_sentences(text: str) -> list[str]:
    """Split sentences, preserving code blocks and K8s syntax."""
    # Preserve code blocks
    code_blocks = []
    text = re.sub(r'```[\s\S]*?```', 
                  lambda m: (code_blocks.append(m.group(0)), f'__CODE_{len(code_blocks)-1}__')[1], 
                  text)
    
    # Split on sentence boundaries (period/question/exclamation + space + capital)
    sentences = re.split(r'(?<![A-Z])\.(?=\s+[A-Z])|\?(?=\s+[A-Z])|!(?=\s+[A-Z])', text)
    
    # Restore code blocks
    result = []
    for sent in sentences:
        for i, block in enumerate(code_blocks):
            sent = sent.replace(f'__CODE_{i}__', block)
        result.append(sent.strip())
    
    return [s for s in result if s]
```

---

## What's Already in the Codebase

### Libraries Imported
- ✅ `re` (regex) - Used everywhere
- ✅ `spacy` - Used for NER, NOT sentence splitting
- ✅ `sentence-transformers` - For embeddings, NOT splitting
- ❌ `nltk` - NOT imported
- ❌ `nltk.sent_tokenize` - NOT used

### Text Processing Patterns Found
1. **Markdown section extraction** (test_small_chunk_extraction.py)
   - Splits by headings (H1-H4)
   - Preserves heading in content
   - Minimum 15 words per section

2. **Code block handling** (run_experiment.py)
   - Removes markdown code blocks (```...```)
   - Extracts JSON from responses
   - Regex: `r"^```(?:json)?\s*"` and `r"\s*```$"`

3. **Text normalization**
   - CamelCase splitting: `re.sub(r"([a-z])([A-Z])", r"\1 \2", term)`
   - Token splitting: `text.split()`

---

## Edge Cases to Handle

| Case | Example | How to Handle |
|------|---------|---------------|
| Code blocks | ```yaml ... ``` | Preserve before splitting |
| Inline code | `kubectl apply` | Preserve before splitting |
| Field paths | `spec.containers[0].image` | Don't split on dots |
| Abbreviations | K8s, CRD, HPA | Maintain abbreviation list |
| URLs | `https://example.com` | Don't split on dots |
| Decimal numbers | `1.2.3` (version) | Don't split on dots |
| Lists | `- Item 1\n- Item 2` | Preserve list structure |

---

## Three Options Compared

### 1. Regex (RECOMMENDED) ⭐⭐⭐⭐⭐
```python
# Pros: Fast, no dependencies, customizable
# Cons: Requires careful regex tuning
# Best for: K8s documentation with special syntax
```

### 2. SpaCy (AVAILABLE) ⭐⭐⭐⭐
```python
import spacy
nlp = spacy.load("en_core_web_sm")
sentences = [sent.text for sent in nlp(text).sents]

# Pros: Already in dependencies, handles complex cases
# Cons: Slower, overkill for simple splitting
# Best for: Complex English prose
```

### 3. NLTK (NEW DEPENDENCY) ⭐⭐⭐
```python
from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(text)

# Pros: Battle-tested, handles abbreviations
# Cons: New dependency, slower
# Best for: General English text
```

---

## Implementation Checklist

- [ ] Decide use case (indexing? LLM context? general NLP?)
- [ ] Choose approach (regex/spacy/nltk)
- [ ] Create `utils/sentence_splitter.py`
- [ ] Add K8s abbreviation list if using regex
- [ ] Write test cases for edge cases
- [ ] Integrate with existing pipeline
- [ ] Benchmark against manual annotations

---

## Files to Reference

| File | Purpose | Key Code |
|------|---------|----------|
| `test_small_chunk_extraction.py` | Markdown section extraction | `extract_sections_simple()` |
| `run_experiment.py` | Code block handling | `parse_json_response()` |
| `pyproject.toml` | Dependencies | Lists all available libraries |

---

## K8s Abbreviations to Preserve

```python
K8S_ABBREVIATIONS = {
    'K8s', 'CRD', 'HPA', 'PVC', 'PV', 'RBAC', 'YAML', 'JSON',
    'API', 'HTTP', 'HTTPS', 'DNS', 'IP', 'TCP', 'UDP', 'TLS',
    'SSL', 'ETCD', 'OIDC', 'SAML', 'LDAP', 'OAuth'
}
```

---

## Next Action

**Choose your approach and let me know if you want me to:**
1. Implement the regex-based splitter
2. Create SpaCy wrapper
3. Add NLTK integration
4. Write comprehensive tests
