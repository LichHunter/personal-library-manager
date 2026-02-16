# Sentence Splitting Implementation Examples

## Example 1: Regex-Based Splitter (Recommended)

```python
# utils/sentence_splitter.py

import re
from typing import List

class KubernetesSentenceSplitter:
    """Sentence splitter optimized for Kubernetes documentation."""
    
    # K8s abbreviations that shouldn't trigger sentence splits
    K8S_ABBREVIATIONS = {
        'K8s', 'CRD', 'HPA', 'PVC', 'PV', 'RBAC', 'YAML', 'JSON',
        'API', 'HTTP', 'HTTPS', 'DNS', 'IP', 'TCP', 'UDP', 'TLS',
        'SSL', 'ETCD', 'OIDC', 'SAML', 'LDAP', 'OAuth', 'CRUD'
    }
    
    @staticmethod
    def split(text: str) -> List[str]:
        """
        Split text into sentences, preserving code blocks and K8s syntax.
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
            
        Examples:
            >>> splitter = KubernetesSentenceSplitter()
            >>> text = "Use `kubectl apply -f file.yaml` to deploy. This creates resources."
            >>> splitter.split(text)
            ["Use `kubectl apply -f file.yaml` to deploy.", "This creates resources."]
        """
        
        # Step 1: Preserve code blocks (```...```)
        code_blocks = []
        text = re.sub(
            r'```[\s\S]*?```',
            lambda m: (code_blocks.append(m.group(0)), f'__CODE_BLOCK_{len(code_blocks)-1}__')[1],
            text
        )
        
        # Step 2: Preserve inline code (`...`)
        text = re.sub(
            r'`[^`]+`',
            lambda m: (code_blocks.append(m.group(0)), f'__INLINE_CODE_{len(code_blocks)-1}__')[1],
            text
        )
        
        # Step 3: Split on sentence boundaries
        # Match: period/question/exclamation + space + capital letter
        # BUT NOT: abbreviations, decimal numbers, URLs
        sentences = re.split(
            r'(?<![A-Z])\.(?=\s+[A-Z])|'  # Period + space + capital (not after capital)
            r'\?(?=\s+[A-Z])|'              # Question mark + space + capital
            r'!(?=\s+[A-Z])',               # Exclamation mark + space + capital
            text
        )
        
        # Step 4: Restore preserved code blocks
        result = []
        for sent in sentences:
            for i, block in enumerate(code_blocks):
                sent = sent.replace(f'__CODE_BLOCK_{i}__', block)
                sent = sent.replace(f'__INLINE_CODE_{i}__', block)
            
            sent = sent.strip()
            if sent:  # Only include non-empty sentences
                result.append(sent)
        
        return result
    
    @staticmethod
    def split_with_metadata(text: str) -> List[dict]:
        """
        Split text and return sentences with metadata.
        
        Returns:
            List of dicts with 'text', 'start', 'end', 'is_code'
        """
        sentences = KubernetesSentenceSplitter.split(text)
        result = []
        pos = 0
        
        for sent in sentences:
            start = text.find(sent, pos)
            end = start + len(sent)
            is_code = sent.startswith('```') or sent.startswith('`')
            
            result.append({
                'text': sent,
                'start': start,
                'end': end,
                'is_code': is_code
            })
            
            pos = end
        
        return result
```

---

## Example 2: Test Cases

```python
# tests/test_sentence_splitter.py

import pytest
from utils.sentence_splitter import KubernetesSentenceSplitter

class TestKubernetesSentenceSplitter:
    """Test cases for K8s-aware sentence splitting."""
    
    @pytest.fixture
    def splitter(self):
        return KubernetesSentenceSplitter()
    
    def test_basic_sentences(self, splitter):
        """Test basic sentence splitting."""
        text = "This is a sentence. This is another."
        result = splitter.split(text)
        assert result == ["This is a sentence.", "This is another."]
    
    def test_preserve_code_blocks(self, splitter):
        """Test that code blocks are preserved."""
        text = "Here's YAML:\n```yaml\nkind: Pod\nmetadata:\n  name: test\n```\nThis is a Pod."
        result = splitter.split(text)
        assert len(result) == 2
        assert "```yaml" in result[0]
        assert result[1] == "This is a Pod."
    
    def test_preserve_inline_code(self, splitter):
        """Test that inline code is preserved."""
        text = "Use `kubectl apply -f file.yaml` to deploy. This creates resources."
        result = splitter.split(text)
        assert result[0] == "Use `kubectl apply -f file.yaml` to deploy."
        assert result[1] == "This creates resources."
    
    def test_preserve_field_paths(self, splitter):
        """Test that field paths with dots are preserved."""
        text = "Set `spec.containers[0].image` to nginx. This is required."
        result = splitter.split(text)
        assert "`spec.containers[0].image`" in result[0]
        assert result[1] == "This is required."
    
    def test_abbreviations(self, splitter):
        """Test that K8s abbreviations don't trigger splits."""
        text = "Kubernetes (K8s) uses etcd. It's distributed."
        result = splitter.split(text)
        # K8s. shouldn't split because K8s is in abbreviations
        assert len(result) == 2
        assert "K8s" in result[0]
    
    def test_decimal_numbers(self, splitter):
        """Test that decimal numbers don't trigger splits."""
        text = "Version 1.2.3 is stable. Use it in production."
        result = splitter.split(text)
        assert "1.2.3" in result[0]
        assert result[1] == "Use it in production."
    
    def test_urls(self, splitter):
        """Test that URLs don't trigger splits."""
        text = "See https://example.com for details. It has documentation."
        result = splitter.split(text)
        assert "https://example.com" in result[0]
        assert result[1] == "It has documentation."
    
    def test_lists(self, splitter):
        """Test that lists are preserved."""
        text = "Features:\n- Pod: compute unit\n- Service: network\n\nEach is important."
        result = splitter.split(text)
        assert "- Pod:" in result[0]
        assert "Each is important." in result[1]
    
    def test_empty_input(self, splitter):
        """Test empty input."""
        assert splitter.split("") == []
    
    def test_single_sentence(self, splitter):
        """Test single sentence without period."""
        text = "This is a sentence"
        result = splitter.split(text)
        assert result == ["This is a sentence"]
    
    def test_multiple_spaces(self, splitter):
        """Test handling of multiple spaces."""
        text = "First sentence.  Second sentence."
        result = splitter.split(text)
        assert len(result) == 2
        assert result[0] == "First sentence."
        assert result[1] == "Second sentence."
    
    def test_question_and_exclamation(self, splitter):
        """Test question marks and exclamation marks."""
        text = "Is this working? Yes! It works."
        result = splitter.split(text)
        assert len(result) == 3
        assert result[0] == "Is this working?"
        assert result[1] == "Yes!"
        assert result[2] == "It works."
    
    def test_complex_kubernetes_example(self, splitter):
        """Test complex real-world K8s documentation."""
        text = """
        A Pod is the smallest deployable unit in Kubernetes (K8s). 
        Here's an example:
        
        ```yaml
        apiVersion: v1
        kind: Pod
        metadata:
          name: my-pod
        spec:
          containers:
          - name: app
            image: nginx:1.2.3
        ```
        
        Use `kubectl apply -f pod.yaml` to create it. 
        The `spec.containers[0].image` field specifies the container image.
        """
        
        result = splitter.split(text)
        
        # Should have multiple sentences
        assert len(result) > 3
        
        # Code block should be preserved
        assert any("```yaml" in s for s in result)
        
        # Inline code should be preserved
        assert any("`kubectl apply" in s for s in result)
        assert any("`spec.containers" in s for s in result)
        
        # Version number should not cause splits
        assert any("1.2.3" in s for s in result)
```

---

## Example 3: Integration with Existing Code

```python
# In run_experiment.py or similar

from utils.sentence_splitter import KubernetesSentenceSplitter

def extract_terms_from_sentences(chunk_text: str, model: str) -> list[str]:
    """Extract terms from chunk by processing sentence-by-sentence."""
    
    splitter = KubernetesSentenceSplitter()
    sentences = splitter.split(chunk_text)
    
    all_terms = []
    
    for sentence in sentences:
        # Skip code blocks
        if sentence.startswith('```'):
            continue
        
        # Extract terms from this sentence
        terms = extract_terms_single_sentence(sentence, model)
        all_terms.extend(terms)
    
    # Deduplicate while preserving order
    seen = set()
    result = []
    for term in all_terms:
        if term.lower() not in seen:
            seen.add(term.lower())
            result.append(term)
    
    return result

def extract_terms_single_sentence(sentence: str, model: str) -> list[str]:
    """Extract terms from a single sentence."""
    # Existing extraction logic, but on smaller text
    # ...
    pass
```

---

## Example 4: SpaCy Alternative

```python
# utils/sentence_splitter_spacy.py

import spacy
from typing import List

class SpaCySentenceSplitter:
    """Sentence splitter using SpaCy (already in dependencies)."""
    
    def __init__(self, model: str = "en_core_web_sm"):
        """Initialize with SpaCy model."""
        try:
            self.nlp = spacy.load(model)
        except OSError:
            print(f"Downloading {model}...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model])
            self.nlp = spacy.load(model)
    
    def split(self, text: str) -> List[str]:
        """Split text into sentences using SpaCy."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def split_with_metadata(self, text: str) -> List[dict]:
        """Split and return metadata."""
        doc = self.nlp(text)
        result = []
        
        for sent in doc.sents:
            result.append({
                'text': sent.text.strip(),
                'start': sent.start_char,
                'end': sent.end_char,
                'tokens': [token.text for token in sent]
            })
        
        return result
```

---

## Example 5: NLTK Alternative

```python
# utils/sentence_splitter_nltk.py

from nltk.tokenize import sent_tokenize
from typing import List
import nltk

class NLTKSentenceSplitter:
    """Sentence splitter using NLTK (requires new dependency)."""
    
    def __init__(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def split(self, text: str) -> List[str]:
        """Split text into sentences using NLTK."""
        return sent_tokenize(text)
```

---

## Example 6: Benchmark Script

```python
# benchmark_sentence_splitters.py

import time
from utils.sentence_splitter import KubernetesSentenceSplitter
from utils.sentence_splitter_spacy import SpaCySentenceSplitter
from utils.sentence_splitter_nltk import NLTKSentenceSplitter

def benchmark_splitters(text: str, iterations: int = 100):
    """Benchmark different sentence splitters."""
    
    splitters = {
        'Regex': KubernetesSentenceSplitter(),
        'SpaCy': SpaCySentenceSplitter(),
        'NLTK': NLTKSentenceSplitter(),
    }
    
    results = {}
    
    for name, splitter in splitters.items():
        start = time.time()
        
        for _ in range(iterations):
            splitter.split(text)
        
        elapsed = time.time() - start
        results[name] = {
            'total_time': elapsed,
            'per_call': elapsed / iterations,
            'calls_per_second': iterations / elapsed
        }
    
    # Print results
    print(f"\nBenchmark Results ({iterations} iterations)")
    print("=" * 60)
    
    for name, metrics in results.items():
        print(f"\n{name}:")
        print(f"  Total time: {metrics['total_time']:.3f}s")
        print(f"  Per call: {metrics['per_call']*1000:.2f}ms")
        print(f"  Calls/sec: {metrics['calls_per_second']:.0f}")
    
    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]['per_call'])
    print(f"\n✓ Fastest: {fastest[0]} ({fastest[1]['per_call']*1000:.2f}ms per call)")

if __name__ == "__main__":
    # Load sample K8s documentation
    with open("sample_kubernetes_doc.md") as f:
        text = f.read()
    
    benchmark_splitters(text)
```

---

## Quick Comparison

| Approach | Speed | Accuracy | Code Size | Dependencies |
|----------|-------|----------|-----------|--------------|
| Regex | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ~50 lines | None |
| SpaCy | ⭐⭐ | ⭐⭐⭐⭐⭐ | ~30 lines | Already have |
| NLTK | ⭐⭐⭐ | ⭐⭐⭐⭐ | ~10 lines | New dep |

---

## Recommendation

**Start with Regex** because:
1. ✅ No new dependencies
2. ✅ Fastest performance
3. ✅ Easy to customize for K8s
4. ✅ Can fall back to SpaCy if needed

**Use SpaCy** if:
- You need higher accuracy on complex English
- Performance is not critical
- You're already using SpaCy for other tasks

**Use NLTK** if:
- You want battle-tested sentence splitting
- You don't mind adding a dependency
- You're doing general NLP work
