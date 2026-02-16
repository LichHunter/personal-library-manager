import pytest
from plm.extraction.chunking import get_chunker, WholeChunker, HeadingChunker


KUBERNETES_DOC = """# Kubernetes Pod Security Standards

Kubernetes defines three different Pod Security Standards to cover a broad spectrum of security needs.

## Privileged

The Privileged policy is an entirely unrestricted policy. This policy allows for known privilege escalations and is intended for system and infrastructure workloads.

The Privileged namespace is useful for:
- Running monitoring agents
- Running CNI plugins
- Running cluster-level logging solutions

## Baseline

The Baseline policy is minimally restrictive while preventing known privilege escalations. This policy is targeted at application operators.

Key restrictions in Baseline:
- Disallow privileged containers
- Disallow host namespaces
- Disallow hostPath volumes
- Restrict capabilities to a safe set

### Container Restrictions

Containers must not:
1. Run as privileged
2. Use host networking
3. Use host PID namespace

### Volume Restrictions

The following volume types are disallowed:
- hostPath
- gcePersistentDisk (deprecated)

## Restricted

The Restricted policy is heavily restricted and follows current pod hardening best practices.

This policy includes all Baseline restrictions plus:
- Must run as non-root
- Seccomp must be RuntimeDefault or Localhost
- All capabilities must be dropped

### Running Non-Root

Containers must set:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
```

### Seccomp Profile

Required seccomp configuration:
```yaml
securityContext:
  seccompProfile:
    type: RuntimeDefault
```

## Enforcement Modes

Pod Security Standards can be enforced in three modes:
- enforce: Policy violations cause pod rejection
- audit: Violations trigger audit log entry
- warn: Violations trigger user-facing warning

## Migration Guide

To migrate from PodSecurityPolicy to Pod Security Standards:
1. Enable the PodSecurity admission controller
2. Apply namespace labels
3. Test with warn and audit modes
4. Enable enforce mode
"""


class TestChunkingIntegration:
    def test_whole_chunker_returns_all_content(self):
        chunker = get_chunker("whole")
        result = chunker.chunk(KUBERNETES_DOC)
        
        assert len(result) == 1
        assert result[0].text == KUBERNETES_DOC
        assert "# Kubernetes Pod Security Standards" in result[0].text
        assert "Migration Guide" in result[0].text
        assert "runAsNonRoot: true" in result[0].text
    
    def test_heading_chunker_creates_multiple_chunks(self):
        chunker = get_chunker("heading")
        result = chunker.chunk(KUBERNETES_DOC)
        
        assert len(result) > 1
        
        all_text = " ".join(chunk.text for chunk in result)
        assert "Privileged" in all_text
        assert "Baseline" in all_text
        assert "Restricted" in all_text
    
    def test_heading_chunker_preserves_heading_context(self):
        chunker = HeadingChunker(min_tokens=20, max_tokens=100)
        result = chunker.chunk(KUBERNETES_DOC)
        
        chunks_with_headings = [c for c in result if c.heading is not None]
        assert len(chunks_with_headings) > 0
        
        heading_texts = [c.heading for c in chunks_with_headings]
        assert any("Privileged" in h for h in heading_texts if h)
    
    def test_chunker_factory_returns_correct_types(self):
        whole = get_chunker("whole")
        heading = get_chunker("heading")
        
        assert isinstance(whole, WholeChunker)
        assert isinstance(heading, HeadingChunker)
    
    def test_all_chunks_have_valid_positions(self):
        chunker = HeadingChunker(min_tokens=20, max_tokens=100)
        result = chunker.chunk(KUBERNETES_DOC)
        
        for chunk in result:
            assert chunk.start_char >= 0
            assert chunk.end_char >= chunk.start_char
            assert chunk.end_char <= len(KUBERNETES_DOC) + 500
