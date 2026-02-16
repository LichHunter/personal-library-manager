# PLM Source Structure Implementation

## TL;DR

> **Quick Summary**: Create production `src/plm/` package structure with fast (heuristic) and slow (V6 LLM) extraction systems, porting proven code from POC-1c.
> 
> **Deliverables**:
> - `src/plm/` package with extraction module
> - Fast system: heuristic-only extraction (CamelCase, backticks, ALL_CAPS, dot.paths)
> - Slow system: V6 pipeline from POC-1c (5 stages)
> - Shared LLM connector (Anthropic + Gemini)
> - Data files (auto_vocab.json, tech_domain_negatives_v2.json)
> 
> **Estimated Effort**: Medium (2-3 days)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Setup → Copy shared/llm → Port slow system → Create fast system → Integration

---

## Context

### Original Request
Design and implement `src/` structure for PLM based on POC-1c V6 strategy, with fast and slow extraction systems.

### Key Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Package name | `plm` | Short, standard Python convention |
| Fast system | Heuristic-only | GLiNER rejected ("garbage results" per DECISION_LOG) |
| Slow system | V6 pipeline | Proven F1=0.932 @ 10 docs |
| LLM provider | `poc/shared/llm/` | Full featured (Anthropic + Gemini) |
| Storage | JSON files | Simple, matches POC pattern |
| Dropped | GLiNER, scoring.py, strategies config, retrieval, scripts | Not needed for core extraction |

### Metis Review Findings (Addressed)
- ✅ GLiNER rejection acknowledged → using heuristic-only
- ✅ LLM provider choice resolved → using poc/shared/llm/
- ✅ tech_domain_negatives version → using v2 (newer)
- ✅ Function-to-module mapping → included in task references

---

## Work Objectives

### Core Objective
Create production-ready `src/plm/` package that implements fast (heuristic) and slow (LLM-based V6) extraction pipelines.

### Concrete Deliverables
1. `src/plm/` package directory with proper structure
2. `src/plm/shared/llm/` - LLM connector (copy from poc/shared/llm)
3. `src/plm/extraction/slow/` - V6 pipeline modules
4. `src/plm/extraction/fast/` - Heuristic extraction
5. `src/plm/extraction/pipeline.py` - Main orchestrator
6. `data/vocabularies/` - Vocabulary files
7. `pyproject.toml` - Package configuration
8. `tests/` - Basic test structure

### Definition of Done
- [x] `python -c "from plm.extraction import extract"` succeeds
- [x] `python -c "from plm.shared.llm import call_llm"` succeeds
- [x] All modules have `__init__.py` with proper exports
- [x] Vocabulary files copied and accessible
- [x] Basic smoke tests pass

### Must Have
- Clean import paths (no `sys.path` hacks)
- Proper `__init__.py` exports
- pyproject.toml with all dependencies
- py.typed marker for type checking

### Must NOT Have (Guardrails)
- NO GLiNER integration (documented failure)
- NO strategy preset system (strip to V6 config only)
- NO benchmark scoring functions in core package
- NO retrieval system (faiss, sentence-transformers) for now
- NO modification of scoring logic (`verify_span`, `normalize_term`)

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL verification is executed by the agent using tools (Bash, Python imports).

### Test Decision
- **Infrastructure exists**: NO (new package)
- **Automated tests**: Tests-after (verify via import checks)
- **Framework**: pytest (to be set up)

### Agent-Executed QA Scenarios

```
Scenario: Package structure is correct
  Tool: Bash
  Steps:
    1. test -d src/plm/extraction/fast && echo "PASS"
    2. test -d src/plm/extraction/slow && echo "PASS"
    3. test -d src/plm/shared/llm && echo "PASS"
    4. test -f src/plm/py.typed && echo "PASS"
  Expected Result: All directories and files exist

Scenario: LLM connector imports successfully
  Tool: Bash
  Steps:
    1. cd src && python -c "from plm.shared.llm import call_llm; print('PASS')"
  Expected Result: Import succeeds, prints PASS

Scenario: Slow extraction modules import
  Tool: Bash
  Steps:
    1. cd src && python -c "from plm.extraction.slow import grounding, noise_filter; print('PASS')"
  Expected Result: Import succeeds

Scenario: Fast extraction modules import
  Tool: Bash
  Steps:
    1. cd src && python -c "from plm.extraction.fast import heuristic; print('PASS')"
  Expected Result: Import succeeds

Scenario: Vocabulary files accessible
  Tool: Bash
  Steps:
    1. test -f data/vocabularies/auto_vocab.json && echo "EXISTS"
    2. python -c "import json; d=json.load(open('data/vocabularies/auto_vocab.json')); print(f'seeds: {len(d.get(\"seeds\", []))}')"
  Expected Result: File exists, seeds count printed
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Create directory structure + pyproject.toml
├── Task 2: Copy shared/llm from poc/shared/llm
└── Task 3: Copy vocabulary files to data/

Wave 2 (After Wave 1):
├── Task 4: Port slow system modules from POC-1c
├── Task 5: Create fast system (heuristic extraction)
└── Task 6: Create pipeline orchestrator + integration

Wave 3 (After Wave 2):
└── Task 7: Create basic test structure + smoke tests

Critical Path: Task 1 → Task 4 → Task 6 → Task 7
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4, 5, 6 | None |
| 2 | 1 | 4, 6 | 3 |
| 3 | 1 | 4 | 2 |
| 4 | 1, 2, 3 | 6 | 5 |
| 5 | 1 | 6 | 4 |
| 6 | 4, 5 | 7 | None |
| 7 | 6 | None | None |

---

## TODOs

### Wave 1: Foundation

- [x] 1. Create directory structure and pyproject.toml

  **What to do**:
  
  Create the full directory structure:
  ```
  src/
  └── plm/
      ├── __init__.py
      ├── py.typed
      ├── extraction/
      │   ├── __init__.py
      │   ├── pipeline.py
      │   ├── fast/
      │   │   ├── __init__.py
      │   │   ├── heuristic.py
      │   │   └── confidence.py
      │   └── slow/
      │       ├── __init__.py
      │       ├── candidate_verify.py
      │       ├── taxonomy.py
      │       ├── validation.py
      │       ├── grounding.py
      │       ├── noise_filter.py
      │       └── postprocess.py
      └── shared/
          ├── __init__.py
          └── llm/
              ├── __init__.py
              ├── base.py
              ├── anthropic_provider.py
              └── gemini_provider.py
  
  tests/
  ├── __init__.py
  └── conftest.py
  
  data/
  └── vocabularies/
  ```

  Create `pyproject.toml`:
  ```toml
  [project]
  name = "plm"
  version = "0.1.0"
  description = "Personal Library Manager - Term Extraction"
  requires-python = ">=3.11"
  dependencies = [
      "anthropic>=0.40.0",
      "httpx>=0.27.0",
      "pydantic>=2.0.0",
      "rapidfuzz>=3.0.0",
  ]

  [project.optional-dependencies]
  dev = [
      "pytest>=8.0.0",
      "pytest-asyncio>=0.23.0",
  ]

  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"

  [tool.hatch.build.targets.wheel]
  packages = ["src/plm"]

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  pythonpath = ["src"]
  ```

  Create minimal `__init__.py` files with version:
  ```python
  # src/plm/__init__.py
  """PLM - Personal Library Manager."""
  __version__ = "0.1.0"
  ```

  **Must NOT do**:
  - Don't create placeholder content in module files yet
  - Don't add retrieval dependencies (faiss, sentence-transformers)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Filesystem operations only

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundational)
  - **Blocks**: All other tasks

  **References**:
  - `poc/poc-1c-scalable-ner/pyproject.toml` — Dependencies reference
  - `poc/shared/pyproject.toml` — Package structure reference

  **Acceptance Criteria**:
  - [ ] `test -d src/plm/extraction/slow` → exists
  - [ ] `test -d src/plm/extraction/fast` → exists
  - [ ] `test -d src/plm/shared/llm` → exists
  - [ ] `test -f src/plm/py.typed` → exists
  - [ ] `test -f pyproject.toml` → exists

  **Commit**: YES
  - Message: `feat(plm): create package directory structure`
  - Files: `src/plm/**`, `tests/**`, `data/**`, `pyproject.toml`

---

- [x] 2. Copy shared/llm from poc/shared/llm

  **What to do**:
  
  Copy the LLM provider files:
  ```bash
  cp poc/shared/llm/__init__.py src/plm/shared/llm/
  cp poc/shared/llm/base.py src/plm/shared/llm/
  cp poc/shared/llm/anthropic_provider.py src/plm/shared/llm/
  cp poc/shared/llm/gemini_provider.py src/plm/shared/llm/
  ```

  Update imports in copied files:
  - Change `from shared.llm.X` → `from plm.shared.llm.X`
  - Or use relative imports: `from .base import ...`

  Update `src/plm/shared/__init__.py`:
  ```python
  """Shared utilities for PLM."""
  from .llm import call_llm

  __all__ = ["call_llm"]
  ```

  Update `src/plm/shared/llm/__init__.py`:
  ```python
  """LLM provider abstraction."""
  from .base import call_llm, get_provider, LLMProvider
  from .anthropic_provider import AnthropicProvider
  from .gemini_provider import GeminiProvider

  __all__ = [
      "call_llm",
      "get_provider", 
      "LLMProvider",
      "AnthropicProvider",
      "GeminiProvider",
  ]
  ```

  **Must NOT do**:
  - Don't modify the core provider logic
  - Don't change auth file paths

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES with Task 3
  - **Blocked By**: Task 1
  - **Blocks**: Tasks 4, 6

  **References**:
  - `poc/shared/llm/__init__.py` — Public API exports
  - `poc/shared/llm/base.py` — call_llm, get_provider, LLMProvider
  - `poc/shared/llm/anthropic_provider.py` — AnthropicProvider
  - `poc/shared/llm/gemini_provider.py` — GeminiProvider

  **Acceptance Criteria**:
  - [ ] `ls src/plm/shared/llm/` shows 4 Python files
  - [ ] `cd src && python -c "from plm.shared.llm import call_llm; print('OK')"` → OK
  - [ ] `cd src && python -c "from plm.shared.llm import AnthropicProvider; print('OK')"` → OK

  **Commit**: YES
  - Message: `feat(plm): add shared LLM provider (Anthropic + Gemini)`
  - Files: `src/plm/shared/llm/*`

---

- [x] 3. Copy vocabulary files to data/

  **What to do**:
  
  Copy vocabulary files from POC-1c artifacts:
  ```bash
  cp poc/poc-1c-scalable-ner/artifacts/auto_vocab.json data/vocabularies/
  cp poc/poc-1c-scalable-ner/artifacts/tech_domain_negatives_v2.json data/vocabularies/tech_domain_negatives.json
  ```

  Note: Using `tech_domain_negatives_v2.json` (renamed to `tech_domain_negatives.json`) as it's the newer version.

  **Must NOT do**:
  - Don't modify the vocabulary content
  - Don't copy v1 of tech_domain_negatives

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES with Task 2
  - **Blocked By**: Task 1
  - **Blocks**: Task 4

  **References**:
  - `poc/poc-1c-scalable-ner/artifacts/auto_vocab.json` — 445 terms (bypass, seeds, contextual_seeds)
  - `poc/poc-1c-scalable-ner/artifacts/tech_domain_negatives_v2.json` — ~6K non-entity words

  **Acceptance Criteria**:
  - [ ] `test -f data/vocabularies/auto_vocab.json` → exists
  - [ ] `test -f data/vocabularies/tech_domain_negatives.json` → exists
  - [ ] `python -c "import json; print(len(json.load(open('data/vocabularies/auto_vocab.json'))['seeds']))"` → prints count

  **Commit**: YES
  - Message: `feat(plm): add vocabulary data files`
  - Files: `data/vocabularies/*`

---

### Wave 2: Core Implementation

- [x] 4. Port slow system modules from POC-1c

  **What to do**:
  
  Extract and refactor functions from `hybrid_ner.py` into separate modules.
  
  **Module Mapping** (from hybrid_ner.py):

  | Source Location | Target Module | Functions |
  |-----------------|---------------|-----------|
  | Lines 120-180 | `slow/grounding.py` | `verify_span()`, `normalize_term()`, `ground_candidates()`, `deduplicate()` |
  | Lines 200-280 | `slow/noise_filter.py` | `filter_noise()`, `load_negatives()`, `is_stop_word()`, `is_generic_phrase()` |
  | Lines 400-500 | `slow/candidate_verify.py` | `extract_candidates_heuristic()`, `classify_candidates_llm()` |
  | Lines 550-650 | `slow/taxonomy.py` | `extract_by_taxonomy()`, `ENTITY_TYPES` |
  | Lines 700-800 | `slow/validation.py` | `validate_with_context()`, `build_validation_prompt()` |
  | Lines 850-950 | `slow/postprocess.py` | `expand_spans()`, `suppress_subspans()`, `final_dedup()` |

  Each module should:
  1. Have clear function signatures with type hints
  2. Use relative imports for sibling modules
  3. Import `call_llm` from `plm.shared.llm`
  4. Load vocabularies from `data/vocabularies/`

  Example structure for `slow/grounding.py`:
  ```python
  """Stage 2: Grounding and deduplication."""
  import re
  from rapidfuzz import fuzz

  def normalize_term(term: str) -> str:
      """Normalize term for comparison."""
      return term.lower().strip().replace("-", " ").replace("_", " ")

  def verify_span(term: str, content: str) -> tuple[bool, str]:
      """Verify term exists in content."""
      # Port from scoring.py lines 16-36
      ...

  def ground_candidates(
      candidates: list[str], 
      content: str
  ) -> list[tuple[str, str]]:
      """Ground candidates to actual spans in content."""
      ...

  def deduplicate(terms: list[str]) -> list[str]:
      """Remove duplicate terms after normalization."""
      ...
  ```

  Create `slow/__init__.py`:
  ```python
  """Slow extraction system - V6 LLM-based pipeline."""
  from .grounding import verify_span, normalize_term, ground_candidates
  from .noise_filter import filter_noise
  from .candidate_verify import extract_candidates, classify_candidates
  from .taxonomy import extract_by_taxonomy
  from .validation import validate_terms
  from .postprocess import postprocess_terms

  __all__ = [
      "verify_span",
      "normalize_term",
      "ground_candidates",
      "filter_noise",
      "extract_candidates",
      "classify_candidates",
      "extract_by_taxonomy",
      "validate_terms",
      "postprocess_terms",
  ]
  ```

  **Must NOT do**:
  - Don't change the scoring/matching logic
  - Don't add retrieval dependencies
  - Don't port strategy presets (only V6 config)
  - Don't include benchmark-only code

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: []
  - Reason: Complex refactoring requiring careful code understanding

  **Parallelization**:
  - **Can Run In Parallel**: YES with Task 5
  - **Blocked By**: Tasks 1, 2, 3
  - **Blocks**: Task 6

  **References**:
  - `poc/poc-1c-scalable-ner/hybrid_ner.py` — Main source (2000+ lines)
  - `poc/poc-1c-scalable-ner/scoring.py` — verify_span, normalize_term
  - `poc/poc-1c-scalable-ner/docs/V6_ARCHITECTURE.md` — Stage descriptions

  **Acceptance Criteria**:
  - [ ] `ls src/plm/extraction/slow/` shows 6 .py files + __init__.py
  - [ ] `cd src && python -c "from plm.extraction.slow import verify_span; print('OK')"` → OK
  - [ ] `cd src && python -c "from plm.extraction.slow import filter_noise; print('OK')"` → OK
  - [ ] `cd src && python -c "from plm.extraction.slow import validate_terms; print('OK')"` → OK
  - [ ] grep for `from plm.shared.llm import call_llm` in slow/*.py shows imports

  **Commit**: YES
  - Message: `feat(plm): port V6 slow extraction pipeline from POC-1c`
  - Files: `src/plm/extraction/slow/*`

---

- [x] 5. Create fast system (heuristic extraction)

  **What to do**:
  
  Create heuristic-based fast extraction system.

  **`fast/heuristic.py`**:
  ```python
  """Fast heuristic extraction - zero LLM cost."""
  import re
  from typing import Iterator

  # Patterns from hybrid_ner.py candidate_verify approach
  CAMEL_CASE = re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b')
  ALL_CAPS = re.compile(r'\b[A-Z][A-Z_]{2,}\b')
  DOT_PATH = re.compile(r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b')
  BACKTICK = re.compile(r'`([^`]+)`')
  FUNCTION_CALL = re.compile(r'\b[a-zA-Z_]\w*\(\)')

  def extract_camel_case(text: str) -> Iterator[str]:
      """Extract CamelCase identifiers."""
      for match in CAMEL_CASE.finditer(text):
          yield match.group()

  def extract_all_caps(text: str) -> Iterator[str]:
      """Extract ALL_CAPS identifiers."""
      for match in ALL_CAPS.finditer(text):
          yield match.group()

  def extract_dot_paths(text: str) -> Iterator[str]:
      """Extract dot.separated.paths."""
      for match in DOT_PATH.finditer(text):
          yield match.group()

  def extract_backticks(text: str) -> Iterator[str]:
      """Extract `backtick` content."""
      for match in BACKTICK.finditer(text):
          yield match.group(1)

  def extract_function_calls(text: str) -> Iterator[str]:
      """Extract function() calls."""
      for match in FUNCTION_CALL.finditer(text):
          yield match.group()

  def extract_all_heuristic(text: str) -> list[str]:
      """Extract all candidates using heuristics."""
      candidates = set()
      for extractor in [
          extract_camel_case,
          extract_all_caps,
          extract_dot_paths,
          extract_backticks,
          extract_function_calls,
      ]:
          candidates.update(extractor(text))
      return list(candidates)
  ```

  **`fast/confidence.py`**:
  ```python
  """Confidence scoring for routing decisions."""
  from dataclasses import dataclass
  from typing import Literal

  ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]

  @dataclass
  class ExtractionResult:
      term: str
      sources: list[str]  # Which extractors found it
      confidence: float
      level: ConfidenceLevel

  def compute_confidence(
      term: str,
      sources: list[str],
      entity_ratio: float | None = None,
  ) -> tuple[float, ConfidenceLevel]:
      """Compute confidence score and level for routing."""
      # Multi-source agreement boosts confidence
      source_score = min(len(sources) / 3, 1.0)
      
      # Entity ratio from training data (if available)
      ratio_score = entity_ratio if entity_ratio else 0.5
      
      confidence = (source_score * 0.6) + (ratio_score * 0.4)
      
      if confidence >= 0.8:
          return confidence, "HIGH"
      elif confidence >= 0.5:
          return confidence, "MEDIUM"
      else:
          return confidence, "LOW"
  ```

  **`fast/__init__.py`**:
  ```python
  """Fast extraction system - heuristic-based, zero LLM cost."""
  from .heuristic import (
      extract_all_heuristic,
      extract_camel_case,
      extract_all_caps,
      extract_dot_paths,
      extract_backticks,
  )
  from .confidence import compute_confidence, ExtractionResult

  __all__ = [
      "extract_all_heuristic",
      "extract_camel_case",
      "extract_all_caps",
      "extract_dot_paths",
      "extract_backticks",
      "compute_confidence",
      "ExtractionResult",
  ]
  ```

  **Must NOT do**:
  - Don't add GLiNER (documented failure)
  - Don't add LLM calls (fast = zero cost)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []
  - Reason: Straightforward regex patterns

  **Parallelization**:
  - **Can Run In Parallel**: YES with Task 4
  - **Blocked By**: Task 1
  - **Blocks**: Task 6

  **References**:
  - `poc/poc-1c-scalable-ner/hybrid_ner.py:400-450` — Heuristic patterns
  - `poc/poc-1c-scalable-ner/DECISION_LOG.md` — Why GLiNER was rejected

  **Acceptance Criteria**:
  - [ ] `ls src/plm/extraction/fast/` shows heuristic.py, confidence.py, __init__.py
  - [ ] `cd src && python -c "from plm.extraction.fast import extract_all_heuristic; print('OK')"` → OK
  - [ ] `cd src && python -c "from plm.extraction.fast.heuristic import extract_camel_case; print(list(extract_camel_case('HelloWorld')))"` → ['HelloWorld']
  - [ ] grep -r "gliner\|GLiNER" src/plm/extraction/fast/ → no matches

  **Commit**: YES
  - Message: `feat(plm): add fast heuristic extraction system`
  - Files: `src/plm/extraction/fast/*`

---

- [x] 6. Create pipeline orchestrator and integration

  **What to do**:
  
  Create the main pipeline that orchestrates fast and slow systems.

  **`extraction/pipeline.py`**:
  ```python
  """Main extraction pipeline orchestrator."""
  from dataclasses import dataclass, field
  from typing import Literal

  from .fast import extract_all_heuristic, compute_confidence
  from .slow import (
      verify_span,
      normalize_term,
      filter_noise,
      validate_terms,
      postprocess_terms,
  )

  @dataclass
  class ExtractionConfig:
      """Configuration for extraction pipeline."""
      use_fast_only: bool = False
      use_slow_only: bool = False
      confidence_threshold: float = 0.7
      validate_medium_confidence: bool = True

  @dataclass  
  class ExtractionResult:
      """Result of extraction pipeline."""
      terms: list[str]
      fast_candidates: list[str] = field(default_factory=list)
      slow_candidates: list[str] = field(default_factory=list)
      filtered_count: int = 0
      validated_count: int = 0

  def extract(
      text: str,
      config: ExtractionConfig | None = None,
  ) -> ExtractionResult:
      """
      Extract technical entities from text.
      
      Args:
          text: Document text to extract from
          config: Pipeline configuration
          
      Returns:
          ExtractionResult with extracted terms and stats
      """
      config = config or ExtractionConfig()
      result = ExtractionResult(terms=[])
      
      # Stage 1: Fast extraction (heuristics)
      if not config.use_slow_only:
          fast_candidates = extract_all_heuristic(text)
          result.fast_candidates = fast_candidates
      
      # Stage 2: Slow extraction (LLM-based) - TODO: implement full V6
      if not config.use_fast_only:
          # Placeholder for full V6 pipeline
          slow_candidates = []  # Will call taxonomy, candidate_verify, etc.
          result.slow_candidates = slow_candidates
      
      # Merge candidates
      all_candidates = list(set(result.fast_candidates + result.slow_candidates))
      
      # Stage 3: Ground and filter
      grounded = [(c, verify_span(c, text)) for c in all_candidates]
      valid = [c for c, (valid, _) in grounded if valid]
      
      # Stage 4: Noise filter
      filtered = filter_noise(valid)
      result.filtered_count = len(valid) - len(filtered)
      
      # Stage 5: Postprocess
      final = postprocess_terms(filtered, text)
      result.terms = final
      
      return result

  # Convenience functions
  def fast_extract(text: str) -> list[str]:
      """Fast-only extraction (zero LLM cost)."""
      return extract(text, ExtractionConfig(use_fast_only=True)).terms

  def slow_extract(text: str) -> list[str]:
      """Slow-only extraction (full V6 pipeline)."""
      return extract(text, ExtractionConfig(use_slow_only=True)).terms
  ```

  **`extraction/__init__.py`**:
  ```python
  """PLM Extraction Module."""
  from .pipeline import (
      extract,
      fast_extract,
      slow_extract,
      ExtractionConfig,
      ExtractionResult,
  )

  __all__ = [
      "extract",
      "fast_extract",
      "slow_extract",
      "ExtractionConfig",
      "ExtractionResult",
  ]
  ```

  **Must NOT do**:
  - Don't implement full V6 slow pipeline yet (stub is fine)
  - Don't add complex routing logic yet

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocked By**: Tasks 4, 5
  - **Blocks**: Task 7

  **References**:
  - `poc/poc-1c-scalable-ner/hybrid_ner.py:1500-1700` — Pipeline orchestration
  - `poc/poc-1c-scalable-ner/docs/V6_ARCHITECTURE.md` — Stage flow

  **Acceptance Criteria**:
  - [ ] `cd src && python -c "from plm.extraction import extract; print('OK')"` → OK
  - [ ] `cd src && python -c "from plm.extraction import fast_extract; r = fast_extract('Use React.Component'); print(r)"` → prints list with 'React.Component'
  - [ ] `cd src && python -c "from plm.extraction import ExtractionConfig; print('OK')"` → OK

  **Commit**: YES
  - Message: `feat(plm): add extraction pipeline orchestrator`
  - Files: `src/plm/extraction/pipeline.py`, `src/plm/extraction/__init__.py`

---

### Wave 3: Testing

- [x] 7. Create basic test structure and smoke tests

  **What to do**:
  
  Create pytest-based test structure with smoke tests.

  **`tests/conftest.py`**:
  ```python
  """Shared test fixtures."""
  import pytest

  @pytest.fixture
  def sample_text():
      return """
      Using React.Component with TypeScript for a NextJS app.
      The CrashLoopBackOff error appears when pods restart.
      Check kubectl logs and docker-compose.yml for issues.
      """

  @pytest.fixture
  def expected_terms():
      return [
          "React.Component",
          "TypeScript", 
          "NextJS",
          "CrashLoopBackOff",
          "kubectl",
          "docker-compose.yml",
      ]
  ```

  **`tests/unit/test_heuristic.py`**:
  ```python
  """Tests for fast heuristic extraction."""
  from plm.extraction.fast import (
      extract_camel_case,
      extract_dot_paths,
      extract_all_heuristic,
  )

  def test_extract_camel_case():
      text = "Use HelloWorld and FooBar"
      result = list(extract_camel_case(text))
      assert "HelloWorld" in result
      assert "FooBar" in result

  def test_extract_dot_paths():
      text = "Import React.Component and os.path.join"
      result = list(extract_dot_paths(text))
      assert "React.Component" in result
      assert "os.path.join" in result

  def test_extract_all_heuristic(sample_text):
      result = extract_all_heuristic(sample_text)
      assert len(result) > 0
      assert "React.Component" in result
  ```

  **`tests/unit/test_grounding.py`**:
  ```python
  """Tests for grounding functions."""
  from plm.extraction.slow.grounding import verify_span, normalize_term

  def test_verify_span_exact():
      assert verify_span("React", "Use React for UI")[0] == True

  def test_verify_span_not_found():
      assert verify_span("Vue", "Use React for UI")[0] == False

  def test_normalize_term():
      assert normalize_term("Hello-World") == "hello world"
      assert normalize_term("HELLO_WORLD") == "hello world"
  ```

  **`tests/integration/test_pipeline.py`**:
  ```python
  """Integration tests for extraction pipeline."""
  from plm.extraction import extract, fast_extract, ExtractionConfig

  def test_fast_extract_basic(sample_text):
      result = fast_extract(sample_text)
      assert isinstance(result, list)
      assert len(result) > 0

  def test_extract_with_config(sample_text):
      config = ExtractionConfig(use_fast_only=True)
      result = extract(sample_text, config)
      assert result.terms is not None
      assert len(result.fast_candidates) > 0
  ```

  **Must NOT do**:
  - Don't mock LLM calls yet (slow system is stub)
  - Don't create exhaustive tests (just smoke tests)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (final task)
  - **Blocked By**: Task 6

  **References**:
  - `poc/shared/tests/test_llm.py` — Test patterns

  **Acceptance Criteria**:
  - [ ] `ls tests/unit/` shows test_heuristic.py, test_grounding.py
  - [ ] `ls tests/integration/` shows test_pipeline.py
  - [ ] `cd src && python -m pytest ../tests/ -v --tb=short` → tests pass

  **Commit**: YES
  - Message: `test(plm): add basic test structure and smoke tests`
  - Files: `tests/**`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(plm): create package directory structure` | src/plm/**, tests/**, data/**, pyproject.toml |
| 2 | `feat(plm): add shared LLM provider (Anthropic + Gemini)` | src/plm/shared/llm/* |
| 3 | `feat(plm): add vocabulary data files` | data/vocabularies/* |
| 4 | `feat(plm): port V6 slow extraction pipeline from POC-1c` | src/plm/extraction/slow/* |
| 5 | `feat(plm): add fast heuristic extraction system` | src/plm/extraction/fast/* |
| 6 | `feat(plm): add extraction pipeline orchestrator` | src/plm/extraction/pipeline.py, __init__.py |
| 7 | `test(plm): add basic test structure and smoke tests` | tests/** |

---

## Success Criteria

### Verification Commands
```bash
# Package imports work
cd src && python -c "from plm.extraction import extract, fast_extract; print('OK')"
cd src && python -c "from plm.shared.llm import call_llm; print('OK')"

# Fast extraction works
cd src && python -c "
from plm.extraction import fast_extract
result = fast_extract('Using React.Component with TypeScript')
print(result)
assert 'React.Component' in result
print('PASS')
"

# Tests pass
python -m pytest tests/ -v
```

### Final Checklist
- [x] `src/plm/` package structure complete
- [x] LLM connector copied and working
- [x] Fast system (heuristic) implemented
- [x] Slow system modules ported (stubs OK for full pipeline)
- [x] Pipeline orchestrator working
- [x] Vocabulary files in data/
- [x] Basic tests passing
