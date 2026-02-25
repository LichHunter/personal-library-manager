# POC-1c: Scalable NER Extraction Without Vocabulary Lists

## TL;DR

> **Quick Summary**: Replace vocabulary-dependent NER extraction with two scalable approaches: (A) Retrieval-augmented few-shot using SO NER train.txt as example database, and (B) SLIMER-style structured prompting with entity definitions. Both eliminate the need for manually curated term lists that don't scale.
> 
> **Deliverables**:
> - New `poc-1c-scalable-ner/` directory with complete implementation
> - Approach A: Retrieval system using 741 train.txt documents
> - Approach B: SLIMER structured prompting (zero-shot)
> - Head-to-head benchmark comparison on 100 test documents
> - Analysis of which approach scales better
> 
> **Estimated Effort**: Large (3-5 days)
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Setup → Implement Approaches (parallel) → Benchmark → Analysis

---

## Context

### Original Request
Test two scalable NER extraction approaches that don't require vocabulary lists:
1. Retrieval-augmented few-shot using SO NER train.txt
2. SLIMER structured prompting with entity definitions

### Problem Statement
Current v6_spanfix strategy achieves 95.1% precision / 94.2% recall on 50 docs but relies on:
- `CONTEXT_VALIDATION_BYPASS`: 67 manually curated terms
- `MUST_EXTRACT_SEEDS`: 42 manually curated terms  
- `GT_NEGATIVE_TERMS`: 47 manually curated terms
- Various regex patterns and multi-word lists

**This doesn't scale**: Every new document set introduces new FPs/FNs requiring vocabulary expansion.

### Research Findings
1. **Retrieval-augmented few-shot**: Similar documents have similar entity patterns. Retrieve K similar annotated docs → use as few-shot examples → LLM learns GT style dynamically.

2. **SLIMER (ACL 2024)**: Rich entity definitions + annotation guidelines replace examples. Zero-shot with semantic descriptions achieves comparable results to few-shot.

3. **Dataset structure**: SO NER has proper train/dev/test splits with zero overlap:
   - train.txt: 741 documents (use for retrieval)
   - test.txt: 249 documents (use for evaluation)

### Base Strategy: Iteration 26 (v6_spanfix)
Best 50-doc result: P=95.1%, R=94.2%, H=4.9%, F1=0.946

Pipeline:
```
Phase 1: Triple extraction (Sonnet exhaustive + Haiku exhaustive + Haiku simple)
Phase 2: Span grounding (verify_span)
Phase 2.5: Span expansion (_try_expand_span)
Phase 3: Vote routing (2+ votes = auto-keep, 1 vote = Sonnet review)
Phase 4: Sonnet review (APPROVE/REJECT)
Phase 5: Noise filter (v6_filter + brace-expansion filter)
Phase 6: Context validation (ENTITY/GENERIC) ← REPLACE WITH NEW APPROACHES
Phase 6.5: Compound entity extraction
Phase 7: Must-extract seeding ← ELIMINATE WITH NEW APPROACHES
```

---

## Work Objectives

### Core Objective
Create poc-1c that implements and benchmarks two scalable NER approaches, eliminating vocabulary dependence while maintaining or improving iter 26 performance.

### Concrete Deliverables
1. `poc/poc-1c-scalable-ner/` directory with complete implementation
2. `retrieval_ner.py` — Approach A implementation
3. `slimer_ner.py` — Approach B implementation
4. `benchmark_comparison.py` — Head-to-head comparison script
5. `artifacts/train_embeddings.npy` — Embedded train documents
6. `artifacts/comparison_results.json` — Benchmark results
7. Documentation: README.md, RESULTS.md

### Definition of Done
- [ ] Both approaches run successfully on 100 test documents
- [ ] Precision ≥ 91% (match or beat iter 29 100-doc baseline)
- [ ] Recall ≥ 91%
- [ ] Hallucination ≤ 9%
- [ ] Zero vocabulary lists used in new approaches
- [ ] Clear winner identified with analysis

### Must Have
- Proper train/test separation (no data leakage)
- Retrieval safety: Jaccard filtering for near-duplicates
- Apples-to-apples comparison with iter 26 baseline
- Per-document metrics for statistical analysis

### Must NOT Have (Guardrails)
- NO vocabulary lists (CONTEXT_VALIDATION_BYPASS, MUST_EXTRACT_SEEDS, etc.)
- NO hard-coded term rules in new approaches
- NO data leakage (test docs in retrieval index)
- NO modification of test.txt or train.txt data

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL verification is executed by the agent using tools (Bash, file inspection).

### Test Decision
- **Infrastructure exists**: YES (existing benchmark framework)
- **Automated tests**: Tests-after (verify via benchmark metrics)
- **Framework**: Python assertions + JSON result validation

### Agent-Executed QA Scenarios

```
Scenario: POC-1c directory structure created correctly
  Tool: Bash
  Steps:
    1. ls -la poc/poc-1c-scalable-ner/
    2. Assert: utils/, artifacts/ directories exist
    3. Assert: pyproject.toml, README.md exist
    4. Assert: retrieval_ner.py, slimer_ner.py exist
  Expected Result: All required files and directories present

Scenario: Train embeddings generated successfully
  Tool: Bash
  Steps:
    1. ls -la poc/poc-1c-scalable-ner/artifacts/train_embeddings.npy
    2. python -c "import numpy as np; e = np.load('artifacts/train_embeddings.npy'); print(f'Shape: {e.shape}')"
    3. Assert: Shape is (741, embedding_dim)
  Expected Result: 741 document embeddings saved

Scenario: Retrieval approach benchmark completes
  Tool: Bash
  Steps:
    1. python benchmark_comparison.py --approach retrieval --n-docs 100
    2. Assert: Exit code 0
    3. Assert: artifacts/retrieval_results.json created
    4. grep "precision" artifacts/retrieval_results.json
  Expected Result: Benchmark completes with metrics logged

Scenario: SLIMER approach benchmark completes
  Tool: Bash
  Steps:
    1. python benchmark_comparison.py --approach slimer --n-docs 100
    2. Assert: Exit code 0
    3. Assert: artifacts/slimer_results.json created
  Expected Result: Benchmark completes with metrics logged

Scenario: No vocabulary lists used in new approaches
  Tool: Bash (grep)
  Steps:
    1. grep -n "CONTEXT_VALIDATION_BYPASS\|MUST_EXTRACT_SEEDS\|GT_NEGATIVE" retrieval_ner.py slimer_ner.py
    2. Assert: No matches found
  Expected Result: Zero vocabulary list references
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Create POC-1c directory structure
├── Task 2: Copy and adapt base files from POC-1b
└── Task 3: Parse train.txt and create document database

Wave 2 (After Wave 1):
├── Task 4: Implement Approach A - Retrieval-augmented
├── Task 5: Implement Approach B - SLIMER structured prompting
└── Task 6: Create benchmark comparison framework

Wave 3 (After Wave 2):
├── Task 7: Run benchmarks on both approaches
├── Task 8: Analyze results and document findings
└── Task 9: Create final comparison report

Critical Path: Task 1 → Task 4 → Task 7 → Task 8
Parallel Speedup: ~40% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3 | None |
| 2 | 1 | 4, 5, 6 | 3 |
| 3 | 1 | 4 | 2 |
| 4 | 2, 3 | 7 | 5, 6 |
| 5 | 2 | 7 | 4, 6 |
| 6 | 2 | 7 | 4, 5 |
| 7 | 4, 5, 6 | 8 | None |
| 8 | 7 | 9 | None |
| 9 | 8 | None | None |

---

## TODOs

### Wave 1: Setup

- [ ] 1. Create POC-1c directory structure

  **What to do**:
  ```bash
  mkdir -p poc/poc-1c-scalable-ner/{utils,artifacts,data}
  touch poc/poc-1c-scalable-ner/utils/__init__.py
  ```
  
  Create directory structure:
  ```
  poc/poc-1c-scalable-ner/
  ├── pyproject.toml
  ├── README.md
  ├── retrieval_ner.py          # Approach A
  ├── slimer_ner.py             # Approach B  
  ├── benchmark_comparison.py   # Comparison runner
  ├── parse_so_ner.py           # Train/test parser
  ├── utils/
  │   ├── __init__.py
  │   ├── llm_provider.py       # Copy from poc-1b
  │   └── logger.py             # Copy from poc-1b
  ├── artifacts/
  │   ├── train_embeddings.npy  # Generated
  │   ├── train_documents.json  # Parsed train.txt
  │   └── results/              # Benchmark results
  └── data/                     # Symlinks to SO NER data
  ```

  **Must NOT do**:
  - Don't copy unnecessary files from poc-1b
  - Don't create artifacts that should be generated

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed - filesystem operations only

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundational)
  - **Blocks**: Tasks 2, 3

  **References**:
  - `poc/poc-1b-llm-extraction-improvements/` — Structure reference
  - `poc/README.md` — POC guidelines

  **Acceptance Criteria**:
  - [ ] Directory structure created
  - [ ] `ls poc/poc-1c-scalable-ner/` shows expected structure

  **Commit**: YES
  - Message: `feat(poc-1c): create directory structure for scalable NER`
  - Files: `poc/poc-1c-scalable-ner/*`

---

- [ ] 2. Copy and adapt base files from POC-1b

  **What to do**:
  
  **2.1 Copy utility files (verbatim)**:
  ```bash
  cp poc/poc-1b-llm-extraction-improvements/utils/llm_provider.py poc/poc-1c-scalable-ner/utils/
  cp poc/poc-1b-llm-extraction-improvements/utils/logger.py poc/poc-1c-scalable-ner/utils/
  ```
  
  **2.2 Create pyproject.toml** (adapted from poc-1b):
  ```toml
  [project]
  name = "poc-1c-scalable-ner"
  version = "0.1.0"
  description = "Scalable NER extraction without vocabulary lists"
  requires-python = ">=3.11"
  dependencies = [
      "anthropic>=0.40.0",
      "httpx>=0.27.0",
      "pydantic>=2.0.0",
      "rapidfuzz>=3.0.0",
      "numpy>=1.26.0",
      "rich>=13.0.0",
      # For embeddings (Approach A)
      "sentence-transformers>=2.2.0",
      "faiss-cpu>=1.7.0",  # Vector search
  ]
  ```
  
  **2.3 Create README.md**:
  ```markdown
  # POC-1c: Scalable NER Extraction
  
  ## What
  Test two scalable NER approaches that eliminate vocabulary dependence:
  - Approach A: Retrieval-augmented few-shot
  - Approach B: SLIMER structured prompting
  
  ## Why
  POC-1b's vocabulary lists (176+ terms) don't scale beyond 100 docs.
  
  ## Dataset
  StackOverflow NER (Tabassum et al., ACL 2020)
  - train.txt: 741 docs (retrieval corpus)
  - test.txt: 249 docs (evaluation)
  
  ## Usage
  python benchmark_comparison.py --approach retrieval --n-docs 100
  python benchmark_comparison.py --approach slimer --n-docs 100
  python benchmark_comparison.py --approach both --n-docs 100
  ```

  **Must NOT do**:
  - Don't copy so_ner_benchmark.py verbatim (needs major refactoring)
  - Don't copy any vocabulary lists
  - Don't copy test_*.py files

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES with Task 3
  - **Blocks**: Tasks 4, 5, 6

  **References**:
  - `poc/poc-1b-llm-extraction-improvements/utils/llm_provider.py` — OAuth provider (copy verbatim)
  - `poc/poc-1b-llm-extraction-improvements/utils/logger.py` — Benchmark logger (copy verbatim)
  - `poc/poc-1b-llm-extraction-improvements/pyproject.toml` — Dependencies reference

  **Acceptance Criteria**:
  - [ ] `utils/llm_provider.py` copied and imports work
  - [ ] `utils/logger.py` copied and imports work
  - [ ] `pyproject.toml` created with correct dependencies
  - [ ] `uv sync` succeeds in poc-1c directory
  - [ ] `README.md` created with usage instructions

  **Commit**: YES
  - Message: `feat(poc-1c): copy utilities and create project config`
  - Files: `poc/poc-1c-scalable-ner/utils/*.py`, `poc/poc-1c-scalable-ner/pyproject.toml`

---

- [ ] 3. Parse train.txt and create document database

  **What to do**:
  
  Create `parse_so_ner.py` that:
  1. Parses BIO-tagged train.txt (741 docs) and test.txt (249 docs)
  2. Extracts document text and GT entities for each
  3. Saves to JSON for easy loading
  
  **Parser logic** (adapt from poc-1b so_ner_benchmark.py lines 90-336):
  ```python
  def parse_so_ner_file(filepath: str) -> list[dict]:
      """Parse SO NER BIO-tagged file into documents.
      
      Returns list of:
      {
          "doc_id": "Q12345",
          "text": "reconstructed document text...",
          "gt_terms": ["React", "JavaScript", "npm"],
          "entities": [{"text": "React", "type": "Library", "start": 10, "end": 15}]
      }
      """
  ```
  
  **Entity types to include** (from SO NER):
  - Library, Library_Class, Library_Function, Library_Variable
  - Language, Application, Operating_System
  - Data_Structure, Data_Type, Algorithm
  - Device, File_Type, Website, Version
  - HTML_XML_Tag, Error_Name, User_Interface_Element
  
  **Entity types to EXCLUDE** (code artifacts):
  - Code_Block, Output_Block, Variable_Name, Value, User_Name
  
  **Output files**:
  - `artifacts/train_documents.json` — 741 parsed train docs
  - `artifacts/test_documents.json` — 249 parsed test docs

  **Must NOT do**:
  - Don't modify original train.txt/test.txt files
  - Don't include Code_Block/Output_Block entities

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES with Task 2
  - **Blocks**: Task 4

  **References**:
  - `poc/poc-1b-llm-extraction-improvements/so_ner_benchmark.py:90-336` — BIO parser logic
  - `/tmp/StackOverflowNER/resources/annotated_ner_data/StackOverflow/train.txt` — Train data
  - `/tmp/StackOverflowNER/resources/annotated_ner_data/StackOverflow/test.txt` — Test data
  - `/tmp/StackOverflowNER/resources/annotated_ner_data/Readme.md` — Data format docs

  **Acceptance Criteria**:
  - [ ] `parse_so_ner.py` created with `parse_so_ner_file()` function
  - [ ] `python parse_so_ner.py` generates both JSON files
  - [ ] `artifacts/train_documents.json` has 741 documents
  - [ ] `artifacts/test_documents.json` has 249 documents
  - [ ] Each document has doc_id, text, gt_terms fields
  - [ ] Sample document spot-check: GT terms match BIO tags

  **Commit**: YES
  - Message: `feat(poc-1c): add SO NER parser with train/test document extraction`
  - Files: `poc/poc-1c-scalable-ner/parse_so_ner.py`, `poc/poc-1c-scalable-ner/artifacts/*.json`

---

### Wave 2: Implementation

- [ ] 4. Implement Approach A — Retrieval-augmented few-shot

  **What to do**:
  
  Create `retrieval_ner.py` with:
  
  **4.1 Document embedding** (one-time setup):
  ```python
  from sentence_transformers import SentenceTransformer
  import faiss
  import numpy as np
  
  def build_retrieval_index(train_docs: list[dict]) -> tuple[faiss.Index, np.ndarray]:
      """Embed all 741 train documents and build FAISS index."""
      model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, good quality
      
      texts = [doc["text"][:2000] for doc in train_docs]  # Truncate for embedding
      embeddings = model.encode(texts, show_progress_bar=True)
      
      # Build FAISS index
      dimension = embeddings.shape[1]
      index = faiss.IndexFlatIP(dimension)  # Inner product (cosine after normalization)
      faiss.normalize_L2(embeddings)
      index.add(embeddings)
      
      return index, embeddings
  ```
  
  **4.2 Safe retrieval with leakage prevention**:
  ```python
  def safe_retrieve(
      test_doc: dict,
      train_docs: list[dict],
      index: faiss.Index,
      embeddings: np.ndarray,
      model: SentenceTransformer,
      k: int = 5,
      jaccard_threshold: float = 0.8
  ) -> list[dict]:
      """Retrieve K similar train docs with near-duplicate filtering."""
      # Embed test document
      test_embedding = model.encode([test_doc["text"][:2000]])
      faiss.normalize_L2(test_embedding)
      
      # Search for top 2*k candidates (to allow filtering)
      distances, indices = index.search(test_embedding, k * 2)
      
      # Filter near-duplicates using Jaccard similarity
      filtered = []
      for idx in indices[0]:
          candidate = train_docs[idx]
          jaccard = compute_jaccard(test_doc["text"], candidate["text"])
          if jaccard < jaccard_threshold:
              filtered.append(candidate)
          if len(filtered) >= k:
              break
      
      return filtered
  
  def compute_jaccard(text1: str, text2: str, n: int = 13) -> float:
      """Character n-gram Jaccard similarity."""
      grams1 = set(text1[i:i+n] for i in range(len(text1)-n+1))
      grams2 = set(text2[i:i+n] for i in range(len(text2)-n+1))
      if not grams1 or not grams2:
          return 0.0
      return len(grams1 & grams2) / len(grams1 | grams2)
  ```
  
  **4.3 Few-shot extraction prompt**:
  ```python
  def build_fewshot_prompt(test_doc: dict, retrieved_docs: list[dict]) -> str:
      """Build few-shot prompt with retrieved examples."""
      examples = []
      for doc in retrieved_docs:
          examples.append(f"""
  TEXT: {doc["text"][:800]}...
  ENTITIES: {json.dumps(doc["gt_terms"])}
  """)
      
      return f"""You are extracting technical entities from StackOverflow posts.
  
  Here are examples of correct entity extraction from similar posts:

  {"".join(examples)}

  Now extract entities from this text following the same annotation style:

  TEXT: {test_doc["text"]}

  ENTITIES (JSON array of strings):"""
  ```
  
  **4.4 Main extraction function**:
  ```python
  def extract_with_retrieval(
      test_doc: dict,
      train_docs: list[dict],
      index: faiss.Index,
      embeddings: np.ndarray,
      model: SentenceTransformer,
      k: int = 5
  ) -> list[str]:
      """Extract entities using retrieval-augmented few-shot."""
      # Retrieve similar train documents
      similar_docs = safe_retrieve(test_doc, train_docs, index, embeddings, model, k)
      
      # Build few-shot prompt
      prompt = build_fewshot_prompt(test_doc, similar_docs)
      
      # Call LLM
      response = call_llm(prompt, model="sonnet", max_tokens=2000, temperature=0.0)
      
      # Parse response
      entities = parse_entity_response(response)
      
      # Verify spans exist in text (from iter 26 pipeline)
      verified = [e for e in entities if verify_span(e, test_doc["text"])]
      
      return verified
  ```

  **Must NOT do**:
  - Don't use any vocabulary lists (BYPASS, SEEDS, GT_NEGATIVE)
  - Don't include test documents in retrieval index
  - Don't skip Jaccard filtering

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None specific needed
  - Reason: Novel implementation requiring careful design

  **Parallelization**:
  - **Can Run In Parallel**: YES with Tasks 5, 6
  - **Blocked By**: Tasks 2, 3
  - **Blocks**: Task 7

  **References**:
  - `poc/poc-1b-llm-extraction-improvements/so_ner_benchmark.py:1420-1600` — verify_span(), parse logic
  - `poc/poc-1b-llm-extraction-improvements/utils/llm_provider.py` — call_llm() function
  - Research: BANER, GEIC papers on retrieval-augmented NER

  **Acceptance Criteria**:
  - [ ] `retrieval_ner.py` created with all functions above
  - [ ] `build_retrieval_index()` creates FAISS index from 741 train docs
  - [ ] `safe_retrieve()` returns K docs with Jaccard filtering
  - [ ] `extract_with_retrieval()` returns list of entity strings
  - [ ] No vocabulary list imports or references (verify with grep)
  - [ ] Unit test: Extract from 1 test doc, get non-empty results

  **Commit**: YES
  - Message: `feat(poc-1c): implement retrieval-augmented NER extraction`
  - Files: `poc/poc-1c-scalable-ner/retrieval_ner.py`

---

- [ ] 5. Implement Approach B — SLIMER structured prompting

  **What to do**:
  
  Create `slimer_ner.py` with:
  
  **5.1 Entity type definitions** (based on SO NER annotation guidelines):
  ```python
  ENTITY_DEFINITIONS = """
  ## Entity Types for StackOverflow Technical Content

  ### LIBRARY_FRAMEWORK
  Definition: Specific software libraries, frameworks, or packages that developers import/use.
  Examples: React, TensorFlow, jQuery, pandas, Spring Boot, Express.js, NumPy, Django
  Boundary: Extract the library name only, not "the React library" → "React"
  NOT entities: "library", "framework", "package", "module" (generic terms)

  ### PROGRAMMING_LANGUAGE
  Definition: Named programming, scripting, or markup languages.
  Examples: Python, JavaScript, C++, TypeScript, HTML5, SQL, Rust, Go, Ruby
  Include versions: "Python 3.9", "C++11", "ES6"
  NOT entities: "language", "code", "script", "markup" (generic terms)

  ### API_FUNCTION_CLASS
  Definition: Specific API endpoints, functions, methods, class names, or interfaces.
  Examples: querySelector(), ArrayList, HttpClient, console.log(), useState(), Promise
  Include: Method signatures with parentheses
  NOT entities: "function", "method", "class", "API", "endpoint" (generic terms)

  ### APPLICATION_TOOL
  Definition: Specific software applications, IDEs, developer tools, or services.
  Examples: Visual Studio, Chrome, Docker, Postman, Git, npm, Webpack, VS Code
  NOT entities: "application", "tool", "editor", "browser", "IDE" (generic terms)

  ### PLATFORM_OS
  Definition: Operating systems, platforms, or runtime environments.
  Examples: Windows, Linux, macOS, Android, iOS, Node.js, .NET, JVM
  NOT entities: "platform", "operating system", "environment" (generic terms)

  ### DATA_STRUCTURE_TYPE
  Definition: Specific data structures, types, or formats.
  Examples: HashMap, ArrayList, JSON, XML, DataFrame, int, string, boolean
  Include: Language-specific type names
  NOT entities: "data structure", "type", "format", "object" (generic terms)

  ### ERROR_EXCEPTION
  Definition: Specific error types, exception classes, or error codes.
  Examples: NullPointerException, TypeError, 404, ECONNREFUSED, StackOverflowError
  NOT entities: "error", "exception", "bug", "issue" (generic terms)

  ### UI_COMPONENT
  Definition: Specific UI component types or widget names.
  Examples: Button, TextField, ListView, RecyclerView, Modal, Dropdown
  Context-dependent: "button" in UI context IS an entity; in "press the button" is NOT
  NOT entities: "component", "widget", "element", "control" (generic terms)

  ### WEBSITE_SERVICE
  Definition: Specific websites, web services, or online platforms.
  Examples: GitHub, StackOverflow, npm registry, AWS, Google Cloud, Heroku
  NOT entities: "website", "service", "cloud", "server" (generic terms)
  """
  ```
  
  **5.2 Annotation guidelines**:
  ```python
  ANNOTATION_GUIDELINES = """
  ## Annotation Guidelines

  ### General Rules
  1. ONLY extract proper nouns or specific product/technology names
  2. Extract the MINIMAL span: "the React library" → extract "React"
  3. Include version numbers when present: "Python 3.9", "C++11"
  4. Context matters: same word can be entity or not depending on usage

  ### What IS an entity
  - Specific named technologies you could look up documentation for
  - Terms that refer to a PARTICULAR implementation, not a category
  - Words that would be capitalized in formal technical writing

  ### What is NOT an entity
  - Generic programming vocabulary: object, class, function, method, variable
  - Descriptive nouns: server, client, request, response, database, file
  - Actions: compile, execute, deploy, install, configure
  - Concepts: authentication, authorization, caching, threading

  ### Context-Dependent Decisions
  - "python" in "I wrote a python script" → ENTITY (the language)
  - "python" in "python is a snake" → NOT entity (the animal)
  - "table" in "HTML table element" → ENTITY (UI component)
  - "table" in "database table" → ENTITY (data structure type)
  - "table" in "put it on the table" → NOT entity (furniture)

  ### Boundary Rules
  - Method calls: include parentheses → "getElementById()"
  - Namespaced: extract full path → "React.Component", "java.util.List"
  - Version-qualified: include version → "Python 3.9", "ES2020"
  - Compound names: keep together → "Visual Studio Code", "Spring Boot"
  """
  ```
  
  **5.3 Extraction prompt with chain-of-thought**:
  ```python
  SLIMER_PROMPT = """
  {entity_definitions}

  {annotation_guidelines}

  ## Task

  Extract all technical entities from the following StackOverflow post.

  TEXT:
  {text}

  Think step-by-step:
  1. Identify all noun phrases that could be technologies
  2. For each, determine if it's a specific named technology or generic term
  3. Apply the annotation guidelines to decide inclusion
  4. Extract minimal spans with proper boundaries

  Reasoning:
  [Provide brief reasoning for key decisions]

  ENTITIES (JSON array of strings):
  """
  
  def extract_with_slimer(doc: dict) -> list[str]:
      """Extract entities using SLIMER structured prompting."""
      prompt = SLIMER_PROMPT.format(
          entity_definitions=ENTITY_DEFINITIONS,
          annotation_guidelines=ANNOTATION_GUIDELINES,
          text=doc["text"][:4000]  # Truncate for context window
      )
      
      response = call_llm(prompt, model="sonnet", max_tokens=2000, temperature=0.0)
      
      # Parse JSON array from response
      entities = parse_entity_response(response)
      
      # Verify spans exist in text
      verified = [e for e in entities if verify_span(e, doc["text"])]
      
      return verified
  ```

  **Must NOT do**:
  - Don't use any vocabulary lists
  - Don't hardcode specific terms to include/exclude
  - Don't reference train.txt data (zero-shot approach)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: None specific needed
  - Reason: Careful prompt engineering required

  **Parallelization**:
  - **Can Run In Parallel**: YES with Tasks 4, 6
  - **Blocked By**: Task 2
  - **Blocks**: Task 7

  **References**:
  - SLIMER paper (ACL 2024): "Show Less, Instruct More"
  - `poc/poc-1b-llm-extraction-improvements/so_ner_benchmark.py:1028-1060` — Context validation prompt structure
  - `/tmp/StackOverflowNER/resources/annotated_ner_data/Readme.md` — Entity type definitions

  **Acceptance Criteria**:
  - [ ] `slimer_ner.py` created with ENTITY_DEFINITIONS, ANNOTATION_GUIDELINES
  - [ ] `extract_with_slimer()` function works standalone
  - [ ] No vocabulary list imports or references (verify with grep)
  - [ ] No references to train.txt (pure zero-shot)
  - [ ] Unit test: Extract from 1 test doc, get non-empty results
  - [ ] Prompt includes chain-of-thought reasoning step

  **Commit**: YES
  - Message: `feat(poc-1c): implement SLIMER structured prompting NER`
  - Files: `poc/poc-1c-scalable-ner/slimer_ner.py`

---

- [ ] 6. Create benchmark comparison framework

  **What to do**:
  
  Create `benchmark_comparison.py` that:
  
  **6.1 Unified benchmark runner**:
  ```python
  def run_benchmark(
      approach: str,  # "retrieval", "slimer", "baseline"
      test_docs: list[dict],
      train_docs: list[dict] = None,  # Only for retrieval
      n_docs: int = 100,
      seed: int = 42
  ) -> dict:
      """Run benchmark for specified approach."""
      
      # Select documents (same as iter 26)
      selected = select_documents(test_docs, n_docs, seed)
      
      results = {
          "approach": approach,
          "n_docs": n_docs,
          "documents": [],
          "totals": {"tp": 0, "fp": 0, "fn": 0}
      }
      
      # Setup for retrieval approach
      if approach == "retrieval":
          index, embeddings, model = setup_retrieval(train_docs)
      
      for doc in tqdm(selected, desc=f"Benchmark ({approach})"):
          # Extract based on approach
          if approach == "retrieval":
              extracted = extract_with_retrieval(doc, train_docs, index, embeddings, model)
          elif approach == "slimer":
              extracted = extract_with_slimer(doc)
          elif approach == "baseline":
              extracted = extract_with_baseline(doc)  # Iter 26 pipeline
          
          # Score
          tp, fp, fn = score_extraction(extracted, doc["gt_terms"], doc["text"])
          
          results["documents"].append({
              "doc_id": doc["doc_id"],
              "extracted": extracted,
              "gt_terms": doc["gt_terms"],
              "tp": tp, "fp": fp, "fn": fn
          })
          results["totals"]["tp"] += tp
          results["totals"]["fp"] += fp
          results["totals"]["fn"] += fn
      
      # Calculate metrics
      tp, fp, fn = results["totals"]["tp"], results["totals"]["fp"], results["totals"]["fn"]
      results["metrics"] = {
          "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
          "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
          "hallucination": fp / (tp + fp) if (tp + fp) > 0 else 0,
          "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
      }
      
      return results
  ```
  
  **6.2 Scoring function** (from iter 26):
  ```python
  def score_extraction(extracted: list[str], gt_terms: list[str], text: str) -> tuple[int, int, int]:
      """Score extraction using m2m_v3 matching (from iter 26)."""
      # Normalize terms
      extracted_norm = {normalize_term(e) for e in extracted}
      gt_norm = {normalize_term(g) for g in gt_terms}
      
      # Many-to-many matching with fuzzy threshold
      tp, fp, fn, matches = many_to_many_score(
          list(extracted_norm), 
          list(gt_norm),
          threshold=80  # Same as iter 26
      )
      
      return tp, fp, fn
  ```
  
  **6.3 CLI interface**:
  ```python
  if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--approach", choices=["retrieval", "slimer", "baseline", "all"], required=True)
      parser.add_argument("--n-docs", type=int, default=100)
      parser.add_argument("--seed", type=int, default=42)
      args = parser.parse_args()
      
      # Load data
      train_docs = load_json("artifacts/train_documents.json")
      test_docs = load_json("artifacts/test_documents.json")
      
      approaches = [args.approach] if args.approach != "all" else ["retrieval", "slimer", "baseline"]
      
      for approach in approaches:
          print(f"\n{'='*60}")
          print(f"Running {approach} approach on {args.n_docs} documents")
          print('='*60)
          
          results = run_benchmark(approach, test_docs, train_docs, args.n_docs, args.seed)
          
          # Save results
          save_json(results, f"artifacts/results/{approach}_results.json")
          
          # Print summary
          m = results["metrics"]
          print(f"\nResults for {approach}:")
          print(f"  Precision:     {m['precision']*100:.1f}%")
          print(f"  Recall:        {m['recall']*100:.1f}%")
          print(f"  Hallucination: {m['hallucination']*100:.1f}%")
          print(f"  F1:            {m['f1']:.3f}")
  ```

  **Must NOT do**:
  - Don't modify scoring logic (use same as iter 26 for fair comparison)
  - Don't change document selection logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: YES with Tasks 4, 5
  - **Blocked By**: Task 2
  - **Blocks**: Task 7

  **References**:
  - `poc/poc-1b-llm-extraction-improvements/so_ner_benchmark.py:1420-1600` — Benchmark runner pattern
  - `poc/poc-1b-llm-extraction-improvements/test_dplus_v3_sweep.py:100-200` — many_to_many_score, normalize_term
  - `poc/poc-1b-llm-extraction-improvements/so_ner_benchmark.py:299-334` — select_documents()

  **Acceptance Criteria**:
  - [ ] `benchmark_comparison.py` created with CLI interface
  - [ ] `python benchmark_comparison.py --approach slimer --n-docs 10` runs successfully
  - [ ] Results saved to `artifacts/results/`
  - [ ] Metrics match expected format (precision, recall, hallucination, f1)
  - [ ] Same document selection as iter 26 (verified by doc_ids)

  **Commit**: YES
  - Message: `feat(poc-1c): add benchmark comparison framework`
  - Files: `poc/poc-1c-scalable-ner/benchmark_comparison.py`

---

### Wave 3: Evaluation

- [ ] 7. Run benchmarks on both approaches

  **What to do**:
  
  Run full benchmarks on 100 test documents for all three approaches:
  
  ```bash
  # Approach A: Retrieval-augmented
  source .venv/bin/activate
  nohup python benchmark_comparison.py --approach retrieval --n-docs 100 --seed 42 \
      > /tmp/poc1c_retrieval.log 2>&1 &
  
  # Approach B: SLIMER
  nohup python benchmark_comparison.py --approach slimer --n-docs 100 --seed 42 \
      > /tmp/poc1c_slimer.log 2>&1 &
  
  # Baseline: Iter 26 equivalent (for comparison)
  nohup python benchmark_comparison.py --approach baseline --n-docs 100 --seed 42 \
      > /tmp/poc1c_baseline.log 2>&1 &
  ```
  
  **Expected runtime**: ~30-45 min per approach (100 docs × ~20-30 sec/doc)
  
  **Monitor progress**:
  ```bash
  tail -f /tmp/poc1c_retrieval.log
  grep "Precision\|Recall" /tmp/poc1c_*.log
  ```

  **Must NOT do**:
  - Don't run inline (use nohup for long runs)
  - Don't skip baseline comparison

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: None needed
  - Note: Long-running background tasks

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on all implementations)
  - **Blocked By**: Tasks 4, 5, 6
  - **Blocks**: Task 8

  **References**:
  - `/tmp/iter29_output.log` — Reference for benchmark output format
  - Iter 26 results: P=95.1%, R=94.2%, H=4.9% (50 docs)
  - Iter 29 results: P=91.0%, R=91.6%, H=9.0% (100 docs)

  **Acceptance Criteria**:
  - [ ] All three benchmarks complete without error
  - [ ] `artifacts/results/retrieval_results.json` exists with 100 docs
  - [ ] `artifacts/results/slimer_results.json` exists with 100 docs
  - [ ] `artifacts/results/baseline_results.json` exists with 100 docs
  - [ ] All results have precision, recall, hallucination, f1 metrics

  **Commit**: NO (benchmark output, not code)

---

- [ ] 8. Analyze results and document findings

  **What to do**:
  
  Create analysis comparing all three approaches:
  
  **8.1 Metric comparison table**:
  ```markdown
  | Approach | Precision | Recall | Hallucination | F1 |
  |----------|-----------|--------|---------------|-----|
  | Retrieval | X% | X% | X% | X.XXX |
  | SLIMER | X% | X% | X% | X.XXX |
  | Baseline (iter 26) | 91.0% | 91.6% | 9.0% | 0.913 |
  ```
  
  **8.2 FP/FN breakdown**:
  - Top 10 FP terms for each approach
  - Top 10 FN terms for each approach
  - Overlap analysis: Do approaches miss same terms?
  
  **8.3 Per-document analysis**:
  - Documents where retrieval > SLIMER
  - Documents where SLIMER > retrieval
  - Correlation with document characteristics
  
  **8.4 Scalability assessment**:
  - Did we eliminate vocabulary dependence?
  - Which approach generalizes better to unseen docs?
  - Cost comparison (API calls, embeddings)
  
  **Output**: `RESULTS.md` in poc-1c directory

  **Must NOT do**:
  - Don't cherry-pick results
  - Don't ignore failure cases

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocked By**: Task 7
  - **Blocks**: Task 9

  **References**:
  - `artifacts/results/*.json` — Raw benchmark results
  - Iter 26/29 FP/FN analysis from session context

  **Acceptance Criteria**:
  - [ ] `RESULTS.md` created with metric comparison
  - [ ] FP/FN breakdown for each approach
  - [ ] Clear winner identified with reasoning
  - [ ] Scalability assessment included
  - [ ] Recommendations for production use

  **Commit**: YES
  - Message: `docs(poc-1c): add benchmark results analysis`
  - Files: `poc/poc-1c-scalable-ner/RESULTS.md`

---

- [ ] 9. Create final comparison report

  **What to do**:
  
  Synthesize findings into executive summary with recommendations:
  
  **9.1 Key findings**:
  - Which approach won?
  - By how much?
  - What are the trade-offs?
  
  **9.2 Scalability verdict**:
  - Did we solve the vocabulary scaling problem?
  - What's the new ceiling?
  - What's the cost (API calls, latency)?
  
  **9.3 Recommendations**:
  - For production: Which approach to use?
  - For further research: What to try next?
  - For poc-1b: Should we migrate?
  
  **Output**: Update `README.md` with results summary

  **Must NOT do**:
  - Don't leave without clear recommendation

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: None needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocked By**: Task 8
  - **Blocks**: None (final task)

  **References**:
  - `RESULTS.md` — Detailed analysis
  - `artifacts/results/*.json` — Raw data

  **Acceptance Criteria**:
  - [ ] `README.md` updated with results summary
  - [ ] Clear recommendation section
  - [ ] Next steps documented
  - [ ] POC marked as complete

  **Commit**: YES
  - Message: `docs(poc-1c): complete POC with final recommendations`
  - Files: `poc/poc-1c-scalable-ner/README.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(poc-1c): create directory structure for scalable NER` | poc-1c/* | ls -la |
| 2 | `feat(poc-1c): copy utilities and create project config` | utils/*.py, pyproject.toml | uv sync |
| 3 | `feat(poc-1c): add SO NER parser with train/test document extraction` | parse_so_ner.py, artifacts/*.json | python parse_so_ner.py |
| 4 | `feat(poc-1c): implement retrieval-augmented NER extraction` | retrieval_ner.py | unit test |
| 5 | `feat(poc-1c): implement SLIMER structured prompting NER` | slimer_ner.py | unit test |
| 6 | `feat(poc-1c): add benchmark comparison framework` | benchmark_comparison.py | --help |
| 8 | `docs(poc-1c): add benchmark results analysis` | RESULTS.md | review |
| 9 | `docs(poc-1c): complete POC with final recommendations` | README.md | review |

---

## Success Criteria

### Verification Commands
```bash
# All files exist
ls poc/poc-1c-scalable-ner/{retrieval_ner.py,slimer_ner.py,benchmark_comparison.py}

# No vocabulary lists
grep -r "CONTEXT_VALIDATION_BYPASS\|MUST_EXTRACT_SEEDS\|GT_NEGATIVE" poc/poc-1c-scalable-ner/*.py

# Benchmarks complete
cat poc/poc-1c-scalable-ner/artifacts/results/*_results.json | jq '.metrics'

# Results documented
head -50 poc/poc-1c-scalable-ner/RESULTS.md
```

### Final Checklist
- [ ] Both approaches implemented without vocabulary lists
- [ ] Benchmarks run on 100 test documents
- [ ] Metrics comparable to iter 26 baseline
- [ ] Clear winner identified
- [ ] Scalability problem addressed
- [ ] Documentation complete

---

## Appendix: Key Code References

### From poc-1b so_ner_benchmark.py (Iteration 26 state)

**CONTEXT_VALIDATION_BYPASS (line 1063-1088)**: 67 terms — TO BE ELIMINATED
**MUST_EXTRACT_SEEDS (line 1314-1333)**: 42 terms — TO BE ELIMINATED
**GT_NEGATIVE_TERMS (line ~763)**: 47 terms — TO BE ELIMINATED

**verify_span() (from test_dplus_v3_sweep.py)**: Keep — span grounding is valid
**normalize_term() (from test_dplus_v3_sweep.py)**: Keep — needed for scoring
**many_to_many_score() (from test_dplus_v3_sweep.py)**: Keep — needed for scoring
**select_documents() (line 299-334)**: Keep — same document selection

### SO NER Data Locations

- Train: `/tmp/StackOverflowNER/resources/annotated_ner_data/StackOverflow/train.txt`
- Test: `/tmp/StackOverflowNER/resources/annotated_ner_data/StackOverflow/test.txt`
- Dev: `/tmp/StackOverflowNER/resources/annotated_ner_data/StackOverflow/dev.txt`
- Format docs: `/tmp/StackOverflowNER/resources/annotated_ner_data/Readme.md`
