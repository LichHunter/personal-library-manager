# Slow Extraction Docker Image

## TL;DR

> **Quick Summary**: Build a Nix-based Docker image for the slow extraction system that watches folders for documents, processes them with V6 pipeline, and outputs JSON with extracted terms. Includes chunking strategies and comprehensive tests.
> 
> **Deliverables**:
> - `src/plm/extraction/chunking/` module with `WholeChunker` and `HeadingChunker`
> - `src/plm/extraction/cli.py` Docker entrypoint
> - `src/plm/extraction/slow/flake.nix` build flake
> - Updated `flake.nix` with slow-extraction packages
> - Comprehensive unit tests for chunking strategies
> - LLM provider env var support for Docker
> 
> **Estimated Effort**: Large
> **Parallel Execution**: Partial - Tasks 1-2 can parallel, rest sequential
> **Critical Path**: Task 1 → Task 3 → Task 4 → Task 5 → Task 6 → Task 7

---

## Context

### Original Request
Build a Docker image (via Nix flake) for the slow extraction system that:
- Watches a folder for new documents
- Processes documents with existing `src/plm/extraction/slow/` V6 pipeline
- Outputs chunks with extracted terms (multiline text preserved)
- Logs low-confidence terms separately
- Supports configurable chunking strategies via env var or CLI flag
- Vocabularies provided via external paths

### Technical Context
- Slow extraction V6 pipeline exists at `src/plm/extraction/slow/`
- LLM providers at `src/plm/shared/llm/` (currently OpenCode OAuth only)
- Chunking reference: `poc/chunking_benchmark_v2/strategies/paragraph_heading.py`
- Build flake pattern from user's Java project example

### Key Decisions
- **LLM auth**: env var (`ANTHROPIC_API_KEY`) OR OpenCode auth file path
- **Vocabularies**: external paths (not embedded in image)
- **Chunking**: default = whole document, optional = heading-based
- **Testing flags**: `PROCESS_ONCE`, `DRY_RUN`
- **Nix approach**: uv2nix with separate build flake
- **Build flake location**: `src/plm/extraction/slow/flake.nix`

---

## Work Objectives

### Core Objective
Create a containerized slow extraction service with configurable chunking, external vocabulary support, and comprehensive test coverage.

### Concrete Deliverables
- `src/plm/extraction/chunking/__init__.py`
- `src/plm/extraction/chunking/base.py`
- `src/plm/extraction/chunking/whole.py`
- `src/plm/extraction/chunking/heading.py`
- `src/plm/extraction/cli.py`
- `src/plm/extraction/slow/flake.nix`
- Updated `flake.nix`
- Updated `src/plm/shared/llm/anthropic_provider.py`
- `tests/unit/test_chunking_base.py`
- `tests/unit/test_chunking_whole.py`
- `tests/unit/test_chunking_heading.py`
- `tests/integration/test_chunking_integration.py`

### Definition of Done
- [ ] `nix build .#slow-extraction` produces working Python venv
- [ ] `nix build .#slow-extraction-docker` produces loadable Docker image
- [ ] Docker container processes documents and outputs JSON
- [ ] Both chunking strategies work (whole/heading)
- [ ] Low-confidence terms logged to JSONL
- [ ] All unit tests pass
- [ ] `nix develop` still works (devShell preserved)

### Must Have
- Chunker class hierarchy with registry pattern
- WholeChunker (default) and HeadingChunker strategies
- CLI with env var and flag support
- Multiline text preservation in output JSON
- External vocabulary path configuration
- API key OR OAuth file authentication
- Comprehensive per-class unit tests

### Must NOT Have (Guardrails)
- No embedded vocabularies in Docker image
- No hardcoded API keys or paths
- No inotify/watchdog dependency (polling only)
- No breaking changes to existing devShell
- No web server or API endpoints (batch processing only)

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (pytest)
- **Automated tests**: YES (tests-after for chunking, TDD optional)
- **Framework**: pytest

### Agent-Executed QA Scenarios

**Scenario: Nix Python venv builds successfully**
```
Tool: Bash
Steps:
  1. nix build .#slow-extraction
  2. Assert: exit code 0
  3. ./result/bin/python -c "from plm.extraction.chunking import get_chunker; print(get_chunker('whole').name)"
  4. Assert: output is "whole"
Evidence: Command output captured
```

**Scenario: Nix Docker image builds successfully**
```
Tool: Bash
Steps:
  1. nix build .#slow-extraction-docker
  2. Assert: exit code 0
  3. docker load < result
  4. Assert: "Loaded image" in output
  5. docker run --rm plm-slow-extraction:0.1.0 python -c "from plm.extraction.chunking import get_chunker; print('OK')"
  6. Assert: output is "OK"
Evidence: Command output captured
```

**Scenario: Docker container processes document**
```
Tool: Bash
Preconditions: Docker image loaded, test directories created
Steps:
  1. mkdir -p /tmp/plm-test/{input,output,logs,vocabularies}
  2. cp data/vocabularies/*.json /tmp/plm-test/vocabularies/
  3. echo "# Test Doc\n\nUsing React with TypeScript." > /tmp/plm-test/input/test.md
  4. docker run --rm \
       -v /tmp/plm-test/input:/data/input \
       -v /tmp/plm-test/output:/data/output \
       -v /tmp/plm-test/logs:/data/logs \
       -v /tmp/plm-test/vocabularies:/data/vocabularies \
       -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
       -e PROCESS_ONCE=true \
       plm-slow-extraction:0.1.0
  5. Assert: /tmp/plm-test/output/test.json exists
  6. cat /tmp/plm-test/output/test.json | jq '.chunks[0].text'
  7. Assert: multiline text preserved (contains \n)
Evidence: Output JSON captured
```

**Scenario: Unit tests pass**
```
Tool: Bash
Steps:
  1. cd /home/susano/Code/personal-library-manager
  2. PYTHONPATH=src pytest tests/unit/test_chunking*.py -v
  3. Assert: exit code 0
  4. Assert: all tests passed
Evidence: pytest output captured
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: LLM Provider API key support
└── Task 2: Chunking module + tests

Wave 2 (After Wave 1):
└── Task 3: CLI entrypoint

Wave 3 (After Wave 2):
└── Task 4: Build flake + verify builds

Wave 4 (After Wave 3):
└── Task 5: Main flake integration

Wave 5 (After Wave 4):
└── Task 6: Docker E2E test

Critical Path: Task 2 → Task 3 → Task 4 → Task 5 → Task 6
```

---

## TODOs

- [x] 1. Add API Key Support to LLM Provider

  **What to do**:
  - Modify `src/plm/shared/llm/anthropic_provider.py`:
    - Check `ANTHROPIC_API_KEY` env var first
    - If set, use direct API key authentication (not OAuth)
    - Fall back to OpenCode auth file if no env var
    - Support `OPENCODE_AUTH_PATH` env var for custom auth file location
  - Update `__init__` to handle both auth modes
  - Add `_use_api_key` flag to switch between direct API and OAuth

  **Must NOT do**:
  - Don't break existing OpenCode OAuth flow
  - Don't store API keys in code

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 2)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:
  - `src/plm/shared/llm/anthropic_provider.py:59-108` - Current auth implementation
  - `src/plm/shared/llm/anthropic_provider.py:263-272` - Current header construction

  **Acceptance Criteria**:
  - [ ] `ANTHROPIC_API_KEY=sk-xxx python -c "from plm.shared.llm import call_llm"` works
  - [ ] Existing OpenCode OAuth still works when env var not set
  - [ ] `OPENCODE_AUTH_PATH=/custom/path` respected

  **Commit**: YES
  - Message: `feat(llm): support ANTHROPIC_API_KEY env var for Docker`
  - Files: `src/plm/shared/llm/anthropic_provider.py`

---

- [x] 2. Create Chunking Module with Tests

  **What to do**:
  - Create `src/plm/extraction/chunking/` directory structure:
    - `__init__.py` - exports Chunk, Chunker, WholeChunker, HeadingChunker, get_chunker
    - `base.py` - Chunk dataclass, abstract Chunker, registry (CHUNKERS, get_chunker, register_chunker)
    - `whole.py` - WholeChunker implementation (single chunk, no splitting)
    - `heading.py` - HeadingChunker (port from `poc/chunking_benchmark_v2/strategies/paragraph_heading.py`)
  
  - Create test files:
    - `tests/unit/test_chunking_base.py` - Chunk dataclass, registry tests
    - `tests/unit/test_chunking_whole.py` - WholeChunker comprehensive tests
    - `tests/unit/test_chunking_heading.py` - HeadingChunker comprehensive tests
    - `tests/integration/test_chunking_integration.py` - Real document tests

  **Must NOT do**:
  - Don't add inotify/watchdog dependencies
  - Don't over-complicate the registry pattern

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
    - Requires careful porting of heading chunker logic + comprehensive test writing

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Task 1)
  - **Parallel Group**: Wave 1
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:
  - `poc/chunking_benchmark_v2/strategies/paragraph_heading.py:1-262` - HeadingChunker reference implementation
  - `poc/chunking_benchmark_v2/strategies/base.py` - Chunk/ChunkingStrategy base classes
  - `.sisyphus/drafts/slow-extraction-docker.md` - Test specifications

  **Acceptance Criteria**:
  - [ ] `from plm.extraction.chunking import get_chunker` works
  - [ ] `get_chunker("whole")` returns WholeChunker
  - [ ] `get_chunker("heading")` returns HeadingChunker
  - [ ] `pytest tests/unit/test_chunking_base.py -v` passes (~12 tests)
  - [ ] `pytest tests/unit/test_chunking_whole.py -v` passes (~18 tests)
  - [ ] `pytest tests/unit/test_chunking_heading.py -v` passes (~25 tests)
  - [ ] `pytest tests/integration/test_chunking_integration.py -v` passes

  **Commit**: YES
  - Message: `feat(extraction): add chunking module with whole/heading strategies`
  - Files: `src/plm/extraction/chunking/*.py`, `tests/unit/test_chunking*.py`, `tests/integration/test_chunking*.py`

---

- [x] 3. Create CLI Entrypoint

  **What to do**:
  - Create `src/plm/extraction/cli.py`:
    - Parse environment variables:
      - `INPUT_DIR` (default: `/data/input`)
      - `OUTPUT_DIR` (default: `/data/output`)
      - `LOG_DIR` (default: `/data/logs`)
      - `VOCAB_NEGATIVES_PATH` (default: `/data/vocabularies/tech_domain_negatives.json`)
      - `VOCAB_SEEDS_PATH` (default: `/data/vocabularies/auto_vocab.json`)
      - `POLL_INTERVAL` (default: `30`)
      - `CONFIDENCE_THRESHOLD` (default: `0.5`)
      - `CHUNKING_STRATEGY` (default: `whole`)
      - `CHUNK_MIN_TOKENS` (default: `50`)
      - `CHUNK_MAX_TOKENS` (default: `256`)
      - `PROCESS_ONCE` (default: `false`)
      - `DRY_RUN` (default: `false`)
      - `LLM_MODEL` (default: `sonnet`)
    - CLI flags: `--chunker`, `--process-once`, `--dry-run`, `--help`
    - Implement polling-based watch loop
    - Process documents using V6 pipeline with chunking
    - Output JSON with multiline text preserved (use `json.dumps(indent=2)`)
    - Log low-confidence terms to `LOG_DIR/low_confidence.jsonl`
    - Graceful shutdown on SIGTERM/SIGINT

  **Must NOT do**:
  - No argparse complexity (env vars primary, flags secondary)
  - No daemon mode (Docker handles that)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 4
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `src/plm/extraction/pipeline.py:38-83` - Pipeline flow
  - `src/plm/extraction/slow/__init__.py` - Slow extraction exports
  - `src/plm/extraction/chunking/` - Chunking module (from Task 2)
  - `.sisyphus/drafts/slow-extraction-docker.md` - Output format specification

  **Acceptance Criteria**:
  - [ ] `python -m plm.extraction.cli --help` shows usage
  - [ ] `PROCESS_ONCE=true python -m plm.extraction.cli` processes and exits
  - [ ] Output JSON has `{"file": ..., "chunks": [{"text": "...\n...", "terms": [...]}]}`
  - [ ] `CHUNKING_STRATEGY=heading` uses HeadingChunker
  - [ ] Low-confidence terms written to `low_confidence.jsonl`
  - [ ] SIGTERM causes graceful shutdown

  **Commit**: YES
  - Message: `feat(extraction): add CLI entrypoint for Docker container`
  - Files: `src/plm/extraction/cli.py`

---

- [x] 4. Create Build Flake and Verify Builds

  **What to do**:
  - Create `src/plm/extraction/slow/flake.nix`:
    - Uses `flake-utils.lib.eachDefaultSystem`
    - Receives inputs from main flake (no inputs defined)
    - Loads workspace from `pyproject.toml` via `uv2nix.lib.workspace.loadWorkspace`
    - Path: `../../../../.` to reach repo root
    - Builds Python venv with `pythonSet.mkVirtualEnv`
    - Builds Docker image with `dockerTools.buildLayeredImage`
    - Exports: `slow-extraction` (venv), `slow-extraction-docker` (image)
  
  - **Verify builds work**:
    ```bash
    # Test venv build (will fail until main flake updated, but flake syntax should be valid)
    cd src/plm/extraction/slow
    nix flake check  # Verify syntax
    ```

  **Must NOT do**:
  - Don't define inputs in build flake (passed from main)
  - Don't embed vocabularies or data

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`
    - Nix flake with uv2nix requires careful setup

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:
  - `.sisyphus/drafts/slow-extraction-docker.md` - Build flake specification
  - `pyproject.toml` - Python dependencies
  - User's Java flake example (backend/flake.nix pattern)

  **Acceptance Criteria**:
  - [ ] `src/plm/extraction/slow/flake.nix` exists
  - [ ] Flake syntax is valid (no nix errors)
  - [ ] Docker image config includes volumes for /data/*
  - [ ] CA certificates included for HTTPS

  **Commit**: YES
  - Message: `feat(nix): add build flake for slow extraction Docker`
  - Files: `src/plm/extraction/slow/flake.nix`

---

- [ ] 5. Update Main Flake and Verify Full Build

  **What to do**:
  - Update `flake.nix`:
    - Add inputs: `flake-utils`, `pyproject-nix`, `uv2nix`, `pyproject-build-systems`
    - Import build flake: `import ./src/plm/extraction/slow/flake.nix`
    - Pass inputs to build flake
    - Expose packages: `slow-extraction`, `slow-extraction-docker`
    - Preserve existing devShell
  
  - **Verify all builds work**:
    ```bash
    # Test Python venv
    nix build .#slow-extraction
    ./result/bin/python -c "from plm.extraction.chunking import get_chunker; print(get_chunker('whole').name)"
    
    # Test Docker image
    nix build .#slow-extraction-docker
    docker load < result
    docker run --rm plm-slow-extraction:0.1.0 python -c "from plm.extraction import extract; print('OK')"
    
    # Test devShell preserved
    nix develop -c python --version
    ```

  **Must NOT do**:
  - Don't break existing devShell
  - Don't remove existing packages

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4
  - **Blocks**: Task 6
  - **Blocked By**: Task 4

  **References**:
  - `flake.nix` - Current flake structure
  - `.sisyphus/drafts/slow-extraction-docker.md` - Main flake specification
  - User's Java main flake example

  **Acceptance Criteria**:
  - [ ] `nix build .#slow-extraction` succeeds
  - [ ] `nix build .#slow-extraction-docker` succeeds
  - [ ] `docker load < result` loads image successfully
  - [ ] `nix develop -c python --version` still works
  - [ ] `nix flake check` passes

  **Commit**: YES
  - Message: `feat(nix): integrate slow extraction build into main flake`
  - Files: `flake.nix`

---

- [ ] 6. Docker End-to-End Integration Test

  **What to do**:
  - Create test script or run manual E2E test:
    ```bash
    # Setup
    mkdir -p /tmp/plm-test/{input,output,logs,vocabularies}
    cp data/vocabularies/*.json /tmp/plm-test/vocabularies/
    
    # Create test document
    cat > /tmp/plm-test/input/kubernetes-test.md << 'EOF'
    # Pod Lifecycle

    A Pod's status field is a PodStatus object,
    which has a phase field.

    ## Container States

    Kubernetes tracks the state of each container inside a Pod.
    Using React with TypeScript and useState hook.
    EOF
    
    # Run with whole chunking
    docker run --rm \
      -v /tmp/plm-test/input:/data/input \
      -v /tmp/plm-test/output:/data/output \
      -v /tmp/plm-test/logs:/data/logs \
      -v /tmp/plm-test/vocabularies:/data/vocabularies \
      -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
      -e PROCESS_ONCE=true \
      -e CHUNKING_STRATEGY=whole \
      plm-slow-extraction:0.1.0
    
    # Verify output
    cat /tmp/plm-test/output/kubernetes-test.json | jq .
    
    # Run with heading chunking
    rm /tmp/plm-test/output/*
    docker run --rm \
      -v /tmp/plm-test/input:/data/input \
      -v /tmp/plm-test/output:/data/output \
      -v /tmp/plm-test/logs:/data/logs \
      -v /tmp/plm-test/vocabularies:/data/vocabularies \
      -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
      -e PROCESS_ONCE=true \
      -e CHUNKING_STRATEGY=heading \
      plm-slow-extraction:0.1.0
    
    # Verify multiple chunks
    cat /tmp/plm-test/output/kubernetes-test.json | jq '.chunks | length'
    ```

  **Must NOT do**:
  - Don't mock LLM calls (real API validation)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 5 (final)
  - **Blocks**: None
  - **Blocked By**: Task 5

  **References**:
  - `.sisyphus/drafts/slow-extraction-docker.md` - E2E test specification

  **Acceptance Criteria**:
  - [ ] Output JSON exists at `/tmp/plm-test/output/kubernetes-test.json`
  - [ ] JSON has correct structure: `{"file": ..., "chunks": [...], "stats": {...}}`
  - [ ] Multiline text preserved in `chunks[].text` (contains `\n`)
  - [ ] Terms extracted with confidence levels
  - [ ] `whole` strategy produces 1 chunk
  - [ ] `heading` strategy produces multiple chunks
  - [ ] Low-confidence terms logged to `low_confidence.jsonl` (if any)

  **Commit**: YES
  - Message: `test(extraction): verify Docker end-to-end workflow`
  - Files: None (verification only) or add `scripts/test-docker-e2e.sh`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(llm): support ANTHROPIC_API_KEY env var for Docker` | `src/plm/shared/llm/anthropic_provider.py` |
| 2 | `feat(extraction): add chunking module with whole/heading strategies` | `src/plm/extraction/chunking/*.py`, `tests/` |
| 3 | `feat(extraction): add CLI entrypoint for Docker container` | `src/plm/extraction/cli.py` |
| 4 | `feat(nix): add build flake for slow extraction Docker` | `src/plm/extraction/slow/flake.nix` |
| 5 | `feat(nix): integrate slow extraction build into main flake` | `flake.nix` |
| 6 | `test(extraction): verify Docker end-to-end workflow` | `scripts/test-docker-e2e.sh` (optional) |

---

## Success Criteria

### Verification Commands
```bash
# Unit tests
PYTHONPATH=src pytest tests/unit/test_chunking*.py -v

# Nix builds
nix build .#slow-extraction
nix build .#slow-extraction-docker

# Docker E2E
docker load < result
docker run --rm -e ANTHROPIC_API_KEY=$KEY -e PROCESS_ONCE=true \
  -v ./test-data:/data plm-slow-extraction:0.1.0
```

### Final Checklist
- [ ] All unit tests pass (~55 tests)
- [ ] `nix build .#slow-extraction` works
- [ ] `nix build .#slow-extraction-docker` works
- [ ] Docker container processes documents
- [ ] Both chunking strategies work
- [ ] Multiline text preserved in output
- [ ] Low-confidence terms logged
- [ ] DevShell still works
- [ ] No hardcoded paths or keys
