# Draft: Slow Extraction Docker Image (v2)

## User's Request
Build a Docker image (via Nix flake) for the slow extraction system that:
- Watches a folder for new documents
- Processes documents with the existing `src/plm/extraction/slow/` V6 pipeline
- Outputs chunks with extracted terms (multiline text preserved)
- Logs low-confidence terms separately

## Confirmed Decisions
- **LLM auth**: env var (`ANTHROPIC_API_KEY`) OR path to opencode file
- **Vocabularies**: external paths (not embedded)
- **Chunking**: default = whole document (like POC), optional = heading-based
- **Testing flags**: available (`PROCESS_ONCE`, `DRY_RUN`)
- **Output**: JSON with multiline text preserved
- **Nix approach**: uv2nix (parse pyproject.toml)

**Key constraint**: Use existing extraction system - no new `ingestion/` folder.

---

## Research Findings

### Existing Slow Extraction System

**Location**: `src/plm/extraction/slow/` (7 modules, V6 strategy, F1=0.932)

**Stages**:
1. `taxonomy.py` - `extract_by_taxonomy(text, model="sonnet")` → LLM extraction
2. `candidate_verify.py` - `extract_candidates_heuristic(text)` → Zero-cost patterns
3. `grounding.py` - `verify_span()`, `ground_candidates()` → Span verification
4. `noise_filter.py` - `filter_noise()`, `load_negatives()` → Remove junk
5. `validation.py` - `validate_terms()` → LLM context validation
6. `postprocess.py` - `expand_spans()`, `suppress_subspans()`, `final_dedup()`

**Dependencies**:
- Internal: `plm.shared.llm.call_llm()` (Anthropic/Gemini via OAuth)
- Python: `anthropic>=0.40.0`, `httpx>=0.27.0`, `pydantic>=2.0.0`, `rapidfuzz>=3.0.0`
- Data: `data/vocabularies/tech_domain_negatives.json`, `auto_vocab.json`

**Current state**: No CLI/entrypoint exists. Pipeline is library-only.

### Nix Python Docker Patterns

**Two approaches identified**:

1. **Simple: `python312.withPackages`**
   - Works if all deps are in nixpkgs
   - Quick to implement
   - `anthropic`, `httpx`, `pydantic` are in nixpkgs
   - `rapidfuzz` is in nixpkgs as `python312Packages.rapidfuzz`

2. **Modern: `uv2nix` (2026 recommended)**
   - Parses `pyproject.toml` directly
   - More complex setup
   - Better for ongoing maintenance

**Critical**: Must include `dockerTools.caCertificates` for HTTPS API calls.

---

## Proposed Architecture

### Option A: Minimal Entrypoint Script (Recommended)

Add a single file `src/plm/extraction/cli.py` as the Docker entrypoint:

```python
#!/usr/bin/env python3
"""CLI entrypoint for slow extraction - Docker container."""
import json
import os
import sys
import time
from pathlib import Path

from plm.extraction.slow import (
    extract_by_taxonomy,
    extract_candidates_heuristic,
    ground_candidates,
    filter_noise,
    load_negatives,
    validate_terms,
    expand_spans,
    suppress_subspans,
    final_dedup,
)
from plm.extraction.fast.confidence import compute_confidence

def extract_with_confidence(text: str, model: str = "sonnet") -> list[dict]:
    """Run V6 pipeline and return terms with confidence scores."""
    # Stage 1: Generate candidates
    taxonomy_terms = extract_by_taxonomy(text, model=model)
    heuristic_terms = extract_candidates_heuristic(text)
    
    # Stage 2-5: Full pipeline
    candidates_by_source = {"taxonomy": taxonomy_terms, "heuristic": heuristic_terms}
    grounded = ground_candidates(candidates_by_source, text)
    terms = [entry["term"] for entry in grounded.values()]
    
    negatives = load_negatives()
    filtered = filter_noise(terms, negatives=negatives, doc_text=text)
    validated = validate_terms(filtered, text, model=model)
    expanded = expand_spans(validated, text)
    suppressed = suppress_subspans(expanded)
    final = final_dedup(suppressed)
    
    # Compute confidence for each term
    results = []
    for term in final:
        sources = []
        if term in taxonomy_terms: sources.append("taxonomy")
        if term in heuristic_terms: sources.append("heuristic")
        conf, level = compute_confidence(term, sources)
        results.append({"term": term, "confidence": conf, "level": level, "sources": sources})
    
    return results

def watch_and_process(input_dir: Path, output_dir: Path, log_dir: Path, poll_interval: int):
    """Watch folder and process new documents."""
    processed = set()
    
    while True:
        for f in input_dir.iterdir():
            if f.is_file() and f.name not in processed:
                process_file(f, output_dir, log_dir)
                processed.add(f.name)
        time.sleep(poll_interval)

def process_file(input_path: Path, output_dir: Path, log_dir: Path):
    """Process single file, write output and low-confidence log."""
    text = input_path.read_text()
    
    # Simple chunking (paragraph-based)
    chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    result = {"file": input_path.name, "chunks": []}
    low_conf_terms = []
    
    for chunk_text in chunks:
        terms = extract_with_confidence(chunk_text)
        result["chunks"].append({"text": chunk_text, "terms": terms})
        
        # Collect low-confidence terms
        for t in terms:
            if t["level"] == "LOW":
                low_conf_terms.append({
                    "file": input_path.name,
                    "term": t["term"],
                    "confidence": t["confidence"],
                    "context": chunk_text[:200]
                })
    
    # Write output
    output_path = output_dir / f"{input_path.stem}.json"
    output_path.write_text(json.dumps(result, indent=2))
    
    # Append low-confidence terms to log
    if low_conf_terms:
        log_path = log_dir / "low_confidence.jsonl"
        with log_path.open("a") as f:
            for t in low_conf_terms:
                f.write(json.dumps(t) + "\n")

if __name__ == "__main__":
    input_dir = Path(os.environ.get("INPUT_DIR", "/data/input"))
    output_dir = Path(os.environ.get("OUTPUT_DIR", "/data/output"))
    log_dir = Path(os.environ.get("LOG_DIR", "/data/logs"))
    poll_interval = int(os.environ.get("POLL_INTERVAL", "30"))
    
    for d in [input_dir, output_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    watch_and_process(input_dir, output_dir, log_dir, poll_interval)
```

### Flake.nix Addition

```nix
# Add to outputs in flake.nix
packages = forAllSystems (system:
  let
    pkgs = nixpkgs.legacyPackages.${system};
    
    # Python environment with all dependencies
    pythonEnv = pkgs.python312.withPackages (ps: with ps; [
      anthropic
      httpx
      pydantic
      rapidfuzz
    ]);
    
  in {
    # Docker image for slow extraction
    docker-slow-extraction = pkgs.dockerTools.buildLayeredImage {
      name = "plm-slow-extraction";
      tag = "latest";
      
      contents = [
        pythonEnv
        pkgs.dockerTools.caCertificates  # For HTTPS API calls
      ];
      
      extraCommands = ''
        mkdir -p app/src app/data
        cp -r ${./src/plm} app/src/plm
        cp -r ${./data/vocabularies} app/data/vocabularies
      '';
      
      config = {
        Cmd = [ "${pythonEnv}/bin/python" "-m" "plm.extraction.cli" ];
        WorkingDir = "/app";
        Env = [
          "PYTHONPATH=/app/src"
        ];
      };
    };
  }
);
```

---

## Open Questions

1. **LLM Authentication**: Current system uses OpenCode OAuth (`~/.local/share/opencode/auth.json`). 
   - Should we support direct API keys via env vars (`ANTHROPIC_API_KEY`)?
   - Need to check if `plm.shared.llm` supports both patterns.

2. **Chunking Strategy**: Draft uses simple paragraph split.
   - Is this sufficient, or do you want heading-aware chunking?
   - Max chunk size?

3. **Process Once Mode**: For testing, should there be a `PROCESS_ONCE=true` mode that processes existing files and exits?

4. **Vocabulary Files**: Should vocabularies be embedded in image or mounted as volume?
   - Embedded: simpler, but requires rebuild to update
   - Volume: flexible, but requires setup

5. **Output Format**: Draft uses:
   ```json
   {
     "file": "doc.txt",
     "chunks": [
       {"text": "...", "terms": [{"term": "React", "confidence": 0.9, "level": "HIGH"}]}
     ]
   }
   ```
   Is this the structure you want?

---

## Decisions Needed

| Decision | Options | Default |
|----------|---------|---------|
| Entrypoint location | `src/plm/extraction/cli.py` vs new module | `cli.py` in extraction |
| LLM auth method | OAuth file vs env var | Need to support env var for Docker |
| Chunking | Paragraph split vs heading-aware | Paragraph (simple) |
| Vocabularies | Embedded vs mounted volume | Embedded |
| Nix approach | `withPackages` vs `uv2nix` | `withPackages` (simpler) |

---

---

## Chunker Class Hierarchy

```
src/plm/extraction/
├── chunking/
│   ├── __init__.py          # Exports: Chunker, WholeDocumentChunker, HeadingChunker
│   ├── base.py              # Abstract Chunker base class
│   ├── whole.py             # WholeDocumentChunker (default - no splitting)
│   └── heading.py           # HeadingChunker (paragraph + heading context)
```

### Base Class (`base.py`)
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    index: int
    heading: str | None = None
    start_char: int = 0
    end_char: int = 0

class Chunker(ABC):
    """Abstract base class for document chunking strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier for CLI/env var selection."""
        ...
    
    @abstractmethod
    def chunk(self, text: str, filename: str = "") -> list[Chunk]:
        """Split document text into chunks."""
        ...

# Registry for strategy selection
CHUNKERS: dict[str, type[Chunker]] = {}

def register_chunker(cls: type[Chunker]) -> type[Chunker]:
    """Decorator to register a chunker strategy."""
    CHUNKERS[cls.__name__.lower().replace("chunker", "")] = cls
    return cls

def get_chunker(name: str) -> Chunker:
    """Get chunker by name (e.g., 'whole', 'heading')."""
    if name not in CHUNKERS:
        raise ValueError(f"Unknown chunker: {name}. Available: {list(CHUNKERS.keys())}")
    return CHUNKERS[name]()
```

### Whole Document (`whole.py`) - Default
```python
from .base import Chunker, Chunk, register_chunker

@register_chunker
class WholeChunker(Chunker):
    """No chunking - process entire document as one chunk (POC default)."""
    
    @property
    def name(self) -> str:
        return "whole"
    
    def chunk(self, text: str, filename: str = "") -> list[Chunk]:
        return [Chunk(
            text=text,
            index=0,
            heading=None,
            start_char=0,
            end_char=len(text),
        )]
```

### Heading-Based (`heading.py`)
```python
from .base import Chunker, Chunk, register_chunker
import re

@register_chunker  
class HeadingChunker(Chunker):
    """Paragraph-based chunking with heading context (from chunking_benchmark_v2)."""
    
    def __init__(self, min_tokens: int = 50, max_tokens: int = 256):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
    
    @property
    def name(self) -> str:
        return "heading"
    
    def chunk(self, text: str, filename: str = "") -> list[Chunk]:
        # Port logic from poc/chunking_benchmark_v2/strategies/paragraph_heading.py
        ...
```

### CLI Usage
```bash
# Via environment variable
CHUNKING_STRATEGY=whole   # default
CHUNKING_STRATEGY=heading

# Or CLI flag
python -m plm.extraction.cli --chunker heading
```

---

## Flake Structure (Build Flake Alongside Code)

```
personal-library-manager/
├── flake.nix                           # Main flake (imports build flake)
├── flake.lock
├── pyproject.toml
├── src/plm/
│   └── extraction/
│       ├── slow/
│       │   ├── flake.nix               # Build flake for slow extraction Docker
│       │   ├── __init__.py
│       │   ├── taxonomy.py
│       │   └── ...
│       ├── chunking/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── whole.py
│       │   └── heading.py
│       └── cli.py                      # CLI entrypoint
```

---

## Build Flake: `src/plm/extraction/slow/flake.nix`

```nix
{
  # This flake is imported by the main flake - inputs are passed in
  outputs = inputs@{ self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
        
        pname = "plm-slow-extraction";
        version = "0.1.0";
        
        # Load workspace from pyproject.toml at repo root
        # Path: src/plm/extraction/slow/ -> ../../../../ = repo root
        workspace = uv2nix.lib.workspace.loadWorkspace { 
          workspaceRoot = ../../../../.;
        };
        
        # Create overlay from pyproject.toml
        overlay = workspace.mkPyprojectOverlay {
          sourcePreference = "wheel";
        };
        
        # Build Python package set
        pythonSet = (pkgs.callPackage pyproject-nix.build.packages {
          inherit python;
        }).overrideScope (
          pkgs.lib.composeManyExtensions [
            pyproject-build-systems.overlays.default
            overlay
          ]
        );
        
        # Virtual environment with all dependencies
        venv = pythonSet.mkVirtualEnv "plm-env" workspace.deps.default;
        
      in {
        packages = rec {
          # The Python package/venv
          slow-extraction = venv;
          
          # Docker image
          slow-extraction-docker = pkgs.dockerTools.buildLayeredImage {
            name = pname;
            tag = version;
            
            contents = [
              venv
              pkgs.dockerTools.caCertificates
              pkgs.coreutils
              pkgs.bash
            ];
            
            config = {
              Cmd = [ "${venv}/bin/python" "-m" "plm.extraction.cli" ];
              WorkingDir = "/app";
              
              Env = [
                "PYTHONUNBUFFERED=1"
                "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                "NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
              ];
              
              Volumes = {
                "/data/input" = {};
                "/data/output" = {};
                "/data/logs" = {};
                "/data/vocabularies" = {};
                "/auth" = {};
              };
            };
          };
          
          default = slow-extraction;
        };
        
        devShell = pkgs.mkShell {
          buildInputs = [ venv ];
          shellHook = ''
            echo "PLM Slow Extraction development shell"
          '';
        };
      }
    );
}
```

---

## Main Flake: `flake.nix` (Updated)

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    
    # uv2nix for Python packaging
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ self, nixpkgs, flake-utils, pyproject-nix, uv2nix, pyproject-build-systems, ... }:
    let
      linuxSystem = "x86_64-linux";
      
      pkgs = import nixpkgs {
        system = linuxSystem;
      };
      
      # Import slow extraction build flake
      slow-extraction-flake = import ./src/plm/extraction/slow/flake.nix;
      slow-extraction-outputs = slow-extraction-flake.outputs {
        inherit self nixpkgs flake-utils pyproject-nix uv2nix pyproject-build-systems;
      };
      linux-slow-extraction-pkgs = slow-extraction-outputs.packages.${linuxSystem};
      
    in {
      packages = {
        "${linuxSystem}" = {
          # Slow extraction packages
          slow-extraction = linux-slow-extraction-pkgs.slow-extraction;
          slow-extraction-docker = linux-slow-extraction-pkgs.slow-extraction-docker;
          
          # Default
          default = linux-slow-extraction-pkgs.default;
        };
      };
      
      devShells = {
        "${linuxSystem}" = {
          default = pkgs.mkShell {
            packages = with pkgs; [
              opencode
              bun
              python312
              python312Packages.pip
              uv
              ollama
              tmux
            ];

            LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
              "/run/opengl-driver"
              pkgs.zlib
              pkgs.stdenv.cc.cc.lib
            ];

            shellHook = ''
              echo "Personal Library Manager dev environment"
              echo "Python $(python --version 2>&1)"

              if [ ! -d ".venv" ]; then
                echo "Creating Python virtual environment..."
                uv venv
              fi

              source .venv/bin/activate
              uv sync --quiet 2>/dev/null || true

              export LD_LIBRARY_PATH="${
                pkgs.lib.makeLibraryPath [
                  pkgs.zlib
                  pkgs.stdenv.cc.cc.lib
                ]
              }/run/opengl-driver/lib:$LD_LIBRARY_PATH"
            '';
          };
        };
      };
    };
}
```

---

## Usage

```bash
# Build Docker image
nix build .#slow-extraction-docker

# Load into Docker
docker load < result

# Run container
docker run -v ./data:/data \
  -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
  -e CHUNKING_STRATEGY=whole \
  plm-slow-extraction:0.1.0

# Build just the Python venv (for local testing)
nix build .#slow-extraction
```

---

## Key Points

### Build Flake Pattern (like your Java example)
- **Location**: `src/plm/extraction/slow/flake.nix` alongside the code
- **No inputs defined**: Receives inputs from main flake
- **Uses `flake-utils.lib.eachDefaultSystem`**: Multi-platform support
- **Workspace path**: `../../../../.` to reach repo root for `pyproject.toml`

### Main Flake Pattern
- **Defines all inputs**: nixpkgs, flake-utils, uv2nix, pyproject-nix, pyproject-build-systems
- **Imports build flake**: `import ./src/plm/extraction/slow/flake.nix`
- **Passes inputs**: `slow-extraction-flake.outputs { inherit self nixpkgs ... }`
- **Exposes packages**: `slow-extraction`, `slow-extraction-docker`

### Docker Image Features
- **CA certificates**: `dockerTools.caCertificates` + env vars for HTTPS
- **Volumes**: External mounting for input/output/logs/vocabularies/auth
- **No embedded data**: Everything mounted at runtime

### Environment Variables (to be supported in CLI)
```bash
# LLM Authentication (one of these)
ANTHROPIC_API_KEY=sk-...              # Direct API key
OPENCODE_AUTH_PATH=/auth/auth.json    # Path to OpenCode OAuth file

# Vocabulary paths
VOCAB_NEGATIVES_PATH=/data/vocabularies/tech_domain_negatives.json
VOCAB_SEEDS_PATH=/data/vocabularies/auto_vocab.json

# Processing options
INPUT_DIR=/data/input
OUTPUT_DIR=/data/output
LOG_DIR=/data/logs
POLL_INTERVAL=30
CONFIDENCE_THRESHOLD=0.5

# Chunking
CHUNKING_MODE=whole|heading          # whole=full document, heading=paragraph+heading
CHUNK_MIN_TOKENS=50
CHUNK_MAX_TOKENS=256

# Testing flags
PROCESS_ONCE=true|false              # Process existing files and exit
DRY_RUN=true|false                   # Don't write output, just log
```

### Output Format (JSON with multiline text)
```json
{
  "file": "kubernetes-pods.md",
  "processed_at": "2026-02-16T14:30:00Z",
  "chunks": [
    {
      "text": "## Pod Lifecycle\n\nA Pod's status field is a PodStatus object,\nwhich has a phase field.\n\nThe phase of a Pod is a simple, high-level\nsummary of where the Pod is in its lifecycle.",
      "chunk_index": 0,
      "heading": "## Pod Lifecycle",
      "terms": [
        {
          "term": "Pod",
          "confidence": 0.95,
          "level": "HIGH",
          "sources": ["taxonomy", "heuristic"]
        },
        {
          "term": "PodStatus",
          "confidence": 0.92,
          "level": "HIGH",
          "sources": ["heuristic"]
        }
      ]
    }
  ],
  "stats": {
    "total_chunks": 15,
    "total_terms": 47,
    "high_confidence": 35,
    "medium_confidence": 10,
    "low_confidence": 2
  }
}
```

### Low Confidence Log (JSONL)
```jsonl
{"file": "doc.md", "term": "ambiguous", "confidence": 0.35, "level": "LOW", "context": "## Pod Lifecycle\n\nA Pod's status..."}
{"file": "doc.md", "term": "maybe", "confidence": 0.42, "level": "LOW", "context": "The phase of a Pod..."}
```

---

## Implementation Plan

### Task 1: Modify LLM Provider for API Key Support
`src/plm/shared/llm/anthropic_provider.py`:
- Check `ANTHROPIC_API_KEY` env var first
- Fall back to OpenCode auth file
- Support custom auth file path via `OPENCODE_AUTH_PATH`

**Commit**: `feat(llm): support ANTHROPIC_API_KEY env var for Docker`

---

### Task 2: Create Chunking Module with Tests
`src/plm/extraction/chunking/`:
- `base.py` - Abstract `Chunker` class + registry
- `whole.py` - `WholeChunker` (default, no splitting)
- `heading.py` - `HeadingChunker` (paragraph + heading context)

`tests/unit/test_chunking.py`:
- Test `WholeChunker` returns single chunk with full text
- Test `HeadingChunker` splits on paragraphs
- Test `HeadingChunker` prepends heading context
- Test `HeadingChunker` merges small paragraphs
- Test `HeadingChunker` splits large paragraphs
- Test chunker registry `get_chunker("whole")` / `get_chunker("heading")`
- Test multiline text preservation in chunks

**Commit**: `feat(extraction): add chunking module with whole/heading strategies`

---

### Task 3: Create CLI Entrypoint
`src/plm/extraction/cli.py`:
- Parse all env vars (INPUT_DIR, OUTPUT_DIR, etc.)
- CLI flags: `--chunker`, `--process-once`, `--dry-run`
- Implement watch loop with polling
- Support chunking strategy selection
- Handle output formatting with multiline preservation
- Log low-confidence terms to JSONL

**Commit**: `feat(extraction): add CLI entrypoint for Docker container`

---

### Task 4: Create Build Flake
`src/plm/extraction/slow/flake.nix`:
- uv2nix workspace loading
- Python venv build (`slow-extraction`)
- Docker image build (`slow-extraction-docker`)

**Test**:
```bash
# Test Python venv build
nix build .#slow-extraction
./result/bin/python -c "from plm.extraction import extract; print('OK')"

# Test Docker image build
nix build .#slow-extraction-docker
docker load < result
docker run --rm plm-slow-extraction:0.1.0 --help
```

**Commit**: `feat(nix): add build flake for slow extraction Docker`

---

### Task 5: Update Main Flake
`flake.nix`:
- Add uv2nix inputs
- Import build flake from `src/plm/extraction/slow/flake.nix`
- Expose packages: `slow-extraction`, `slow-extraction-docker`
- Keep existing devShell

**Test**:
```bash
nix build .#slow-extraction-docker
nix build .#slow-extraction
nix develop  # Verify devShell still works
```

**Commit**: `feat(nix): integrate slow extraction build into main flake`

---

### Task 6: Integration Test - Docker End-to-End
Test full Docker workflow:
```bash
# Setup test data
mkdir -p /tmp/plm-test/{input,output,logs,vocabularies}
cp data/vocabularies/*.json /tmp/plm-test/vocabularies/
echo "Using React with TypeScript and useState hook" > /tmp/plm-test/input/test.txt

# Run container with process-once mode
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
cat /tmp/plm-test/output/test.json | jq '.chunks[0].terms'
cat /tmp/plm-test/logs/low_confidence.jsonl
```

**Verify**:
- [ ] Output JSON exists with correct structure
- [ ] Text is multiline preserved
- [ ] Terms extracted with confidence levels
- [ ] Low-confidence terms logged to JSONL
- [ ] Both `whole` and `heading` chunking strategies work

**Commit**: `test(extraction): verify Docker end-to-end workflow`

---

## Commit Summary

| Task | Commit Message |
|------|----------------|
| 1 | `feat(llm): support ANTHROPIC_API_KEY env var for Docker` |
| 2 | `feat(extraction): add chunking module with whole/heading strategies` |
| 3 | `feat(extraction): add CLI entrypoint for Docker container` |
| 4 | `feat(nix): add build flake for slow extraction Docker` |
| 5 | `feat(nix): integrate slow extraction build into main flake` |
| 6 | `test(extraction): verify Docker end-to-end workflow` |

---

## Test Coverage

### File: `tests/unit/test_chunking_base.py`

```python
"""Tests for chunking base classes and registry."""
import pytest
from plm.extraction.chunking.base import Chunk, Chunker, get_chunker, CHUNKERS


class TestChunkDataclass:
    """Tests for Chunk dataclass."""
    
    def test_chunk_required_fields(self):
        """Chunk requires text and index."""
        chunk = Chunk(text="hello", index=0)
        assert chunk.text == "hello"
        assert chunk.index == 0
    
    def test_chunk_optional_fields_defaults(self):
        """Chunk optional fields have correct defaults."""
        chunk = Chunk(text="hello", index=0)
        assert chunk.heading is None
        assert chunk.start_char == 0
        assert chunk.end_char == 0
    
    def test_chunk_all_fields(self):
        """Chunk accepts all fields."""
        chunk = Chunk(
            text="content",
            index=1,
            heading="## Section",
            start_char=100,
            end_char=200,
        )
        assert chunk.heading == "## Section"
        assert chunk.start_char == 100
        assert chunk.end_char == 200
    
    def test_chunk_multiline_text(self):
        """Chunk preserves multiline text exactly."""
        text = "Line 1\nLine 2\n\nLine 3"
        chunk = Chunk(text=text, index=0)
        assert chunk.text == text
        assert "\n\n" in chunk.text


class TestChunkerRegistry:
    """Tests for chunker registry functions."""
    
    def test_get_chunker_whole(self):
        """get_chunker('whole') returns WholeChunker instance."""
        chunker = get_chunker("whole")
        assert chunker.name == "whole"
    
    def test_get_chunker_heading(self):
        """get_chunker('heading') returns HeadingChunker instance."""
        chunker = get_chunker("heading")
        assert chunker.name == "heading"
    
    def test_get_chunker_unknown_raises(self):
        """get_chunker with unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown chunker"):
            get_chunker("nonexistent")
    
    def test_get_chunker_lists_available(self):
        """Error message lists available chunkers."""
        with pytest.raises(ValueError) as exc_info:
            get_chunker("bad")
        assert "whole" in str(exc_info.value)
        assert "heading" in str(exc_info.value)
    
    def test_registry_contains_expected_chunkers(self):
        """CHUNKERS registry has whole and heading."""
        assert "whole" in CHUNKERS
        assert "heading" in CHUNKERS
```

---

### File: `tests/unit/test_chunking_whole.py`

```python
"""Tests for WholeChunker strategy."""
import pytest
from plm.extraction.chunking import WholeChunker, Chunk


class TestWholeChunkerBasic:
    """Basic functionality tests."""
    
    @pytest.fixture
    def chunker(self):
        return WholeChunker()
    
    def test_name_property(self, chunker):
        """WholeChunker.name returns 'whole'."""
        assert chunker.name == "whole"
    
    def test_returns_list_of_chunks(self, chunker):
        """chunk() returns list of Chunk objects."""
        result = chunker.chunk("hello world")
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)
    
    def test_single_chunk_returned(self, chunker):
        """WholeChunker always returns exactly one chunk."""
        result = chunker.chunk("any text here")
        assert len(result) == 1
    
    def test_chunk_index_is_zero(self, chunker):
        """Single chunk has index 0."""
        result = chunker.chunk("text")
        assert result[0].index == 0


class TestWholeChunkerTextPreservation:
    """Tests for text content preservation."""
    
    @pytest.fixture
    def chunker(self):
        return WholeChunker()
    
    def test_preserves_exact_text(self, chunker):
        """Chunk text equals input exactly."""
        text = "Hello World"
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_preserves_single_newlines(self, chunker):
        """Single newlines preserved."""
        text = "Line 1\nLine 2\nLine 3"
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_preserves_double_newlines(self, chunker):
        """Paragraph breaks (double newlines) preserved."""
        text = "Para 1\n\nPara 2\n\nPara 3"
        result = chunker.chunk(text)
        assert result[0].text == text
        assert result[0].text.count("\n\n") == 2
    
    def test_preserves_mixed_whitespace(self, chunker):
        """Mixed whitespace preserved exactly."""
        text = "Start\n\n\n  indented\n\ttabbed\nend"
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_preserves_markdown_formatting(self, chunker):
        """Markdown headings and formatting preserved."""
        text = "# Title\n\n## Section\n\nParagraph with **bold**."
        result = chunker.chunk(text)
        assert "# Title" in result[0].text
        assert "## Section" in result[0].text


class TestWholeChunkerCharPositions:
    """Tests for character position tracking."""
    
    @pytest.fixture
    def chunker(self):
        return WholeChunker()
    
    def test_start_char_is_zero(self, chunker):
        """start_char is always 0."""
        result = chunker.chunk("any text")
        assert result[0].start_char == 0
    
    def test_end_char_equals_length(self, chunker):
        """end_char equals text length."""
        text = "12345"
        result = chunker.chunk(text)
        assert result[0].end_char == 5
    
    def test_end_char_with_unicode(self, chunker):
        """end_char correct with unicode characters."""
        text = "héllo 世界"
        result = chunker.chunk(text)
        assert result[0].end_char == len(text)


class TestWholeChunkerEdgeCases:
    """Edge case tests."""
    
    @pytest.fixture
    def chunker(self):
        return WholeChunker()
    
    def test_empty_string(self, chunker):
        """Empty string returns chunk with empty text."""
        result = chunker.chunk("")
        assert len(result) == 1
        assert result[0].text == ""
    
    def test_whitespace_only(self, chunker):
        """Whitespace-only text preserved."""
        result = chunker.chunk("   \n\n   ")
        assert result[0].text == "   \n\n   "
    
    def test_very_long_text(self, chunker):
        """Very long text returned as single chunk."""
        text = "word " * 10000
        result = chunker.chunk(text)
        assert len(result) == 1
        assert len(result[0].text) == len(text)
    
    def test_heading_is_none(self, chunker):
        """WholeChunker sets heading to None."""
        result = chunker.chunk("# Heading\n\nText")
        assert result[0].heading is None
```

---

### File: `tests/unit/test_chunking_heading.py`

```python
"""Tests for HeadingChunker strategy."""
import pytest
from plm.extraction.chunking import HeadingChunker, Chunk


class TestHeadingChunkerBasic:
    """Basic functionality tests."""
    
    @pytest.fixture
    def chunker(self):
        return HeadingChunker(min_tokens=50, max_tokens=256)
    
    def test_name_property(self, chunker):
        """HeadingChunker.name returns 'heading'."""
        assert chunker.name == "heading"
    
    def test_returns_list_of_chunks(self, chunker):
        """chunk() returns list of Chunk objects."""
        result = chunker.chunk("# Title\n\nParagraph one.\n\nParagraph two.")
        assert isinstance(result, list)
        assert all(isinstance(c, Chunk) for c in result)
    
    def test_configurable_min_tokens(self):
        """min_tokens parameter is configurable."""
        chunker = HeadingChunker(min_tokens=100)
        assert chunker.min_tokens == 100
    
    def test_configurable_max_tokens(self):
        """max_tokens parameter is configurable."""
        chunker = HeadingChunker(max_tokens=500)
        assert chunker.max_tokens == 500


class TestHeadingChunkerParagraphSplitting:
    """Tests for paragraph-based splitting."""
    
    @pytest.fixture
    def chunker(self):
        # Small limits to test splitting behavior
        return HeadingChunker(min_tokens=5, max_tokens=50)
    
    def test_splits_on_double_newline(self, chunker):
        """Paragraphs separated by double newline become separate chunks."""
        text = "First paragraph here.\n\nSecond paragraph here."
        result = chunker.chunk(text)
        assert len(result) >= 2
    
    def test_single_paragraph_single_chunk(self, chunker):
        """Single paragraph without breaks is one chunk."""
        text = "This is a single paragraph with no breaks."
        result = chunker.chunk(text)
        assert len(result) == 1
    
    def test_preserves_paragraph_text(self, chunker):
        """Paragraph text content preserved in chunks."""
        text = "Para one content.\n\nPara two content."
        result = chunker.chunk(text)
        all_text = " ".join(c.text for c in result)
        assert "Para one" in all_text
        assert "Para two" in all_text


class TestHeadingChunkerHeadingContext:
    """Tests for heading context prepending."""
    
    @pytest.fixture
    def chunker(self):
        return HeadingChunker(min_tokens=5, max_tokens=100)
    
    def test_prepends_heading_to_chunk(self, chunker):
        """Chunks under a heading have heading prepended."""
        text = "## My Section\n\nParagraph under the section."
        result = chunker.chunk(text)
        # At least one chunk should have the heading
        assert any("## My Section" in c.text for c in result)
    
    def test_heading_in_chunk_metadata(self, chunker):
        """Chunk.heading field contains current heading."""
        text = "## Section\n\nContent here."
        result = chunker.chunk(text)
        assert any(c.heading == "## Section" for c in result)
    
    def test_nested_headings(self, chunker):
        """Nested headings tracked correctly."""
        text = "# Title\n\n## Section\n\nContent.\n\n### Subsection\n\nMore content."
        result = chunker.chunk(text)
        headings = [c.heading for c in result if c.heading]
        assert any("##" in h for h in headings)
    
    def test_no_heading_document(self, chunker):
        """Document without headings still chunks correctly."""
        text = "Para one.\n\nPara two.\n\nPara three."
        result = chunker.chunk(text)
        assert len(result) >= 1
        # heading should be None for all
        assert all(c.heading is None for c in result)


class TestHeadingChunkerMerging:
    """Tests for small paragraph merging."""
    
    def test_merges_small_paragraphs(self):
        """Paragraphs below min_tokens merged together."""
        chunker = HeadingChunker(min_tokens=20, max_tokens=100)
        # Each "Hi." is ~1 token, so they should merge
        text = "Hi.\n\nHello.\n\nHey."
        result = chunker.chunk(text)
        # Should merge into fewer chunks than paragraphs
        assert len(result) < 3
    
    def test_no_merge_across_heading_boundary(self):
        """Don't merge paragraphs across different headings."""
        chunker = HeadingChunker(min_tokens=20, max_tokens=100)
        text = "## Section A\n\nSmall.\n\n## Section B\n\nAlso small."
        result = chunker.chunk(text)
        # Should have separate chunks for each section
        section_a_chunks = [c for c in result if c.heading and "Section A" in c.heading]
        section_b_chunks = [c for c in result if c.heading and "Section B" in c.heading]
        # Content shouldn't be merged across sections
        for chunk in section_a_chunks:
            assert "Also small" not in chunk.text
    
    def test_merge_respects_max_tokens(self):
        """Merging stops before exceeding max_tokens."""
        chunker = HeadingChunker(min_tokens=5, max_tokens=20)
        # Create paragraphs that would exceed max if all merged
        text = "Word one two.\n\nWord three four.\n\nWord five six."
        result = chunker.chunk(text)
        for chunk in result:
            words = chunk.text.split()
            assert len(words) <= 20  # rough token estimate


class TestHeadingChunkerSplitting:
    """Tests for large paragraph splitting."""
    
    def test_splits_large_paragraph(self):
        """Paragraph exceeding max_tokens is split."""
        chunker = HeadingChunker(min_tokens=5, max_tokens=10)
        # Create paragraph with many words
        text = " ".join(["word"] * 50)
        result = chunker.chunk(text)
        assert len(result) > 1
    
    def test_split_tries_sentence_boundary(self):
        """Large paragraph split tries to end at sentence."""
        chunker = HeadingChunker(min_tokens=5, max_tokens=20)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        result = chunker.chunk(text)
        # Chunks should end with period when possible
        for chunk in result[:-1]:  # Last chunk may not end with period
            stripped = chunk.text.strip()
            # Should end with sentence punctuation
            assert stripped.endswith(('.', '!', '?')) or len(stripped.split()) <= 20


class TestHeadingChunkerTextPreservation:
    """Tests for text content preservation."""
    
    @pytest.fixture
    def chunker(self):
        return HeadingChunker(min_tokens=5, max_tokens=200)
    
    def test_preserves_newlines_in_paragraph(self, chunker):
        """Single newlines within paragraph preserved."""
        text = "## Section\n\nLine one\nLine two\nLine three"
        result = chunker.chunk(text)
        combined = "\n".join(c.text for c in result)
        assert "Line one\nLine two" in combined or "Line one" in combined
    
    def test_preserves_code_blocks(self, chunker):
        """Code block formatting preserved."""
        text = "## Code\n\n```python\ndef foo():\n    pass\n```"
        result = chunker.chunk(text)
        combined = "\n".join(c.text for c in result)
        assert "```python" in combined
        assert "def foo():" in combined
    
    def test_preserves_indentation(self, chunker):
        """Indentation preserved in chunks."""
        text = "## List\n\n- Item 1\n  - Nested\n- Item 2"
        result = chunker.chunk(text)
        combined = "\n".join(c.text for c in result)
        assert "  - Nested" in combined


class TestHeadingChunkerCharPositions:
    """Tests for character position tracking."""
    
    @pytest.fixture
    def chunker(self):
        return HeadingChunker(min_tokens=5, max_tokens=100)
    
    def test_chunk_indices_sequential(self, chunker):
        """Chunk indices are sequential starting from 0."""
        text = "## A\n\nPara 1.\n\n## B\n\nPara 2."
        result = chunker.chunk(text)
        indices = [c.index for c in result]
        assert indices == list(range(len(result)))
    
    def test_start_end_char_valid(self, chunker):
        """start_char < end_char for non-empty chunks."""
        text = "## Section\n\nSome content here."
        result = chunker.chunk(text)
        for chunk in result:
            if chunk.text.strip():
                assert chunk.start_char < chunk.end_char


class TestHeadingChunkerEdgeCases:
    """Edge case tests."""
    
    @pytest.fixture
    def chunker(self):
        return HeadingChunker(min_tokens=5, max_tokens=100)
    
    def test_empty_string(self, chunker):
        """Empty string returns empty list."""
        result = chunker.chunk("")
        assert result == []
    
    def test_only_headings(self, chunker):
        """Document with only headings (no content)."""
        text = "# Title\n\n## Section\n\n### Subsection"
        result = chunker.chunk(text)
        # Should handle gracefully - may return empty or heading-only chunks
        assert isinstance(result, list)
    
    def test_heading_at_end(self, chunker):
        """Heading at end of document."""
        text = "Content here.\n\n## Trailing Heading"
        result = chunker.chunk(text)
        assert len(result) >= 1
    
    def test_multiple_blank_lines(self, chunker):
        """Multiple blank lines between paragraphs."""
        text = "Para 1.\n\n\n\nPara 2."
        result = chunker.chunk(text)
        assert len(result) >= 1
```

---

### Integration Tests (`tests/integration/test_chunking_integration.py`)

```python
"""Integration tests for chunking with extraction pipeline."""
import pytest
from plm.extraction.chunking import get_chunker, WholeChunker, HeadingChunker


class TestChunkerWithRealDocuments:
    """Test chunkers with realistic document content."""
    
    @pytest.fixture
    def kubernetes_doc(self):
        return '''# Pod Lifecycle

A Pod's status field is a PodStatus object, which has a phase field.

The phase of a Pod is a simple, high-level summary of where the Pod is in its lifecycle.

## Pod Phase

The following are possible values for phase:

**Pending**: The Pod has been accepted by the Kubernetes cluster, but one or more containers has not been set up.

**Running**: The Pod has been bound to a node, and all containers have been created.

**Succeeded**: All containers in the Pod have terminated in success.

## Container States

Kubernetes tracks the state of each container inside a Pod.

### Waiting

A container in Waiting state is still running operations.

### Running  

The Running status indicates that a container is executing without issues.
'''
    
    def test_whole_chunker_real_doc(self, kubernetes_doc):
        """WholeChunker handles real Kubernetes doc."""
        chunker = get_chunker("whole")
        result = chunker.chunk(kubernetes_doc)
        
        assert len(result) == 1
        assert "Pod Lifecycle" in result[0].text
        assert "Container States" in result[0].text
    
    def test_heading_chunker_real_doc(self, kubernetes_doc):
        """HeadingChunker handles real Kubernetes doc."""
        chunker = get_chunker("heading")
        result = chunker.chunk(kubernetes_doc)
        
        # Should create multiple chunks
        assert len(result) > 1
        
        # Each chunk should have heading context
        for chunk in result:
            # Either has heading or is at document start
            assert chunk.heading is not None or chunk.index == 0
        
        # Content should be distributed across chunks
        all_text = " ".join(c.text for c in result)
        assert "Pod Lifecycle" in all_text
        assert "Container States" in all_text
        assert "Waiting" in all_text
```

---

### Nix Build Tests (Manual/CI)

```bash
# Test 1: Python venv builds
nix build .#slow-extraction
./result/bin/python -c "from plm.extraction.chunking import get_chunker; print(get_chunker('whole').name)"
# Expected: whole

# Test 2: Docker image builds
nix build .#slow-extraction-docker
docker load < result
# Expected: Loaded image: plm-slow-extraction:0.1.0

# Test 3: Docker container runs
docker run --rm plm-slow-extraction:0.1.0 python -c "from plm.extraction.chunking import get_chunker; print('OK')"
# Expected: OK

# Test 4: Docker help/usage
docker run --rm plm-slow-extraction:0.1.0 --help
# Expected: Usage information

# Test 5: DevShell still works
nix develop -c python --version
# Expected: Python 3.12.x
```
