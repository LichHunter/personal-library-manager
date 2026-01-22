# Proof of Concept (POC) Directory

This directory contains isolated experiments to validate ideas before integrating them into the main project.

## POC Guidelines

### Purpose

POCs are **throwaway experiments** designed to:
- Test a hypothesis quickly
- Validate technical feasibility
- Gather metrics and benchmarks
- Inform design decisions

POCs are NOT:
- Production code
- Part of the main application
- Expected to be maintained long-term

### Structure

Each POC should be a self-contained directory with:

```
poc/{poc-name}/
├── README.md           # REQUIRED: What, Why, How, Results
├── pyproject.toml      # Dependencies (use uv for package management)
├── *.py                # Implementation files
├── results.json        # Benchmark results (if applicable)
└── .venv/              # Local virtual environment (gitignored)
```

### README Template

Every POC README must answer:

```markdown
# {POC Name}

## What
One-paragraph description of what this POC tests.

## Why
What question are we trying to answer? What decision does this inform?

## Hypothesis
What do we expect to find?

## Setup
Step-by-step instructions to run the POC.

## Usage
Commands to run the experiment.

## Results
What did we find? Include:
- Key metrics
- Surprising findings
- Limitations discovered

## Conclusion
What did we learn? How does this affect our design?

## Files
Brief description of each file's purpose.
```

### Lifecycle

1. **Create**: Start with a clear hypothesis
2. **Implement**: Keep it minimal - just enough to test the hypothesis
3. **Document**: Record results immediately after running
4. **Decide**: Use findings to inform RESEARCH.md decisions
5. **Archive**: POCs stay in the repo for reference but are not maintained

### Best Practices

1. **Isolate dependencies**: Each POC has its own `pyproject.toml` and `.venv`
2. **Pin versions**: Use exact versions for reproducibility
3. **Document immediately**: Write results while they're fresh
4. **Link to research**: Reference the POC in RESEARCH.md when making decisions
5. **Keep it small**: A POC should take hours, not days

### Running POCs

All POCs assume you're in the project's Nix development shell:

```bash
cd /home/fujin/Code/personal-library-manager
direnv allow  # or: nix develop

# Then for each POC:
cd poc/{poc-name}
uv sync
source .venv/bin/activate
python {script}.py
```

---

## Current POCs

| POC | Status | Purpose | Key Finding |
|-----|--------|---------|-------------|
| [raptor_test](./raptor_test/) | Complete | Benchmark RAPTOR with local models | Summarization is 98.6% of indexing time |
| [test_data](./test_data/) | Complete | Generate test datasets for retrieval evaluation | Evidence extraction requires post-processing |
| [retrieval_benchmark](./retrieval_benchmark/) | In Progress | Compare retrieval strategies (flat, RAPTOR, LOD) | - |

---

## Adding a New POC

```bash
mkdir poc/my-new-poc
cd poc/my-new-poc

# Initialize with uv
cat > pyproject.toml << 'EOF'
[project]
name = "my-new-poc"
version = "0.1.0"
description = "Brief description"
requires-python = ">=3.11"
dependencies = []

[dependency-groups]
dev = []
EOF

# Create README from template
touch README.md

# Set up virtual environment
uv sync
```
