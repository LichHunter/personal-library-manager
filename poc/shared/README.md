# Shared LLM Providers

Unified LLM provider interface for all POCs. Supports **Anthropic (Claude)**, **Google Gemini** via OpenCode OAuth, and **Ollama** for local models.

## Quick Start

```python
import sys
sys.path.insert(0, "/path/to/poc")  # or use PYTHONPATH

from shared.llm import call_llm

# Anthropic (default — routes by model name)
response = call_llm("Extract terms from: kubectl get pods", model="sonnet")

# Gemini (auto-routed when model starts with "gemini")
response = call_llm("Extract terms from: kubectl get pods", model="gemini-2.5-flash")

# Ollama (local models — auto-downloads if needed)
response = call_llm("Extract terms from: kubectl get pods", model="llama3.2")
```

## Setup

### Prerequisites

1. **OpenCode CLI** authenticated with both providers:
   ```bash
   opencode auth  # connects to Anthropic + Google
   ```

2. **Auth file** at `~/.local/share/opencode/auth.json`:
   ```json
   {
     "anthropic": {
       "type": "oauth",
       "access": "...",
       "refresh": "...",
       "expires": 1234567890000
     },
     "google": {
       "type": "oauth",
       "access": "...",
       "refresh": "refresh_token|project_id|managed_project_id",
       "expires": 1234567890000
     }
   }
   ```

3. **Install httpx** (the only runtime dependency):
   ```bash
   pip install httpx
   # or add to your POC's pyproject.toml: "httpx>=0.27.0"
   ```

### Importing from a POC

**Option A: sys.path (quick & dirty)**
```python
import sys
from pathlib import Path

# Add poc/ to path so `shared.llm` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.llm import call_llm
```

**Option B: PYTHONPATH (recommended for scripts)**
```bash
export PYTHONPATH="/path/to/poc:$PYTHONPATH"
python my_script.py
```

**Option C: pip install -e (recommended for development)**
```bash
cd poc/shared
pip install -e .
```

## API Reference

### `call_llm(prompt, model, timeout, max_tokens, temperature, thinking_budget)`

Convenience wrapper — routes to the correct provider based on model name.

```python
from shared.llm import call_llm

# Anthropic models
call_llm("Hello", model="sonnet")                    # Claude Sonnet 4.5
call_llm("Hello", model="haiku")                     # Claude Haiku 3.5
call_llm("Hello", model="opus")                      # Claude Opus 4.5

# Gemini models (starts with "gemini")
call_llm("Hello", model="gemini-2.5-pro")            # Gemini 2.5 Pro
call_llm("Hello", model="gemini-2.5-flash")          # Gemini 2.5 Flash

# Ollama models (local — starts with llama/mistral/qwen/phi/deepseek/gemma)
call_llm("Hello", model="llama3.2")                  # LLaMA 3.2 3B (default)
call_llm("Hello", model="mistral")                   # Mistral 7B
call_llm("Hello", model="qwen2.5")                   # Qwen 2.5 7B
call_llm("Hello", model="ollama/phi4")               # Explicit ollama/ prefix
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *(required)* | User prompt text |
| `model` | `str` | `"claude-haiku-4-5"` | Model name or alias |
| `timeout` | `int` | `90` | Request timeout (seconds) |
| `max_tokens` | `int` | `2000` | Maximum output tokens |
| `temperature` | `float` | `0.0` | Sampling temperature (0.0–1.0) |
| `thinking_budget` | `int \| None` | `None` | Extended thinking tokens (Anthropic/Gemini 2.5+) |

**Returns:** `str` — generated text. Empty string `""` on failure (never raises).

### Model Aliases

#### Anthropic

| Alias | Resolves To |
|-------|-------------|
| `haiku` | `claude-3-5-haiku-latest` |
| `sonnet` | `claude-sonnet-4-5-20250929` |
| `opus` | `claude-opus-4-5-20251101` |
| `claude-haiku-4-5` | `claude-3-5-haiku-latest` |
| `claude-sonnet-4` | `claude-sonnet-4-20250514` |

#### Gemini

The Code Assist API uses short model names directly (not preview-dated versions, not `models/` prefix).

| Alias | Resolves To |
|-------|-------------|
| `gemini-pro` | `gemini-2.5-pro` |
| `gemini-flash` | `gemini-2.5-flash` |
| `gemini-3-pro` | `gemini-3-pro-preview` |
| `gemini-3-flash` | `gemini-3-flash-preview` |

Canonical names pass through unchanged:

| Model | Status |
|-------|--------|
| `gemini-3-pro-preview` | Working (Gemini 3 Pro) |
| `gemini-3-flash-preview` | Working (Gemini 3 Flash) |
| `gemini-2.5-pro` | Working |
| `gemini-2.5-flash` | Working |
| `gemini-2.5-flash-lite` | Working |
| `gemini-2.0-flash` | Working (deprecated Mar 2026) |

**Not supported on Code Assist:** Gemma models, `gemini-2.0-flash-lite`, `gemini-1.5-*`, `models/*` prefix, preview-dated versions (e.g., `gemini-2.5-flash-preview-05-20`), `-latest` aliases.

#### Ollama (Local)

Models tested in POCs:

| Alias | Resolves To | Notes |
|-------|-------------|-------|
| `llama`, `llama3.2` | `llama3.2:3b` | Recommended default |
| `llama3.2-1b` | `llama3.2:1b` | Fastest, lower quality |
| `llama3.1`, `llama3` | `llama3.1:8b` | Better quality |
| `mistral` | `mistral:7b` | Alternative architecture |
| `qwen`, `qwen2.5` | `qwen2.5:7b` | Strong reasoning |
| `deepseek` | `deepseek-r1:8b` | DeepSeek R1 |
| `phi`, `phi4` | `phi4:14b` | Microsoft Phi-4 |
| `gemma2` | `gemma2:9b` | Gemma 2 (local via Ollama) |

**Auto-download:** If a model isn't available locally, OllamaProvider will automatically download it on first use.

### Direct Provider Access

```python
from shared.llm import AnthropicProvider, GeminiProvider, OllamaProvider

# Anthropic
anthropic = AnthropicProvider()
response = anthropic.generate("Hello", model="sonnet", max_tokens=1000)

# Gemini
gemini = GeminiProvider()
response = gemini.generate("Hello", model="gemini-2.5-flash", max_tokens=1000)

# Ollama (local)
ollama = OllamaProvider()
response = ollama.generate("Hello", model="llama3.2", max_tokens=1000)

# Custom auth path / host
provider = AnthropicProvider(auth_path="/custom/path/auth.json")
ollama = OllamaProvider(host="http://remote-server:11434")
```

### Extended Thinking

```python
# Anthropic extended thinking
response = call_llm(
    "Solve this complex problem...",
    model="sonnet",
    thinking_budget=5000,  # 5K tokens for internal reasoning
    max_tokens=2000,
)

# Gemini extended thinking (2.5+ models only)
response = call_llm(
    "Solve this complex problem...",
    model="gemini-2.5-pro",
    thinking_budget=5000,
)
```

### Logging

The module uses stdlib `logging`. Enable debug output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or target specific providers
logging.getLogger("shared.llm.anthropic").setLevel(logging.DEBUG)
logging.getLogger("shared.llm.gemini").setLevel(logging.DEBUG)
```

## Architecture

```
poc/shared/
├── __init__.py
├── pyproject.toml
├── README.md
├── llm/
│   ├── __init__.py              # Public API: call_llm, get_provider, providers
│   ├── base.py                  # LLMProvider ABC + routing + call_llm()
│   ├── anthropic_provider.py    # Anthropic Claude via OpenCode OAuth
│   ├── gemini_provider.py       # Google Gemini via OpenCode OAuth (Code Assist)
│   └── ollama_provider.py       # Local models via Ollama (auto-download)
└── tests/
    └── test_llm.py              # Unit tests (mocked, no real API calls)
```

### Provider Routing

`call_llm()` routes based on model name:
- Starts with `ollama/` or local model names (`llama`, `mistral`, `qwen`, `phi`, `deepseek`, `gemma`) → `OllamaProvider`
- Starts with `gemini` → `GeminiProvider`
- Everything else → `AnthropicProvider`

### Error Handling

Both providers:
- Return empty string `""` on failure (never raise from `generate()`)
- Automatic retry with exponential backoff + jitter (up to 8 attempts)
- Retry on: 429 (rate limit), 500, 502, 503 (and 529 for Anthropic)
- Respect `Retry-After` headers
- Gemini: terminal `QUOTA_EXHAUSTED` errors are NOT retried
- Auto-refresh expired OAuth tokens before each request

### Authentication Flow

**Anthropic:**
1. Read `~/.local/share/opencode/auth.json` → `anthropic` key
2. Check token expiry (60s buffer)
3. If expired: POST to `https://console.anthropic.com/v1/oauth/token` with refresh token
4. Save refreshed tokens back to auth file

**Gemini:**
1. Read `~/.local/share/opencode/auth.json` → `google` key
2. Parse refresh token parts: `token|projectId|managedProjectId`
3. Resolve project ID (from env var, token parts, or `loadCodeAssist` API)
4. If expired: POST to `https://oauth2.googleapis.com/token` with refresh token
5. Wrap request in Code Assist envelope: `{project, model, request}`
6. POST to `https://cloudcode-pa.googleapis.com/v1internal:generateContent`

## Environment Variables

### Ollama

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

**Setup:**
```bash
# Start Ollama server
ollama serve

# Models auto-download on first use, or pull manually:
ollama pull llama3.2:3b
ollama pull mistral:7b
```

### Gemini

**Required** (for token refresh — public OAuth app credentials from [google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)):

| Variable | Purpose |
|----------|---------|
| `GEMINI_CLIENT_ID` | OAuth client ID for Gemini Code Assist |
| `GEMINI_CLIENT_SECRET` | OAuth client secret for Gemini Code Assist |

**Optional:**

| Variable | Purpose |
|----------|---------|
| `OPENCODE_GEMINI_PROJECT_ID` | Force a specific Google Cloud project |
| `GOOGLE_CLOUD_PROJECT` | Standard GCP project ID |

Find the credential values in the source of [google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli) or [jenslys/opencode-gemini-auth](https://github.com/jenslys/opencode-gemini-auth).

## Tests

```bash
cd poc/shared
uv sync
uv run pytest tests/ -v
```

Tests are fully mocked — no real API calls or auth files needed.
