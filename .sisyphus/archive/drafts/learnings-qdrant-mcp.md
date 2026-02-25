# Learnings: Qdrant MCP Server

## Key Patterns to Adopt

### 1. FastMCP-Based Architecture
- Migrated from custom MCP implementation (-137 lines of boilerplate)
- Single entry point, settings-driven server
- **Action**: Use FastMCP from day 1, don't build custom MCP infrastructure

### 2. Environment-Only Configuration
- Removed CLI arguments entirely
- Docker-friendly, Claude Desktop compatible
- Pydantic BaseSettings for validation
- **Action**: All config via environment variables

### 3. Dynamic Tool Registration
```python
# If default collection is set, hide the parameter from LLM
if self.settings.default_collection:
    search_func = make_partial_function(search_func, {"collection": default})

# Read-only mode: don't register write tools
if not self.settings.read_only:
    self.tool(store_func, ...)
```

### 4. Custom Tool Descriptions
```python
class ToolSettings(BaseSettings):
    search_description: str = Field(
        default="Search the corpus...",
        validation_alias="PLM_SEARCH_DESCRIPTION",
    )
```
- Same server, different prompts = different use cases
- Code search vs memory storage via config only

### 5. Graceful Error Handling
- Return `None` for no results (not empty list)
- Don't throw errors for missing collections
- LLMs handle `None` better than `[]`

## Documentation Patterns

### Mental Model
- "Semantic memory layer" not "vector search"
- Frame for LLM users, not database users

### Multi-Client Support
- Claude Desktop (JSON config)
- Cursor/Windsurf (SSE transport)
- VS Code (one-click install buttons)
- Use-case-specific examples for each

### Onboarding Flow
1. Zero-install: `uvx mcp-server-qdrant`
2. Claude Desktop integration (copy-paste JSON)
3. Docker deployment
- Progressive complexity

## Testing Strategy
- Unit tests for Pydantic settings
- Integration tests with in-memory Qdrant
- CI on Python 3.10-3.13
- Pre-commit: ruff, isort, mypy

## Evolution Insights
- v0.5→v0.8: Progressive feature addition
- Custom→FastMCP: Framework migration saved complexity
- All major features came from user requests

## What They DON'T Have (Our Advantage)
- No hierarchical search (chunk/heading/document)
- No hybrid search (BM25 + semantic)
- No query understanding/routing
- No reranking
- No context expansion

---
*Source: Investigation of qdrant/mcp-server-qdrant @ 860ab93*
