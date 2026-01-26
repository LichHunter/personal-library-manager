# Issues & Gotchas - Realistic Questions Benchmark

## [2026-01-27] JSON Parsing
- **Issue**: Claude sometimes wraps JSON in markdown code blocks
- **Solution**: Strip ```json and ``` markers before parsing (lines 164-172)

## [2026-01-27] Retry Logic
- **Issue**: LLM calls can fail intermittently
- **Solution**: 3 retry attempts with 1s delay between attempts

## [2026-01-27] API Key Required
- **Issue**: `transform_question()` requires `ANTHROPIC_API_KEY` environment variable
- **Error**: "Could not resolve authentication method. Expected either api_key or auth_token to be set"
- **Solution**: User must set `export ANTHROPIC_API_KEY=sk-...` before running `--test-prompt` or `--generate`
- **Impact**: Cannot test prompt quality or generate questions without API key
