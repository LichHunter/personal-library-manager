# Learnings

## uv.lock Must Be Git-Tracked for Nix Flakes

**Issue**: Nix flakes only see files tracked by git. The `uv.lock` file was in the directory but not tracked.

**Error**: `opening file '/nix/store/.../uv.lock': No such file or directory`

**Solution**: `git add uv.lock` before running nix build commands.

**Note**: The `.gitignore` already has a comment saying uv.lock should be tracked, but it wasn't actually added.

## Main Flake Already Had All Changes

The `flake.nix` was already updated with:
- All uv2nix inputs (pyproject-nix, uv2nix, pyproject-build-systems)
- Import of build flake from `./src/plm/extraction/slow/flake.nix`
- Exposed packages (slow-extraction, slow-extraction-docker)
- Preserved devShells using forAllSystems

Only verification was needed after adding uv.lock to git.

## Task 6: Docker E2E Integration Test - Infrastructure Verified

**Status**: Infrastructure verified, test script created. Full E2E test blocked by missing `ANTHROPIC_API_KEY`.

### Infrastructure Verification Completed

1. **Docker Image Build**: ✅ SUCCESS
   - Command: `nix build .#slow-extraction-docker`
   - Result: `/nix/store/qlsggbb4d9qiyqvg011ksrgf8r3zg9il-plm-slow-extraction.tar.gz`
   - Size: 487MB (compressed), 236MB (uncompressed)

2. **Docker Image Load**: ✅ SUCCESS
   - Command: `docker load < result`
   - Image: `plm-slow-extraction:0.1.0`
   - Verified with: `docker images | grep plm-slow-extraction`

3. **Test Directory Structure**: ✅ SUCCESS
   - Created: `/tmp/plm-test/{input,output,logs,vocabularies}`
   - All directories ready for test execution

4. **Vocabulary Files**: ✅ SUCCESS
   - Copied from: `data/vocabularies/`
   - Files: `auto_vocab.json` (17K), `tech_domain_negatives.json` (24K)
   - Destination: `/tmp/plm-test/vocabularies/`

5. **Sample Test Document**: ✅ SUCCESS
   - Created: `/tmp/plm-test/input/kubernetes-test.md`
   - Content: Markdown with known terms (Kubernetes, Pod, React, TypeScript, useState)
   - Multiline structure verified

### Test Script Created

**Location**: `scripts/test-docker-e2e.sh`
**Size**: 7.6K
**Status**: Executable, syntax validated

**Features**:
- API key validation at startup
- Docker image existence check
- Test directory setup
- Vocabulary file copying
- Sample document creation
- Two test runs:
  1. `CHUNKING_STRATEGY=whole` (expects 1 chunk)
  2. `CHUNKING_STRATEGY=heading` (expects multiple chunks)
- JSON output validation
- Multiline text preservation check
- Term extraction verification
- Low-confidence log checking
- Automatic cleanup

**Usage**:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
./scripts/test-docker-e2e.sh
```

### Blocker: Missing ANTHROPIC_API_KEY

The full E2E test cannot run without the API key because:
- Docker container calls Anthropic API for term extraction
- Real LLM validation is required (no mocking allowed per task spec)
- Container will fail with exit code non-zero if API key is missing

**To Complete Task 6**:
1. Set `ANTHROPIC_API_KEY` environment variable
2. Run: `./scripts/test-docker-e2e.sh`
3. Verify all acceptance criteria pass
4. Commit with message: `test(extraction): verify Docker end-to-end workflow`

### Acceptance Criteria Status

- [x] Docker image builds successfully
- [x] Docker image loads into Docker daemon
- [x] Test directories created
- [x] Vocabularies copied
- [x] Sample document created
- [x] Test script created and executable
- [ ] Full E2E test runs (blocked by API key)
- [ ] Output JSON exists and is valid (blocked by API key)
- [ ] Multiline text preserved (blocked by API key)
- [ ] Terms extracted with confidence (blocked by API key)
- [ ] `whole` strategy produces 1 chunk (blocked by API key)
- [ ] `heading` strategy produces multiple chunks (blocked by API key)
- [ ] Low-confidence logs created (blocked by API key)

