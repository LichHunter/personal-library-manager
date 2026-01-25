## Task 0: Baguetter Installation Issues

### Problem
Baguetter was added to pyproject.toml but cannot be imported due to environment issues:

1. **POC venv (`.venv/`)**: Has Python 3.12 and baguetter installed, but missing system libraries (libz.so.1)
   - Error: `ImportError: libz.so.1: cannot open shared object file: No such file or directory`
   - This breaks numpy, which breaks baguetter

2. **Main project venv**: Nix-managed, read-only, cannot install packages
   - Error: `[Errno 30] Read-only file system`

3. **README says**: Use main venv (`source .venv/bin/activate` from project root)
   - But main venv doesn't have baguetter and can't install it

### Root Cause
- The POC has its own pyproject.toml and venv, but the venv is broken (missing system libs)
- The benchmark is meant to run from the main project venv (per README)
- But the main venv is Nix-managed and immutable

### Solution Needed
The benchmark should run from the main project venv. We need to either:
1. Add baguetter to the Nix flake (proper solution)
2. OR: Fix the POC venv to have proper system libraries from Nix
3. OR: Use the main venv but with `--break-system-packages` and install to user site-packages

### Current State
- pyproject.toml updated with baguetter>=0.1.1 ✓
- uv.lock updated ✓  
- But import fails due to environment issues ✗

### Next Steps
Need to properly integrate baguetter into the Nix environment or fix the POC venv.

## RESOLUTION ATTEMPT SUMMARY

### What Was Tried
1. ✗ Install to POC venv: Works but venv has missing system libraries (libz.so.1)
2. ✗ Install to main venv user site-packages: Main venv disables user site-packages by default
3. ✗ Install to main venv directly: Main venv site-packages is read-only (Nix-managed)
4. ✗ Copy packages from Python 3.12 to 3.13: Compiled extensions don't work across versions
5. ✗ Use POC venv with nix develop: Wheels incompatible with Nix environment

### Root Cause Analysis
The fundamental issue is **environment mismatch**:
- POC venv uses pre-built wheels (manylinux) that expect system libraries
- Nix environment provides isolated, immutable Python without those system libs
- Main venv is Nix-managed and read-only
- User site-packages disabled in Nix venv

### PROPER SOLUTION REQUIRED
**Add baguetter to flake.nix** - This is the only sustainable approach:
1. Add baguetter to the Nix flake's Python packages
2. Rebuild the dev environment: `nix flake update && nix develop`
3. All dependencies will be properly resolved by Nix

### Current Status
- pyproject.toml: ✓ Updated with baguetter>=0.1.1
- uv.lock: ✓ Updated
- Import verification: ✗ BLOCKED - Requires Nix flake update
- Commit: ✓ Created (493911c)

### RESOLUTION ✓

**Solution Implemented**: Updated `flake.nix` to add required system libraries to `LD_LIBRARY_PATH`

**Changes Made**:
1. Added `pkgs.zlib` to LD_LIBRARY_PATH (for numpy's libz.so.1)
2. Added `pkgs.stdenv.cc.cc.lib` to LD_LIBRARY_PATH (for faiss's libstdc++.so.6)
3. Updated shellHook to export the path explicitly

**Verification**:
```bash
nix develop
cd poc/chunking_benchmark_v2
source .venv/bin/activate
python -c "from baguetter.indices import BMXSparseIndex; idx = BMXSparseIndex(); print('✓ Works!')"
# Output: ✓ Works!
```

**Commits**:
- `493911c`: chore(benchmark): add baguetter dependency for BMX
- `0dd12fb`: fix(nix): add zlib and libstdc++ to LD_LIBRARY_PATH for Python C extensions

**Status**: ✅ RESOLVED - Baguetter now works in `nix develop` shell

**Time to Resolution**: ~2.5 hours
