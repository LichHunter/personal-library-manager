# Decisions: BMX Optimal Strategy

## Architectural Choices

(Subagents will append findings here)

## [2026-01-25] Baguetter Environment Fix

### Problem
Baguetter import failed with "libz.so.1: cannot open shared object file"

### Root Cause
- Baguetter depends on numpy/faiss which need system C libraries
- Libraries (zlib, libstdc++) not in LD_LIBRARY_PATH when running from agent context

### Solution (Already in flake.nix)
Commit 0dd12fb added required libraries to LD_LIBRARY_PATH:
```nix
LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
  "/run/opengl-driver"
  pkgs.zlib              # provides libz.so.1
  pkgs.stdenv.cc.cc.lib  # provides libstdc++.so.6
];
```

### Verification
```bash
export LD_LIBRARY_PATH="/nix/store/.../zlib-1.3.1/lib:/nix/store/.../gcc-15.2.0-lib/lib:/run/opengl-driver/lib"
cd poc/chunking_benchmark_v2
source .venv/bin/activate
python -c "from baguetter.indices import BMXSparseIndex; BMXSparseIndex()"
# ✓ Works!
```

### Action Required
Benchmarks must be run with LD_LIBRARY_PATH set (automatically done in `nix develop` shell)
