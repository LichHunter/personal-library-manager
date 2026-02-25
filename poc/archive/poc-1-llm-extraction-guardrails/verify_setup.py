#!/usr/bin/env python3
"""Phase 1: Environment Setup Verification

Verifies all required infrastructure:
1. Claude API access (Haiku, Sonnet, Opus) via OpenCode OAuth
2. Ollama with local models (Llama 3 8B, Mistral 7B) - optional
3. Required packages
4. Corpus availability
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CORPUS_DIR = (
    Path(__file__).parent.parent
    / "chunking_benchmark_v2"
    / "corpus"
    / "kubernetes_sample_200"
)

TEST_PROMPT = "Extract Kubernetes terms from: 'The pod entered CrashLoopBackOff state after the kubelet failed to pull the image.' Return JSON list."

CLAUDE_MODELS = ["claude-haiku", "claude-sonnet", "claude-opus"]
OLLAMA_MODELS = ["llama3:8b", "mistral:7b"]


def verify_anthropic_api() -> dict:
    results = {"available": False, "models": {}}

    try:
        from utils.llm_provider import call_llm, get_provider

        get_provider("claude-haiku")
    except FileNotFoundError as e:
        console.print(f"[red]Auth file not found: {e}[/red]")
        return results
    except Exception as e:
        console.print(f"[red]Failed to initialize provider: {e}[/red]")
        return results

    for model in CLAUDE_MODELS:
        try:
            start = time.time()
            response = call_llm(TEST_PROMPT, model=model, max_tokens=200, temperature=0)
            latency = (time.time() - start) * 1000

            if response:
                results["models"][model] = {
                    "status": "ok",
                    "latency_ms": round(latency),
                    "response_preview": response[:100],
                }
                console.print(f"[green]✓[/green] {model}: {latency:.0f}ms")
            else:
                results["models"][model] = {
                    "status": "error",
                    "error": "Empty response",
                }
                console.print(f"[red]✗[/red] {model}: Empty response")
        except Exception as e:
            results["models"][model] = {"status": "error", "error": str(e)}
            console.print(f"[red]✗[/red] {model}: {e}")

    results["available"] = all(
        m.get("status") == "ok" for m in results["models"].values()
    )
    return results


def verify_ollama() -> dict:
    results = {"available": False, "models": {}}

    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            console.print(
                f"[yellow]Ollama not available: {result.stderr[:100]}[/yellow]"
            )
            return results
        available_models = result.stdout
    except FileNotFoundError:
        console.print("[yellow]Ollama not installed (optional)[/yellow]")
        return results
    except Exception as e:
        console.print(f"[yellow]Ollama check failed: {e}[/yellow]")
        return results

    for model in OLLAMA_MODELS:
        model_base = model.split(":")[0]
        found = model_base in available_models

        if not found:
            results["models"][model] = {"status": "not_installed"}
            console.print(
                f"[yellow]○[/yellow] {model}: not installed (run: ollama pull {model})"
            )
            continue

        try:
            start = time.time()
            result = subprocess.run(
                ["ollama", "run", model],
                input=TEST_PROMPT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            latency = (time.time() - start) * 1000

            if result.returncode == 0:
                results["models"][model] = {
                    "status": "ok",
                    "latency_ms": round(latency),
                    "response_preview": result.stdout[:100],
                }
                console.print(f"[green]✓[/green] {model}: {latency:.0f}ms")
            else:
                results["models"][model] = {
                    "status": "error",
                    "error": result.stderr[:100],
                }
                console.print(f"[red]✗[/red] {model}: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            results["models"][model] = {"status": "timeout"}
            console.print(f"[yellow]○[/yellow] {model}: timeout (>120s)")
        except Exception as e:
            results["models"][model] = {"status": "error", "error": str(e)}
            console.print(f"[red]✗[/red] {model}: {e}")

    results["available"] = all(
        m.get("status") == "ok" for m in results["models"].values()
    )
    return results


def verify_corpus() -> dict:
    results = {"available": False, "files": [], "total_files": 0}

    if not CORPUS_DIR.exists():
        console.print(f"[red]Corpus directory not found: {CORPUS_DIR}[/red]")
        return results

    files = list(CORPUS_DIR.glob("*.md"))
    results["total_files"] = len(files)
    results["files"] = [f.name for f in files[:10]]
    results["available"] = len(files) >= 50

    if results["available"]:
        console.print(f"[green]✓[/green] Corpus: {len(files)} files available")
    else:
        console.print(f"[yellow]○[/yellow] Corpus: only {len(files)} files (need 50)")

    return results


def verify_packages() -> dict:
    results = {"all_available": True, "packages": {}}

    required = ["anthropic", "rapidfuzz", "numpy", "scipy", "pydantic", "rich", "httpx"]

    for pkg in required:
        try:
            __import__(pkg)
            results["packages"][pkg] = "ok"
        except ImportError:
            results["packages"][pkg] = "missing"
            results["all_available"] = False

    status = "[green]✓[/green]" if results["all_available"] else "[red]✗[/red]"
    console.print(
        f"{status} Packages: {len([p for p in results['packages'].values() if p == 'ok'])}/{len(required)} available"
    )

    return results


def main():
    console.print("\n[bold]Phase 1: Environment Setup Verification[/bold]\n")

    ARTIFACTS_DIR.mkdir(exist_ok=True)

    console.print("[bold]1. Checking packages...[/bold]")
    packages = verify_packages()

    console.print("\n[bold]2. Checking Claude API (via OpenCode OAuth)...[/bold]")
    anthropic_result = verify_anthropic_api()

    console.print("\n[bold]3. Checking Ollama (local models - optional)...[/bold]")
    ollama_result = verify_ollama()

    console.print("\n[bold]4. Checking corpus...[/bold]")
    corpus_result = verify_corpus()

    table = Table(title="\nSetup Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Notes")

    table.add_row(
        "Packages",
        "[green]OK[/green]" if packages["all_available"] else "[red]FAIL[/red]",
        f"{len([p for p in packages['packages'].values() if p == 'ok'])}/{len(packages['packages'])} packages",
    )
    table.add_row(
        "Claude API",
        "[green]OK[/green]" if anthropic_result["available"] else "[red]FAIL[/red]",
        f"{len([m for m in anthropic_result['models'].values() if m.get('status') == 'ok'])}/3 models",
    )
    table.add_row(
        "Ollama",
        "[green]OK[/green]" if ollama_result["available"] else "[yellow]SKIP[/yellow]",
        f"{len([m for m in ollama_result['models'].values() if m.get('status') == 'ok'])}/2 models (optional)",
    )
    table.add_row(
        "Corpus",
        "[green]OK[/green]"
        if corpus_result["available"]
        else "[yellow]PARTIAL[/yellow]",
        f"{corpus_result['total_files']} files",
    )

    console.print(table)

    all_critical_ok = (
        packages["all_available"]
        and anthropic_result["available"]
        and corpus_result["available"]
    )

    artifact = {
        "phase": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models_verified": {
            "claude": anthropic_result,
            "ollama": ollama_result,
        },
        "dependencies": packages,
        "corpus": corpus_result,
        "api_status": {
            "anthropic": "ok" if anthropic_result["available"] else "error",
            "ollama": "ok" if ollama_result["available"] else "skip",
        },
        "phase_status": "COMPLETE" if all_critical_ok else "INCOMPLETE",
    }

    artifact_path = ARTIFACTS_DIR / "phase-1-setup.json"
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2)
    console.print(f"\n[dim]Artifact saved: {artifact_path}[/dim]")

    if all_critical_ok:
        console.print(
            "\n[green bold]✓ Phase 1 COMPLETE - Ready for Phase 2[/green bold]"
        )
        if not ollama_result["available"]:
            console.print(
                "[yellow]Note: Ollama models skipped - POC will use Claude only[/yellow]"
            )
        return 0
    else:
        console.print("\n[red bold]✗ Phase 1 INCOMPLETE - Fix issues above[/red bold]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
