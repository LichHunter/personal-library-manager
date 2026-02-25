from __future__ import annotations

import re
import subprocess
from pathlib import Path

from plm.benchmark.metrics import RequestTrace, TraceEntry


TRACE_PATTERN = re.compile(
    r"\[[\d:]+\]\s+\[\s*[\d.]+s\]\s+TRACE\s+\|\s+"
    r"\[([a-f0-9-]+)\]\s+\[(\w+(?::\w+)?)\]\s+(.+)$"
)


def parse_trace_log(path: Path) -> dict[str, RequestTrace]:
    traces: dict[str, RequestTrace] = {}
    
    if not path.exists():
        return traces
    
    with open(path, encoding="utf-8") as f:
        for line in f:
            match = TRACE_PATTERN.match(line.strip())
            if match:
                request_id, stage, message = match.groups()
                if request_id not in traces:
                    traces[request_id] = RequestTrace(request_id=request_id)
                traces[request_id].entries.append(TraceEntry(stage=stage, message=message))
    
    return traces


def discover_trace_log(container_name: str = "plm-search-service") -> Path | None:
    try:
        result = subprocess.run(
            [
                "docker", "inspect", container_name,
                "--format",
                '{{range .Mounts}}{{if eq .Destination "/data/logs"}}{{.Source}}{{end}}{{end}}',
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode != 0:
            return None
        
        source_path = result.stdout.strip()
        if not source_path:
            return None
        
        trace_log = Path(source_path) / "search_trace.log"
        if trace_log.exists():
            return trace_log
        
        return None
        
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def attach_traces(
    results: list,
    traces: dict[str, RequestTrace],
) -> None:
    for result in results:
        if result.request_id and result.request_id in traces:
            result.trace = traces[result.request_id]
