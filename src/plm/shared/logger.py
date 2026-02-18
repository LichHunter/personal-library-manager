from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO


@dataclass
class _TimerEntry:
    name: str
    start: float
    end: float | None = None

    @property
    def elapsed(self) -> float:
        return (self.end or time.perf_counter()) - self.start


class PipelineLogger:
    """Structured logger with three independent sinks.

    - console   : INFO+  (human-readable, always on)
    - info_file : INFO+  (same as console but persisted)
    - trace_file: TRACE+ (every log line, full detail)

    All three write simultaneously; each has its own level gate.
    """

    LEVELS: dict[str, int] = {
        "TRACE": -1,
        "DEBUG":  0,
        "INFO":   1,
        "PROG":   1,
        "METRIC": 1,
        "WARN":   2,
        "ERROR":  3,
    }

    def __init__(
        self,
        log_file: str | Path | None = None,
        trace_file: str | Path | None = None,
        console: bool = True,
        min_level: str = "INFO",
    ) -> None:
        self.console = console
        self.min_level = self.LEVELS.get(min_level.upper(), 1)
        self._info_file: TextIO | None = None
        self._trace_file: TextIO | None = None
        self.log_path: Path | None = None
        self.trace_path: Path | None = None
        self._timers: dict[str, _TimerEntry] = {}
        self._metrics: dict[str, list[tuple[float, Any]]] = {}
        self._start = time.perf_counter()

        if log_file:
            self.log_path = Path(log_file)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._info_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
            self._raw_info("=" * 80)
            self._raw_info(f"PLM Log — {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self._raw_info("=" * 80)
            self._raw_info("")

        if trace_file:
            self.trace_path = Path(trace_file)
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace_file = open(self.trace_path, "w", encoding="utf-8", buffering=1)
            self._raw_trace("=" * 80)
            self._raw_trace(f"PLM Trace Log — {time.strftime('%Y-%m-%d %H:%M:%S')}")
            self._raw_trace("=" * 80)
            self._raw_trace("")

    def _raw_info(self, line: str) -> None:
        if self._info_file:
            self._info_file.write(line + "\n")

    def _raw_trace(self, line: str) -> None:
        if self._trace_file:
            self._trace_file.write(line + "\n")

    def _emit(self, level: str, msg: str) -> None:
        level_int = self.LEVELS.get(level, 1)
        ts = time.strftime("%H:%M:%S")
        elapsed = time.perf_counter() - self._start
        line = f"[{ts}] [{elapsed:7.2f}s] {level:6} | {msg}"

        if self.console and level_int >= self.min_level:
            print(line, flush=True)

        if level_int >= 1:
            self._raw_info(line)

        self._raw_trace(line)

    def trace(self, msg: str) -> None:
        self._emit("TRACE", msg)

    def debug(self, msg: str) -> None:
        self._emit("DEBUG", msg)

    def info(self, msg: str) -> None:
        self._emit("INFO", msg)

    def warn(self, msg: str) -> None:
        self._emit("WARN", msg)

    def error(self, msg: str) -> None:
        self._emit("ERROR", msg)

    def section(self, title: str) -> None:
        sep = "=" * 80
        for line in ("", sep, f"  {title}", sep):
            if self.console:
                print(line, flush=True)
            self._raw_info(line)
            self._raw_trace(line)

    def subsection(self, title: str) -> None:
        sep = "-" * 60
        for line in ("", sep, f"  {title}", sep):
            if self.console:
                print(line, flush=True)
            self._raw_info(line)
            self._raw_trace(line)

    def progress(self, current: int, total: int, label: str = "") -> None:
        pct = (current / total * 100) if total else 0
        filled = int(20 * current / total) if total else 0
        bar = "█" * filled + "░" * (20 - filled)
        msg = f"[{current:>4}/{total}] {bar} {pct:5.1f}%"
        if label:
            msg += f"  {label}"
        self._emit("PROG", msg)

    def metric(self, name: str, value: Any, unit: str = "") -> None:
        t = time.perf_counter() - self._start
        self._metrics.setdefault(name, []).append((t, value))
        vstr = f"{value:.3f}" if isinstance(value, float) else str(value)
        if unit:
            vstr += f" {unit}"
        self._emit("METRIC", f"{name} = {vstr}")

    def timer_start(self, name: str) -> None:
        self._timers[name] = _TimerEntry(name=name, start=time.perf_counter())

    def timer_end(self, name: str) -> float:
        t = self._timers.get(name)
        if t is None:
            self.warn(f"Timer '{name}' never started")
            return 0.0
        t.end = time.perf_counter()
        return t.elapsed

    @contextmanager
    def timer(self, name: str):
        self.timer_start(name)
        try:
            yield
        finally:
            elapsed = self.timer_end(name)
            self._emit("METRIC", f"timer:{name} = {elapsed:.3f}s")

    def summary(self) -> None:
        self.section("RUN SUMMARY")
        total = time.perf_counter() - self._start
        self.info(f"Total wall time: {total:.2f}s")

        completed = {n: t.elapsed for n, t in self._timers.items() if t.end}
        if completed:
            self.subsection("Timer breakdown")
            for name, elapsed in sorted(completed.items(), key=lambda x: -x[1])[:20]:
                self.info(f"  {name:<50} {elapsed:>8.3f}s")

        if self.log_path:
            self.info(f"Info log : {self.log_path}")
        if self.trace_path:
            self.info(f"Trace log: {self.trace_path}")

    def install_stdlib_bridge(self, root_logger: str = "", level: int = logging.INFO) -> None:
        handler = _BridgeHandler(self)
        handler.setLevel(level)
        root = logging.getLogger(root_logger)
        root.setLevel(min(root.level or logging.DEBUG, level))
        if not any(isinstance(h, _BridgeHandler) for h in root.handlers):
            root.addHandler(handler)

    def close(self) -> None:
        if self._info_file:
            self._info_file.close()
            self._info_file = None
        if self._trace_file:
            self._trace_file.close()
            self._trace_file = None

    def __enter__(self) -> "PipelineLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class _BridgeHandler(logging.Handler):
    _MAP = {
        logging.DEBUG:    "debug",
        logging.INFO:     "info",
        logging.WARNING:  "warn",
        logging.ERROR:    "error",
        logging.CRITICAL: "error",
    }

    def __init__(self, logger: PipelineLogger) -> None:
        super().__init__()
        self._plm = logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            getattr(self._plm, self._MAP.get(record.levelno, "info"))(
                f"[{record.name}] {msg}"
            )
        except Exception:
            self.handleError(record)


_default: PipelineLogger | None = None


def get_logger() -> PipelineLogger:
    global _default
    if _default is None:
        _default = PipelineLogger()
    return _default


def set_logger(logger: PipelineLogger) -> None:
    global _default
    _default = logger
