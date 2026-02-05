"""Centralized logging for benchmark runs.

Provides structured logging with:
- Console output with timestamps
- File output for persistence
- Section headers for visual organization
- Progress tracking
- Metric logging
- Timing utilities
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class TimerEntry:
    """Track a named timer."""

    name: str
    start_time: float
    end_time: float | None = None

    @property
    def elapsed(self) -> float:
        if self.end_time is None:
            return time.perf_counter() - self.start_time
        return self.end_time - self.start_time


class BenchmarkLogger:
    """Centralized logging for benchmark runs.

    Usage:
        logger = BenchmarkLogger(log_dir="results")
        logger.info("Starting benchmark")
        logger.section("Loading Data")
        logger.metric("documents", 52)

        with logger.timer("indexing"):
            # do work
            pass

        logger.progress(5, 100, "Processing queries")
        logger.close()
    """

    LEVELS = {"TRACE": -1, "DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}

    def __init__(
        self,
        log_dir: str | Path | None = None,
        log_file: str | None = None,
        console: bool = True,
        min_level: str = "INFO",
    ):
        """Initialize the logger.

        Args:
            log_dir: Directory for log files. If None, no file logging.
            log_file: Specific log filename. If None, auto-generates timestamp-based name.
            console: Whether to output to console.
            min_level: Minimum log level to output (DEBUG, INFO, WARN, ERROR).
        """
        self.console = console
        self.min_level = self.LEVELS.get(min_level.upper(), 1)
        self.file_handle: TextIO | None = None
        self.log_path: Path | None = None
        self.timers: dict[str, TimerEntry] = {}
        self.metrics: dict[str, list[tuple[float, Any]]] = {}
        self.start_time = time.perf_counter()

        # Set up file logging
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            if log_file:
                self.log_path = log_dir / log_file
            else:
                timestamp = time.strftime("%Y-%m-%d_%H%M%S")
                self.log_path = log_dir / f"benchmark_{timestamp}.log"

            self.file_handle = open(self.log_path, "w", encoding="utf-8")
            self._write_header()

    def _write_header(self):
        """Write log file header."""
        if self.file_handle:
            self.file_handle.write("=" * 80 + "\n")
            self.file_handle.write(
                f"Benchmark Log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            self.file_handle.write("=" * 80 + "\n\n")
            self.file_handle.flush()

    def _format_message(self, level: str, msg: str) -> str:
        """Format a log message with timestamp and level."""
        timestamp = time.strftime("%H:%M:%S")
        return f"[{timestamp}] {level:5} | {msg}"

    def _log(self, level: str, msg: str):
        """Internal logging method."""
        if self.LEVELS.get(level, 1) < self.min_level:
            return

        formatted = self._format_message(level, msg)

        if self.console:
            print(formatted, flush=True)

        if self.file_handle:
            self.file_handle.write(formatted + "\n")
            self.file_handle.flush()

    def trace(self, msg: str):
        self._log("TRACE", msg)

    def debug(self, msg: str):
        self._log("DEBUG", msg)

    def info(self, msg: str):
        """Log info message."""
        self._log("INFO", msg)

    def warn(self, msg: str):
        """Log warning message."""
        self._log("WARN", msg)

    def error(self, msg: str):
        """Log error message."""
        self._log("ERROR", msg)

    def section(self, title: str):
        """Log a section header for visual organization."""
        separator = "=" * 80

        if self.console:
            print(f"\n{separator}", flush=True)

        if self.file_handle:
            self.file_handle.write(f"\n{separator}\n")

        self._log("INFO", title)

        if self.console:
            print(separator, flush=True)

        if self.file_handle:
            self.file_handle.write(separator + "\n")
            self.file_handle.flush()

    def subsection(self, title: str):
        """Log a subsection header."""
        separator = "-" * 60

        if self.console:
            print(f"\n{separator}", flush=True)

        if self.file_handle:
            self.file_handle.write(f"\n{separator}\n")

        self._log("INFO", title)

        if self.console:
            print(separator, flush=True)

        if self.file_handle:
            self.file_handle.write(separator + "\n")
            self.file_handle.flush()

    def progress(self, current: int, total: int, msg: str = ""):
        """Log progress indicator."""
        pct = (current / total * 100) if total > 0 else 0
        bar_width = 20
        filled = int(bar_width * current / total) if total > 0 else 0
        bar = "█" * filled + "░" * (bar_width - filled)

        progress_str = f"[{current:3}/{total:3}] {bar} {pct:5.1f}%"
        if msg:
            progress_str += f" | {msg}"

        self._log("PROG", progress_str)

    def metric(self, name: str, value: Any, unit: str = ""):
        """Log a metric value."""
        # Store metric
        timestamp = time.perf_counter() - self.start_time
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append((timestamp, value))

        # Format value
        if isinstance(value, float):
            if unit in ("s", "ms", "us"):
                value_str = f"{value:.3f}{unit}"
            elif unit == "%":
                display_value = value * 100 if value <= 1.0 else value
                value_str = f"{display_value:.1f}%"
            else:
                value_str = f"{value:.4f}"
                if unit:
                    value_str += f" {unit}"
        else:
            value_str = str(value)
            if unit:
                value_str += f" {unit}"

        self._log("METRIC", f"{name}={value_str}")

    def table(self, headers: list[str], rows: list[list[Any]], title: str = ""):
        """Log tabular data."""
        if not rows:
            return

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(str(cell)))

        # Format header
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        separator = "-+-".join("-" * w for w in widths)

        lines = []
        if title:
            lines.append(f"\n{title}")
        lines.append(header_line)
        lines.append(separator)

        for row in rows:
            row_line = " | ".join(
                str(cell).ljust(widths[i]) if i < len(widths) else str(cell)
                for i, cell in enumerate(row)
            )
            lines.append(row_line)

        output = "\n".join(lines)

        if self.console:
            print(output, flush=True)

        if self.file_handle:
            self.file_handle.write(output + "\n")
            self.file_handle.flush()

    def timer_start(self, name: str):
        """Start a named timer."""
        self.timers[name] = TimerEntry(name=name, start_time=time.perf_counter())
        self.debug(f"Timer '{name}' started")

    def timer_end(self, name: str) -> float:
        """End a named timer and return elapsed time."""
        if name not in self.timers:
            self.warn(f"Timer '{name}' was not started")
            return 0.0

        timer = self.timers[name]
        timer.end_time = time.perf_counter()
        elapsed = timer.elapsed

        self.metric(f"{name}_time", elapsed, "s")
        return elapsed

    class _TimerContext:
        """Context manager for timing blocks."""

        def __init__(self, logger: "BenchmarkLogger", name: str):
            self.logger = logger
            self.name = name

        def __enter__(self):
            self.logger.timer_start(self.name)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.logger.timer_end(self.name)
            return False

    def timer(self, name: str) -> _TimerContext:
        """Context manager for timing a block of code.

        Usage:
            with logger.timer("indexing"):
                # do work
                pass
        """
        return self._TimerContext(self, name)

    def indent(self, msg: str, level: int = 1) -> str:
        """Create an indented message with tree-like prefix."""
        if level <= 0:
            return msg
        prefix = "│  " * (level - 1) + "├─ "
        return prefix + msg

    def indent_last(self, msg: str, level: int = 1) -> str:
        """Create an indented message for last item in a group."""
        if level <= 0:
            return msg
        prefix = "│  " * (level - 1) + "└─ "
        return prefix + msg

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all logged metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if not values:
                continue
            vals = [v for _, v in values]
            if all(isinstance(v, (int, float)) for v in vals):
                summary[name] = {
                    "count": len(vals),
                    "min": min(vals),
                    "max": max(vals),
                    "avg": sum(vals) / len(vals),
                    "last": vals[-1],
                }
            else:
                summary[name] = {"count": len(vals), "last": vals[-1]}
        return summary

    def get_timers_summary(self) -> dict[str, float]:
        """Get summary of all timer durations."""
        return {
            name: timer.elapsed
            for name, timer in self.timers.items()
            if timer.end_time is not None
        }

    def summary(self):
        """Log a summary of the benchmark run."""
        self.section("RUN SUMMARY")

        total_time = time.perf_counter() - self.start_time
        self.info(f"Total runtime: {total_time:.2f}s")

        # Timer summary
        timers = self.get_timers_summary()
        if timers:
            self.subsection("Timing Breakdown")
            rows = [
                [name, f"{elapsed:.2f}s"] for name, elapsed in sorted(timers.items())
            ]
            self.table(["Phase", "Duration"], rows)

        if self.log_path:
            self.info(f"Log saved to: {self.log_path}")

    def close(self):
        """Close the logger and file handle."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Convenience function for simple usage
_default_logger: BenchmarkLogger | None = None


def get_logger() -> BenchmarkLogger:
    """Get or create the default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = BenchmarkLogger()
    return _default_logger


def set_logger(logger: BenchmarkLogger):
    """Set the default logger."""
    global _default_logger
    _default_logger = logger
