from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from plm.extraction.fast.document_processor import DocumentResult, process_document
from plm.extraction.fast.gliner import ExtractedEntity
from plm.shared.logger import PipelineLogger


def _serialize_result(result: DocumentResult) -> dict:
    data = asdict(result)
    for section in data["headings"]:
        for chunk in section["chunks"]:
            chunk["entities"] = [
                {"text": e["text"], "label": e["label"], "score": e["score"],
                 "start": e["start"], "end": e["end"]}
                for e in chunk["entities"]
            ]
    return data


def _collect_files(input_dir: Path, patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(input_dir.glob(pattern))
    return sorted(set(files))


def _write_manifest(low_confidence_dir: Path, flagged: list[dict]) -> None:
    manifest = {
        "description": "Documents flagged as low-confidence by GLiNER extraction",
        "total_flagged": len(flagged),
        "documents": flagged,
    }
    (low_confidence_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch extract entities from documents using GLiNER.",
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log-file", type=Path, default=None,
                        help="INFO+ log file (readable summary)")
    parser.add_argument("--trace-file", type=Path, default=None,
                        help="TRACE+ log file (full detail, every line)")
    parser.add_argument("--low-confidence-dir", type=Path, default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--extraction-threshold", type=float, default=0.3)
    parser.add_argument("--pattern", type=str, default="**/*.md,**/*.txt")
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of parallel threads for document processing. "
            "GLiNER (the bottleneck at ~97%% of processing time) partially "
            "releases the GIL during PyTorch inference, so threading helps. "
            "Recommended: --workers 8 with torch threads auto-set to "
            "max(1, cpu_count // workers). On a 16-core machine this gives "
            "~3x speedup at ~200MB extra RAM. Diminishing returns beyond 8. "
            "Note: per-file log output is interleaved when workers > 1."
        ),
    )

    args = parser.parse_args(argv)

    if not args.input.is_dir():
        print(f"Error: Input directory does not exist: {args.input}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    if args.low_confidence_dir:
        args.low_confidence_dir.mkdir(parents=True, exist_ok=True)

    log = PipelineLogger(
        log_file=args.log_file,
        trace_file=args.trace_file,
        console=True,
        min_level="INFO",
    )
    log.install_stdlib_bridge(root_logger="plm", level=10)
    log.install_stdlib_bridge(root_logger="", level=30)

    log.section("PLM Fast Extraction")
    log.info(f"Input:                {args.input}")
    log.info(f"Output:               {args.output}")
    log.info(f"Confidence threshold: {args.confidence_threshold}")
    log.info(f"Extraction threshold: {args.extraction_threshold}")
    if args.log_file:
        log.info(f"Info log:             {args.log_file}")
    if args.trace_file:
        log.info(f"Trace log:            {args.trace_file}")

    patterns = [p.strip() for p in args.pattern.split(",")]
    files = _collect_files(args.input, patterns)

    if not files:
        log.warn(f"No files matching {patterns} found in {args.input}")
        log.close()
        return 0

    log.metric("files_found", len(files))

    # Configure torch threading for parallel workers.
    # GLiNER is ~97% of processing time and partially releases the GIL
    # during PyTorch inference. To avoid thread contention:
    #   total_threads = workers × torch_threads ≈ cpu_count
    # Benchmarked on 16-core CPU (200 K8s docs, GLiNER medium-v2.1):
    #   1 worker, 16 torch threads: 3.2s/doc (baseline)
    #   4 workers,  4 torch threads: 2.1s/doc (1.5x)
    #   8 workers,  2 torch threads: 2.0s/doc (1.6x, best throughput)
    #   8 workers,  1 torch thread:  ~same, slight regression
    # RAM: ~1.3GB base (model) + ~25MB per extra worker thread.
    # Diminishing returns beyond 8 workers on 16 cores.
    n_workers = max(1, args.workers)
    if n_workers > 1:
        import torch

        cpu_count = os.cpu_count() or 4
        torch_threads = max(1, cpu_count // n_workers)
        torch.set_num_threads(torch_threads)
        log.info(f"Workers:              {n_workers} (torch_threads={torch_threads})")

    total_entities = 0
    total_chunks = 0
    total_keywords = 0
    total_sections = 0
    flagged_docs: list[dict] = []
    errors: list[str] = []
    file_times: list[float] = []

    _lock = threading.Lock()

    def _process_one(filepath: Path, index: int) -> None:
        """Process a single file: extract, log, write output. Thread-safe."""
        nonlocal total_entities, total_chunks, total_keywords, total_sections
        rel_path = filepath.relative_to(args.input)

        t0 = time.perf_counter()
        result = process_document(
            filepath,
            confidence_threshold=args.confidence_threshold,
            extraction_threshold=args.extraction_threshold,
        )
        elapsed = time.perf_counter() - t0

        with _lock:
            file_times.append(elapsed)

            if result.error:
                log.error(f"[{index}/{len(files)}] FAILED {rel_path}: {result.error}")
                errors.append(str(rel_path))
                return

            n_sections = len(result.headings)
            n_chunks = sum(len(s.chunks) for s in result.headings)
            n_entities = result.total_entities
            n_keywords = sum(
                len(c.keywords) for s in result.headings for c in s.chunks
            )
            avg_conf = result.avg_confidence

            total_entities += n_entities
            total_chunks += n_chunks
            total_keywords += n_keywords
            total_sections += n_sections

            log.progress(index, len(files), str(rel_path))

            if n_workers == 1:
                log.metric("sections", n_sections)
                log.metric("chunks", n_chunks)
                log.metric("entities", n_entities)
                log.metric("keywords", n_keywords)
                log.metric("avg_confidence", round(avg_conf, 4))
                log.metric("file_time_s", round(elapsed, 3), "s")

                for sec in result.headings:
                    sec_entities = sum(len(c.entities) for c in sec.chunks)
                    sec_keywords = sum(len(c.keywords) for c in sec.chunks)
                    log.trace(
                        f"  section {sec.heading!r:50s} "
                        f"chunks={len(sec.chunks):2d}  "
                        f"ents={sec_entities:3d}  kw={sec_keywords:3d}"
                    )
                    for chunk in sec.chunks:
                        log.trace(
                            f"    chunk [{chunk.start_char}:{chunk.end_char}]  "
                            f"kw={chunk.keywords}  "
                            f"ents={[e.text for e in chunk.entities]}"
                        )

            if result.is_low_confidence:
                log.warn(
                    f"Low confidence: {rel_path} avg={avg_conf:.3f}"
                    f" < {args.confidence_threshold}"
                )

        output_path = args.output / rel_path.with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = _serialize_result(result)
        output_path.write_text(json.dumps(output_data, indent=2))

        if result.is_low_confidence and args.low_confidence_dir:
            dest = args.low_confidence_dir / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, dest)
            json_dest = args.low_confidence_dir / rel_path.with_suffix(".json")
            json_dest.write_text(json.dumps(output_data, indent=2))
            with _lock:
                flagged_docs.append({
                    "file": str(rel_path),
                    "avg_confidence": round(avg_conf, 4),
                    "total_entities": n_entities,
                })

    log.section(f"Processing {len(files)} files")

    with log.timer("total_extraction"):
        if n_workers == 1:
            for i, filepath in enumerate(files, 1):
                log.subsection(f"[{i}/{len(files)}] {filepath.relative_to(args.input)}")
                _process_one(filepath, i)
        else:
            log.info("Warming up GLiNER model...")
            from plm.extraction.fast.gliner import get_model
            get_model()

            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(_process_one, fp, i): fp
                    for i, fp in enumerate(files, 1)
                }
                for future in as_completed(futures):
                    exc = future.exception()
                    if exc is not None:
                        fp = futures[future]
                        with _lock:
                            log.error(f"Exception processing {fp}: {exc}")
                            errors.append(str(fp))

    if args.low_confidence_dir and flagged_docs:
        _write_manifest(args.low_confidence_dir, flagged_docs)

    log.section("Extraction Summary")
    log.metric("files_processed", len(files) - len(errors))
    log.metric("files_errored",   len(errors))
    log.metric("files_flagged",   len(flagged_docs))
    log.metric("total_sections",  total_sections)
    log.metric("total_chunks",    total_chunks)
    log.metric("total_entities",  total_entities)
    log.metric("total_keywords",  total_keywords)

    if file_times:
        avg_t = sum(file_times) / len(file_times)
        log.metric("avg_file_time_s", round(avg_t, 3), "s")
        log.metric("min_file_time_s", round(min(file_times), 3), "s")
        log.metric("max_file_time_s", round(max(file_times), 3), "s")

    if errors:
        log.subsection("Errors")
        for e in errors:
            log.error(f"  {e}")

    log.summary()
    log.close()
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
