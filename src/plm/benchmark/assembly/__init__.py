"""Final assembly and audit metadata for benchmark framework.

Merges verified and regenerated cases, splits by tier, adds comprehensive
audit metadata, and generates statistics report.
"""

from plm.benchmark.assembly.assembler import (
    BenchmarkCaseAudit,
    assemble_datasets,
)

__all__ = [
    "BenchmarkCaseAudit",
    "assemble_datasets",
]
