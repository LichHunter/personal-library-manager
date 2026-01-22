# RAPTOR Reverse Engineering Documentation

> Comprehensive analysis of [RAPTOR](https://github.com/parthsarthi03/raptor) (Recursive Abstractive Processing for Tree-Organized Retrieval)

## Repository Structure

```
raptor/
├── source/                    # Original RAPTOR source code (cloned)
├── README.md                  # This file - documentation index
│
├── 01-discovery/              # Discovery Documents
│   ├── codebase-inventory.md  # File-by-file inventory and roles
│   ├── dependency-analysis.md # External dependencies and rationale
│   └── api-surface.md         # Public API and entry points
│
├── 02-structure/              # Structural Documents (Static View)
│   ├── component-diagram.md   # High-level modules and boundaries
│   ├── class-diagram.md       # Class relationships and hierarchy
│   ├── data-model.md          # Data structures and schemas
│   └── package-diagram.md     # Code organization and layers
│
├── 03-behavior/               # Behavioral Documents (Dynamic View)
│   ├── sequence-diagrams.md   # Step-by-step flows for key operations
│   ├── state-diagrams.md      # States and transitions
│   ├── activity-flowcharts.md # Decision logic and branching
│   └── data-flow.md           # Data transformation through system
│
├── 04-algorithms/             # Algorithm Documents
│   ├── clustering.md          # RAPTOR clustering algorithm details
│   ├── tree-building.md       # Tree construction algorithm
│   ├── retrieval.md           # Query and retrieval algorithms
│   └── complexity-analysis.md # Time/space complexity analysis
│
├── 05-integration/            # Integration Documents
│   ├── extension-points.md    # How to customize/extend
│   ├── configuration.md       # All config options and effects
│   └── error-handling.md      # Error propagation and recovery
│
└── 06-assessment/             # Quality Assessment
    ├── code-quality.md        # Issues, tech debt, anti-patterns
    ├── test-coverage.md       # What's tested, gaps
    └── performance.md         # Benchmarks, bottlenecks
```

## Documentation Status

| Section | Document | Status | Lines |
|---------|----------|--------|-------|
| Discovery | [Codebase Inventory](01-discovery/codebase-inventory.md) | ✅ Complete | 357 |
| Discovery | [Dependency Analysis](01-discovery/dependency-analysis.md) | ✅ Complete | 287 |
| Discovery | [API Surface](01-discovery/api-surface.md) | ✅ Complete | 520 |
| Structure | [Component Diagram](02-structure/component-diagram.md) | ✅ Complete | 238 |
| Structure | [Class Diagram](02-structure/class-diagram.md) | ✅ Complete | 643 |
| Structure | [Data Model](02-structure/data-model.md) | ✅ Complete | 378 |
| Structure | [Package Diagram](02-structure/package-diagram.md) | ✅ Complete | 367 |
| Behavior | [Sequence Diagrams](03-behavior/sequence-diagrams.md) | ✅ Complete | 471 |
| Behavior | [State Diagrams](03-behavior/state-diagrams.md) | ✅ Complete | 489 |
| Behavior | [Activity Flowcharts](03-behavior/activity-flowcharts.md) | ✅ Complete | 757 |
| Behavior | [Data Flow](03-behavior/data-flow.md) | ✅ Complete | 575 |
| Algorithms | [Clustering](04-algorithms/clustering.md) | ✅ Complete | 437 |
| Algorithms | [Tree Building](04-algorithms/tree-building.md) | ✅ Complete | 619 |
| Algorithms | [Retrieval](04-algorithms/retrieval.md) | ✅ Complete | 651 |
| Algorithms | [Complexity Analysis](04-algorithms/complexity-analysis.md) | ✅ Complete | 436 |
| Integration | [Extension Points](05-integration/extension-points.md) | ✅ Complete | 880 |
| Integration | [Configuration](05-integration/configuration.md) | ✅ Complete | 565 |
| Integration | [Error Handling](05-integration/error-handling.md) | ✅ Complete | 640 |
| Assessment | [Code Quality](06-assessment/code-quality.md) | ✅ Complete | 315 |
| Assessment | [Test Coverage](06-assessment/test-coverage.md) | ✅ Complete | 455 |
| Assessment | [Performance](06-assessment/performance.md) | ✅ Complete | 394 |

**Total Documentation:** ~10,500 lines across 21 documents

## Source Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `raptor/__init__.py` | ~17 | Public API exports |
| `raptor/tree_structures.py` | ~29 | Node and Tree data classes |
| `raptor/tree_builder.py` | ~370 | Abstract TreeBuilder + config |
| `raptor/cluster_tree_builder.py` | ~152 | ClusterTreeBuilder implementation |
| `raptor/cluster_utils.py` | ~186 | Clustering algorithms (UMAP, GMM) |
| `raptor/tree_retriever.py` | ~328 | TreeRetriever + config |
| `raptor/RetrievalAugmentation.py` | ~307 | Main facade class |
| `raptor/EmbeddingModels.py` | ~38 | Embedding model interfaces |
| `raptor/SummarizationModels.py` | ~75 | Summarization model interfaces |
| `raptor/QAModels.py` | ~186 | QA model interfaces |
| `raptor/Retrievers.py` | ~9 | BaseRetriever abstract class |
| `raptor/utils.py` | ~209 | Utility functions |
| `raptor/FaissRetriever.py` | ? | FAISS-based retriever |

## Key Questions to Answer

1. **How does RAPTOR build the hierarchical tree?**
2. **How does the clustering algorithm work in detail?**
3. **What are the two retrieval strategies and when to use each?**
4. **How extensible is the system? What can be swapped?**
5. **What are the performance characteristics and limitations?**
6. **What design decisions were made and why?**
7. **What are the code quality issues or areas for improvement?**
