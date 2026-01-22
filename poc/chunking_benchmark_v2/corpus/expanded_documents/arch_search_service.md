# Search Service Architecture

## Overview

The Search Service provides full-text and semantic search across workflows, executions, and documentation. Built on Elasticsearch with custom analyzers for technical content.

## Design Principles

- **Relevance First**: Results ranked by relevance, not recency
- **Typo Tolerance**: Handles misspellings and variations

## Components

### Indexer

Indexes documents in near real-time. Typical indexing latency is 500ms.

**Technology:** Custom Go service
**Scaling:** Queue-based with backpressure

### Query Engine

Executes search queries. Supports filters, facets, and aggregations.

**Technology:** Elasticsearch 8.x
**Scaling:** 3-node cluster with replicas

### Suggestion Service

Provides autocomplete suggestions. Returns suggestions within 20ms.

**Technology:** Completion suggester
**Scaling:** In-memory FST

## Data Flow

Document → Analyzer → Index → Query → Rank → Results

## Performance Characteristics

- **Latency P50:** 25ms
- **Latency P99:** 100ms
- **Throughput:** 2000 queries/second
