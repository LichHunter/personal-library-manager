# Caching Layer Architecture

## Overview

The Caching Layer reduces latency and database load by caching frequently accessed data. Uses a multi-tier strategy with local and distributed caches.

## Design Principles

- **Cache Invalidation**: Explicit invalidation on writes
- **Graceful Degradation**: System works without cache

## Components

### Local Cache

First-level cache with 100ms TTL. Reduces Redis round-trips for hot data.

**Technology:** Ristretto in-memory cache
**Scaling:** Per-pod memory allocation

### Distributed Cache

Second-level cache with configurable TTL. Maximum key size 512MB.

**Technology:** Redis Cluster 7.x
**Scaling:** 6-node cluster with sharding

### CDN

Caches static assets and API responses. Cache hit ratio target 95%.

**Technology:** CloudFront with Lambda@Edge
**Scaling:** Global edge locations

## Data Flow

Request → Local Cache → Redis → Database → Cache Population

## Performance Characteristics

- **Latency P50:** 1ms (hit), 50ms (miss)
- **Latency P99:** 5ms (hit), 200ms (miss)
- **Throughput:** 100000 reads/second
