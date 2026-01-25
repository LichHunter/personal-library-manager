# Manual Grading Analysis - 2026-01-25

## Test Run Details
- **Strategy**: enriched_hybrid_llm
- **Questions**: 10
- **Haiku's Average**: 7.2/10 (INCONCLUSIVE)
- **My Manual Review**: Below

---

## Question-by-Question Manual Grading

### Question 1: What is the default API request limit per minute?

**Expected**: 100 requests per minute per authenticated user, 20 requests per minute for unauthenticated requests

**Retrieved Chunk 1**: 
> "## Rate Limiting To ensure fair usage and system stability, CloudFlow enforces rate limits on all API endpoints. **Default Limits:** - 100 requests per minute per authenticated user - 20 requests per ..."

**My Grade**: **10/10** ✅
**Reasoning**: PERFECT. Chunk 1 contains the EXACT answer word-for-word. Both limits are stated clearly.

**Haiku's Grade**: 10/10 ✅
**Agreement**: YES - Haiku got this right.

---

### Question 2: How long do CloudFlow JWT tokens remain valid?

**Expected**: All tokens expire after 3600 seconds (1 hour). Implement token refresh logic in your application.

**Retrieved Chunks**: 
- Chunk 1: API Reference overview (generic)
- Chunk 2: JWT token validation (no expiry time)
- Chunk 4: JWT time diagnostics (no expiry stated)
- Chunk 5: Cache TTL strategies mentioning "1 hour (workflow definitions, templates)"

**My Grade**: **3/10** ❌
**Reasoning**: FAIL. None of the chunks explicitly state "JWT tokens expire after 3600 seconds (1 hour)". Chunk 5 mentions "1 hour" but for workflow definitions/templates, NOT JWT tokens. This is a retrieval failure.

**Haiku's Grade**: 5/10 (PARTIAL)
**Agreement**: NO - Haiku was too generous. This should be FAIL (3/10 or lower).

---

### Question 3: What authentication methods does CloudFlow support?

**Expected**: CloudFlow supports three authentication methods: API Keys, OAuth 2.0, and JWT Tokens

**Retrieved Chunk 1**: 
> "# CloudFlow API Reference Version 2.1.0... The CloudFlow API is a RESTful service that enables developers to programmatically manage cloud workflows, data pipelin..."

**My Grade**: **4/10** ❌
**Reasoning**: POOR. The chunks are truncated and don't show the actual authentication methods section. We can't see the answer in the retrieved content. This is a chunking/truncation problem.

**Haiku's Grade**: 8/10 (PASS)
**Agreement**: NO - Haiku hallucinated or inferred. The retrieved chunks don't show the authentication methods clearly.

---

### Question 4: How can I handle a 429 Too Many Requests error from the CloudFlow API?

**Expected**: When you exceed the rate limit, you'll receive a 429 Too Many Requests response. Best practices include monitoring `X-RateLimit-Remaining` header values, implementing exponential backoff, and caching responses to reduce API calls.

**Retrieved Chunks**:
- Chunk 1: Rate limiting info (100 req/min, 20 req/min)
- Chunk 5: Code snippet with retry logic and `X-RateLimit-Remaining` checking

**My Grade**: **9/10** ✅
**Reasoning**: EXCELLENT. Chunk 5 shows actual code for handling rate limits with `X-RateLimit-Remaining` header checking and retry logic. Chunk 1 provides context. Very good retrieval.

**Haiku's Grade**: 9/10 (PASS)
**Agreement**: YES - Haiku got this right.

---

### Question 5: What are the maximum concurrent workflow executions allowed?

**Expected**: Rate Limit: 100 concurrent executions per workflow, 10 executions per second

**Retrieved Chunks**:
- Chunk 3: "Execution start rate: 500 per second" (different metric)
- Chunk 5: "100 requests per minute" (API rate limit, not workflow executions)

**My Grade**: **2/10** ❌
**Reasoning**: FAIL. None of the chunks mention "100 concurrent executions per workflow" or "10 executions per second". The retrieved content is about different metrics (API rate limits, execution start rate). Complete miss.

**Haiku's Grade**: 5/10 (PARTIAL)
**Agreement**: NO - Haiku was too generous. This is a clear FAIL.

---

### Question 6: How do I add error handling to a workflow?

**Expected**: Always implement error handling for external API calls and database operations. Use retry policies with exponential backoff and set up fallback actions for critical steps.

**Retrieved Chunk 4**:
> "Handle Errors Gracefully Always implement error handling for external API calls and database operations: ```yaml - id: fetch_data action: http_request config: url: 'https://api.example.com/data' retry..."

**My Grade**: **10/10** ✅
**Reasoning**: PERFECT. Chunk 4 contains the EXACT answer with code example showing retry policies and error handling for external API calls.

**Haiku's Grade**: 9/10 (PASS)
**Agreement**: MOSTLY - I'd give it 10/10, Haiku gave 9/10. Close enough.

---

### Question 7: What database connection limits exist in CloudFlow?

**Expected**: PostgreSQL is configured with a maximum of 100 connections, with connection pooling handled via PgBouncer.

**Retrieved Chunk 5**: 
> "- Execution start rate: 500 per second... **Database**: - Read throughput: 50,000 queries per second... - Write throughp..."

**My Grade**: **2/10** ❌
**Reasoning**: FAIL. None of the chunks mention "100 connections" or "PgBouncer". Chunk 5 mentions "2,000 concurrent connections" which contradicts the expected answer. Wrong information retrieved.

**Haiku's Grade**: 6/10 (PARTIAL)
**Agreement**: NO - Haiku was too generous. This should be FAIL.

---

### Question 8: What are the recovery time objectives for CloudFlow?

**Expected**: Recovery Point Objective (RPO): 1 hour, Recovery Time Objective (RTO): 4 hours

**Retrieved Chunk 3**:
> "**Recovery Time Objective (RTO)**: 4 hours **Recovery Point Objective (RPO)**: 24 hours"

**My Grade**: **6/10** ⚠️
**Reasoning**: PARTIAL. RTO is correct (4 hours), but RPO is WRONG (24 hours vs expected 1 hour). Half right, half wrong.

**Haiku's Grade**: 8/10 (PASS)
**Agreement**: NO - Haiku missed that RPO is incorrect. Should be PARTIAL, not PASS.

---

### Question 9: What are the three supported authentication methods?

**Expected**: CloudFlow supports three authentication methods: API Keys, OAuth 2.0, and JWT Tokens with RS256 signing algorithm

**Retrieved Chunk 2**:
> "**Method**: GET, POST, PUT, PATCH, DELETE, HEAD... **Header... Authentication: Basic Auth, Bearer Token, API Key, OAuth 2.0"

**My Grade**: **7/10** ✅
**Reasoning**: GOOD. Chunk 2 mentions "API Key, OAuth 2.0" and "Bearer Token" (which could be JWT). Missing the specific "RS256 signing algorithm" detail. Mostly correct.

**Haiku's Grade**: 7/10 (PASS)
**Agreement**: YES - Haiku got this right.

---

### Question 10: How do I troubleshoot workflow execution failures?

**Expected**: Use the workflows executions get command with verbose flag to view execution details, check step-by-step breakdown, and identify bottleneck steps.

**Retrieved Chunks**:
- Chunk 2: Troubleshooting guide header (no details)
- Chunks 1,3,4,5: Deployment configs, metrics, database (not relevant)

**My Grade**: **1/10** ❌
**Reasoning**: COMPLETE FAIL. None of the chunks mention the "workflows executions get" command or how to troubleshoot execution failures. Just headers and unrelated content.

**Haiku's Grade**: 5/10 (PARTIAL)
**Agreement**: NO - Haiku was WAY too generous. This is a complete miss.

---

## Summary: My Manual Grades vs Haiku's Grades

| # | Question | My Grade | Haiku Grade | Difference | Agreement |
|---|----------|----------|-------------|------------|-----------|
| 1 | API rate limit | 10/10 | 10/10 | 0 | ✅ YES |
| 2 | JWT token validity | 3/10 | 5/10 | -2 | ❌ NO (Haiku too generous) |
| 3 | Auth methods | 4/10 | 8/10 | -4 | ❌ NO (Haiku hallucinated) |
| 4 | 429 error handling | 9/10 | 9/10 | 0 | ✅ YES |
| 5 | Concurrent executions | 2/10 | 5/10 | -3 | ❌ NO (Haiku too generous) |
| 6 | Error handling | 10/10 | 9/10 | +1 | ✅ MOSTLY |
| 7 | DB connection limits | 2/10 | 6/10 | -4 | ❌ NO (Haiku too generous) |
| 8 | Recovery objectives | 6/10 | 8/10 | -2 | ❌ NO (Haiku missed error) |
| 9 | Three auth methods | 7/10 | 7/10 | 0 | ✅ YES |
| 10 | Troubleshoot failures | 1/10 | 5/10 | -4 | ❌ NO (Haiku too generous) |

**My Average**: **5.4/10** → **INVALIDATED** (below 5.5 threshold)
**Haiku's Average**: **7.2/10** → **INCONCLUSIVE**

**Difference**: -1.8 points (Haiku is consistently too generous)

---

## Key Findings

### 1. Haiku Grading Bias
**Haiku consistently over-grades** by ~1.8 points on average. It tends to give credit for:
- Tangentially related content (Questions 2, 5, 7, 10)
- Truncated/incomplete answers (Question 3)
- Partially correct answers (Question 8)

### 2. Retrieval Quality Issues

**EXCELLENT Retrieval** (9-10/10):
- Question 1: API rate limits - PERFECT match
- Question 4: 429 error handling - Code examples included
- Question 6: Error handling - Exact answer with examples

**GOOD Retrieval** (7-8/10):
- Question 9: Auth methods - Mostly correct, missing details

**PARTIAL Retrieval** (4-6/10):
- Question 3: Auth methods - Truncated, can't see answer
- Question 8: Recovery objectives - Half right (RTO), half wrong (RPO)

**FAILED Retrieval** (1-3/10):
- Question 2: JWT validity - No explicit answer
- Question 5: Concurrent executions - Wrong metrics retrieved
- Question 7: DB connections - Wrong information (2000 vs 100)
- Question 10: Troubleshooting - No relevant content

### 3. Root Causes of Failures

**Chunking Issues**:
- Question 3: Answer truncated mid-sentence
- Question 2: JWT expiry info likely in a different chunk

**Vocabulary Mismatch**:
- Question 5: "concurrent executions" vs "execution start rate"
- Question 7: "connection limits" vs "concurrent connections"

**Missing Content**:
- Question 10: Specific CLI command not in retrieved chunks
- Question 2: JWT expiry time not explicitly stated

### 4. Benchmark Validation

**My Verdict**: **INVALIDATED** (5.4/10 average)

The enriched_hybrid_llm strategy shows:
- **30% excellent retrieval** (3/10 questions scored 9-10)
- **10% good retrieval** (1/10 questions scored 7-8)
- **10% partial retrieval** (1/10 questions scored 4-6)
- **50% failed retrieval** (5/10 questions scored 1-3)

**The 88.7% benchmark claim is OVERSTATED**. The strategy fails to retrieve correct information for half the questions.

---

## Recommendations

### Immediate Actions
1. **Don't trust LLM grading** - Haiku is too generous (+1.8 point bias)
2. **Investigate chunking** - Many answers are truncated
3. **Check vocabulary matching** - BM25 missing semantic equivalents

### Potential Improvements
1. **Larger chunks** - 512 tokens may be too small, cutting off answers
2. **Contextual chunking** - Add document/section context to each chunk
3. **Better query rewriting** - "concurrent executions" → "execution limits"
4. **Proposition chunking** - Extract atomic facts instead of fixed-size chunks

### Next Steps
1. Run same test with **semantic-only** strategy (baseline)
2. Run same test with **hybrid** strategy (no LLM rewriting)
3. Compare results to identify if LLM query rewriting helps or hurts
4. Investigate the 5 failed questions - why did retrieval miss?
