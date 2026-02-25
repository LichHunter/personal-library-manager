"""LLM-based query rewriting for improved retrieval.

Converts user queries into documentation-aligned queries using Claude Haiku.
Addresses vocabulary mismatch by rewriting problem descriptions as feature questions
and expanding abbreviations and technical jargon.
"""

import time
from typing import Optional

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from enrichment.provider import call_llm
from logger import get_logger


QUERY_REWRITE_PROMPT = """You are a technical documentation search expert. Your task is to rewrite user questions as direct documentation lookup queries.

Guidelines:
1. Convert problem descriptions to feature/capability questions
2. Expand abbreviations and acronyms to full terms
3. Replace casual language with technical terminology
4. Align with documentation vocabulary and structure
5. Keep the rewritten query concise (one line)

Examples:
- "Why can't I schedule workflows every 30 seconds?" → "workflow scheduling minimum interval frequency constraints"
- "Why does my token stop working after 3600 seconds?" → "token expiration TTL lifetime 3600 seconds"
- "What's the RPO and RTO?" → "recovery point objective recovery time objective disaster recovery"

User question: {query}

Rewritten query (one line, no explanation):"""


def rewrite_query(query: str, timeout: float = 5.0, debug: bool = False) -> str:
    """Rewrite query using Claude Haiku for better documentation retrieval.

    Converts user queries into documentation-aligned queries by:
    - Converting problem descriptions to feature questions
    - Expanding abbreviations and technical jargon
    - Aligning with documentation terminology

    Args:
        query: Original user query
        timeout: Timeout in seconds for LLM call (default 5.0)
        debug: If True, log rewriting details

    Returns:
        Rewritten query string, or original query if rewriting fails
    """
    log = get_logger()

    if not query or not query.strip():
        return query

    if debug:
        log.debug(f"[query-rewrite] Rewriting query: {query}")

    start_time = time.time()

    try:
        prompt = QUERY_REWRITE_PROMPT.format(query=query)
        rewritten = call_llm(prompt, model="claude-haiku", timeout=int(timeout))
        elapsed = time.time() - start_time

        if rewritten and rewritten.strip():
            rewritten = rewritten.strip()
            if debug:
                log.debug(
                    f"[query-rewrite] SUCCESS in {elapsed:.3f}s: {query} → {rewritten}"
                )
            return rewritten
        else:
            if debug:
                log.debug(
                    f"[query-rewrite] Empty response in {elapsed:.3f}s, using original"
                )
            return query

    except Exception as e:
        elapsed = time.time() - start_time
        log.warn(f"[query-rewrite] ERROR after {elapsed:.3f}s: {type(e).__name__}: {e}")
        return query
