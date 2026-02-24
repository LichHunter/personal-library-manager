"""Prompt templates for benchmark case generation.

Generator prompts for converting SignalBundles to benchmark cases.
"""

GENERATOR_SYSTEM_PROMPT = """\
You are generating a benchmark question for evaluating a documentation search system.
Your task is to create a natural language query and identify evidence from the source materials.
Output ONLY valid JSON - no additional text, explanations, or markdown formatting."""


def build_generator_prompt(
    question_title: str,
    question_body: str,
    answer_body: str,
    chunk_contents: list[str],
    signals_summary: str,
    tier: str,
) -> str:
    """Build the generator prompt from a SignalBundle.

    Args:
        question_title: SO question title
        question_body: SO question body (may contain HTML)
        answer_body: SO answer body (may contain HTML)
        chunk_contents: List of documentation chunk texts
        signals_summary: Human-readable summary of detected signals
        tier: Target tier (gold, silver, bronze)

    Returns:
        Formatted prompt string
    """
    # Truncate long content to fit in context window
    max_chunk_chars = 8000
    max_question_chars = 2000
    max_answer_chars = 4000

    truncated_question = question_body[:max_question_chars]
    if len(question_body) > max_question_chars:
        truncated_question += "\n... [truncated]"

    truncated_answer = answer_body[:max_answer_chars]
    if len(answer_body) > max_answer_chars:
        truncated_answer += "\n... [truncated]"

    # Combine chunk contents with separators
    combined_chunks = ""
    total_chars = 0
    for i, chunk in enumerate(chunk_contents):
        if total_chars + len(chunk) > max_chunk_chars:
            remaining = max_chunk_chars - total_chars
            if remaining > 100:
                combined_chunks += f"\n--- CHUNK {i+1} ---\n{chunk[:remaining]}... [truncated]\n"
            break
        combined_chunks += f"\n--- CHUNK {i+1} ---\n{chunk}\n"
        total_chars += len(chunk)

    # Build tier-specific instructions
    if tier == "gold":
        quote_instruction = """2. Find an EXACT quote from the documentation chunks:
   - The quote MUST appear VERBATIM in the chunk content above
   - The quote MUST be at least 30 characters long
   - Copy it character-for-character with no modifications
   - This is REQUIRED for GOLD tier"""
    else:
        quote_instruction = """2. If possible, find an EXACT quote from the documentation chunks:
   - The quote MUST appear VERBATIM in the chunk content above
   - If you find a quote, it should be at least 30 characters
   - Return null if no good quote is found"""

    prompt = f"""\
## Source Information

### Stack Overflow Question
**Title:** {question_title}

**Body:**
{truncated_question}

### Stack Overflow Answer (Links to Documentation)
{truncated_answer}

### Documentation Chunks
{combined_chunks}

### Signal Summary
{signals_summary}

## Your Task

1. Create a NATURAL LANGUAGE QUESTION that:
   - Captures the user's information need from the SO question
   - Is reformulated (NOT a copy of the SO title)
   - Is between 5 and 100 words long
   - Would be answered by the documentation chunks above

{quote_instruction}

3. Write a brief EVIDENCE explanation (1-2 sentences) explaining why these chunks answer the question.

## Self-Check Requirements (Your output MUST pass these)
- Query MUST be 5-100 words (count carefully)
- If matched_quote is provided, it MUST appear EXACTLY in one of the chunks above
- If matched_quote is provided for GOLD tier, length MUST be >= 30 characters
- Query should be a genuine question, not a statement

## Output Format
Return ONLY a JSON object with this exact structure:
{{
  "query": "Your natural language question here",
  "matched_quote": "Exact verbatim text from chunk" or null,
  "evidence_text": "Brief explanation of why chunks answer the question",
  "reasoning": "Your reasoning about how SO question maps to documentation"
}}

IMPORTANT: Output ONLY the JSON object, no other text."""

    return prompt


def build_signals_summary(
    quote_matches: list[dict],
    reciprocal_matches: list[dict],
    fragment_matches_heading: bool,
    url_fragment: str | None,
    answer_upvotes: int,
    is_accepted: bool,
) -> str:
    """Build human-readable summary of detected signals.

    Args:
        quote_matches: List of QuoteMatch dicts
        reciprocal_matches: List of ReciprocalMatch dicts
        fragment_matches_heading: Whether URL fragment matches a heading
        url_fragment: URL fragment if present
        answer_upvotes: Answer upvote count
        is_accepted: Whether answer is accepted

    Returns:
        Human-readable signal summary
    """
    signals = []

    if fragment_matches_heading and url_fragment:
        signals.append(f"URL fragment #{url_fragment} matches a documentation heading")

    if quote_matches:
        max_len = max(m.get("match_length", 0) for m in quote_matches)
        signals.append(f"{len(quote_matches)} quote match(es) found (max length: {max_len} chars)")

    if reciprocal_matches:
        max_words = max(m.get("word_count", 0) for m in reciprocal_matches)
        signals.append(f"{len(reciprocal_matches)} reciprocal match(es) found (max: {max_words} words)")

    if is_accepted:
        signals.append("This is the ACCEPTED answer")

    signals.append(f"Answer has {answer_upvotes} upvotes")

    return "\n".join(f"- {s}" for s in signals) if signals else "No special signals detected"
