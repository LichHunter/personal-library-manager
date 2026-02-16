"""Fast heuristic extraction - zero LLM cost."""
import re
from typing import Iterator

# Regex patterns
CAMEL_CASE = re.compile(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b')
ALL_CAPS = re.compile(r'\b[A-Z][A-Z_]{2,}\b')
DOT_PATH = re.compile(r'\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+\b')
BACKTICK = re.compile(r'`([^`]+)`')
FUNCTION_CALL = re.compile(r'\b[a-zA-Z_]\w*\(\)')


def extract_camel_case(text: str) -> Iterator[str]:
    """Extract CamelCase identifiers from text."""
    for match in CAMEL_CASE.finditer(text):
        yield match.group()


def extract_all_caps(text: str) -> Iterator[str]:
    """Extract ALL_CAPS constants from text."""
    for match in ALL_CAPS.finditer(text):
        yield match.group()


def extract_dot_paths(text: str) -> Iterator[str]:
    """Extract dot.path.notation identifiers from text."""
    for match in DOT_PATH.finditer(text):
        yield match.group()


def extract_backticks(text: str) -> Iterator[str]:
    """Extract content within backticks from text."""
    for match in BACKTICK.finditer(text):
        yield match.group(1)


def extract_function_calls(text: str) -> Iterator[str]:
    """Extract function call patterns (name()) from text."""
    for match in FUNCTION_CALL.finditer(text):
        yield match.group()


def extract_all_heuristic(text: str) -> list[str]:
    """Extract all heuristic candidates using all extractors."""
    candidates = set()
    for extractor in [
        extract_camel_case,
        extract_all_caps,
        extract_dot_paths,
        extract_backticks,
        extract_function_calls,
    ]:
        candidates.update(extractor(text))
    return list(candidates)
