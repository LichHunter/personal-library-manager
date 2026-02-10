#!/usr/bin/env python3
"""Approach B: SLIMER structured prompting NER extraction.

Uses rich entity type definitions and annotation guidelines instead of examples.
Zero-shot — no retrieval database, no vocabulary lists.

Based on: "Show Less, Instruct More" (SLIMER, ACL 2024)
"""

import json
import re

from scoring import normalize_term, verify_span
from utils.llm_provider import call_llm


ENTITY_DEFINITIONS = """\
## Entity Types for StackOverflow Technical Content

### LIBRARY_FRAMEWORK
Specific software libraries, frameworks, or packages that developers import/use.
Includes: React, TensorFlow, jQuery, pandas, Spring Boot, Express.js, NumPy, Django, Prism, Silverlight
Boundary: extract the library name only — "the React library" becomes "React"
NOT entities: "library", "framework", "package", "module" (generic terms)

### PROGRAMMING_LANGUAGE
Named programming, scripting, or markup languages.
Includes: Python, JavaScript, C++, C#, TypeScript, HTML, SQL, Rust, Go, Ruby, Objective-C, xaml
Include versions when present: "Python 3.9", "C++11", "ES6"
NOT entities: "language", "code", "script", "markup" (generic terms)

### API_FUNCTION_CLASS
Specific API endpoints, functions, methods, class names, interfaces, or protocols.
Includes: querySelector(), ArrayList, HttpClient, console.log(), IEnumerable, NetConnection, WebClient
Include: method signatures with parentheses, interface names starting with I
NOT entities: "function", "method", "class", "API", "endpoint", "interface" (generic terms)

### APPLICATION_TOOL
Specific software applications, IDEs, developer tools, or services.
Includes: Visual Studio, Chrome, Docker, Postman, Git, npm, Webpack, VS Code, composer, codeplex
NOT entities: "application", "tool", "editor", "browser", "IDE" (generic terms)

### PLATFORM_OS
Operating systems, platforms, or runtime environments.
Includes: Windows, Linux, macOS, Android, iOS, Node.js, .NET, JVM
NOT entities: "platform", "operating system", "environment" (generic terms)

### DATA_STRUCTURE_TYPE
Specific data structures, data types, or file/data formats.
Includes: HashMap, ArrayList, JSON, XML, DataFrame, int, string, boolean, float, double, char, void, array, dict, tuple, set, queue, stack
Include: language-specific type keywords used as entities
NOT entities: "data structure", "type", "format", "object", "variable" (generic terms)

### ERROR_EXCEPTION
Specific error types, exception classes, or error codes.
Includes: NullPointerException, TypeError, 404, ECONNREFUSED, StackOverflowError
NOT entities: "error", "exception", "bug", "issue" (generic terms)

### UI_COMPONENT
Specific UI component types or widget names when used as technical entities.
Includes: Button, TextField, ListView, RecyclerView, Modal, Dropdown, slider, checkbox, trackbar, scrollbar, cell, menu, cursor
Context-dependent: "button" in a UI programming context IS an entity; in "press the button to submit" is NOT
NOT entities: "component", "widget", "element", "control" in generic sense

### WEBSITE_SERVICE_ORGANIZATION
Specific websites, web services, online platforms, or tech organizations.
Includes: GitHub, StackOverflow, Google, Microsoft, npm registry, AWS, Heroku, codeplex, codepen
NOT entities: "website", "service", "cloud", "server" (generic terms)

### DEVICE_HARDWARE
Specific devices, hardware components, or hardware-related technical terms.
Includes: iPhone, Arduino, Raspberry Pi, GPU, CPU, microphone, keyboard
NOT entities: "device", "hardware", "computer", "phone" in generic sense

### FILE_FORMAT
Specific file formats, extensions, or file types.
Includes: JSON, XML, CSV, PDF, JPEG, YAML, jpg, pdf
NOT entities: "file", "document", "format" (generic terms)"""


ANNOTATION_GUIDELINES = """\
## Annotation Guidelines

### General Rules
1. ONLY extract proper nouns or specific product/technology names that a developer could look up documentation for
2. Extract the MINIMAL span: "the React library" → extract "React"
3. Include version numbers when present: "Python 3.9", "C++11"
4. Context matters: same word can be entity or not depending on usage
5. When in doubt about whether a term is a specific technology or generic vocabulary, lean toward extraction if it appears in a technical programming context

### What IS an entity
- Specific named technologies with documentation (React, Python, jQuery)
- Specific API names, class names, function names (getElementById(), ArrayList)
- Language-specific type keywords used as technical terms (string, int, boolean, array, float)
- Specific UI component types in programming context (button, checkbox, slider, table, form)
- Named file formats (JSON, XML, PDF)
- Named tools, IDEs, platforms (Docker, VS Code, GitHub)

### What is NOT an entity
- Generic programming vocabulary: object, class, function, method, variable, property, field, instance, type, module, tag, handler, endpoint, event, collection, action, model
- Descriptive nouns in non-technical context: title, demo, nav, popup, thumb, slide
- Actions: compile, execute, deploy, install, configure
- Concepts: authentication, authorization, caching, threading, inheritance

### Context-Dependent Decisions
- "table" in "HTML table element" or "SQL table" → ENTITY (UI component / data structure)
- "table" in "put it on the table" → NOT entity
- "server" in "Express server" → extract "Express", NOT "server"
- "button" in "add a button to the form" → ENTITY (UI component in programming context)
- "button" in "press the submit button" → may still be ENTITY if discussing UI programming
- "private" as access modifier keyword → ENTITY (language keyword)
- "java" referring to the language → ENTITY
- "window" in UI programming context → ENTITY (Window object/class)

### Boundary Rules
- Method calls: include parentheses → "getElementById()", "console.log()"
- Namespaced: extract full qualified name → "React.Component", "java.util.List"
- Version-qualified: include version → "Python 3.9", "C++11", "ES2020"
- Compound names: keep together → "Visual Studio Code", "Spring Boot", "ASP.NET Web API"
- File paths and filenames: extract when they are specific (e.g. "composer.json", "config/modules.config.php")"""


SLIMER_PROMPT = """\
{entity_definitions}

{annotation_guidelines}

## Task

Extract ALL technical named entities from the following StackOverflow post.
Be thorough — extract every specific technology, library, framework, language, API, function, class, tool, platform, data type, UI component, file format, error type, and website mentioned.

TEXT:
{text}

Return ONLY a JSON array of entity strings. No explanations, no categories, just the entities.
ENTITIES:"""


def extract_with_slimer(doc: dict) -> list[str]:
    prompt = SLIMER_PROMPT.format(
        entity_definitions=ENTITY_DEFINITIONS,
        annotation_guidelines=ANNOTATION_GUIDELINES,
        text=doc["text"][:5000],
    )

    response = call_llm(prompt, model="sonnet", max_tokens=2000, temperature=0.0)
    entities = _parse_entity_response(response)

    verified: list[str] = []
    seen: set[str] = set()
    for e in entities:
        ok, _ = verify_span(e, doc["text"])
        if ok:
            key = normalize_term(e)
            if key not in seen:
                seen.add(key)
                verified.append(e)

    return verified


def _parse_entity_response(response: str) -> list[str]:
    response = response.strip()
    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if json_match:
        try:
            entities = json.loads(json_match.group())
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if isinstance(e, str) and e.strip()]
        except json.JSONDecodeError:
            pass

    entities: list[str] = []
    for line in response.splitlines():
        line = line.strip().strip("-*•").strip()
        line = line.strip('"').strip("'").strip(",").strip()
        if line and len(line) >= 2 and not line.startswith("{") and not line.startswith("["):
            entities.append(line)
    return entities
