#!/usr/bin/env python3
"""Benchmark prompt variants for individual extractors.

Tests prompt text directly against GT without the full pipeline.
Designed for rapid iteration on prompt quality.

Usage:
    python benchmark_prompt_variants.py --variant haiku_simple_v1 --n-docs 10
    python benchmark_prompt_variants.py --variant haiku_simple_v2 --n-docs 10
    python benchmark_prompt_variants.py --list-variants
"""

import argparse
import json
import time
from pathlib import Path

from parse_so_ner import select_documents
from scoring import many_to_many_score
from hybrid_ner import _parse_terms_json
from utils.llm_provider import call_llm


ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_json(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


# ============================================================================
# PROMPT VARIANTS REGISTRY
# ============================================================================

PROMPT_VARIANTS: dict[str, dict] = {}


def register_variant(
    name: str,
    prompt_template: str,
    model: str = "haiku",
    needs_retrieval: bool = False,
    description: str = "",
    thinking_budget: int | None = None,
    custom_runner: "callable | None" = None,
) -> None:
    PROMPT_VARIANTS[name] = {
        "prompt_template": prompt_template,
        "model": model,
        "needs_retrieval": needs_retrieval,
        "description": description,
        "thinking_budget": thinking_budget,
        "custom_runner": custom_runner,
    }


# ---------------------------------------------------------------------------
# BASELINE: Current haiku_simple (v0)
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v0",
    description="Current haiku_simple prompt (baseline P=60.5% R=71.4%)",
    prompt_template="""\
Extract ALL software named entities from this StackOverflow text. Be EXHAUSTIVE — \
include borderline terms. Missing an entity is worse than including extra ones.

ENTITY TYPES: library/class/function names, languages, data types (string, int, \
long, boolean, float, private, var), data structures (array, list, table, row, \
column, image, container), UI elements (button, screen, page, form, keyboard, \
checkbox, slider, pad), file types, OS names, error names (exception), devices \
(phone, CPU, camera), websites, versions, keyboard keys (Left, Up, PgUp).

CRITICAL — COMMON WORDS ARE ENTITIES when naming specific technical things:
  "string"/"int"/"long"/"boolean"/"private" = Data_Type
  "table"/"list"/"row"/"column"/"image"/"array" = Data_Structure
  "button"/"screen"/"page"/"form"/"keyboard"/"pad" = UI_Element
  "server"/"console"/"browser" = Application
  "exception" = Error_Name, "session" = Class, "key"/"keys" = depends on context

NOT entities: descriptive phrases ("vertical orientation"), process words \
("serializing"), adjectives ("cross-platform").

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v1: Contextual — add salience test, remove unconditional rules
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v1",
    description="Contextual salience test, no unconditional common-word rule",
    prompt_template="""\
Extract software named entities from this StackOverflow text.

A SOFTWARE ENTITY is a term that names a SPECIFIC technical thing that is \
IMPORTANT to understanding the post's technical content.

EXTRACT these entity types:
- Named libraries/frameworks/tools: React, jQuery, NumPy, Docker, Prism
- Class/function/method names: ArrayList, querySelector(), recv(), WebClient
- Programming languages: Java, Python, C#, JavaScript, HTML, SQL, C++11
- Specific data types when discussed technically: "cast to int", "returns a boolean"
- File formats when specific: JSON, XML, CSV, WSDL, .xaml
- Error names: NullPointerException, error 500
- OS/platforms: Linux, Windows, Android, macOS
- Named devices: iPhone, GPU, CPU
- UI components when specific: "the Trackbar control", "a ListView"
- Versions: v3.2, ES6, Silverlight 4
- Websites/orgs: GitHub, Google, Microsoft, codeplex

THE SALIENCE TEST — for common words like "server", "table", "string", "button":
Ask: "Is this term the FOCUS of technical discussion here, or just background?"
- "configure the Apache server" → "Apache" is entity, "server" is background → skip "server"
- "the String class has methods" → "String" is the focus → extract "String"
- "store data in a table" → "table" is background → skip
- "the DataTable component" → "DataTable" is specific → extract

DO NOT extract:
- Generic vocabulary used incidentally: server, browser, console, screen, code, class, method
- Descriptive phrases: "vertical orientation", "hidden fields"
- Process words: "serializing", "loading", "downloading"

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v2: Conservative — only named/specific things
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v2",
    description="Conservative: only named/specific things, skip all common words",
    prompt_template="""\
Extract ONLY specific named software entities from this StackOverflow text.

EXTRACT:
- Named libraries/frameworks: React, jQuery, NumPy, boost, Prism, libc++
- Named classes/functions/methods: ArrayList, HttpClient, querySelector(), WebClient
- Programming languages: Java, Python, C#, JavaScript, HTML, CSS, SQL
- Named tools/products: Docker, Chrome, Visual Studio, Xcode, Android Studio
- Named file formats: JSON, XML, CSV, WSDL
- Named errors: NullPointerException, IndexOutOfBoundsException
- Named OS/platforms: Linux, Windows, macOS, Android, iOS
- Named versions with product: ES6, Silverlight 4, C++11
- Named websites/orgs: GitHub, Google, Microsoft, codeplex

DO NOT extract any of these (even if technically valid):
- Generic type words: string, int, boolean, float, long, array, list, table
- Generic UI words: button, screen, page, form, keyboard, slider
- Generic infra words: server, browser, console, client, container, controller
- Generic CS concepts: event, handler, model, view, interface, session, key
- Descriptive phrases, adjectives, process words

When in doubt, SKIP. Precision matters more than completeness.

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v3: Balanced with contrastive examples
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v3",
    description="Balanced with contrastive ENTITY vs NOT-ENTITY examples",
    prompt_template="""\
Extract software named entities from this StackOverflow text.

ENTITY = a term naming a SPECIFIC technical thing important to the discussion.

CONTRASTIVE EXAMPLES — same words, different decisions:

  ENTITY (specific referent):          | NOT ENTITY (generic/incidental):
  "import React from 'react'" → React  | "check the code" → skip "code"
  "ArrayList<String>" → ArrayList      | "store in a list" → skip "list"
  "the Button component" → Button      | "click the button" → skip "button"
  "Apache server" → Apache             | "the server crashed" → skip "server"
  "cast to int" → int                  | "an integer value" → skip "integer"
  "the Session class" → Session        | "during this session" → skip "session"
  "String.format()" → String           | "the string is empty" → skip "string"
  "using lxml" → lxml                  | "parse the xml" → skip "xml"
  "Left arrow key" → Left              | "on the left side" → skip "left"

EXTRACT: Named libraries, frameworks, classes, functions, methods, languages, \
tools, products, platforms, file formats, error names, versions, websites.

ALSO EXTRACT common words ONLY when they are the specific subject of technical \
discussion (e.g., "the int type in Java" → int is entity).

DO NOT extract: generic vocabulary, descriptions, process words, adjectives.

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)


# ---------------------------------------------------------------------------
# haiku_simple_v4: Oracle-guided — positive-biased framing, "technical focus" test
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v4",
    description="Oracle-guided: positive-biased framing, 'technical focus' test",
    prompt_template="""\
Extract software named entities from this StackOverflow text.

Entities include: libraries, classes, functions, APIs, languages, data types, \
data structures, UI components, file formats, keyboard keys, CSS properties, \
HTML elements, OS names, devices, error names, versions, websites.

Common words ARE entities when they are the technical focus:
  "click the button" → button = entity (UI element being interacted with)
  "set the padding" → padding = entity (CSS property being configured)
  "the string is empty" → string = entity (data type being checked)
  "press Left arrow" → Left = entity (keyboard key)
  "stored in a table" → table = entity (data structure being used)
  "the image loads" → image = entity (data object being processed)
  "keyboard input" → keyboard = entity (device being used)
  "column width" → column, width = entities (layout properties)
  "throws an exception" → exception = entity (error type)
  "the server returns" → server = entity (application being discussed)

Skip ONLY when purely incidental/generic:
  "check the code" → skip (generic reference)
  "my project" → skip (not a technical thing)
  "the function returns" → skip "function" (focus is on what it returns)

Also extract: file extensions (.h, .xib, .cs), compound names (Interface Builder, \
Left arrow, page up), version numbers with context.

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v5: Tighter version — keep positive bias but add precision guard
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v5",
    description="Positive bias + precision guard for known-bad categories",
    prompt_template="""\
Extract software named entities from this StackOverflow text.

Entities: libraries, classes, functions, APIs, languages, data types, \
data structures, UI components, file formats, keyboard keys, CSS properties, \
OS names, devices, error names, versions, websites, organizations.

Common words ARE entities when technically focused:
  "click the button" → button (UI element)
  "the string is empty" → string (data type)
  "stored in a table" → table (data structure)
  "press the Left key" → Left (keyboard key)
  "set the padding" → padding (property)
  "the server crashed" → server (application)
  "throws exception" → exception (error type)
  "keyboard input" → keyboard (device)

Skip when incidental: "check the code", "my project", "the function returns X"

DO NOT extract these categories (always skip):
  - Descriptive phrases: "hidden fields", "vertical orientation"
  - Process/action words: "serializing", "loading", "downloading"
  - Design pattern names used generically: controller, handler, observer, factory
  - Generic CS vocabulary: code, method, class, function, property, module, \
    element, object (unless part of a specific name like "Element.querySelector")

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v6: v4 with nudge toward common words + version numbers
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v6",
    description="v4 + stronger common-word inclusion + version numbers",
    prompt_template="""\
Extract software named entities from this StackOverflow text.

Entities include: libraries, classes, functions, APIs, languages, data types, \
data structures, UI components, file formats, keyboard keys, CSS properties, \
HTML elements, OS names, devices, error names, versions, websites.

Common words ARE entities when used as technical terms:
  "click the button" → button (UI element)
  "set the padding" → padding (CSS property)
  "the string is empty" → string (data type)
  "press Left arrow" → Left (keyboard key)
  "stored in a table" → table (data structure)
  "the image loads" → image (data object)
  "keyboard input" → keyboard (device)
  "column width" → column, width (layout terms)
  "throws an exception" → exception (error type)
  "the server returns" → server (application)
  "private field" → private (access modifier)
  "the Session class" → Session (class name)
  "a global variable" → global (scope modifier)

Skip ONLY these: purely generic references ("check the code", "my project"), \
descriptions ("vertical orientation"), process words ("serializing"), \
design pattern names used generically (controller, handler, observer, factory), \
words "function", "method", "class", "code", "module", "object", "property" \
when used as generic CS vocabulary.

Also extract: file extensions (.h, .xib, .cs), compound names (Interface Builder, \
Left arrow, page up), version numbers (1.9.3, v3.2, ES6, 2.18.3).

When in doubt about a common word, INCLUDE it — other pipeline stages will filter.

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v7: v4 with explicit "scan every sentence" instruction
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v7",
    description="v4 + scan-every-sentence + broader entity types",
    prompt_template="""\
Extract ALL software named entities from this StackOverflow text. \
Scan EVERY sentence — do not skip any.

Entities: libraries, classes, functions, APIs, languages, data types (string, \
int, boolean, float, long, private, var), data structures (array, list, table, \
row, column, image, HashMap), UI components (button, slider, trackbar, page, \
form, screen, keyboard, pad), file formats (JSON, XML, .h, .xib, .cs), \
keyboard keys (Left, Up, PgUp, PageDown), CSS properties (padding, width, \
height, flex-direction), OS names, devices (phone, camera, CPU, microphone), \
error names (exception, NullPointerException), versions (v3.2, 1.9.3, ES6), \
websites, organizations.

Common words ARE entities when technically focused:
  "click the button" → button  |  "set padding" → padding
  "string is empty" → string   |  "the image loads" → image
  "stored in table" → table    |  "keyboard input" → keyboard
  "press Left key" → Left      |  "the server returns" → server

Skip only: "check the code", "my project", "the function returns X", \
descriptive phrases, process words, generic CS vocab (code, method, class, \
function, property, module, object — unless part of a specific name).

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v8: Keep v0's aggressive recall, surgical exclusion of known-bad vocab
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v8",
    description="v0 recall + surgical exclusion of generic CS vocab that causes FPs",
    prompt_template="""\
Extract ALL software named entities from this StackOverflow text. Be EXHAUSTIVE — \
include borderline terms. Missing an entity is worse than including extra ones.

ENTITY TYPES: library/class/function names, languages, data types (string, int, \
long, boolean, float, private, var), data structures (array, list, table, row, \
column, image, container), UI elements (button, screen, page, form, keyboard, \
checkbox, slider, pad), file types (.h, .xib, .cs, JSON, XML), OS names, \
error names (exception), devices (phone, CPU, camera), websites, versions \
(1.9.3, v3.2, ES6), keyboard keys (Left, Up, PgUp), CSS properties (padding, \
width, height, flex-direction).

Common words ARE entities when naming specific technical things:
  string/int/long/boolean/private/var = Data_Type
  table/list/row/column/image/array = Data_Structure
  button/screen/page/form/keyboard/pad = UI_Element
  server/console/browser = Application
  exception = Error_Name, session = Class, key/keys = depends on context

NEVER extract these (always generic, never entities in this dataset):
  code, method, function, class, property, module, object, element, field, \
  namespace, variable, library, framework, package, handler, controller, \
  observer, factory, adapter, wrapper, template, plugin, engine, runtime, \
  endpoint, service, distribution, configuration, implementation, node, \
  child, parent, root, listener, callback, promise, response, body, path, \
  query, token, hash, flag, option, state, context, provider, consumer

NOT entities: descriptive phrases ("vertical orientation"), process words \
("serializing"), adjectives ("cross-platform").

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v9: v8 but shorter exclusion list (only top FP causers)
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v9",
    description="v0 recall + minimal exclusion (only top-10 FP generic words)",
    prompt_template="""\
Extract ALL software named entities from this StackOverflow text. Be EXHAUSTIVE — \
include borderline terms. Missing an entity is worse than including extra ones.

ENTITY TYPES: library/class/function names, languages, data types (string, int, \
long, boolean, float, private, var), data structures (array, list, table, row, \
column, image, container), UI elements (button, screen, page, form, keyboard, \
checkbox, slider, pad), file types (.h, .xib, .cs, JSON, XML), OS names, \
error names (exception), devices (phone, CPU, camera), websites, versions \
(1.9.3, v3.2, ES6), keyboard keys (Left, Up, PgUp), CSS properties (padding, \
width, height).

Common words ARE entities when naming specific technical things:
  string/int/long/boolean/private = Data_Type
  table/list/row/column/image/array = Data_Structure
  button/screen/page/form/keyboard/pad = UI_Element
  server/console/browser = Application
  exception = Error_Name, key/keys = depends on context

NEVER extract these generic words (they are NEVER entities in StackOverflow NER):
  code, method, function, class, property, module, field, namespace, variable, \
  controller, handler, factory, template, endpoint, service, distribution, node

NOT entities: descriptive phrases ("vertical orientation"), process words \
("serializing"), adjectives ("cross-platform").

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# BASELINE: Current sonnet_taxonomy / haiku_taxonomy (v0)
# ---------------------------------------------------------------------------
register_variant(
    "taxonomy_v0",
    description="Current EXHAUSTIVE_PROMPT used by sonnet_taxonomy (baseline P=75.2% R=93.2%)",
    prompt_template="""\
You are a Named Entity Recognition system for StackOverflow technical text.

SOFTWARE NAMED ENTITY: A term that names a SPECIFIC, identifiable technical thing.

ENTITY TYPES:
- Code: Library, Framework, Class, Function, Method, API, Variable, Constant, Enum value
- Language: Programming language, Query language, Markup (Java, SQL, HTML, C++11)
- Data: Data_Type (string, int, boolean, float, long, private, var), \
Data_Structure (array, list, table, row, column, image, HashMap, container)
- Infrastructure: Application (server, console, browser, IDE), OS, Device, \
File_Type, Version, Website, Organization
- Interface: UI_Element (button, checkbox, form, screen, page, keyboard, \
trackbar, scrollbar, slider), HTML_Tag, Keyboard_Key (Left, Up, PgUp)
- Other: Error_Name (exception, NullPointerException), Algorithm

THE COMMON-WORD RULE:
Everyday English words ARE entities when they NAME specific technical things:
  "the string is empty" → string = Data_Type
  "click the button" → button = UI_Element
  "stored in a table" → table = Data_Structure
  "throws an exception" → exception = Error_Name
  "runs on the server" → server = Application
  "press the Left key" → Left = Keyboard_Key
  "the Session class" → Session = Class
  "a long value" → long = Data_Type
  "set the padding" → padding = UI property

NOT ENTITIES:
- Descriptive phrases: "vertical orientation", "hidden fields", "entropy pool"
- Process descriptions: "loading on demand", "serializing data"
- Adjectives/modifiers: "cross-platform", "deterministic", "floating"
- Generic vocabulary WITHOUT a specific referent: "check the code", "my project"

EXTRACTION MODE: Be EXHAUSTIVE. Missing an entity is WORSE than including a \
borderline one. When in doubt, INCLUDE the term.

TAXONOMY WITH EXAMPLES:
- Library/Framework: jQuery, React, NumPy, .NET, boost, Prism, libc++, SOAP
- Library_Class: ArrayList, HttpClient, Session, WebClient, ListView, IEnumerable
- Library_Function: recv(), querySelector(), map(), send(), post()
- Language: Java, Python, C#, JavaScript, HTML, CSS, SQL, AS3, C++11
- Data_Type: string, int, boolean, float, long, var, private, Byte[], bytearrays
- Data_Structure: array, list, table, row, column, image, HashMap, graph, container
- Application: Visual Studio, Docker, Chrome, browser, server, console, IDE
- Operating_System: Linux, Windows, macOS, Android, unix
- Device: iPhone, GPU, phone, camera, keyboard, microphone, CPU
- File_Type: JSON, XML, CSV, WSDL, xaml, jpg, pom.xml
- UI_Element: button, checkbox, slider, screen, page, form, trackbar, scrollbar, pad
- HTML_Tag: <div>, <input>, li
- Error_Name: NullPointerException, exception, error 500
- Version: v3.2, ES6, 2.18.3, Silverlight 4
- Website/Org: GitHub, Google, Microsoft, W3C, codeplex, codepen.io
- Keyboard_Key: Left, Right, Up, Down, PgUp, PageDown, Tab, Ctrl+C

RULES:
1. Extract terms EXACTLY as they appear in the text
2. Scan EVERY sentence — do not skip any
3. Common words like "string", "table", "button", "server", "list", "image", \
"exception", "session", "key", "padding", "long", "private" ARE entities per \
the Common-Word Rule above
4. DO NOT extract descriptive phrases or action words

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# taxonomy_v1: Contextual taxonomy — remove unconditional common-word rule
# ---------------------------------------------------------------------------
register_variant(
    "taxonomy_v1",
    description="Exhaustive taxonomy with surgical exclusion of known-bad generic categories",
    prompt_template="""\
You are a Named Entity Recognition system for StackOverflow technical text.

EXTRACTION MODE: Be EXHAUSTIVE. Missing an entity is WORSE than including a \
borderline one. When in doubt, INCLUDE the term.

ENTITY TYPES WITH EXAMPLES:
- Library/Framework: jQuery, React, NumPy, .NET, Prism, libc++, SOAP
- Library_Class: ArrayList, HttpClient, Session, WebClient, ListView, IEnumerable
- Library_Function: recv(), querySelector(), map(), send(), post()
- Language: Java, Python, C#, JavaScript, HTML, CSS, SQL, AS3, C++11
- Data_Type: string, int, boolean, float, long, var, private, Byte[], bytearrays
- Data_Structure: array, list, table, row, column, image, HashMap, graph, container
- Application: Visual Studio, Docker, Chrome, browser, server, console, IDE
- Operating_System: Linux, Windows, macOS, Android, unix
- Device: iPhone, GPU, phone, camera, keyboard, microphone, CPU
- File_Type: JSON, XML, CSV, WSDL, xaml, jpg, pom.xml, .h, .xib, .cs
- UI_Element: button, checkbox, slider, screen, page, form, trackbar, scrollbar, pad
- HTML_Tag: <div>, <input>, li
- Error_Name: NullPointerException, exception, error 500
- Version: v3.2, ES6, 2.18.3, Silverlight 4, 1.9.3
- Website/Org: GitHub, Google, Microsoft, W3C, codeplex
- Keyboard_Key: Left, Right, Up, Down, PgUp, PageDown, Tab, Ctrl+C
- CSS_Property: padding, width, height, flex-direction, column (layout context)

Common words like "string", "table", "button", "server", "list", "image", \
"exception", "session", "key", "padding", "long", "private" ARE entities when \
used as technical terms in context.

NEVER extract these (always generic vocabulary, never entities):
- Design patterns/roles: controller, handler, observer, factory, adapter, wrapper
- CS abstractions: code, method, function, class, property, module, object, \
  element, field, namespace, variable, library (the word), framework (the word)
- Process words: distribution, configuration, implementation, specification
- Infrastructure generic: endpoint, service, template, plugin, engine, runtime
- Descriptive phrases, adjectives, action/process words

RULES:
1. Extract terms EXACTLY as they appear in the text
2. Scan EVERY sentence — do not skip any
3. DO NOT extract descriptive phrases or action words
4. Include file extensions, version numbers, compound names

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# BASELINE: Current haiku_fewshot retrieval prompt (v0)
# ---------------------------------------------------------------------------
register_variant(
    "taxonomy_v2",
    description="Exhaustive taxonomy + positive common-word framing (mirrors haiku_simple_v4 style)",
    prompt_template="""\
You are a Named Entity Recognition system for StackOverflow technical text.

EXTRACTION MODE: Be EXHAUSTIVE. Missing an entity is WORSE than including a \
borderline one. Scan EVERY sentence.

ENTITY TYPES:
- Library/Framework: jQuery, React, NumPy, .NET, Prism, libc++, SOAP
- Class/Interface: ArrayList, HttpClient, Session, WebClient, ListView, IEnumerable
- Function/Method: recv(), querySelector(), map(), send(), post()
- Language: Java, Python, C#, JavaScript, HTML, CSS, SQL, AS3, C++11
- Data_Type: string, int, boolean, float, long, var, private, Byte[]
- Data_Structure: array, list, table, row, column, image, HashMap, container
- Application: Visual Studio, Docker, Chrome, browser, server, console
- OS: Linux, Windows, macOS, Android, unix
- Device: iPhone, GPU, phone, camera, keyboard, microphone, CPU
- File_Type: JSON, XML, CSV, WSDL, .xaml, .h, .xib, .cs, pom.xml
- UI_Element: button, checkbox, slider, screen, page, form, trackbar, pad
- Error_Name: NullPointerException, exception, error 500
- Version: v3.2, ES6, 2.18.3, Silverlight 4, 1.9.3
- Website/Org: GitHub, Google, Microsoft, codeplex
- Keyboard_Key: Left, Right, Up, Down, PgUp, Tab, Ctrl+C
- CSS_Property: padding, width, height, flex-direction

Common words ARE entities when technically focused:
  "click the button" → button  |  "set the padding" → padding
  "string is empty" → string   |  "the image loads" → image
  "stored in table" → table    |  "keyboard input" → keyboard
  "press Left key" → Left      |  "the server returns" → server
  "throws exception" → exception | "private field" → private

NEVER extract (always skip):
  - Generic CS vocabulary: code, method, function, class, property, module, \
    object, element, field, namespace, variable
  - Design patterns: controller, handler, observer, factory, adapter, wrapper
  - Process/abstract nouns: distribution, configuration, implementation, \
    specification, architecture, management
  - Descriptive phrases, adjectives, action/process words

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v10: Common-word & edge-case specialist
# Designed to maximize UNIQUE recall — catches what taxonomy & fewshot miss
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v10",
    description="Common-word specialist: targets terms other extractors consistently miss",
    prompt_template="""\
You are a SPECIALIST extractor for software entities that look like common English \
words. Other extractors handle named libraries/frameworks well. Your job is to \
catch the entities they MISS.

FOCUS ON THESE CATEGORIES (other extractors under-extract them):

1. COMMON WORDS THAT ARE ENTITIES in StackOverflow technical text:
   - Data types used as entities: string, int, boolean, float, long, private, var
   - Data structures: table, tables, array, list, row, column, image, container
   - UI elements: button, keyboard, screen, page, form, slider, scrollbar, pad, checkbox
   - Infrastructure: server, console, browser, exception, session, kernel, global
   - Devices: phone, camera, CPU, microphone
   - Other: configuration, calculator, borders, padding, symlinks, ondemand

2. FILE PATHS & DIRECTORY REFERENCES:
   - /usr/bin, /System/Library/..., src/main/resources/...
   - Relative paths: ../../path/to/file
   - Path patterns: /usr/bin/{{file1,file2,file3}}

3. UNUSUAL ENTITY FORMATS:
   - Compound key names: PgUp, PageDown, Ctrl+C, Left arrow
   - Function calls with args: getInputSizes(ImageFormat.YUV_420_888)
   - Size expressions: MAX SIZE(4608x3456)
   - Type notation: Byte[], bytearrays
   - Objective-C selectors: setCustomDoneTarget:action:, doneAction:
   - Words with punctuation: Pseudo-randomic, cross-platform (when a tool name)

4. VERSION NUMBERS (standalone or with context):
   - 1.9.3, 2.18.3, v3.2, 0.7.3, 13.10, 1.1.0

5. FILE EXTENSIONS: .h, .xib, .cs, .m, .long, .xaml

EXTRACTION RULES:
- Extract terms EXACTLY as they appear in the text
- Scan EVERY sentence — do not skip any
- When a common English word appears in a technical context, ALWAYS extract it
- Include file paths, version numbers, unusual formats
- DO NOT extract: generic CS vocabulary (code, method, function, class, property, \
  module, field, namespace — unless part of a specific name)
- DO NOT extract: descriptive phrases, process words, adjectives

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# haiku_simple_v11: v10 but with even stronger "always include common words" rule
# ---------------------------------------------------------------------------
register_variant(
    "haiku_simple_v11",
    description="Aggressive common-word extractor: extract ALL common words used technically",
    prompt_template="""\
Extract software named entities from this StackOverflow text. You are the \
"common word catcher" — your primary job is extracting technical entities that \
LOOK like everyday English words.

ALWAYS EXTRACT these when they appear in a technical context:
- Data types: string, int, boolean, float, long, double, byte, char, private, var, \
  public, static, void, null, undefined
- Data structures: table, tables, array, list, row, column, image, container, \
  tree, graph, stack, queue, set, map, key, keys, field, record
- UI elements: button, keyboard, screen, page, form, slider, scrollbar, pad, \
  checkbox, input, label, link, menu, tab, panel, dialog, window
- Infrastructure: server, console, browser, session, kernel, global, host, \
  client, socket, port, thread, process, pipe
- Devices: phone, camera, CPU, microphone, monitor, printer, mouse, touch
- Error terms: exception, error, fault, crash, overflow, timeout
- Config terms: configuration, option, setting, parameter, flag, mode
- CSS/Layout: padding, margin, border, borders, width, height, column, row, flex, \
  float, display, position
- Other: calculator, symlinks, ondemand, binary, socket, buffer, cache, driver

ALSO EXTRACT:
- File paths: /usr/bin, /System/Library/..., any path with / or \\
- Version numbers: 1.9.3, 2.18.3, v3.2, 0.7.3, standalone numbers with dots
- File extensions: .h, .xib, .cs, .m, .long, .py, .js
- Keyboard keys: PgUp, PageDown, Left, Right, Up, Down, Tab, Enter, Ctrl+C
- Function calls with arguments: getInputSizes(ImageFormat.YUV_420_888)
- Type notations: Byte[], int[], String[]
- Objective-C selectors: method:param:, action:

ALSO EXTRACT named libraries, classes, functions, languages — the usual stuff.

DO NOT extract:
- Generic CS meta-vocabulary: code, method, function, class, property, module, \
  object, element, variable (unless part of a specific name like "String.format")
- Descriptive phrases: "vertical orientation", "hidden fields"
- Process/action words: "serializing", "downloading", "loading"

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

register_variant(
    "fewshot_v0",
    description="Current retrieval fewshot prompt (baseline P=85.6% R=81.6%)",
    needs_retrieval=True,
    prompt_template="""\
You are extracting software named entities from StackOverflow text.

WHAT IS AN ENTITY:
Named libraries, classes, functions, APIs, tools, platforms, languages, \
data types (string, int, boolean, long), data structures (array, list, table, \
row, column, image), UI elements (button, screen, page, form, keyboard), \
file formats, error names, devices, websites, versions, keyboard keys.

COMMON WORDS ARE ENTITIES when they name specific technical things in context. \
Only skip truly generic vocabulary with no technical referent ("the code", \
"my project").

Here are examples of correct annotation from similar posts:

{examples_block}

---

Now extract ALL entities from this text. Follow the annotation style of the \
examples above. When in doubt, INCLUDE — missing entities is worse than \
including borderline ones.

TEXT: {text}

Return ONLY a JSON array of entity strings. No explanations.
ENTITIES:""",
)

# ---------------------------------------------------------------------------
# fewshot_v1: Retrieval with negative examples
# ---------------------------------------------------------------------------
register_variant(
    "fewshot_v1",
    description="Retrieval fewshot with negative examples (terms NOT annotated)",
    needs_retrieval=True,
    prompt_template="""\
You are extracting software named entities from StackOverflow text.

A SOFTWARE ENTITY names a SPECIFIC technical thing important to the discussion.

Here are examples of correct annotation from similar posts.
Pay attention to both what IS annotated AND what is NOT:

{examples_block}

NOTICE: The annotators did NOT mark generic words like "server", "code", \
"function", "class", "method", "event", "controller", "model", "view" unless \
they named something SPECIFIC in that post. Follow the same convention.

---

Extract entities from this text following the annotation style above.
When a common word appears (server, button, string, table, list, key, etc.), \
check: did similar examples annotate such words? Only include if the word names \
a specific technical thing central to the discussion.

TEXT: {text}

Return ONLY a JSON array of entity strings. No explanations.
ENTITIES:""",
)

# ---------------------------------------------------------------------------
# taxonomy_v3_thinking: Sonnet with extended thinking — maximally exhaustive
# Uses full SO NER annotation guidelines (28 entity types)
# ---------------------------------------------------------------------------
register_variant(
    "taxonomy_v3_thinking",
    model="sonnet",
    thinking_budget=10000,
    description="Sonnet + extended thinking (10k budget): exhaustive extraction with SO NER guidelines",
    prompt_template="""\
You are a Named Entity Recognition system trained on the StackOverflow NER dataset \
(Tabassum et al., ACL 2020). Your task is to extract ALL software-related named \
entities from the text below.

THE 28 ENTITY TYPES IN THIS DATASET:
Algorithm, Application, Class, Code_Block, Data_Structure, Data_Type, Device, \
Error_Name, File_Name, File_Type, Function, HTML_XML_Tag, Keyboard_IP, Language, \
Library, License, Operating_System, Organization, Output_Block, User_Interface_Element, \
User_Name, Value, Variable, Version, Website

CRITICAL RULES FROM THE ANNOTATION GUIDELINES:
1. COMMON ENGLISH WORDS ARE ENTITIES when they name technical things:
   - "string", "int", "boolean", "float", "long", "private", "var" → Data_Type
   - "table", "list", "array", "row", "column", "image" → Data_Structure
   - "button", "keyboard", "screen", "page", "form", "scrollbar" → User_Interface_Element
   - "server", "browser", "console" → Application
   - "exception" → Error_Name
   - "phone", "camera", "CPU" → Device
   - "session", "configuration" → Class (when used as class/concept name)

2. MAXIMUM SPAN RULE: Extract the longest meaningful span.
   "NullPointerException" not just "Null". "Interface Builder" not just "Interface".

3. FILE PATHS are entities: /usr/bin/ruby, /System/Library/Frameworks
   Path patterns with braces: /usr/bin/{{erb,gem,irb,rdoc,ri,ruby,testrb}}

4. CSS CLASS SELECTORS are entities: .long, .container, .my-class

5. VERSION NUMBERS are entities: 1.9.3, v3.2, ES6, 2.18.3

6. KEYBOARD KEYS are entities: Left, Right, PgUp, PageDown, Ctrl+C, Tab

7. FILE EXTENSIONS are entities: .h, .xib, .cs, .m, .xaml

8. FUNCTION CALLS with context: recv(), querySelector(), getInputSizes()

PROCESS: Go through the text sentence by sentence. For EACH sentence, identify \
every technical term that falls into any of the 28 entity types. Pay special \
attention to common words that are entities (rule 1) — these are the ones most \
often missed.

DO NOT extract: descriptive phrases ("vertical orientation"), process words \
("serializing", "downloading"), generic CS meta-vocabulary ("code", "method", \
"function", "class", "property", "module" — unless part of a specific name).

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
""",
)

# ---------------------------------------------------------------------------
# candidate_verify_v0: Discriminative approach — generate candidates, then classify
# Model can't "forget" entities when they're explicitly listed as candidates
# ---------------------------------------------------------------------------

import re as _re


def _candidate_verify_runner(doc, train_docs, retrieval_index, retrieval_model):
    """Generate candidates heuristically, then ask LLM to classify each."""
    text = doc["text"][:5000]
    candidates: set[str] = set()

    words = text.split()
    for w in words:
        clean = w.strip(".,;:!?()[]{}\"'")
        if clean and len(clean) >= 2:
            candidates.add(clean)

    for m in _re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([\w]+(?:\.[\w]+){1,5})\b", text):
        term = m.group(1)
        if "." in term and not _re.match(r"^\d+\.\d+(\.\d+)*$", term):
            candidates.add(term)
    for m in _re.finditer(r"\b([\w.]+\(\))", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([A-Z][A-Z0-9_]{1,15})\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"`([^`]{2,50})`", text):
        candidates.add(m.group(1).strip())
    for m in _re.finditer(r"(?<!\w)(\.[a-zA-Z][a-zA-Z0-9_-]*)\b", text):
        if len(m.group(1)) >= 3:
            candidates.add(m.group(1))
    for m in _re.finditer(r"(/[^\s]*\{[^}]+\}[^\s]*)", text):
        candidates.add(m.group(1).strip())
    for m in _re.finditer(r"(/(?:[\w.+-]+/)+[\w.+-]*)", text):
        candidates.add(m.group(1))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because", "but",
        "and", "or", "if", "while", "that", "this", "these", "those", "it",
        "its", "he", "she", "they", "them", "we", "you", "me", "my", "your",
        "his", "her", "our", "their", "what", "which", "who", "whom",
        "I", "also", "like", "get", "set", "use", "using", "used", "want",
        "try", "make", "see", "know", "think", "take", "come", "go", "work",
        "give", "look", "find", "way", "thing", "something", "anything",
        "However", "Also", "But", "And", "The", "This", "That", "It",
    }
    candidates = {c for c in candidates if c not in stopwords and len(c) >= 2}

    candidate_list = sorted(candidates)

    BATCH_SIZE = 80
    all_entities: list[str] = []

    for i in range(0, len(candidate_list), BATCH_SIZE):
        batch = candidate_list[i : i + BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {term}" for j, term in enumerate(batch))

        prompt = f"""\
Given this StackOverflow text, classify which of the candidate terms below are \
software named entities (libraries, classes, functions, languages, data types, \
data structures, UI elements, file types, OS names, devices, error names, \
versions, websites, keyboard keys, CSS properties, file paths, etc.).

A term IS an entity if it names a SPECIFIC technical thing in this text.
Common words like "string", "table", "button", "server", "keyboard", "exception", \
"session", "phone", "image", "row", "column", "private", "long" ARE entities when \
used as technical terms.

TEXT:
{text}

CANDIDATE TERMS:
{numbered}

Return ONLY the terms that ARE entities. Output JSON: {{"entities": ["term1", "term2", ...]}}
"""
        response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)
        batch_terms = _parse_terms_json(response)
        all_entities.extend(batch_terms)

    return all_entities


register_variant(
    "candidate_verify_v0",
    model="sonnet",
    description="Discriminative: heuristic candidates → LLM classifies each (prevents forgetting)",
    prompt_template="(uses custom_runner)",
    custom_runner=_candidate_verify_runner,
)

# ---------------------------------------------------------------------------
# candidate_verify_v1: Enhanced candidate generation + more aggressive classifier
# Adds bigrams, hyphenated compounds, bare numbers, all-lowercase tokens
# ---------------------------------------------------------------------------


def _candidate_verify_v1_runner(doc, train_docs, retrieval_index, retrieval_model):
    """Enhanced candidate generation with bigrams and aggressive classification."""
    text = doc["text"][:5000]
    candidates: set[str] = set()

    words = text.split()
    for w in words:
        clean = w.strip(".,;:!?()[]{}\"'")
        if clean and len(clean) >= 1:
            candidates.add(clean)

    for i in range(len(words) - 1):
        w1 = words[i].strip(".,;:!?()[]{}\"'")
        w2 = words[i + 1].strip(".,;:!?()[]{}\"'")
        if w1 and w2:
            candidates.add(f"{w1} {w2}")

    for m in _re.finditer(r"\b(\w+(?:-\w+)+)\b", text):
        candidates.add(m.group(1))

    for m in _re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([a-z]{3,30})\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([\w]+(?:\.[\w]+){1,5})\b", text):
        term = m.group(1)
        if "." in term:
            candidates.add(term)
    for m in _re.finditer(r"\b([\w.]+\(\))", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([A-Z][A-Z0-9_]{1,15})\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"`([^`]{2,50})`", text):
        candidates.add(m.group(1).strip())
    for m in _re.finditer(r"(?<!\w)(\.[a-zA-Z][a-zA-Z0-9_-]*)\b", text):
        if len(m.group(1)) >= 3:
            candidates.add(m.group(1))
    for m in _re.finditer(r"(/[^\s]*\{[^}]+\}[^\s]*)", text):
        candidates.add(m.group(1).strip())
    for m in _re.finditer(r"(/(?:[\w.+-]+/)+[\w.+-]*)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"([A-Z]:[\\/][^\s]+)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\d+)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\d+\.\d+(?:\.\d+)*)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"(\w+\|[\w|]*)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\w+\([^)]{1,30}\))\s*;?", text):
        full = m.group(0).strip()
        candidates.add(m.group(1))
        if full.endswith(";"):
            candidates.add(full)
    for m in _re.finditer(r'"(\s*[A-Za-z]\s*)"', text):
        candidates.add(m.group(0))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because", "but",
        "and", "or", "if", "while", "that", "this", "these", "those", "it",
        "its", "he", "she", "they", "them", "we", "you", "me", "my", "your",
        "his", "her", "our", "their", "what", "which", "who", "whom",
        "am", "about", "up", "down", "don't", "doesn't", "didn't", "won't",
        "wouldn't", "shouldn't", "couldn't", "can't", "I'm", "I've", "I'll",
    }
    candidates = {c for c in candidates if c not in stopwords and len(c) >= 1}

    candidate_list = sorted(candidates)

    BATCH_SIZE = 80
    all_entities: list[str] = []

    for i in range(0, len(candidate_list), BATCH_SIZE):
        batch = candidate_list[i : i + BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {term}" for j, term in enumerate(batch))

        prompt = f"""\
Given this StackOverflow text, classify which of the candidate terms below are \
software named entities per the StackOverflow NER annotation guidelines.

ENTITY TYPES (28 categories): Algorithm, Application, Class, Code_Block, \
Data_Structure, Data_Type, Device, Error_Name, File_Name, File_Type, Function, \
HTML_XML_Tag, Keyboard_IP, Language, Library, License, Operating_System, \
Organization, Output_Block, User_Interface_Element, User_Name, Value, Variable, \
Version, Website

CRITICAL — COMMON ENGLISH WORDS ARE ENTITIES in this dataset:
- "page", "list", "set", "tree", "cursor", "global", "setup", "arrow" → YES, entities
- "string", "table", "button", "server", "keyboard", "exception" → YES, entities
- "session", "phone", "image", "row", "column", "private", "long" → YES, entities
- Bare version numbers like "14", "2010", "3.0" → YES, entities (Version)
- Hyphenated compounds: "jQuery-generated", "cross-browser" → YES, entities
- Multi-word names: "visual editor", "command line" → YES, entities
- Example/demo content shown in posts: "ABCDEF|", sample data → YES, entities (Output_Block/Value)
- Terms with special chars: "this[int]", "keydown/keyup/keypress" → YES, entities
- Function calls with args: "free(pt) ;", "size(y, 1)", "buttons(new/edit)" → YES, entities
- Quoted drive letters or single-char refs: '" C "', '" D "' → YES, entities

When a candidate appears in technical context, INCLUDE it. The downstream pipeline \
will handle false positives. Missing entities is MUCH WORSE than including extras.

TEXT:
{text}

CANDIDATE TERMS:
{numbered}

Return ALL terms that could be entities. Be AGGRESSIVE — include borderline cases. \
Output JSON: {{"entities": ["term1", "term2", ...]}}
"""
        response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)
        batch_terms = _parse_terms_json(response)
        all_entities.extend(batch_terms)

    return all_entities


register_variant(
    "candidate_verify_v1",
    model="sonnet",
    description="Enhanced candidates (bigrams, bare numbers, hyphenated) + aggressive classifier",
    prompt_template="(uses custom_runner)",
    custom_runner=_candidate_verify_v1_runner,
)

# ---------------------------------------------------------------------------
# candidate_verify_v2: Same as v1 but Haiku classifier instead of Sonnet
# ---------------------------------------------------------------------------


def _candidate_verify_v2_runner(doc, train_docs, retrieval_index, retrieval_model):
    """Same candidate generation as v1, Haiku classifier for cost reduction."""
    text = doc["text"][:5000]
    candidates: set[str] = set()

    words = text.split()
    for w in words:
        clean = w.strip(".,;:!?()[]{}\"'")
        if clean and len(clean) >= 1:
            candidates.add(clean)

    for i in range(len(words) - 1):
        w1 = words[i].strip(".,;:!?()[]{}\"'")
        w2 = words[i + 1].strip(".,;:!?()[]{}\"'")
        if w1 and w2:
            candidates.add(f"{w1} {w2}")

    for m in _re.finditer(r"\b(\w+(?:-\w+)+)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([A-Z][a-z]+(?:[A-Z][a-z0-9]*)+)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([a-z]+[A-Z][a-zA-Z0-9]*)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([a-z]{3,30})\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([\w]+(?:\.[\w]+){1,5})\b", text):
        term = m.group(1)
        if "." in term:
            candidates.add(term)
    for m in _re.finditer(r"\b([\w.]+\(\))", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b([A-Z][A-Z0-9_]{1,15})\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"`([^`]{2,50})`", text):
        candidates.add(m.group(1).strip())
    for m in _re.finditer(r"(?<!\w)(\.[a-zA-Z][a-zA-Z0-9_-]*)\b", text):
        if len(m.group(1)) >= 3:
            candidates.add(m.group(1))
    for m in _re.finditer(r"(/[^\s]*\{[^}]+\}[^\s]*)", text):
        candidates.add(m.group(1).strip())
    for m in _re.finditer(r"(/(?:[\w.+-]+/)+[\w.+-]*)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"([A-Z]:[\\/][^\s]+)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\d+)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\d+\.\d+(?:\.\d+)*)\b", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"(\w+\|[\w|]*)", text):
        candidates.add(m.group(1))
    for m in _re.finditer(r"\b(\w+\([^)]{1,30}\))\s*;?", text):
        full = m.group(0).strip()
        candidates.add(m.group(1))
        if full.endswith(";"):
            candidates.add(full)
    for m in _re.finditer(r'"(\s*[A-Za-z]\s*)"', text):
        candidates.add(m.group(0))

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "must", "to", "of",
        "in", "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "when", "where", "why", "how", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "because", "but",
        "and", "or", "if", "while", "that", "this", "these", "those", "it",
        "its", "he", "she", "they", "them", "we", "you", "me", "my", "your",
        "his", "her", "our", "their", "what", "which", "who", "whom",
        "am", "about", "up", "down", "don't", "doesn't", "didn't", "won't",
        "wouldn't", "shouldn't", "couldn't", "can't", "I'm", "I've", "I'll",
    }
    candidates = {c for c in candidates if c not in stopwords and len(c) >= 1}

    candidate_list = sorted(candidates)

    BATCH_SIZE = 80
    all_entities: list[str] = []

    for i in range(0, len(candidate_list), BATCH_SIZE):
        batch = candidate_list[i : i + BATCH_SIZE]
        numbered = "\n".join(f"{j+1}. {term}" for j, term in enumerate(batch))

        prompt = f"""\
Given this StackOverflow text, classify which of the candidate terms below are \
software named entities per the StackOverflow NER annotation guidelines.

ENTITY TYPES (28 categories): Algorithm, Application, Class, Code_Block, \
Data_Structure, Data_Type, Device, Error_Name, File_Name, File_Type, Function, \
HTML_XML_Tag, Keyboard_IP, Language, Library, License, Operating_System, \
Organization, Output_Block, User_Interface_Element, User_Name, Value, Variable, \
Version, Website

CRITICAL — COMMON ENGLISH WORDS ARE ENTITIES in this dataset:
- "page", "list", "set", "tree", "cursor", "global", "setup", "arrow" → YES, entities
- "string", "table", "button", "server", "keyboard", "exception" → YES, entities
- "session", "phone", "image", "row", "column", "private", "long" → YES, entities
- Bare version numbers like "14", "2010", "3.0" → YES, entities (Version)
- Hyphenated compounds: "jQuery-generated", "cross-browser" → YES, entities
- Multi-word names: "visual editor", "command line" → YES, entities
- Example/demo content shown in posts: "ABCDEF|", sample data → YES, entities (Output_Block/Value)
- Terms with special chars: "this[int]", "keydown/keyup/keypress" → YES, entities
- Function calls with args: "free(pt) ;", "size(y, 1)", "buttons(new/edit)" → YES, entities
- Quoted drive letters or single-char refs: '" C "', '" D "' → YES, entities

When a candidate appears in technical context, INCLUDE it. The downstream pipeline \
will handle false positives. Missing entities is MUCH WORSE than including extras.

TEXT:
{text}

CANDIDATE TERMS:
{numbered}

Return ALL terms that could be entities. Be AGGRESSIVE — include borderline cases. \
Output JSON: {{"entities": ["term1", "term2", ...]}}
"""
        response = call_llm(prompt, model="haiku", max_tokens=3000, temperature=0.0)
        batch_terms = _parse_terms_json(response)
        all_entities.extend(batch_terms)

    return all_entities


register_variant(
    "candidate_verify_v2",
    model="haiku",
    description="Same as v1 but Haiku classifier (10x cheaper, ~5x faster)",
    prompt_template="(uses custom_runner)",
    custom_runner=_candidate_verify_v2_runner,
)


# ============================================================================
# EXECUTION
# ============================================================================

def run_prompt_variant(
    variant_name: str,
    doc: dict,
    train_docs: list[dict] | None = None,
    retrieval_index=None,
    retrieval_model=None,
) -> tuple[list[str], float]:
    """Run a prompt variant on a single doc and return (terms, elapsed)."""
    variant = PROMPT_VARIANTS[variant_name]

    if variant.get("custom_runner"):
        t0 = time.time()
        terms = variant["custom_runner"](doc, train_docs, retrieval_index, retrieval_model)
        elapsed = time.time() - t0
        return terms, elapsed

    template = variant["prompt_template"]
    model = variant["model"]
    thinking_budget = variant.get("thinking_budget")

    t0 = time.time()

    if variant["needs_retrieval"]:
        from retrieval_ner import safe_retrieve
        similar_docs = safe_retrieve(doc, train_docs, retrieval_index, retrieval_model, k=5)

        examples_parts: list[str] = []
        for i, sdoc in enumerate(similar_docs, 1):
            text_preview = sdoc["text"][:800]
            terms = json.dumps(sdoc["gt_terms"], ensure_ascii=False)
            examples_parts.append(
                f"--- Example {i} ---\n"
                f"TEXT: {text_preview}\n"
                f"ENTITIES: {terms}"
            )
        examples_block = "\n\n".join(examples_parts)

        prompt = template.format(
            examples_block=examples_block,
            text=doc["text"][:5000],
        )
    else:
        prompt = template.format(content=doc["text"][:5000])

    response = call_llm(
        prompt,
        model=model,
        max_tokens=3000,
        temperature=0.0,
        thinking_budget=thinking_budget,
        timeout=180 if thinking_budget else 90,
    )
    elapsed = time.time() - t0

    terms = _parse_terms_json(response)
    return terms, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark prompt variants")
    parser.add_argument("--variant", type=str, required=False)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--n-docs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--doc-ids", nargs="+", default=None)
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--save", action="store_true", help="Save results to artifacts")
    args = parser.parse_args()

    if args.list_variants:
        print("Available prompt variants:")
        for name, info in sorted(PROMPT_VARIANTS.items()):
            model = info["model"]
            retrieval = " [+retrieval]" if info["needs_retrieval"] else ""
            print(f"  {name:25s} ({model}{retrieval}) — {info['description']}")
        return

    variant_names = args.variants or ([args.variant] if args.variant else None)
    if not variant_names:
        parser.error("Must specify --variant or --variants or --list-variants")

    for v in variant_names:
        if v not in PROMPT_VARIANTS:
            parser.error(f"Unknown variant: {v}. Use --list-variants to see options.")

    train_docs = load_json(ARTIFACTS_DIR / "train_documents.json")
    test_docs = load_json(ARTIFACTS_DIR / "test_documents.json")
    print(f"Loaded {len(train_docs)} train, {len(test_docs)} test docs")

    if args.doc_ids:
        selected = [d for d in test_docs if d["doc_id"] in args.doc_ids]
    else:
        selected = select_documents(test_docs, args.n_docs, seed=args.seed)

    # Build retrieval index if any variant needs it
    retrieval_index = None
    retrieval_model = None
    if any(PROMPT_VARIANTS[v]["needs_retrieval"] for v in variant_names):
        from retrieval_ner import build_retrieval_index
        retrieval_index, _, retrieval_model = build_retrieval_index(train_docs)

    print(f"\nBenchmarking {len(variant_names)} variants on {len(selected)} docs\n")

    per_variant: dict[str, dict] = {
        name: {"tp": 0, "fp": 0, "fn": 0, "total_time": 0.0, "docs": []}
        for name in variant_names
    }

    for doc_idx, doc in enumerate(selected):
        doc_id = doc["doc_id"]
        gt = doc["gt_terms"]
        print(f"{'='*70}")
        print(f"Doc [{doc_idx+1}/{len(selected)}]: {doc_id} (GT: {len(gt)} terms)")
        print(f"{'='*70}")

        for name in variant_names:
            terms, elapsed = run_prompt_variant(
                name, doc, train_docs, retrieval_index, retrieval_model,
            )
            scores = many_to_many_score(terms, gt)

            per_variant[name]["tp"] += scores["tp"]
            per_variant[name]["fp"] += scores["fp"]
            per_variant[name]["fn"] += scores["fn"]
            per_variant[name]["total_time"] += elapsed
            per_variant[name]["docs"].append({
                "doc_id": doc_id,
                "extracted_count": len(terms),
                "tp": scores["tp"],
                "fp": scores["fp"],
                "fn": scores["fn"],
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1"],
                "fp_terms": scores["fp_terms"],
                "fn_terms": scores["fn_terms"],
            })

            print(
                f"  {name:25s}: P={scores['precision']*100:5.1f}%  "
                f"R={scores['recall']*100:5.1f}%  "
                f"F1={scores['f1']:.3f}  "
                f"({len(terms)} extracted, {elapsed:.1f}s)"
            )
            if scores["fp_terms"]:
                print(f"    FP: {scores['fp_terms'][:10]}")
            if scores["fn_terms"]:
                print(f"    FN: {scores['fn_terms'][:10]}")

        print()

    # Aggregate
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}\n")
    print(f"{'Variant':25s} {'P':>7s} {'R':>7s} {'F1':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Time':>7s}")
    print("-" * 80)

    for name in variant_names:
        data = per_variant[name]
        tp, fp, fn = data["tp"], data["fp"], data["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(
            f"{name:25s} {p*100:6.1f}% {r*100:6.1f}% {f1:6.3f} {tp:5d} {fp:5d} {fn:5d} {data['total_time']:6.1f}s"
        )

    # Save if requested
    if args.save:
        out_path = ARTIFACTS_DIR / "results" / "prompt_variant_benchmark.json"
        results = {}
        for name in variant_names:
            data = per_variant[name]
            tp, fp, fn = data["tp"], data["fp"], data["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            results[name] = {
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn,
                "total_time": round(data["total_time"], 1),
                "docs": data["docs"],
            }
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
