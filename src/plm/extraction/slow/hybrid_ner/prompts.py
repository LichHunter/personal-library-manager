"""Prompt templates for extraction and validation stages."""

# ---------------------------------------------------------------------------
# SHARED ENTITY DEFINITION (single source of truth for all prompts)
# ---------------------------------------------------------------------------

SHARED_ENTITY_DEFINITION = """\
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
- Generic vocabulary WITHOUT a specific referent: "check the code", "my project"\
"""

# ---------------------------------------------------------------------------
# EXTRACTION PROMPTS (optimize for RECALL — when in doubt, include)
# ---------------------------------------------------------------------------

EXHAUSTIVE_PROMPT = """\
You are a Named Entity Recognition system for StackOverflow technical text.

""" + SHARED_ENTITY_DEFINITION + """

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
"""

HAIKU_SIMPLE_PROMPT = """\
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
"""

RETRIEVAL_PROMPT_TEMPLATE = """\
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
ENTITIES:"""

# ---------------------------------------------------------------------------
# VALIDATION PROMPTS (same entity definition, entity_ratio as calibration)
# ---------------------------------------------------------------------------

REVIEW_PROMPT = """\
You are reviewing candidate software named entities extracted from a StackOverflow \
post. Each candidate was found by only ONE extractor (low confidence).

ENTITY DEFINITION:
A term is an ENTITY if it names a SPECIFIC technical thing: library, class, \
function, language, data type (string, int, boolean, long, private), data \
structure (array, list, table, row, column, image), UI element (button, page, \
form, keyboard, pad), file format, error name (exception), device, website, \
application (server, console, browser), keyboard key, version, constant, enum.

COMMON WORDS ARE ENTITIES when naming specific technical things in context.

NOT entities: descriptive phrases, process descriptions, adjectives, generic \
vocabulary without a specific referent.

TEXT:
{content}

CANDIDATES:
{terms_json}

APPROVE if the term names a specific technical thing per the definition above. \
REJECT if it is a descriptive phrase, process description, adjective, or generic \
vocabulary. When in doubt about a recognized tech term, APPROVE.

Output JSON: {{"terms": [{{"term": "...", "decision": "APPROVE|REJECT"}}]}}
"""

CONTEXT_VALIDATION_PROMPT = """\
You are classifying candidate software named entities from a StackOverflow post.

ENTITY DEFINITION:
A term is an ENTITY if it names a SPECIFIC technical thing: library, class, \
function, language, data type (string, int, boolean, long, private), data \
structure (array, list, table, row, column, image), UI element (button, page, \
form, keyboard), file format, error name (exception), device, website, \
application (server, console, browser), keyboard key, version.

COMMON WORDS ARE ENTITIES when naming specific technical things:
  "the string is empty" → string = ENTITY (Data_Type)
  "stored in a table" → table = ENTITY (Data_Structure)
  "check the model properties" → model = GENERIC, properties = GENERIC

NOT entities: descriptive phrases, process descriptions, adjectives, generic \
vocabulary without a specific referent.

TEXT:
{content}

TERMS TO CLASSIFY:
{terms_json}

Output JSON: {{"terms": [{{"term": "...", "decision": "ENTITY|GENERIC"}}]}}
"""

# ---------------------------------------------------------------------------
# TERM-RETRIEVAL VALIDATION PROMPTS
# ---------------------------------------------------------------------------

TERM_RETRIEVAL_PROMPT = """\
You are classifying candidate software named entities. For each term, I show you \
how it was labeled in SIMILAR StackOverflow documents from our training data, plus \
the current document where you must decide.

CURRENT DOCUMENT:
{content}

CANDIDATES TO CLASSIFY:
{candidates_block}

For each term, classify as ENTITY or GENERIC based on context:
- ENTITY: The term names a specific software thing (language, library, class, \
data type, UI element, file format, error, device, OS, website)
- GENERIC: The term is general programming vocabulary, a description, or a \
common English word not referring to a specific named software thing

Use the training examples as calibration — they show how human annotators \
labeled the same term in similar contexts.

Output JSON: {{"terms": [{{"term": "...", "decision": "ENTITY|GENERIC"}}]}}
"""

TERM_RETRIEVAL_PROMPT_WITH_REASONING = """\
You are classifying candidate software named entities. For each term, I show you \
how it was labeled in SIMILAR StackOverflow documents from our training data, plus \
the current document where you must decide.

CURRENT DOCUMENT:
{content}

CANDIDATES TO CLASSIFY:
{candidates_block}

For each term, classify as ENTITY or GENERIC based on context:
- ENTITY: The term names a specific software thing (language, library, class, \
data type, UI element, file format, error, device, OS, website)
- GENERIC: The term is general programming vocabulary, a description, or a \
common English word not referring to a specific named software thing

Use the training examples as STRONG calibration. The entity_ratio tells you how \
often human annotators labeled this term as an entity vs generic. A term with \
entity_ratio=80%+ is almost always an entity. A term with entity_ratio<20% is \
almost always generic. For ratios between 20-80%, carefully examine how the term \
is used in the CURRENT DOCUMENT compared to the training snippets.

For each term, briefly explain your reasoning before deciding.

Output JSON: {{"terms": [{{"term": "...", "reasoning": "...", "decision": "ENTITY|GENERIC"}}]}}
"""

TERM_RETRIEVAL_REVIEW_PROMPT = """\
You are reviewing LOW-CONFIDENCE candidates (found by only 1 extractor). For each \
term, I show training data evidence and the current document.

ENTITY DEFINITION:
A term is an ENTITY if it names a SPECIFIC technical thing: library, class, \
function, language, data type (string, int, boolean, long, private), data \
structure (array, list, table, row, column, image), UI element (button, page, \
form, keyboard, pad), file format, error name (exception), device, website, \
application (server, console, browser), keyboard key, version.

COMMON WORDS ARE ENTITIES when naming specific technical things in context.
NOT entities: descriptive phrases, process descriptions, adjectives, generic \
vocabulary without a specific referent.

CURRENT DOCUMENT:
{content}

LOW-CONFIDENCE CANDIDATES:
{candidates_block}

REVIEW GUIDELINES:
- entity_ratio is calibration: high (>70%) lean APPROVE, low (<30%) lean REJECT, \
middle = examine context
- A low ratio does NOT mean "always reject" — APPROVE if this document clearly \
uses the term as a named entity
- REJECT descriptive phrases, process descriptions, and adjectives regardless

Output JSON: {{"terms": [{{"term": "...", "decision": "APPROVE|REJECT"}}]}}
"""
