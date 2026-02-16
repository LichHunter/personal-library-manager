"""Stage 1 part: Taxonomy-based entity extraction.

Defines the 20 software entity types and provides taxonomy-driven
LLM extraction using structured prompts.
Ported from poc-1c-scalable-ner/hybrid_ner.py.
"""

import json
import re

from plm.shared.llm import call_llm


# 20 Software entity types for technical NER
ENTITY_TYPES: dict[str, str] = {
    "Library": "Named libraries, frameworks, SDKs (e.g., jQuery, React, NumPy, boost)",
    "Library_Class": "Classes from libraries (e.g., ArrayList, HttpClient, Session)",
    "Library_Function": "Functions/methods from libraries (e.g., recv(), querySelector())",
    "Language": "Programming/markup languages (e.g., Java, Python, HTML, SQL, C++11)",
    "Data_Type": "Primitive/built-in types (e.g., string, int, boolean, float, long)",
    "Data_Structure": "Data structures (e.g., array, list, table, HashMap, graph)",
    "Application": "Software applications (e.g., Visual Studio, Chrome, browser, server)",
    "Operating_System": "OS names (e.g., Linux, Windows, macOS, Android)",
    "Device": "Hardware devices (e.g., iPhone, GPU, keyboard, camera)",
    "File_Type": "File formats/extensions (e.g., JSON, XML, CSV, .xaml)",
    "UI_Element": "UI components (e.g., button, checkbox, slider, form, page)",
    "HTML_Tag": "HTML elements (e.g., <div>, <input>, li)",
    "Error_Name": "Error/exception types (e.g., NullPointerException, error 500)",
    "Version": "Version identifiers (e.g., v3.2, ES6, 2.18.3)",
    "Website_Org": "Websites/organizations (e.g., GitHub, Google, W3C)",
    "Keyboard_Key": "Keyboard keys (e.g., Left, PgUp, Ctrl+C, Tab)",
    "API": "API names and endpoints",
    "Variable_Constant": "Named variables and constants",
    "Enum_Value": "Enumeration values",
    "Algorithm": "Named algorithms (e.g., Dijkstra, quicksort)",
}


# Entity definition shared across prompts
_ENTITY_DEFINITION = """\
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
  "the string is empty" -> string = Data_Type
  "click the button" -> button = UI_Element
  "stored in a table" -> table = Data_Structure
  "throws an exception" -> exception = Error_Name
  "runs on the server" -> server = Application
  "press the Left key" -> Left = Keyboard_Key
  "the Session class" -> Session = Class
  "a long value" -> long = Data_Type
  "set the padding" -> padding = UI property

NOT ENTITIES:
- Descriptive phrases: "vertical orientation", "hidden fields", "entropy pool"
- Process descriptions: "loading on demand", "serializing data"
- Adjectives/modifiers: "cross-platform", "deterministic", "floating"
- Generic vocabulary WITHOUT a specific referent: "check the code", "my project"\
"""


# Exhaustive extraction prompt optimized for recall
_EXTRACTION_PROMPT = """\
You are a Named Entity Recognition system for StackOverflow technical text.

{entity_definition}

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
2. Scan EVERY sentence -- do not skip any
3. Common words like "string", "table", "button", "server", "list", "image", \
"exception", "session", "key", "padding", "long", "private" ARE entities per \
the Common-Word Rule above
4. DO NOT extract descriptive phrases or action words

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""


def extract_by_taxonomy(
    text: str,
    model: str = "sonnet",
    max_tokens: int = 3000,
) -> list[str]:
    """Extract entities using taxonomy-driven LLM prompt.
    
    Uses structured entity definitions and examples to guide
    exhaustive extraction optimized for recall.
    
    Args:
        text: Document text to extract from (truncated to 5000 chars)
        model: LLM model to use (default: "sonnet")
        max_tokens: Max tokens for response
        
    Returns:
        List of extracted entity terms
    """
    prompt = _EXTRACTION_PROMPT.format(
        entity_definition=_ENTITY_DEFINITION,
        content=text[:5000],
    )
    
    response = call_llm(prompt, model=model, max_tokens=max_tokens, temperature=0.0)
    
    return _parse_terms_json(response)


def _parse_terms_json(response: str) -> list[str]:
    """Parse LLM response that returns {"terms": [...]} or [...] format."""
    response = response.strip()
    
    # Try {"terms": [...]}
    obj_match = re.search(r"\{.*\}", response, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict) and "terms" in parsed:
                terms = parsed["terms"]
                if isinstance(terms, list):
                    return [str(t).strip() for t in terms if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass
    
    # Try bare [...]
    arr_match = re.search(r"\[.*\]", response, re.DOTALL)
    if arr_match:
        try:
            terms = json.loads(arr_match.group())
            if isinstance(terms, list):
                return [str(t).strip() for t in terms if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass
    
    # Fallback: line-by-line
    entities: list[str] = []
    for line in response.splitlines():
        line = line.strip().strip("-*").strip().strip('"').strip("'").strip(",").strip()
        if line and len(line) >= 2 and not line.startswith("{") and not line.startswith("["):
            entities.append(line)
    
    return entities
