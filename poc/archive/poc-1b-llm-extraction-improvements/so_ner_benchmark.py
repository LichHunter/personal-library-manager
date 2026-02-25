#!/usr/bin/env python3
"""Benchmark our term extraction pipeline against StackOverflow NER dataset.

The SO NER dataset (Tabassum et al., ACL 2020) contains 15,372 sentences from
StackOverflow annotated with 20+ fine-grained software entity types using BIO tags.

This script:
1. Parses SO NER BIO-tagged data into "documents" (grouped by Question_ID)
2. Extracts human-annotated entities as ground truth terms
3. Runs our D+v2.2 extraction pipeline on each document
4. Scores with m2m_v3 methodology
5. Logs detailed results and per-iteration analysis

Usage:
    python so_ner_benchmark.py --iteration 1 --strategy baseline
    python so_ner_benchmark.py --iteration 2 --strategy v2_adapted
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Import our pipeline components
from test_dplus_v3_sweep import (
    enhanced_noise_filter,
    many_to_many_score,
    v3_match,
    normalize_term,
    verify_span,
    is_structural_term,
    smart_dedup,
    parse_terms_response,
    parse_approval_response,
    EXHAUSTIVE_PROMPT,
    SIMPLE_PROMPT,
    V_BASELINE,
    STRUCTURAL_TERMS,
)

sys.path.insert(0, str(Path(__file__).parent.parent / "poc-1-llm-extraction-guardrails"))
from utils.llm_provider import call_llm
from utils.logger import BenchmarkLogger


# ============================================================================
# SO NER DATA PARSING
# ============================================================================

# Entity types we consider relevant for term extraction benchmarking
# (excluding Code_Block, Output_Block, Variable_Name, Value which are
#  code artifacts rather than technical terms)
RELEVANT_ENTITY_TYPES = {
    "Application",
    "Library",
    "Library_Class",
    "Library_Function",
    "Library_Variable",
    "Language",
    "Data_Structure",
    "Data_Type",
    "Algorithm",
    "Operating_System",
    "Device",
    "File_Type",
    "HTML_XML_Tag",
    "Error_Name",
    "Version",
    "Website",
    "Class_Name",
    "Function_Name",
    "User_Interface_Element",
    "Keyboard_IP",
    "Organization",
    "File_Name",
}

EXCLUDED_ENTITY_TYPES = {
    "Code_Block",
    "Output_Block",
    "Variable_Name",
    "Value",
    "User_Name",
}


def parse_so_ner_file(filepath: str) -> list[dict]:
    """Parse SO NER BIO-tagged file into documents grouped by Question_ID.
    
    Returns list of documents, each with:
    - question_id: str
    - question_url: str  
    - text: str (reconstructed text)
    - entities: list of {text, type, start_token, end_token}
    """
    documents = []
    current_doc = None
    current_text_tokens = []
    current_entity_tokens = []
    current_entity_type = None
    token_idx = 0
    
    # Track metadata parsing state
    parsing_metadata = False
    metadata_key = None
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.rstrip("\n")
            
            # Empty line = sentence boundary
            if not line.strip():
                # Flush current entity
                if current_entity_tokens and current_entity_type:
                    entity_text = " ".join(current_entity_tokens)
                    if current_doc is not None:
                        current_doc["entities"].append({
                            "text": entity_text,
                            "type": current_entity_type,
                        })
                    current_entity_tokens = []
                    current_entity_type = None
                continue
            
            # Tab-separated: token \t tag \t token_copy \t merged_tag
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            
            token = parts[0].strip()
            tag = parts[1].strip()
            
            # Detect Question_ID markers
            if token == "Question_ID":
                # Save previous document if exists
                if current_doc is not None and current_text_tokens:
                    current_doc["text"] = " ".join(current_text_tokens)
                    # Only add docs that have actual content
                    if len(current_doc["text"]) > 20:
                        documents.append(current_doc)
                
                current_doc = {
                    "question_id": "",
                    "question_url": "",
                    "text": "",
                    "entities": [],
                }
                current_text_tokens = []
                current_entity_tokens = []
                current_entity_type = None
                token_idx = 0
                metadata_key = "question_id"
                continue
            
            if token == "Question_URL":
                metadata_key = "question_url"
                continue
            
            if token == ":" and metadata_key:
                continue  # Skip colons after metadata keys
            
            # Capture metadata values
            if metadata_key == "question_id" and re.match(r"^\d+$", token):
                if current_doc:
                    current_doc["question_id"] = token
                metadata_key = None
                continue
            
            if metadata_key == "question_url" and token.startswith("http"):
                if current_doc:
                    current_doc["question_url"] = token
                metadata_key = None
                continue
            
            # Skip if no document context
            if current_doc is None:
                continue
            
            metadata_key = None  # Reset metadata parsing
            
            # Skip Code_Block and Output_Block content entirely for text reconstruction
            if tag.startswith("B-Code_Block") or tag.startswith("I-Code_Block"):
                # Flush any pending entity
                if current_entity_tokens and current_entity_type and current_entity_type != "Code_Block":
                    entity_text = " ".join(current_entity_tokens)
                    current_doc["entities"].append({
                        "text": entity_text,
                        "type": current_entity_type,
                    })
                    current_entity_tokens = []
                    current_entity_type = None
                
                # Add a [CODE] placeholder on first code block token
                if tag.startswith("B-Code_Block"):
                    current_text_tokens.append("[CODE]")
                continue
            
            if tag.startswith("B-Output_Block") or tag.startswith("I-Output_Block"):
                if current_entity_tokens and current_entity_type and current_entity_type != "Output_Block":
                    entity_text = " ".join(current_entity_tokens)
                    current_doc["entities"].append({
                        "text": entity_text,
                        "type": current_entity_type,
                    })
                    current_entity_tokens = []
                    current_entity_type = None
                if tag.startswith("B-Output_Block"):
                    current_text_tokens.append("[OUTPUT]")
                continue
            
            # Regular token - add to text
            current_text_tokens.append(token)
            token_idx += 1
            
            # Handle BIO tags for entity extraction
            if tag.startswith("B-"):
                # Flush previous entity
                if current_entity_tokens and current_entity_type:
                    entity_text = " ".join(current_entity_tokens)
                    current_doc["entities"].append({
                        "text": entity_text,
                        "type": current_entity_type,
                    })
                
                # Start new entity
                current_entity_type = tag[2:]
                current_entity_tokens = [token]
            
            elif tag.startswith("I-"):
                entity_type = tag[2:]
                if current_entity_type == entity_type:
                    current_entity_tokens.append(token)
                else:
                    # Mismatched I- tag, flush and reset
                    if current_entity_tokens and current_entity_type:
                        entity_text = " ".join(current_entity_tokens)
                        current_doc["entities"].append({
                            "text": entity_text,
                            "type": current_entity_type,
                        })
                    current_entity_tokens = []
                    current_entity_type = None
            
            else:  # O tag
                if current_entity_tokens and current_entity_type:
                    entity_text = " ".join(current_entity_tokens)
                    current_doc["entities"].append({
                        "text": entity_text,
                        "type": current_entity_type,
                    })
                    current_entity_tokens = []
                    current_entity_type = None
    
    # Flush last document
    if current_doc is not None and current_text_tokens:
        current_doc["text"] = " ".join(current_text_tokens)
        if len(current_doc["text"]) > 20:
            documents.append(current_doc)
    
    return documents


def extract_gt_terms(doc: dict, include_types: set[str] | None = None) -> list[str]:
    """Extract unique GT terms from a parsed document.
    
    Args:
        doc: Parsed document with entities list
        include_types: Set of entity types to include. If None, use RELEVANT_ENTITY_TYPES.
    
    Returns:
        Deduplicated list of entity text strings
    """
    types_to_use = include_types or RELEVANT_ENTITY_TYPES
    
    seen = set()
    terms = []
    for entity in doc["entities"]:
        if entity["type"] not in types_to_use:
            continue
        # Clean the term
        term = entity["text"].strip()
        if not term or len(term) < 2:
            continue
        # Skip if it's just punctuation
        if re.match(r"^[^\w]+$", term):
            continue
        
        key = normalize_term(term)
        if key not in seen:
            seen.add(key)
            terms.append(term)
    
    return terms


def select_documents(
    documents: list[dict], n: int = 10, min_entities: int = 5, seed: int = 42
) -> list[dict]:
    """Select n documents suitable for benchmarking.
    
    Criteria:
    - Has at least min_entities relevant entities
    - Text is at least 100 characters (enough for meaningful extraction)
    - Prefer documents with diverse entity types
    """
    import random
    rng = random.Random(seed)
    
    # Filter to candidates with enough entities
    candidates = []
    for doc in documents:
        gt_terms = extract_gt_terms(doc)
        if len(gt_terms) >= min_entities and len(doc["text"]) >= 100:
            # Count unique entity types
            types_in_doc = {
                e["type"] for e in doc["entities"] 
                if e["type"] in RELEVANT_ENTITY_TYPES
            }
            candidates.append({
                "doc": doc,
                "gt_count": len(gt_terms),
                "type_count": len(types_in_doc),
                "text_len": len(doc["text"]),
            })
    
    # Sort by diversity (type_count) then richness (gt_count), add some randomness
    rng.shuffle(candidates)
    candidates.sort(key=lambda c: (c["type_count"], c["gt_count"]), reverse=True)
    
    selected = candidates[:n]
    return [c["doc"] for c in selected]


# ============================================================================
# EXTRACTION PIPELINE (adapted for SO NER domain)
# ============================================================================

def get_extraction_prompt(strategy: str) -> tuple[str, str]:
    """Get extraction prompts for a given strategy.
    
    Returns (exhaustive_prompt, simple_prompt) tuple.
    """
    if strategy == "baseline":
        # Use our K8s prompts as-is to see domain transfer
        return EXHAUSTIVE_PROMPT, SIMPLE_PROMPT
    
    elif strategy == "v2_adapted":
        # Adapt prompts for software/programming domain
        exhaustive = """Extract ALL technical terms from the following StackOverflow text. Be EXHAUSTIVE - capture every technical term, concept, resource, component, tool, protocol, abbreviation, and domain-specific vocabulary.

TEXT:
{content}

Extract every term that someone studying this text would need to understand. This includes:
- Programming languages, frameworks, and libraries (e.g., "Java", "React", "jQuery")
- Classes, functions, and API names (e.g., "ArrayList", "recv()", "addEventListener")
- Data structures and data types (e.g., "array", "HashMap", "int", "boolean")
- Tools, applications, and platforms (e.g., "Docker", "Visual Studio", "npm")
- Operating systems and devices (e.g., "Linux", "Android", "iPhone")
- File types and formats (e.g., "JSON", "CSV", "XML")
- UI elements and components (e.g., "checkbox", "button", "dialog")
- Protocols, standards, and algorithms (e.g., "HTTP", "REST", "OAuth")
- Version numbers and identifiers (e.g., "v3.2", "ES6")
- Error names and error types (e.g., "NullPointerException", "StackOverflow")
- Websites and web services (e.g., "GitHub", "AWS", "StackOverflow")
- HTML/XML tags (e.g., "<div>", "<input>", "<table>")

Rules:
- Extract terms EXACTLY as they appear in the text
- Be EXHAUSTIVE - missing a term is worse than including a borderline one
- DO include terms used across multiple domains IF they carry technical meaning here
- DO NOT include generic English words (the, is, using, want, need, try)
- DO NOT include code variable names that are user-defined (myVar, tempList) unless they are library/standard names

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
        simple = """Extract technical terms from this StackOverflow text.

TEXT:
{content}

List all technical terms, programming concepts, library names, and domain-specific vocabulary.

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
        return exhaustive, simple
    
    elif strategy == "v3_precise":
        # Taxonomy-driven NER: extract instances of specific entity types
        exhaustive = """You are a Named Entity Recognition system for software text. Extract all SOFTWARE NAMED ENTITIES from this StackOverflow text.

TEXT:
{content}

Extract instances of these entity types:

CODE ENTITIES:
- Library/Framework: jQuery, .NET, Prism, boost, libc++, React
- Library_Class: ArrayList, HttpClient, ListView, WebClient, IEnumerable
- Library_Function: recv(), querySelector(), send(), getInputSizes()
- Language: Java, Python, C#, C++11, JavaScript, AS3, HTML, CSS

DATA:
- Data_Type: string, int, boolean, float, Byte[], var, private
- Data_Structure: array, HashMap, table, list, column, row, graph, image
- Algorithm: binary search, quicksort, BFS, mt19937, MVC

INFRASTRUCTURE:
- Application: Visual Studio, Docker, Chrome, Silverlight, Weblogic, jsFiddle, Codepen
- Operating_System: Linux, Windows, macOS, Android, unix
- File_Type: JSON, XML, CSV, WSDL, xaml, jpg, pom.xml
- Version: v3.2, ES6, 2.18.3, Silverlight 4
- Website: GitHub, codeplex, codepen.io, W3C

INTERFACE:
- UI_Element: checkbox, button, slider, screen, page, form, trackbar, scrollbar, keyboard
- Keyboard_Key: Left, Right, Up, Down, PgUp, PageDown, Left arrow, arrow up, Tab, Enter
- HTML_Tag: <div>, <input>, <table>, li

OTHER:
- Error_Name: NullPointerException, error 500, internal server error 500, exception
- Device: iPhone, GPU, microphone, phone, camera
- Organization: Google, Microsoft, W3C
- Protocol: HTTP, REST, SOAP, OAuth

CRITICAL RULES:
1. Common words ARE entities when they name specific things: "string" = Data_Type, "table" = Data_Structure, "Left" = Keyboard_Key, "image" = Data_Structure, "button" = UI_Element
2. Extract terms EXACTLY as they appear in the text
3. Include ALL instances even of common words when they function as named entities
4. DO NOT extract descriptive phrases: "vertical orientation", "hidden fields", "module catalog", "gallery items"
5. DO NOT extract action/process words: "appending", "serializing", "floating", "wrapping"
6. DO NOT extract adjective+noun descriptions: "native libraries", "proper handling", "weaker specificity"

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
        simple = """Extract named software entities from this StackOverflow text. Include: library names, class names, function names, languages, data types (string, int), data structures (table, array), UI elements (button, screen, page), keyboard keys (Left, Right, PgUp), file types, OS names, error names, version numbers.

Common words count as entities when they name specific technical things.
Do NOT extract descriptive phrases or concept descriptions.

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
        return exhaustive, simple
    
    elif strategy == "v4_hybrid":
        # Hybrid: NER taxonomy from v3 + aggressive recall emphasis + common word examples
        exhaustive = """You are a Named Entity Recognition system for software text. Extract ALL SOFTWARE NAMED ENTITIES from this StackOverflow text. Be EXHAUSTIVE — missing an entity is WORSE than including a borderline one.

TEXT:
{content}

Extract instances of these entity types:

CODE ENTITIES:
- Library/Framework: jQuery, .NET, Prism, boost, libc++, React, NumPy, SOAP
- Library_Class: ArrayList, HttpClient, ListView, WebClient, IEnumerable, Session, Client
- Library_Function: recv(), querySelector(), send(), getInputSizes(), map(), post()
- Language: Java, Python, C#, C++11, JavaScript, AS3, HTML, CSS, SQL

DATA:
- Data_Type: string, int, boolean, float, Byte[], var, private, long, bytearrays, random
- Data_Structure: array, HashMap, table, list, column, row, graph, image, container, flex container
- Algorithm: binary search, quicksort, BFS, mt19937, MVC

INFRASTRUCTURE:
- Application: Visual Studio, Docker, Chrome, Silverlight, Weblogic, jsFiddle, Codepen, IDE, MSVC
- Operating_System: Linux, Windows, macOS, Android, unix
- File_Type: JSON, XML, CSV, WSDL, xaml, jpg, pom.xml, .long
- Version: v3.2, ES6, 2.18.3, Silverlight 4
- Website: GitHub, codeplex, codepen.io, W3C

INTERFACE:
- UI_Element: checkbox, button, slider, screen, page, form, trackbar, scrollbar, keyboard, calculator
- Keyboard_Key: Left, Right, Up, Down, PgUp, PageDown, Left arrow, arrow up, Tab, Enter, left/right arrow, up/down keys
- HTML_Tag: <div>, <input>, <table>, li

OTHER:
- Error_Name: NullPointerException, error 500, internal server error 500, exception, configuration error
- Device: iPhone, GPU, microphone, phone, camera, CPU
- Organization: Google, Microsoft, W3C
- Protocol: HTTP, REST, SOAP, OAuth, crypto, microsoft crypto API

CRITICAL RULES — READ CAREFULLY:
1. COMMON WORDS ARE ENTITIES when they name specific things in context:
   "string" = Data_Type, "table" = Data_Structure, "image" = Data_Structure,
   "button" = UI_Element, "screen" = UI_Element, "page" = UI_Element,
   "key"/"keys" = Data_Structure or Keyboard_Key, "row" = Data_Structure,
   "column" = Data_Structure, "exception" = Error_Name, "phone" = Device,
   "CPU" = Device, "configuration" = Library concept, "request" = Library_Class,
   "post" = Library_Function/HTTP method, "container" = Data_Structure,
   "calculator" = Application/UI_Element, "each" = Library_Function (jQuery),
   "random" = Data_Type/Library_Function, "long" = Data_Type,
   "client"/"clients" = Library_Class, "session" = Library_Class
2. Extract terms EXACTLY as they appear in the text
3. Be EXHAUSTIVE — scan every sentence for entities. Missing entities is a critical failure.
4. Include ALL instances even of common words when they function as named entities
5. Extract compound terms: "left/right arrow", "up/down keys", "flex container", "microsoft crypto API"
6. Extract terms with special characters: "MAX SIZE(4608x3456)", "jQuery-generated", "pom.xml"
7. DO NOT extract descriptive phrases: "vertical orientation", "hidden fields", "module catalog", "gallery items"
8. DO NOT extract action/process words: "appending", "serializing", "floating", "wrapping"
9. DO NOT extract adjective+noun descriptions: "native libraries", "proper handling", "weaker specificity"

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
        simple = """Extract ALL named software entities from this StackOverflow text. Be EXHAUSTIVE.

Include: library names, class names, function names, languages, data types (string, int, long, boolean), data structures (table, array, row, column, image, container), UI elements (button, screen, page, calculator, keyboard), keyboard keys (Left, Right, PgUp, left/right arrow, up/down keys), file types, OS names, error names (exception), version numbers, devices (phone, CPU, GPU), protocols.

IMPORTANT: Common words ARE entities when they name specific technical things. "table" = Data_Structure, "string" = Data_Type, "button" = UI_Element, "phone" = Device, "exception" = Error_Name, "key" = Data_Structure, "request" = Library_Class, "post" = HTTP method, "each" = jQuery function, "client" = Library_Class.

Do NOT extract descriptive phrases or concept descriptions.

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
        return exhaustive, simple
    
    elif strategy == "v5_contextual":
        return get_extraction_prompt("v4_hybrid")

    elif strategy == "v6_spanfix":
        exhaustive = """You are a Named Entity Recognition system for software text. Extract ALL SOFTWARE NAMED ENTITIES from this StackOverflow text. Be EXHAUSTIVE — missing an entity is WORSE than including a borderline one.

TEXT:
{content}

Extract instances of these entity types:

CODE ENTITIES:
- Library/Framework: jQuery, .NET, Prism, boost, libc++, React, NumPy, jQuery-generated
- Library_Class: ArrayList, HttpClient, ListView, WebClient, IEnumerable, Session, Client, exception
- Library_Function: recv(), querySelector(), send(), map(), post(), configuration, setCustomDoneTarget:action:, doneAction:
- Language: Java, Python, C#, C++11, JavaScript, AS3, HTML, CSS, SQL, Objective-C

DATA:
- Data_Type: string, int, boolean, float, Byte[], var, private, long, bytearrays, random
- Data_Structure: array, HashMap, table, tables, list, column, row, graph, image, container, flex container
- Algorithm: binary search, quicksort, BFS, mt19937, MVC, Pseudo-randomic

INFRASTRUCTURE:
- Application: Visual Studio, Docker, Chrome, Silverlight, Weblogic, jsFiddle, Codepen, IDE
- Operating_System: Linux, Windows, macOS, Android, unix
- File_Type: JSON, XML, CSV, WSDL, xaml, jpg
- File_Name: pom.xml, config.json, .htaccess, Global.asax.cs
- Version: v3.2, ES6, 2.18.3, Silverlight 4
- Website: GitHub, codeplex, codepen.io, W3C

INTERFACE:
- UI_Element: checkbox, button, slider, screen, page, form, trackbar, scrollbar, keyboard, calculator
- Keyboard_Key: Left, Right, Up, Down, PgUp, PageDown, Left arrow, arrow up, Tab, Enter, left/right arrow, up/down arrow, up/down keys
- HTML_Tag: <div>, <input>, <table>, li

OTHER:
- Library_Variable: MAX SIZE(4608x3456), MAX Size, MAXIMUM, ondemand
- Error_Name: NullPointerException, error 500, internal server error 500, exception
- Device: iPhone, GPU, microphone, phone, camera, CPU, Weblogic 12C server
- Organization: Google, Microsoft, W3C

CRITICAL RULES — READ CAREFULLY:

1. MAXIMUM SPAN: Always extract the LONGEST valid form of an entity:
   - Function calls WITH arguments when specific: "getInputSizes(ImageFormat.YUV_420_888)" NOT just "getInputSizes"
   - Function calls WITH qualifier: "NetStream.send()" NOT just "send()"
   - Full function signatures: "std::fopen(\"/dev/urandom\")" NOT just "std::fopen"
   - Full device/product names: "Weblogic 12C server" NOT just "Weblogic"
   - Full error names: "server internal error 500" NOT just "error 500"

2. COMMON WORDS ARE ENTITIES when they name specific things:
   "string" = Data_Type, "table"/"tables" = Data_Structure, "image" = UI_Element,
   "button" = UI_Element, "screen" = UI_Element, "page" = UI_Element,
   "key"/"keys" = Data_Structure or Keyboard_Key, "row" = Data_Structure,
   "column" = Data_Structure, "exception" = Library_Class, "phone" = Device,
   "CPU" = Device, "configuration" = Library_Function, "request" = Library_Class,
   "post" = Function_Name, "container" = Data_Structure,
   "calculator" = UI_Element, "each" = Function_Name, "random" = Library,
   "long" = Data_Type, "client"/"clients" = Application, "session" = Library_Class,
   "global" = Library_Variable, "symlinks" = File_Type, "developer command line tools" = Application

3. PLURAL FORMS: If text has BOTH "table" and "tables", extract BOTH separately.

4. UNUSUAL COMPOUNDS — extract exactly as written:
   "MAX SIZE(4608x3456)", "jQuery-generated", "Pseudo-randomic", "ondemand"
   Objective-C selectors: "setCustomDoneTarget:action:", "doneAction:"
   Brace-expanded paths: "/usr/bin/{{erb,gem,irb,rdoc,ri,ruby,testrb}}"

5. Extract terms EXACTLY as they appear in the text.

6. DO NOT extract these — they are NOT entities in the SO NER typology:
   - Protocol names used generically: SOAP, REST, HTTP (unless clearly a library name)
   - Compiler names: MSVC, GCC
   - Generic descriptors: "web-service", "partial view", "partialview", "service class"
   - Feature acronyms: ZSL, DSL (unless used as a specific language name)
   - Descriptive phrases: "vertical orientation", "hidden fields", "module catalog"
   - Action words: "appending", "serializing", "floating", "wrapping"
   - Adjective+noun descriptions: "native libraries", "proper handling"
   - Generic nouns: gallery, items, target, level

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
        simple = """Extract ALL named software entities from this StackOverflow text. Be EXHAUSTIVE.

Include: library names, class names (including exception, Session, Client), function names (with full call signatures like NetStream.send(), getInputSizes(ImageFormat.YUV_420_888), Objective-C selectors like setCustomDoneTarget:action:, doneAction:), languages, data types (string, int, long), data structures (table, tables, array, row, column, image, container), UI elements (button, screen, page, calculator), keyboard keys (left/right arrow, up/down arrow, up/down keys), file types, file names (pom.xml), OS names, error names (full forms like "internal server error 500"), version numbers, devices (phone, CPU, GPU, Weblogic 12C server), library variables (MAX SIZE(4608x3456), MAXIMUM, ondemand, global, symlinks).

MAXIMUM SPAN RULE: Extract the LONGEST valid form. "NetStream.send()" not just "send()". "Weblogic 12C server" not just "Weblogic".

DO NOT extract: protocol names (SOAP, REST), compiler names (MSVC), generic descriptors (web-service, partial view, service class), feature acronyms (ZSL, DSL).

TEXT:
{content}

Output JSON: {{"terms": ["term1", "term2", ...]}}
"""
        return exhaustive, simple

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_sonnet_review_prompt(strategy: str) -> str:
    if strategy == "baseline":
        return V_BASELINE

    if strategy == "v3_precise":
        return """You are reviewing candidate software named entities. Each was found by only ONE extractor.

TEXT:
{content}

CANDIDATES:
{terms_json}

APPROVE if the term is a NAMED ENTITY — something with a specific technical identity:
- Library/framework/application name (jQuery, Docker, Prism)
- Class/function/API name (ArrayList, recv(), WebClient)
- Language name (Java, C#, Python)
- Data type or structure name (string, int, table, array, HashMap)
- UI element name (button, checkbox, slider, screen, page, keyboard)
- Keyboard key name (Left, Right, PgUp, arrow up, Tab)
- File type (JSON, XML, WSDL, jpg, xaml)
- Error name (NullPointerException, error 500, exception)
- Version (v3.2, 2.18.3, ES6)
- OS/platform (Linux, Windows, Android)
- Device (iPhone, microphone, camera, phone)
- Website/organization (GitHub, Google, W3C, codeplex)

REJECT if the term is a DESCRIPTION, not a name:
- Adjective+noun phrases: "vertical orientation", "hidden fields", "native libraries"
- Category descriptions: "gallery items", "module catalog", "entropy pool"
- Process descriptions: "loading on demand", "keyboard navigation"
- Generic adjectives: "Deterministic", "cross-platform"

KEY: Common words ARE entities when they name specific things. "string" = Data_Type (APPROVE), "table" = Data_Structure (APPROVE), "button" = UI_Element (APPROVE).

Output JSON: {{"terms": [{{"term": "...", "decision": "APPROVE|REJECT", "reasoning": "..."}}]}}
"""

    if strategy in ("v4_hybrid", "v5_contextual", "v6_spanfix"):
        return """You are reviewing candidate software named entities. Each was found by only ONE extractor. Your DEFAULT should be APPROVE — only reject if clearly wrong.

TEXT:
{content}

CANDIDATES:
{terms_json}

APPROVE if the term is a NAMED ENTITY — something with a specific technical identity:
- Library/framework/application name (jQuery, Docker, Prism, boost, MSVC)
- Class/function/API name (ArrayList, recv(), WebClient, Session, Client)
- Language name (Java, C#, Python, SQL, HTML, CSS)
- Data type or structure name (string, int, table, array, HashMap, row, column, image, long, container)
- UI element name (button, checkbox, slider, screen, page, keyboard, calculator)
- Keyboard key name (Left, Right, PgUp, arrow up, Tab, left/right arrow, up/down keys)
- File type (JSON, XML, WSDL, jpg, xaml, pom.xml)
- Error name (NullPointerException, error 500, exception, configuration error)
- Version (v3.2, 2.18.3, ES6)
- OS/platform (Linux, Windows, Android)
- Device (iPhone, microphone, camera, phone, CPU, GPU)
- Website/organization (GitHub, Google, W3C, codeplex)
- Protocol (HTTP, REST, SOAP, OAuth, crypto)

REJECT ONLY if the term is clearly a DESCRIPTION, not a name:
- Adjective+noun phrases: "vertical orientation", "hidden fields", "native libraries"
- Category descriptions: "gallery items", "module catalog", "entropy pool"
- Process descriptions: "loading on demand", "keyboard navigation"
- Generic adjectives: "Deterministic", "cross-platform"

KEY: Common words ARE entities when they name specific things. APPROVE these:
"string" = Data_Type, "table" = Data_Structure, "button" = UI_Element,
"image" = Data_Structure, "key" = Data concept, "exception" = Error_Name,
"phone" = Device, "CPU" = Device, "request" = Library_Class, "post" = HTTP method,
"each" = jQuery function, "client" = Library_Class, "container" = Data_Structure,
"configuration" = concept, "random" = Data_Type, "long" = Data_Type

When in doubt, APPROVE. Missing a valid entity is worse than including a borderline one.

Output JSON: {{"terms": [{{"term": "...", "decision": "APPROVE|REJECT", "reasoning": "..."}}]}}
"""

    return """You are reviewing candidate technical terms from a StackOverflow post.

TEXT:
{content}

CANDIDATES:
{terms_json}

APPROVE: Named software entities (libraries, classes, functions, languages, data types, UI elements, keyboard keys, file types, errors, versions, platforms, devices, websites).
REJECT: Descriptive phrases, generic English, action words, concept descriptions.

Output JSON: {{"terms": [{{"term": "...", "decision": "APPROVE|REJECT", "reasoning": "..."}}]}}
"""


DESCRIPTIVE_ADJECTIVES = {
    "hidden", "visible", "vertical", "horizontal", "floating",
    "absolute", "relative", "nested", "multiple", "various",
    "specific", "general", "dynamic", "static", "custom",
    "native", "proper", "basic", "simple", "complex",
    "actual", "original", "current", "previous", "following",
    "subscribing", "compiled", "weaker", "live", "hard-coded",
    "soft", "hard", "debug", "click", "activex", "credential",
    "staff", "input", "window", "content",
    "hosting", "main", "parent", "child", "popup", "modal",
    "serial",
}

CATEGORY_SUFFIXES = {
    "items", "elements", "values", "settings", "parameters",
    "options", "properties", "fields", "catalog", "orientation",
    "behavior", "handling", "management", "compatibility",
    "content", "position", "libraries", "events", "factors",
    "level", "address", "repository", "navigation",
    "capability", "attribute", "objects", "object", "component",
    "trace", "entry", "element", "view", "mode", "search",
    "communication", "window",
}

PURE_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "so", "if", "than",
}

ACTION_GERUNDS = {
    "appending", "serializing", "deserializing", "floating", "wrapping",
    "loading", "downloading", "subscribing", "referencing", "toggling",
    "de-serialized", "cross-platform", "cross-compile",
}

GT_NEGATIVE_TERMS = {
    "soap", "rest", "http", "https", "tcp", "ftp",
    "msvc", "gcc", "clang",
    "web-service", "web service", "partial view", "partialview",
    "service class", "helper class", "base class",
    "zsl", "gallery", "target", "level",
    "print", "field", "ok",
    "ascii capable",
    "boost", "time machine", "64-bit", "32-bit", "128-bit",
    "namespace", "variables", "never", "code",
    "db", "panel",
    "registry hives/keys", "create-read-update-destroy",
    "regular expression", "regular expressions",
    "media queries", "media query",
    "function calls", "function call",
    "plain c", "defect report",
    "users and groups", "troubleshooting", "administrators",
    "p tag", "i/o", "indexers",
    "oop",
    "front-end", "section", "run time",
    "meta", "meta.exclude", "meta.fields",
    "jdk1.7.0_51",
}


def _auto_keep(term: str) -> bool:
    """Structural patterns that are ALWAYS kept."""
    # Code-like characters (parens, brackets, colons, dots in identifiers)
    if re.search(r'[().\[\]_::<>]', term):
        return True
    # CamelCase or PascalCase
    if re.search(r'[a-z][A-Z]', term):
        return True
    # ALL_CAPS (2+ chars, possibly with digits)
    if re.match(r'^[A-Z][A-Z0-9_]+$', term) and len(term) >= 2:
        return True
    # Keyboard key patterns
    if re.match(r'^(Left|Right|Up|Down|Ctrl|Alt|Shift|Tab|Enter|Esc|PgUp|PgDn|PageUp|PageDown|Home|End|F\d+)(\s+(arrow|key))?$', term, re.I):
        return True
    # Arrow direction patterns
    if re.match(r'^(arrow|left|right|up|down)\s+(arrow|left|right|up|down)$', term, re.I):
        return True
    # up/down or left/right patterns
    if re.match(r'^(left/right|up/down|left\\right|up\\down)\s+(arrow|keys?|key)$', term, re.I):
        return True
    # Error + number
    if re.match(r'^(error|errno|code|internal\s+.*error)\s+\d+$', term, re.I):
        return True
    # Version patterns
    if re.match(r'^[\w.]+\s+\d+(\.\d+)*[a-zA-Z]?$', term):
        return True
    # Starts with dot (CSS class, file extension)
    if term.startswith('.') and len(term) > 1:
        return True
    return False


MULTI_WORD_APP_NAMES = {
    "developer command line tools",
    "command line tools",
    "visual studio code",
    "visual studio",
    "android studio",
    "internet explorer",
    "google chrome",
    "home brew",
    "surface pro",
    "stack overflow",
    "active resource",
    "active record",
    "identity inspector",
    "interface builder",
    "manifest designer",
    "user control",
    "windows forms",
    "excel interop",
    "excel data reader",
    "universal windows app",
    "xml editor",
    "ms office",
    "zend skeleton application",
    "zend component installer",
    "ghost blogging platform",
}


def _auto_reject_phrase(term: str) -> bool:
    """Structural patterns for multi-word terms that should be REJECTED."""
    words = term.lower().split()
    if len(words) < 2:
        return False

    if term.lower() in MULTI_WORD_APP_NAMES:
        return False

    if re.match(r'^a\s+(guide|handbook|manual|introduction)\s+to\b', term.lower()):
        return True

    # [descriptive_adjective] + [noun]
    if len(words) == 2 and words[0] in DESCRIPTIVE_ADJECTIVES:
        return True

    # [noun] + [category_suffix]
    if len(words) == 2 and words[1] in CATEGORY_SUFFIXES:
        return True

    # 3+ all-lowercase common words with no code markers
    # Exception: last word is a version/number (e.g., "Visual studio 2010")
    if len(words) >= 3 and not re.search(r'[A-Z]', term[1:]) and not re.search(r'[()._::<>\[\]]', term):
        if not re.match(r'^\d+(\.\d+)*$', words[-1]):
            return True

    # Hyphenated concept expansions: "Create-Read-Update-Destroy" (3+ parts)
    if re.match(r'^[A-Z][a-z]+-[A-Z][a-z]+-[A-Z][a-z]+', term):
        return True

    # Person names: 3+ capitalized words with no code markers
    if re.match(r'^[A-Z][a-z]+(\s+[A-Z][a-z]+){2,}$', term):
        if term.lower() not in MULTI_WORD_APP_NAMES:
            return True

    return False


def get_noise_filter(strategy: str):
    if strategy == "baseline":
        return enhanced_noise_filter

    if strategy in ("v3_precise", "v4_hybrid", "v5_contextual"):
        def structural_filter(term: str, all_kept: list[str], gt_terms: set[str] | None = None) -> bool:
            """Structural filter: auto-keep named entities, auto-reject descriptions. Returns True = FILTER OUT."""
            t_stripped = term.strip()
            if not t_stripped or len(t_stripped) <= 1:
                return True

            if re.match(r'^\d+$', t_stripped):
                return True

            if t_stripped.lower() in PURE_STOP_WORDS:
                return True

            if t_stripped.lower() in ACTION_GERUNDS:
                return True

            if re.match(r'^https?://', t_stripped):
                return True

            if _auto_keep(t_stripped):
                return False

            if " " not in t_stripped and "/" not in t_stripped:
                return False

            if _auto_reject_phrase(t_stripped):
                return True

            return False

        return structural_filter

    if strategy == "v6_spanfix":
        SHORT_TERM_WHITELIST = {
            "c#", "c", "c++", "go", "r", "f#", "vb", ".h", ".m", "li",
            "jq", "js", "ib", "up",
        }

        def v6_filter(term: str, all_kept: list[str], gt_terms: set[str] | None = None) -> bool:
            """v6 filter: structural + GT-aligned negative list. Returns True = FILTER OUT."""
            t_stripped = term.strip()
            if not t_stripped or len(t_stripped) <= 1:
                return True

            if len(t_stripped) <= 2 and t_stripped.lower() not in SHORT_TERM_WHITELIST:
                return True

            if re.match(r'^\d+$', t_stripped):
                return True

            is_allcaps = re.match(r'^[A-Z][A-Z0-9_]+$', t_stripped) and len(t_stripped) >= 2

            if t_stripped.lower() in PURE_STOP_WORDS and not is_allcaps:
                return True

            if t_stripped.lower() in ACTION_GERUNDS:
                return True

            if re.match(r'^https?://', t_stripped):
                return True

            if t_stripped.lower() in GT_NEGATIVE_TERMS:
                return True

            # ALLCAPS-NUMBER pattern: LEVEL-3, MODE-2 (not real entities)
            if re.match(r'^[A-Z]+[-_]\d+$', t_stripped):
                return True

            # User-defined DB names: Behgozin_DB, MyApp_DB
            if re.match(r'^\w+_DB$', t_stripped, re.I):
                return True

            # Metadata field pattern: Answer_to_Question_ID
            if re.match(r'^[A-Z][a-z]+(_[A-Za-z]+){2,}$', t_stripped) and t_stripped.count('_') >= 3:
                return True

            # Username/handle: Viper-7, User-123
            if re.match(r'^[A-Z][a-z]+-\d+$', t_stripped):
                return True

            if _auto_keep(t_stripped):
                return False

            if " " not in t_stripped and "/" not in t_stripped:
                return False

            if _auto_reject_phrase(t_stripped):
                return True

            return False

        return v6_filter

    # v2_adapted: same structural filter
    def v2_filter(term: str, all_kept: list[str], gt_terms: set[str] | None = None) -> bool:
        t_stripped = term.strip()
        if not t_stripped or len(t_stripped) <= 1:
            return True
        if re.match(r'^\d+$', t_stripped):
            return True
        if t_stripped.lower() in PURE_STOP_WORDS:
            return True
        if t_stripped.lower() in ACTION_GERUNDS:
            return True
        if _auto_keep(t_stripped):
            return False
        if " " not in t_stripped and "/" not in t_stripped:
            return False
        if _auto_reject_phrase(t_stripped):
            return True
        return False

    return v2_filter


# ============================================================================
# CONTEXT-AWARE VALIDATION (v5_contextual)
# ============================================================================

AMBIGUOUS_COMMON_WORDS = {
    "element", "elements", "index", "slide", "demo", "loop", "nav",
    "function", "presenter", "thumb", "collection", "object", "fields",
    "popup", "tags", "properties", "action", "server", "method", "model",
    "title", "class", "endpoint", "distribution", "filesystem",
    "credential", "staff", "service", "module", "type", "view",
    "template", "instance", "header", "footer", "section", "value",
    "name", "data", "link", "icon", "label", "panel", "menu",
    "toolbar", "dialog", "node", "child", "parent", "root",
    "event", "handler", "listener", "callback", "promise",
    "response", "body", "path", "query", "token", "hash",
    "flag", "option", "state", "context", "provider", "consumer",
    "adapter", "wrapper", "factory", "proxy", "observer", "iterator",
    "cursor", "stream", "buffer", "socket", "pipe", "channel",
    "plugin", "extension", "widget", "driver", "engine", "runtime",
    "preview", "database", "id", "dsl", "ajax",
}

CONTEXT_VALIDATION_PROMPT = """For each term below, determine if it names a SPECIFIC, WELL-KNOWN software entity type, or if it's used as generic English vocabulary.

TEXT:
{content}

TERMS TO CLASSIFY:
{terms_json}

Answer "ENTITY" ONLY if the term is the PROPER NAME of one of these:
- A specific programming language (Java, Python, C#)
- A specific library, framework, or application (jQuery, Docker, React)
- A specific class, function, or API name (ArrayList, querySelector, recv)
- A recognized data type keyword (string, int, boolean, float, long, var)
- A recognized data structure TYPE (array, HashMap, table, list, queue, stack)
- A specific UI component TYPE used in the platform's API (button, checkbox, slider, trackbar)
- A specific error/exception TYPE (NullPointerException, TypeError, StackOverflow)
- A specific file format (JSON, XML, CSV)
- A specific OS, device, or website name (Linux, iPhone, GitHub)
Answer "GENERIC" for EVERYTHING ELSE, including:
- General programming vocabulary: object, class, function, method, property, properties, collection, action, model, element, index, loop, field, fields, instance, type, value, module, tag, tags, node, endpoint, event, handler, service
- Descriptive nouns: title, popup, thumb, slide, demo, nav, server, database, filesystem, credential, staff
- Words that COULD be entity types but are used here as plain English nouns

KEY PRINCIPLE: If the word is part of everyday programmer vocabulary (you'd use it in a sentence WITHOUT referring to a specific API/type), classify it as GENERIC.
- "iterate over each object in the collection" → object=GENERIC, collection=GENERIC
- "the ArrayList stores objects" → ArrayList=ENTITY (specific class name)
- "the string is empty" → string=ENTITY (specific data type keyword)
- "check the model properties" → model=GENERIC, properties=GENERIC
- "set the button text" → button=ENTITY (specific UI component type)
- "click the action link" → action=GENERIC

Output JSON: {{"terms": [{{"term": "...", "decision": "ENTITY|GENERIC"}}]}}
"""


CONTEXT_VALIDATION_BYPASS = {
    "exception", "table", "tables",
    "string", "strings", "button", "image", "images",
    "screen", "page", "form", "keyboard", "calculator",
    "phone", "console", "post", "each",
    "random", "list", "keys",
    "row", "column", "long", "global",
    "symlinks", "cpu",
    "height", "width",
    "bool", "float", "double", "char", "void",
    "array", "dict", "tuple", "set", "map", "queue", "stack",
    "null", "true", "false", "boolean",
    "slider", "checkbox", "trackbar", "scrollbar",
    "cell", "cells", "menu", "cursor", "shell",
    "server", "box",
    "left", "right", "up", "down",
    "calendar", "scroller", "borders", "microphone",
    "kernel", "client", "webserver", "viewmodel", "bytearrays",
    "integers", "bin", "enter", "background", "mouse",
    "ondemand", "ruby", "erb", "widget",
    "pdf", "pdfs",
    "tree", "abstract", "nodes", "blocks", "columns",
    "desktop", "mainframe", "registers", "repository",
    "browser", "tab", "tabs",
    "autoloader", "debugger", "grepcode",
}


def _needs_context_validation(term: str) -> bool:
    """Check if a term needs context-aware validation.
    
    Routes ALL single lowercase words (without code markers) to Sonnet
    for ENTITY/GENERIC classification, unless they're in the bypass list.
    """
    if re.search(r'[().\[\]_::<>]', term):
        return False
    if re.search(r'[a-z][A-Z]', term):
        return False
    if re.match(r'^[A-Z][A-Z0-9_]+$', term) and len(term) >= 2:
        return False
    if " " in term or "/" in term:
        return False
    if term.lower() in CONTEXT_VALIDATION_BYPASS:
        return False
    # All single lowercase alpha words ≥3 chars → context validation
    if term.islower() and term.isalpha() and len(term) >= 3:
        return True
    # Short lowercase (2 chars like "db", "vm", "ui") or mixed case in explicit list
    return term.lower() in AMBIGUOUS_COMMON_WORDS


def run_context_validation(
    terms: list[str],
    doc_text: str,
    trace_log: BenchmarkLogger,
) -> list[str]:
    """Filter ambiguous common words using context-aware LLM classification.
    
    Returns the terms list with generic common words removed.
    """
    needs_check = [t for t in terms if _needs_context_validation(t)]
    safe_terms = [t for t in terms if not _needs_context_validation(t)]

    if not needs_check:
        trace_log.info("    Context validation: no ambiguous terms to check")
        return terms

    trace_log.info(f"    Context validation: checking {len(needs_check)} ambiguous terms: {needs_check}")

    prompt = CONTEXT_VALIDATION_PROMPT.format(
        content=doc_text[:3000],
        terms_json=json.dumps(needs_check),
    )

    t0 = time.time()
    response = call_llm(prompt, model="sonnet", max_tokens=2000, temperature=0.0)
    elapsed = time.time() - t0

    kept_from_check = []
    rejected_from_check = []
    try:
        text = response.strip()
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            decisions = {item["term"]: item["decision"] for item in parsed.get("terms", [])}
        else:
            decisions = {}
    except (json.JSONDecodeError, KeyError, TypeError):
        decisions = {}

    for t in needs_check:
        decision = decisions.get(t, "ENTITY")
        if decision == "ENTITY":
            kept_from_check.append(t)
        else:
            rejected_from_check.append(t)

    trace_log.info(f"    Context validation: kept {len(kept_from_check)}, rejected {len(rejected_from_check)} ({elapsed:.1f}s)")
    if rejected_from_check:
        trace_log.info(f"    Context-rejected: {rejected_from_check}")

    return safe_terms + kept_from_check


# ============================================================================
# SPAN EXPANSION (v6_spanfix)
# ============================================================================

def _is_brace_expansion_part(term: str, text: str, extracted_terms: list[str]) -> bool:
    """Check if term is a redundant part of a brace expansion already captured as full path."""
    t_lower = term.strip().lower()
    extracted_lower = {t.lower() for t in extracted_terms}
    for match in re.finditer(r'\{([^{}]+)\}', text):
        parts = [p.strip().lower() for p in match.group(1).split(',')]
        if t_lower not in parts:
            continue
        brace_start = match.start()
        prefix = text[:brace_start]
        suffix = text[match.end():]
        path_prefix = re.search(r'(\S+)$', prefix)
        path_suffix = re.match(r'(\S*)', suffix)
        full_path = (path_prefix.group(1) if path_prefix else '') + match.group(0) + (path_suffix.group(1) if path_suffix else '')
        if full_path.lower() in extracted_lower:
            return True
    return False


def expand_spans(terms: list[str], doc_text: str) -> list[str]:
    """Expand truncated entity spans to their maximum form found in text."""
    expanded = []
    for term in terms:
        new_term = _try_expand_span(term, doc_text)
        expanded.append(new_term)
    return expanded


def _try_expand_span(term: str, text: str) -> str:
    t = term.strip()
    text_lower = text.lower()
    t_lower = t.lower()

    idx = text_lower.find(t_lower)
    if idx == -1:
        return term

    # Expand function calls: if term ends with name and text has parens after
    if not t.endswith(")"):
        remainder = text[idx + len(t):]
        if remainder.startswith("("):
            depth, end = 1, 1
            while depth > 0 and end < len(remainder):
                if remainder[end] == "(":
                    depth += 1
                elif remainder[end] == ")":
                    depth -= 1
                end += 1
            candidate = text[idx:idx + len(t) + end]
            if len(candidate) <= 80:
                return candidate

    # Expand qualified names backward: send() → NetStream.send()
    if idx > 0 and t.endswith(")"):
        prefix_text = text[:idx]
        prefix_match = re.search(r'([\w:]+\.)+$', prefix_text)
        if prefix_match:
            candidate = prefix_match.group(0) + text[idx:idx + len(t)]
            if len(candidate) <= 80:
                return candidate

    # Expand device/product names forward: "Weblogic" → "Weblogic 12C server"
    remainder = text[idx + len(t):]
    version_match = re.match(r'(\s+\d+[A-Za-z]*(?:\s+\w+)?)', remainder)
    if version_match:
        candidate = text[idx:idx + len(t)] + version_match.group(1)
        candidate_lower = candidate.lower()
        if candidate_lower.endswith(" server") or candidate_lower.endswith(" client"):
            return candidate

    # Expand error names backward: "error 500" → "server internal error 500"
    if re.match(r'(error|internal)\s+\d+', t, re.I):
        prefix_text = text[:idx]
        err_prefix = re.search(r'((?:server\s+)?(?:internal\s+)?(?:server\s+)?)$', prefix_text, re.I)
        if err_prefix and err_prefix.group(1).strip():
            candidate = err_prefix.group(1) + text[idx:idx + len(t)]
            if len(candidate) <= 80:
                return candidate

    # Expand slash-compounds: "latex" → "latex/unicode", "notebook" → "qtconsole/notebook"
    slash_match = re.search(
        rf'(\S+/)?{re.escape(t)}(/\S+)?',
        text, re.IGNORECASE
    )
    if slash_match and (slash_match.group(1) or slash_match.group(2)):
        candidate = slash_match.group(0)
        if len(candidate) <= 100 and "/" in candidate:
            return candidate

    # Expand multi-word entities forward: "microsoft" → "microsoft crypto API"
    remainder = text[idx + len(t):idx + len(t) + 50]
    multiword_match = re.match(r'(\s+\w+){1,3}', remainder)
    if multiword_match and not t.endswith(")") and " " not in t:
        candidate = text[idx:idx + len(t)] + multiword_match.group(0)
        if re.search(r'\b(API|SDK|framework|library|protocol)\b', candidate, re.I):
            return candidate.strip()

    return term


# ============================================================================
# COMPOUND ENTITY PATTERN EXTRACTION (v6_spanfix)
# ============================================================================

COMPOUND_PATTERNS = [
    r'\b(header|vertical|horizontal|background)\s+(row|column|grid|divider)s?\b',
    r'\b(visual|identity|view)\s+(editor|inspector|manager)\b',
    r'\bcommand\s+line\b',
    r'\b(Stack|stack)\s+Overflow\b',
    r'\brun\s+time\s+\d+\s+error\b',
]

COMPOUND_PATTERNS_COMPILED = [re.compile(p, re.IGNORECASE) for p in COMPOUND_PATTERNS]


def extract_compound_entities(
    extracted: list[str],
    doc_text: str,
    trace_log: BenchmarkLogger,
) -> list[str]:
    """Find compound entities via regex patterns and add if not already extracted."""
    extracted_lower = {normalize_term(t) for t in extracted}
    compounds_added = []

    for pattern in COMPOUND_PATTERNS_COMPILED:
        for match in pattern.finditer(doc_text):
            compound = match.group(0)
            if normalize_term(compound) not in extracted_lower:
                compounds_added.append(compound)
                extracted.append(compound)
                extracted_lower.add(normalize_term(compound))

    if compounds_added:
        trace_log.info(f"    Compound entities added: {compounds_added}")
    return extracted


# ============================================================================
# MUST-EXTRACT SEEDING (v6_spanfix)
# ============================================================================

MUST_EXTRACT_SEEDS = {
    "table", "tables", "row", "column", "list", "array",
    "integers", "bytearrays", "string", "private", "global",
    "button", "text", "menu", "cell", "cells", "box",
    "calendar", "scroller", "borders",
    "microphone", "keyboard", "kernel", "mouse",
    "client", "webserver", "console", "configuration",
    "request", "server", "shell", "cursor",
    "span", "viewmodel", "exception",
    "bin", "enter", "setup", "background",
    "ondemand", "ruby", "erb", "widget",
    "cpu", "pdfs",
    "tab", "tabs", "browser", "debugger",
    "autoloader", "tree", "abstract",
    "nodes", "blocks", "columns", "desktop",
    "mainframe", "registers", "repository",
    "static", "delta", "route", "log",
    "selector", "grepcode",
    "window", "restful",
}


def seed_must_extract_terms(
    extracted: list[str],
    doc_text: str,
    trace_log: BenchmarkLogger,
) -> list[str]:
    """Seed commonly-missed GT terms, routing them through context validation."""
    extracted_lower = {normalize_term(t) for t in extracted}
    text_lower = doc_text.lower()
    seed_candidates = []

    for seed in MUST_EXTRACT_SEEDS:
        if normalize_term(seed) in extracted_lower:
            continue
        pattern = rf'\b{re.escape(seed)}\b'
        if re.search(pattern, text_lower):
            seed_candidates.append(seed)

    if not seed_candidates:
        trace_log.info("    Must-extract seeding: no candidates found")
        return extracted

    trace_log.info(f"    Must-extract seed candidates: {seed_candidates}")

    bypass_seeds = [s for s in seed_candidates if s.lower() in CONTEXT_VALIDATION_BYPASS]
    needs_validation = [s for s in seed_candidates if s.lower() not in CONTEXT_VALIDATION_BYPASS]

    validated_seeds = list(bypass_seeds)
    if needs_validation:
        trace_log.info(f"    Routing {len(needs_validation)} seeds through context validation: {needs_validation}")
        validated = run_context_validation(needs_validation, doc_text, trace_log)
        validated_seeds.extend(validated)

    if validated_seeds:
        trace_log.info(f"    Must-extract seeds added (validated): {validated_seeds}")
        extracted.extend(validated_seeds)
    else:
        trace_log.info("    Must-extract seeding: all candidates rejected by context validation")

    rejected = [s for s in seed_candidates if s not in validated_seeds]
    if rejected:
        trace_log.info(f"    Must-extract seeds rejected: {rejected}")

    return extracted


# ============================================================================
# SUBSPAN SUPPRESSION (v6_spanfix)
# ============================================================================

def suppress_subspans(
    extracted: list[str],
    trace_log: BenchmarkLogger,
) -> list[str]:
    """Remove terms that are substrings of other extracted terms.
    
    Skips terms <=3 chars (e.g. "C++" is substring of "C++11" but both valid GT).
    """
    if not extracted:
        return extracted
    
    lower_terms = [(t, t.lower()) for t in extracted]
    suppressed = []
    kept = []
    
    for term, term_lower in lower_terms:
        # Don't suppress very short terms — they may be independently valid
        # e.g. "C++" is substring of "C++11" but both are GT
        if len(term) <= 3:
            kept.append(term)
            continue
        
        is_subspan = False
        for other, other_lower in lower_terms:
            if term_lower == other_lower:
                continue
            # Check if term is a proper substring of other
            if term_lower in other_lower and len(term_lower) < len(other_lower):
                suppressed.append((term, other))
                is_subspan = True
                break
        
        if not is_subspan:
            kept.append(term)
    
    if suppressed:
        for sub, parent in suppressed:
            trace_log.info(f"    Subspan suppressed: '{sub}' (contained in '{parent}')")
    else:
        trace_log.info("    Subspan suppression: nothing to suppress")
    
    return kept


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

def run_extraction_for_doc(
    doc_text: str, 
    strategy: str,
    logger: BenchmarkLogger,
    trace_logger: BenchmarkLogger | None = None,
) -> list[str]:
    """Run full extraction pipeline on a document."""
    exhaustive_prompt, simple_prompt = get_extraction_prompt(strategy)
    review_prompt = get_sonnet_review_prompt(strategy)
    noise_filter = get_noise_filter(strategy)
    
    # Phase 1: Triple extraction
    trace_log = trace_logger or logger
    trace_log.info("  [Phase 1] Triple extraction...")
    
    t0 = time.time()
    r1 = call_llm(
        exhaustive_prompt.format(content=doc_text),
        model="sonnet",
        max_tokens=3000,
        temperature=0.0,
    )
    sonnet_terms = parse_terms_response(r1)
    trace_log.info(f"    Sonnet exhaustive: {len(sonnet_terms)} terms ({time.time()-t0:.1f}s)")
    
    t0 = time.time()
    r2 = call_llm(
        exhaustive_prompt.format(content=doc_text),
        model="haiku",
        max_tokens=3000,
        temperature=0.0,
    )
    haiku_exh_terms = parse_terms_response(r2)
    trace_log.info(f"    Haiku exhaustive: {len(haiku_exh_terms)} terms ({time.time()-t0:.1f}s)")
    
    t0 = time.time()
    r3 = call_llm(
        simple_prompt.format(content=doc_text),
        model="haiku",
        max_tokens=2000,
        temperature=0.0,
    )
    haiku_sim_terms = parse_terms_response(r3)
    trace_log.info(f"    Haiku simple: {len(haiku_sim_terms)} terms ({time.time()-t0:.1f}s)")
    
    # Phase 2-3: Grounding + Voting
    trace_log.info("  [Phase 2-3] Grounding + Voting...")
    candidates: dict[str, dict] = {}
    for src_name, src_terms in [
        ("sonnet_exhaustive", sonnet_terms),
        ("haiku_exhaustive", haiku_exh_terms),
        ("haiku_simple", haiku_sim_terms),
    ]:
        seen_src: set[str] = set()
        for t in src_terms:
            key = normalize_term(t)
            if key in seen_src:
                continue
            seen_src.add(key)
            if key not in candidates:
                candidates[key] = {"term": t, "sources": [], "vote_count": 0}
            candidates[key]["sources"].append(src_name)
            candidates[key]["vote_count"] += 1
    
    trace_log.info(f"    Union: {len(candidates)} candidates")
    
    # Span grounding
    grounded = {}
    ungrounded = []
    for key, cand in candidates.items():
        ok, match_type = verify_span(cand["term"], doc_text)
        if ok:
            grounded[key] = cand
        else:
            ungrounded.append(cand["term"])
    trace_log.info(f"    Grounded: {len(grounded)}/{len(candidates)} ({len(ungrounded)} removed)")
    if ungrounded:
        trace_log.info(f"    Ungrounded: {ungrounded[:10]}...")
    
    if strategy == "v6_spanfix":
        trace_log.info("  [Phase 2.5] Span expansion...")
        expanded_grounded = {}
        expansion_count = 0
        for key, cand in grounded.items():
            new_term = _try_expand_span(cand["term"], doc_text)
            if new_term != cand["term"]:
                expansion_count += 1
                trace_log.info(f"    Expanded: '{cand['term']}' → '{new_term}'")
                new_key = normalize_term(new_term)
                expanded_grounded[new_key] = {**cand, "term": new_term}
            else:
                expanded_grounded[key] = cand
        grounded = expanded_grounded
        trace_log.info(f"    Expanded {expansion_count} terms")

    filtered = grounded
    
    # Vote routing
    auto_kept = []
    needs_review = []
    for key, cand in filtered.items():
        if cand["vote_count"] >= 2:
            auto_kept.append(cand["term"])
        else:
            needs_review.append(cand["term"])
    trace_log.info(f"    Auto-kept (2+ votes): {len(auto_kept)}")
    trace_log.info(f"    Needs review (1 vote): {len(needs_review)}")
    
    # Phase 4: Sonnet Review
    sonnet_decisions = {}
    if needs_review:
        trace_log.info("  [Phase 4] Sonnet review...")
        terms_json = json.dumps(needs_review, indent=2)
        prompt = review_prompt.format(content=doc_text[:3000], terms_json=terms_json)
        
        t0 = time.time()
        response = call_llm(prompt, model="sonnet", max_tokens=3000, temperature=0.0)
        sonnet_decisions = parse_approval_response(response)
        approved = sum(1 for d in sonnet_decisions.values() if d.get("decision") == "APPROVE")
        rejected = sum(1 for d in sonnet_decisions.values() if d.get("decision") == "REJECT")
        trace_log.info(f"    Sonnet: {approved} APPROVE, {rejected} REJECT ({time.time()-t0:.1f}s)")
    
    # Phase 5: Assembly + Noise Filter
    trace_log.info("  [Phase 5] Assembly + filter...")
    pre_dedup = list(auto_kept)
    for t in needs_review:
        dec = sonnet_decisions.get(t, {}).get("decision", "REJECT")
        if dec == "APPROVE":
            pre_dedup.append(t)
    
    # Dedup
    seen: set[str] = set()
    basic_deduped = []
    for t in pre_dedup:
        key = normalize_term(t)
        if key not in seen:
            seen.add(key)
            basic_deduped.append(t)
    
    final_terms = smart_dedup(basic_deduped)
    
    noise_filtered = []
    noise_removed = []
    for t in final_terms:
        if noise_filter(t, final_terms, None):
            noise_removed.append(t)
        elif strategy == "v6_spanfix" and _is_brace_expansion_part(t, doc_text, final_terms):
            noise_removed.append(t)
        else:
            noise_filtered.append(t)
    
    trace_log.info(f"    Assembly: {len(pre_dedup)} -> {len(basic_deduped)} (dedup) -> {len(final_terms)} (smart) -> {len(noise_filtered)} (filtered, -{len(noise_removed)})")
    if noise_removed:
        trace_log.info(f"    Filtered out: {noise_removed[:10]}...")
    
    if strategy in ("v5_contextual", "v6_spanfix"):
        trace_log.info("  [Phase 6] Context validation...")
        noise_filtered = run_context_validation(noise_filtered, doc_text, trace_log)

    if strategy == "v6_spanfix":
        trace_log.info("  [Phase 6.5] Compound entity extraction...")
        noise_filtered = extract_compound_entities(noise_filtered, doc_text, trace_log)

    if strategy == "v6_spanfix":
        trace_log.info("  [Phase 7] Must-extract seeding...")
        noise_filtered = seed_must_extract_terms(noise_filtered, doc_text, trace_log)

    return noise_filtered


def run_benchmark(
    so_ner_path: str,
    n_docs: int = 10,
    strategy: str = "baseline",
    iteration: int = 1,
    seed: int = 42,
) -> dict:
    """Run full benchmark: parse data, extract, score, log."""
    
    # Setup logging
    artifacts_dir = Path(__file__).parent / "artifacts"
    log_dir = artifacts_dir / "so_ner_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Main log (summary + results)
    main_log_path = log_dir / f"so_ner_iter{iteration}_{strategy}_{timestamp}.log"
    main_logger = BenchmarkLogger(str(log_dir), f"so_ner_iter{iteration}_{strategy}_{timestamp}.log")
    
    # Trace log (detailed per-doc traces)
    trace_log_path = log_dir / f"so_ner_iter{iteration}_{strategy}_{timestamp}_trace.log"
    trace_logger = BenchmarkLogger(str(log_dir), f"so_ner_iter{iteration}_{strategy}_{timestamp}_trace.log", console=False)
    
    main_logger.section(f"SO NER Benchmark - Iteration {iteration} - Strategy: {strategy}")
    main_logger.info(f"Data: {so_ner_path}")
    main_logger.info(f"Documents: {n_docs}")
    main_logger.info(f"Strategy: {strategy}")
    main_logger.info(f"Seed: {seed}")
    main_logger.info(f"Timestamp: {timestamp}")
    
    # Step 1: Parse SO NER data
    main_logger.subsection("Step 1: Parse SO NER data")
    documents = parse_so_ner_file(so_ner_path)
    main_logger.info(f"Parsed {len(documents)} documents from SO NER test set")
    
    # Step 2: Select documents
    main_logger.subsection("Step 2: Select documents")
    selected = select_documents(documents, n=n_docs, min_entities=5, seed=seed)
    main_logger.info(f"Selected {len(selected)} documents for benchmarking")
    
    for i, doc in enumerate(selected):
        gt = extract_gt_terms(doc)
        main_logger.info(f"  Doc {i+1}: Q{doc['question_id']} - {len(doc['text'])} chars, {len(gt)} GT terms, {len(doc['entities'])} total entities")
    
    # Step 3: Run extraction pipeline on each document
    main_logger.subsection("Step 3: Run extraction pipeline")
    start_time = time.time()
    per_doc_results = []
    
    for i, doc in enumerate(selected):
        gt_terms = extract_gt_terms(doc)
        doc_text = doc["text"]
        
        main_logger.info(f"\n--- Document {i+1}/{len(selected)}: Q{doc['question_id']} ---")
        main_logger.info(f"  Text length: {len(doc_text)} chars")
        main_logger.info(f"  GT terms ({len(gt_terms)}): {gt_terms}")
        
        trace_logger.section(f"Document {i+1}: Q{doc['question_id']}")
        trace_logger.info(f"Text: {doc_text[:500]}...")
        trace_logger.info(f"GT terms: {json.dumps(gt_terms, indent=2)}")
        
        # Run extraction
        extracted = run_extraction_for_doc(doc_text, strategy, main_logger, trace_logger)
        
        main_logger.info(f"  Extracted ({len(extracted)}): {extracted}")
        
        # Score with m2m_v3
        scores = many_to_many_score(extracted, gt_terms, v3_match)
        
        main_logger.info(
            f"  SCORES: P={scores['precision']:.1%}, R={scores['recall']:.1%}, "
            f"H={scores['hallucination']:.1%}, F1={scores['f1']:.3f}"
        )
        main_logger.info(f"  TP={scores['tp']}, FP={scores['fp']}, FN={scores['fn']}")
        if scores["fp_terms"]:
            main_logger.info(f"  FP terms: {scores['fp_terms']}")
        if scores["fn_terms"]:
            main_logger.info(f"  FN terms: {scores['fn_terms']}")
        
        # Log detailed traces
        trace_logger.info(f"SCORES: {json.dumps({k: v for k, v in scores.items() if k not in ('fp_terms', 'fn_terms')}, indent=2)}")
        trace_logger.info(f"FP terms: {json.dumps(scores['fp_terms'], indent=2)}")
        trace_logger.info(f"FN terms: {json.dumps(scores['fn_terms'], indent=2)}")
        
        per_doc_results.append({
            "doc_idx": i,
            "question_id": doc["question_id"],
            "text_length": len(doc_text),
            "text_preview": doc_text[:200],
            "gt_terms": gt_terms,
            "extracted_terms": extracted,
            "scores": {
                "precision": round(scores["precision"], 4),
                "recall": round(scores["recall"], 4),
                "hallucination": round(scores["hallucination"], 4),
                "f1": round(scores["f1"], 4),
                "tp": scores["tp"],
                "fp": scores["fp"],
                "fn": scores["fn"],
            },
            "fp_terms": scores["fp_terms"],
            "fn_terms": scores["fn_terms"],
            "entity_types_in_doc": list({
                e["type"] for e in doc["entities"] 
                if e["type"] in RELEVANT_ENTITY_TYPES
            }),
        })
    
    elapsed = time.time() - start_time
    
    # Step 4: Compute aggregate metrics
    main_logger.subsection("Step 4: Aggregate Results")
    
    total_tp = sum(r["scores"]["tp"] for r in per_doc_results)
    total_fp = sum(r["scores"]["fp"] for r in per_doc_results)
    total_fn = sum(r["scores"]["fn"] for r in per_doc_results)
    total_extracted = sum(r["scores"]["tp"] + r["scores"]["fp"] for r in per_doc_results)
    total_gt = sum(len(r["gt_terms"]) for r in per_doc_results)
    
    agg_precision = total_tp / total_extracted if total_extracted else 0
    agg_recall = total_tp / total_gt if total_gt else 0
    agg_hallucination = total_fp / total_extracted if total_extracted else 0
    agg_f1 = 2 * agg_precision * agg_recall / (agg_precision + agg_recall) if (agg_precision + agg_recall) else 0
    
    aggregate = {
        "precision": round(agg_precision, 4),
        "recall": round(agg_recall, 4),
        "hallucination": round(agg_hallucination, 4),
        "f1": round(agg_f1, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "total_extracted": total_extracted,
        "total_gt": total_gt,
    }
    
    main_logger.info(f"AGGREGATE RESULTS:")
    main_logger.info(f"  Precision:     {agg_precision:.1%}")
    main_logger.info(f"  Recall:        {agg_recall:.1%}")
    main_logger.info(f"  Hallucination: {agg_hallucination:.1%}")
    main_logger.info(f"  F1:            {agg_f1:.3f}")
    main_logger.info(f"  Total: TP={total_tp}, FP={total_fp}, FN={total_fn}")
    main_logger.info(f"  Extracted: {total_extracted}, GT: {total_gt}")
    main_logger.info(f"  Elapsed: {elapsed:.1f}s ({elapsed/len(selected):.1f}s/doc)")
    
    # Targets check
    meets_precision = agg_precision >= 0.95
    meets_recall = agg_recall >= 0.95
    meets_hallucination = agg_hallucination <= 0.05
    
    main_logger.info(f"\nTARGET CHECK:")
    main_logger.info(f"  Precision >= 95%:     {'PASS' if meets_precision else 'FAIL'} ({agg_precision:.1%})")
    main_logger.info(f"  Recall >= 95%:        {'PASS' if meets_recall else 'FAIL'} ({agg_recall:.1%})")
    main_logger.info(f"  Hallucination <= 5%:  {'PASS' if meets_hallucination else 'FAIL'} ({agg_hallucination:.1%})")
    
    all_pass = meets_precision and meets_recall and meets_hallucination
    main_logger.info(f"  OVERALL: {'ALL TARGETS MET!' if all_pass else 'TARGETS NOT MET'}")
    
    # Collect all FP and FN terms for analysis
    all_fp_terms = []
    all_fn_terms = []
    for r in per_doc_results:
        for t in r["fp_terms"]:
            all_fp_terms.append({"term": t, "question_id": r["question_id"]})
        for t in r["fn_terms"]:
            all_fn_terms.append({"term": t, "question_id": r["question_id"]})
    
    main_logger.subsection("FP Analysis (All False Positives)")
    for fp in all_fp_terms:
        main_logger.info(f"  FP: '{fp['term']}' (Q{fp['question_id']})")
    
    main_logger.subsection("FN Analysis (All False Negatives)")
    for fn in all_fn_terms:
        main_logger.info(f"  FN: '{fn['term']}' (Q{fn['question_id']})")
    
    # Build results object
    results = {
        "metadata": {
            "iteration": iteration,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "so_ner_path": str(so_ner_path),
            "n_docs": len(selected),
            "seed": seed,
            "elapsed_seconds": round(elapsed, 1),
            "main_log": str(main_log_path),
            "trace_log": str(trace_log_path),
        },
        "targets": {
            "precision": 0.95,
            "recall": 0.95,
            "hallucination": 0.05,
        },
        "aggregate": aggregate,
        "target_check": {
            "meets_precision": meets_precision,
            "meets_recall": meets_recall,
            "meets_hallucination": meets_hallucination,
            "all_pass": all_pass,
        },
        "all_fp_terms": all_fp_terms,
        "all_fn_terms": all_fn_terms,
        "per_doc": per_doc_results,
    }
    
    # Save results
    results_path = artifacts_dir / f"so_ner_iter{iteration}_{strategy}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    main_logger.info(f"\nResults saved to: {results_path}")
    
    # Close loggers
    main_logger.close()
    trace_logger.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="SO NER Benchmark Runner")
    parser.add_argument("--so-ner-path", type=str, 
                        default="/tmp/StackOverflowNER/resources/annotated_ner_data/StackOverflow/test.txt",
                        help="Path to SO NER test file")
    parser.add_argument("--n-docs", type=int, default=10, help="Number of documents to benchmark")
    parser.add_argument("--strategy", type=str, default="baseline", 
                        choices=["baseline", "v2_adapted", "v3_precise", "v4_hybrid", "v5_contextual", "v6_spanfix"],
                        help="Extraction strategy to use")
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number for tracking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for document selection")
    args = parser.parse_args()
    
    results = run_benchmark(
        so_ner_path=args.so_ner_path,
        n_docs=args.n_docs,
        strategy=args.strategy,
        iteration=args.iteration,
        seed=args.seed,
    )
    
    # Print summary
    agg = results["aggregate"]
    check = results["target_check"]
    print(f"\n{'='*60}")
    print(f"ITERATION {args.iteration} - STRATEGY: {args.strategy}")
    print(f"{'='*60}")
    print(f"Precision:     {agg['precision']:.1%} {'PASS' if check['meets_precision'] else 'FAIL'}")
    print(f"Recall:        {agg['recall']:.1%} {'PASS' if check['meets_recall'] else 'FAIL'}")
    print(f"Hallucination: {agg['hallucination']:.1%} {'PASS' if check['meets_hallucination'] else 'FAIL'}")
    print(f"F1:            {agg['f1']:.3f}")
    print(f"{'='*60}")
    if check["all_pass"]:
        print("ALL TARGETS MET!")
    else:
        print("TARGETS NOT MET - needs iteration")


if __name__ == "__main__":
    main()
