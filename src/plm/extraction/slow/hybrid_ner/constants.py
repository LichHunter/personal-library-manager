"""Constants for noise filtering and heuristic extraction."""

PURE_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can", "shall",
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "so", "if", "than",
    "about", "up", "out", "no", "just", "also", "more",
    "some", "any", "all", "each", "every", "both",
}

ACTION_GERUNDS = {
    "appending", "serializing", "deserializing", "floating", "wrapping",
    "loading", "downloading", "subscribing", "referencing", "toggling",
    "de-serialized", "cross-platform", "cross-compile",
}

DESCRIPTIVE_ADJECTIVES = {
    "hidden", "visible", "vertical", "horizontal", "floating",
    "absolute", "relative", "nested", "multiple", "various",
    "specific", "general", "dynamic", "static", "custom",
    "native", "proper", "basic", "simple", "complex",
    "actual", "original", "current", "previous", "following",
    "hardware", "software", "entropy", "random", "external",
    "internal", "main", "local", "global", "primary", "secondary",
    "default", "existing", "standard", "typical", "generic",
    "certain", "optional", "required", "initial", "final",
    "entire", "single", "separate", "different", "additional",
}

CATEGORY_SUFFIXES = {
    "items", "elements", "values", "settings", "parameters",
    "options", "properties", "fields", "catalog", "orientation",
    "behavior", "handling", "management", "compatibility",
    "content", "position", "libraries", "events", "factors",
    "pool", "level", "keys", "engine", "mode", "navigation",
    "access", "support", "system", "design", "architecture",
    "configuration", "implementation", "specification",
}

_ALLCAPS_EXCLUDE = {
    "THE", "AND", "BUT", "NOT", "FOR", "ARE", "WAS", "HAS", "HAD",
    "SO", "IT", "IS", "OR", "MY", "IN", "TO", "OF", "AT", "BY",
    "ON", "AN", "AS", "IF", "NO", "DO", "UP", "BE", "AM", "HE",
    "OUTPUT", "UPDATE", "EDIT", "OK", "ERROR", "WARNING", "INFO",
    "DEBUG", "TRUE", "FALSE", "NULL", "NONE", "TODO", "FIXME",
    "NOTE", "CODE", "LEVEL", "NEW", "END", "SET", "GET", "ADD",
    "PUT", "DELETE", "POST", "THEN", "ELSE", "CASE", "WHEN",
    "WITH", "FROM", "INTO", "LIKE", "WHERE", "ORDER", "GROUP",
}
