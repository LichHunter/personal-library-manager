#!/usr/bin/env python3
"""
Generate out-of-domain (OOD) test dataset by renaming entities in SO NER data.

Strategy: Take SO NER documents and systematically replace known entities with
invented names that no NER model has seen during training. This tests whether
models generalize to novel technical terms vs just memorizing known ones.

Renaming rules:
- Library names: React→Xyloph, Django→Brentar, Flask→Quelm
- Language names: Python→Vexlang, Java→Driftcode, C++→Quiron
- Class/function names: CamelCase preserved with new roots
- Keeps document structure and context intact
"""

import json
import random
import re
from pathlib import Path


ENTITY_RENAMES = {
    "React": "Xyloph",
    "Angular": "Brentar",
    "Vue": "Quelm",
    "Django": "Tremvik",
    "Flask": "Plynth",
    "Express": "Vorpal",
    "Spring": "Dravok",
    "Rails": "Klendis",
    "Laravel": "Zenthar",
    "jQuery": "mQuirk",
    "Node": "Brin",
    "Python": "Vexlang",
    "Java": "Driftcode",
    "JavaScript": "Quiroscript",
    "TypeScript": "Delvscript",
    "Ruby": "Opaline",
    "PHP": "GHL",
    "Rust": "Tarnish",
    "Swift": "Glide",
    "Kotlin": "Nexlin",
    "Go": "Stride",
    "MySQL": "TronDB",
    "PostgreSQL": "NovusDB",
    "MongoDB": "ArcStore",
    "Redis": "Cacheon",
    "Docker": "Podwrap",
    "Kubernetes": "Clustrix",
    "AWS": "NebulCloud",
    "Azure": "SkyForge",
    "Git": "Versor",
    "GitHub": "CodeNest",
    "npm": "pkr",
    "pip": "gex",
    "webpack": "bundlox",
    "Babel": "Transplex",
    "CSS": "StyleQL",
    "HTML": "MarkupX",
    "REST": "QLINK",
    "GraphQL": "QueryMesh",
    "API": "SVC",
    "JSON": "DSON",
    "XML": "TML",
    "SQL": "QRL",
    "HTTP": "NTTP",
    "HTTPS": "NTTPS",
    "TCP": "NCP",
    "OAuth": "AuthMesh",
    "JWT": "TKW",
    "React Native": "Xyloph Mobile",
    "Node.js": "Brin.js",
    "Vue.js": "Quelm.js",
    "Next.js": "Fwd.js",
    "Nuxt": "Prevx",
    "Svelte": "Nimblr",
    "Bootstrap": "Gridweave",
    "Tailwind": "Windstyle",
    "Material UI": "Fabric UI",
    "Redux": "Stateflow",
    "MobX": "ReactiStore",
    "Axios": "Fetchron",
    "Nginx": "Servox",
    "Apache": "Webforge",
    "Linux": "Kernux",
    "Ubuntu": "Distriva",
    "Windows": "PaneOS",
    "macOS": "FruitOS",
    "Android": "MobilDroid",
    "iOS": "FruitMobile",
    "TensorFlow": "NeuralForge",
    "PyTorch": "TorchNet",
    "Pandas": "DataFramer",
    "NumPy": "ArrayCalc",
    "Matplotlib": "PlotCraft",
}


def rename_entities_in_text(text: str, rename_map: dict[str, str]) -> str:
    sorted_keys = sorted(rename_map.keys(), key=len, reverse=True)
    result = text
    for original in sorted_keys:
        replacement = rename_map[original]
        result = re.sub(re.escape(original), replacement, result)
        lower_orig = original.lower()
        if lower_orig != original:
            result = re.sub(re.escape(lower_orig), replacement.lower(), result)
    return result


def rename_entities_in_terms(terms: list[str], rename_map: dict[str, str]) -> list[str]:
    renamed = []
    for term in terms:
        new_term = term
        for original, replacement in sorted(rename_map.items(), key=lambda x: len(x[0]), reverse=True):
            new_term = new_term.replace(original, replacement)
            new_term = new_term.replace(original.lower(), replacement.lower())
        renamed.append(new_term)
    return renamed


def generate_ood_dataset(n_docs: int = 50, seed: int = 123) -> list[dict]:
    test_path = (
        Path(__file__).parent.parent
        / "poc-1c-scalable-ner"
        / "artifacts"
        / "test_documents.json"
    )
    with open(test_path) as f:
        all_docs = json.load(f)

    random.seed(seed)
    selected = random.sample(all_docs, min(n_docs, len(all_docs)))

    ood_docs = []
    for doc in selected:
        new_text = rename_entities_in_text(doc["text"], ENTITY_RENAMES)
        new_gt = rename_entities_in_terms(doc["gt_terms"], ENTITY_RENAMES)

        ood_docs.append({
            "doc_id": f"OOD-{doc['doc_id']}",
            "text": new_text,
            "gt_terms": new_gt,
            "original_doc_id": doc["doc_id"],
            "entity_types": doc.get("entity_types", []),
        })

    return ood_docs


def main():
    print("Generating OOD dataset with renamed entities...")
    ood_docs = generate_ood_dataset(n_docs=50, seed=123)

    output_path = Path(__file__).parent / "artifacts" / "ood_dataset.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ood_docs, f, indent=2)

    print(f"Generated {len(ood_docs)} OOD documents")
    print(f"Saved to {output_path}")

    rename_count = 0
    for doc in ood_docs:
        for term in doc["gt_terms"]:
            for replacement in ENTITY_RENAMES.values():
                if replacement in term or replacement.lower() in term.lower():
                    rename_count += 1
                    break
    print(f"Terms containing renamed entities: {rename_count}")


if __name__ == "__main__":
    main()
