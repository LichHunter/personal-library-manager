#!/usr/bin/env python3
"""
Fine-tune GLiNER on SO NER training data and evaluate improvement.

GLiNER training data format:
  {"tokenized_text": ["word1", ...], "ner": [[start, end, "type"], ...]}
  Indices are token-level and INCLUSIVE on both ends.
"""

import json
import re
import sys
from pathlib import Path

import torch

SO_NER_DIR = Path("/tmp/StackOverflowNER/resources/annotated_ner_data/StackOverflow")
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

ENTITY_TYPES = [
    "Application", "Library", "Library_Class", "Library_Function",
    "Library_Variable", "Language", "Data_Structure", "Data_Type",
    "Algorithm", "Operating_System", "Device", "File_Type",
    "HTML_XML_Tag", "Error_Name", "Version", "Website",
    "Class_Name", "Function_Name", "User_Interface_Element",
    "Keyboard_IP", "Organization", "File_Name",
]

EXCLUDED_TYPES = {"Code_Block", "Output_Block", "Variable_Name", "Value", "User_Name"}


def parse_bio_to_gliner_format(filepath: str, max_docs: int | None = None) -> list[dict]:
    samples = []
    current_tokens: list[str] = []
    current_entities: list[list] = []
    current_entity_start: int | None = None
    current_entity_type: str | None = None
    in_metadata = False

    def flush_entity(end_idx: int | None = None):
        nonlocal current_entity_start, current_entity_type
        if current_entity_start is not None and current_entity_type:
            # GLiNER uses INCLUSIVE end index
            if end_idx is None:
                end_idx = len(current_tokens) - 1
            current_entities.append([current_entity_start, end_idx, current_entity_type])
        current_entity_start = None
        current_entity_type = None

    def flush_sample():
        nonlocal current_tokens, current_entities
        flush_entity()
        if current_tokens and current_entities:
            samples.append({
                "tokenized_text": current_tokens,
                "ner": current_entities,
            })

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if not line.strip():
                flush_entity()
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            token, tag = parts[0].strip(), parts[1].strip()

            if token in ("Question_ID", "Question_URL", "Answer_to_Question_ID"):
                flush_sample()
                if max_docs and len(samples) >= max_docs:
                    return samples
                if token == "Question_ID":
                    current_tokens = []
                    current_entities = []
                current_entity_start = None
                current_entity_type = None
                in_metadata = True
                continue

            if in_metadata:
                if re.match(r"^\d+$", token) or token.startswith("http") or token == ":":
                    continue
                in_metadata = False

            if tag.startswith("B-Code_Block") or tag.startswith("I-Code_Block"):
                flush_entity()
                if tag.startswith("B-Code_Block"):
                    current_tokens.append("[CODE]")
                continue

            if tag.startswith("B-Output_Block") or tag.startswith("I-Output_Block"):
                flush_entity()
                if tag.startswith("B-Output_Block"):
                    current_tokens.append("[OUTPUT]")
                continue

            token_idx = len(current_tokens)
            current_tokens.append(token)

            if tag.startswith("B-"):
                etype = tag[2:]
                # Entity ended at previous token (token_idx - 1)
                flush_entity(end_idx=token_idx - 1)
                if etype not in EXCLUDED_TYPES:
                    current_entity_start = token_idx
                    current_entity_type = etype

            elif tag.startswith("I-"):
                etype = tag[2:]
                if current_entity_type != etype:
                    # Type mismatch — entity ended at previous token
                    flush_entity(end_idx=token_idx - 1)

            else:  # O tag
                # Entity ended at previous token
                flush_entity(end_idx=token_idx - 1)

    flush_sample()
    return samples


def train_gliner(
    train_data: list[dict],
    model_id: str = "urchade/gliner_medium-v2.1",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 5e-6,
    save_path: str | None = None,
):
    from gliner import GLiNER

    print(f"Loading base model: {model_id}")
    model = GLiNER.from_pretrained(model_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}")

    output_dir = save_path or str(ARTIFACTS_DIR / "gliner-finetuned")

    print(f"Training: {len(train_data)} samples, {epochs} epochs, bs={batch_size}, lr={lr}")
    trainer = model.train_model(
        train_dataset=train_data,
        eval_dataset=None,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        others_lr=1e-5,
        weight_decay=0.01,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        logging_steps=10,
        save_steps=99999,
        report_to="none",
        dataloader_num_workers=0,
        use_cpu=(not torch.cuda.is_available()),
    )

    model.save_pretrained(output_dir)
    print(f"Saved to {output_dir}")
    return model


def evaluate_on_test(model, n_docs: int = 100) -> dict:
    from eval_framework import evaluate_model, load_so_ner_test, report_to_dict, print_report
    from ner_models import ExtractedEntity

    class WrappedModel:
        def __init__(self, gliner_model, labels):
            self._model = gliner_model
            self._labels = labels

        @property
        def name(self):
            return "gliner-finetuned"

        def extract(self, text):
            truncated = text[:512] if len(text) > 512 else text
            raw = self._model.predict_entities(truncated, self._labels, threshold=0.3)
            seen = set()
            entities = []
            for ent in raw:
                span = ent["text"]
                if span in seen:
                    continue
                seen.add(span)
                entities.append(
                    ExtractedEntity(text=span, confidence=float(ent["score"]), label=ent["label"])
                )
            return entities

    wrapped = WrappedModel(model, ENTITY_TYPES)

    print(f"\nEvaluating on SO NER test ({n_docs} docs)...")
    so_docs = load_so_ner_test(n_docs=n_docs)
    report = evaluate_model(wrapped, so_docs, dataset_name="so-ner")
    print_report(report)

    return report_to_dict(report)


def main():
    print("=" * 60)
    print("  GLiNER Fine-Tuning Experiment")
    print("=" * 60)

    train_path = SO_NER_DIR / "train.txt"
    if not train_path.exists():
        print(f"ERROR: {train_path} not found")
        sys.exit(1)

    print(f"\n[1/4] Parsing training data...")
    train_data = parse_bio_to_gliner_format(str(train_path), max_docs=200)
    print(f"  {len(train_data)} samples, {sum(len(s['ner']) for s in train_data)} entities")

    sample = train_data[0]
    print(f"\n  Verifying format:")
    for start, end, etype in sample["ner"][:3]:
        entity_text = " ".join(sample["tokenized_text"][start : end + 1])
        print(f"    [{start},{end}] '{entity_text}' -> {etype}")

    save_path = str(ARTIFACTS_DIR / "gliner-finetuned")
    print(f"\n[2/4] Fine-tuning GLiNER...")
    model = train_gliner(train_data, epochs=3, batch_size=4, lr=5e-6, save_path=save_path)

    print(f"\n[3/4] Evaluating fine-tuned model...")
    model.eval()
    ft_results = evaluate_on_test(model, n_docs=100)

    output = {
        "experiment": "gliner-finetune",
        "train_samples": len(train_data),
        "epochs": 3,
        "finetuned_results": ft_results,
    }
    output_path = ARTIFACTS_DIR / "gliner_finetune_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[4/4] Results saved to {output_path}")

    print(f"\n{'=' * 60}")
    print("  COMPARISON: Zero-shot vs Fine-tuned")
    print(f"{'=' * 60}")
    print(f"  {'Metric':<20} {'Zero-shot':>12} {'Fine-tuned':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    zs = {"P": 0.581, "R": 0.547, "F1": 0.518, "Hall": 0.419, "Sep": 0.100}
    ft = ft_results
    print(f"  {'Precision':<20} {zs['P']:>12.3f} {ft['avg_precision']:>12.3f}")
    print(f"  {'Recall':<20} {zs['R']:>12.3f} {ft['avg_recall']:>12.3f}")
    print(f"  {'F1':<20} {zs['F1']:>12.3f} {ft['avg_f1']:>12.3f}")
    print(f"  {'Hallucination':<20} {zs['Hall']:>12.3f} {ft['avg_hallucination']:>12.3f}")
    print(f"  {'Conf Separation':<20} {zs['Sep']:>+12.3f} {ft['confidence_separation']:>+12.3f}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
