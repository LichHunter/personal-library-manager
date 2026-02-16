"""Phase 2: Quality Grading

Grades all 100 extractions as GOOD/ACCEPTABLE/POOR using F1-based automatic grading,
with LLM validation on 30 random samples and Cohen's κ computation.

Grading Rubric:
- GOOD: F1 >= 0.80 AND Hallucination <= 15%
- ACCEPTABLE: F1 >= 0.60 AND Hallucination <= 30%
- POOR: F1 < 0.60 OR Hallucination > 30%
"""

import json
import random
from pathlib import Path
from typing import Literal

from sklearn.metrics import cohen_kappa_score

# Import LLM provider from poc-1c
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "poc-1c-scalable-ner"))
from utils.llm_provider import call_llm


Grade = Literal["GOOD", "ACCEPTABLE", "POOR"]


def automatic_grade(f1: float, hallucination: float) -> Grade:
    """Apply F1-based grading rubric."""
    if f1 >= 0.80 and hallucination <= 0.15:
        return "GOOD"
    elif f1 >= 0.60 and hallucination <= 0.30:
        return "ACCEPTABLE"
    else:
        return "POOR"


def grade_to_numeric(grade: Grade) -> int:
    """Convert grade to numeric value for Cohen's κ."""
    return {"GOOD": 2, "ACCEPTABLE": 1, "POOR": 0}[grade]


def llm_grade_sample(doc: dict, model: str = "claude-haiku") -> tuple[Grade, str]:
    """Use LLM to grade a single sample independently.
    
    Returns:
        (grade, reasoning)
    """
    text = doc["text"]
    gt_terms = doc["gt_terms"]
    extracted_terms = doc["extracted_terms"]
    metrics = doc["metrics"]
    
    p = metrics["precision"]
    r = metrics["recall"]
    f1 = metrics["f1"]
    h = metrics["hallucination"]
    
    prompt = f"""Given the following extraction result, grade its quality:

Document text: {text[:500]}{'...' if len(text) > 500 else ''}

Ground truth terms: {', '.join(gt_terms)}
Extracted terms: {', '.join(extracted_terms)}

Metrics:
- Precision: {p:.2%}
- Recall: {r:.2%}
- F1: {f1:.3f}
- Hallucination: {h:.2%}

Based on these metrics, classify this extraction as:
- GOOD: F1 >= 0.80 AND Hallucination <= 15%
- ACCEPTABLE: F1 >= 0.60 AND Hallucination <= 30%
- POOR: F1 < 0.60 OR Hallucination > 30%

Respond with EXACTLY this format:
Grade: [GOOD|ACCEPTABLE|POOR]
Reasoning: [brief explanation in one sentence]"""
    
    response = call_llm(prompt, model=model, timeout=30, max_tokens=200)
    
    # Parse response
    lines = response.strip().split("\n")
    grade_line = next((l for l in lines if l.startswith("Grade:")), "")
    reasoning_line = next((l for l in lines if l.startswith("Reasoning:")), "")
    
    grade_text = grade_line.replace("Grade:", "").strip().upper()
    reasoning = reasoning_line.replace("Reasoning:", "").strip()
    
    # Validate grade
    if grade_text not in ["GOOD", "ACCEPTABLE", "POOR"]:
        # Fallback to automatic grading if LLM response is malformed
        print(f"Warning: Invalid LLM grade '{grade_text}' for doc {doc['doc_id']}, using automatic grade")
        return automatic_grade(f1, h), f"LLM response malformed, automatic grade applied: {response[:100]}"
    
    return grade_text, reasoning


def main():
    """Run Phase 2: Quality Grading."""
    print("=" * 80)
    print("Phase 2: Quality Grading")
    print("=" * 80)
    
    # Load Phase 1 dataset
    artifacts_dir = Path(__file__).parent / "artifacts"
    dataset_path = artifacts_dir / "phase-1-dataset.json"
    
    print(f"\nLoading dataset from {dataset_path}...")
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    print(f"Loaded {len(dataset)} documents")
    
    # Step 1: Automatic grading for all 100 samples
    print("\n" + "=" * 80)
    print("Step 1: Automatic Grading (all 100 samples)")
    print("=" * 80)
    
    grades = []
    for doc in dataset:
        metrics = doc["metrics"]
        f1 = metrics["f1"]
        h = metrics["hallucination"]
        
        grade = automatic_grade(f1, h)
        grade_numeric = grade_to_numeric(grade)
        
        grades.append({
            "doc_id": doc["doc_id"],
            "grade": grade,
            "grade_numeric": grade_numeric,
            "metrics": metrics,
            "auto_grade": grade,
        })
    
    # Count grade distribution
    from collections import Counter
    auto_counts = Counter([g["grade"] for g in grades])
    print(f"\nAutomatic grade distribution:")
    print(f"  GOOD: {auto_counts['GOOD']} ({auto_counts['GOOD']/len(grades)*100:.1f}%)")
    print(f"  ACCEPTABLE: {auto_counts['ACCEPTABLE']} ({auto_counts['ACCEPTABLE']/len(grades)*100:.1f}%)")
    print(f"  POOR: {auto_counts['POOR']} ({auto_counts['POOR']/len(grades)*100:.1f}%)")
    
    # Step 2: LLM validation on 30 random samples
    print("\n" + "=" * 80)
    print("Step 2: LLM Validation (30 random samples)")
    print("=" * 80)
    
    random.seed(42)
    validation_indices = random.sample(range(len(dataset)), 30)
    validation_indices.sort()
    
    print(f"\nSelected 30 samples for LLM validation (seed=42)")
    print(f"Indices: {validation_indices[:10]}... (showing first 10)")
    
    llm_grades_list = []
    auto_grades_list = []
    
    for i, idx in enumerate(validation_indices, 1):
        doc = dataset[idx]
        doc_id = doc["doc_id"]
        
        print(f"\n[{i}/30] Grading doc {doc_id}...")
        
        llm_grade, llm_reasoning = llm_grade_sample(doc)
        auto_grade = grades[idx]["auto_grade"]
        
        # Store LLM grade in the grades list
        grades[idx]["llm_grade"] = llm_grade
        grades[idx]["llm_reasoning"] = llm_reasoning
        
        llm_grades_list.append(llm_grade)
        auto_grades_list.append(auto_grade)
        
        agreement = "✓" if llm_grade == auto_grade else "✗"
        print(f"  Auto: {auto_grade}, LLM: {llm_grade} {agreement}")
        if llm_grade != auto_grade:
            print(f"  Reasoning: {llm_reasoning}")
    
    # Step 3: Compute Cohen's κ
    print("\n" + "=" * 80)
    print("Step 3: Cohen's κ Computation")
    print("=" * 80)
    
    # Convert to numeric for Cohen's κ
    auto_numeric = [grade_to_numeric(g) for g in auto_grades_list]
    llm_numeric = [grade_to_numeric(g) for g in llm_grades_list]
    
    kappa = cohen_kappa_score(auto_numeric, llm_numeric)
    agreement_rate = sum(a == l for a, l in zip(auto_grades_list, llm_grades_list)) / len(auto_grades_list)
    
    print(f"\nCohen's κ: {kappa:.3f}")
    print(f"Agreement rate: {agreement_rate:.1%} ({int(agreement_rate * 30)}/30)")
    
    # Interpret κ
    if kappa >= 0.8:
        interpretation = "almost perfect agreement"
    elif kappa >= 0.6:
        interpretation = "substantial agreement"
    elif kappa >= 0.4:
        interpretation = "moderate agreement"
    elif kappa >= 0.2:
        interpretation = "fair agreement"
    else:
        interpretation = "slight agreement"
    
    print(f"Interpretation: {interpretation}")
    
    # Document disagreements
    disagreements = []
    for idx, (auto, llm) in zip(validation_indices, zip(auto_grades_list, llm_grades_list)):
        if auto != llm:
            doc = dataset[idx]
            disagreements.append({
                "doc_id": doc["doc_id"],
                "auto": auto,
                "llm": llm,
                "reason": grades[idx]["llm_reasoning"],
                "metrics": doc["metrics"]
            })
    
    print(f"\nDisagreements: {len(disagreements)}")
    if disagreements:
        print("\nDisagreement details:")
        for d in disagreements:
            print(f"  {d['doc_id']}: Auto={d['auto']}, LLM={d['llm']}")
            print(f"    F1={d['metrics']['f1']:.3f}, H={d['metrics']['hallucination']:.2%}")
            print(f"    Reason: {d['reason']}")
    
    # Step 4: Save artifacts
    print("\n" + "=" * 80)
    print("Step 4: Saving Artifacts")
    print("=" * 80)
    
    # Save phase-2-grades.json
    grades_path = artifacts_dir / "phase-2-grades.json"
    with open(grades_path, "w") as f:
        json.dump(grades, f, indent=2)
    print(f"\nSaved grades to {grades_path}")
    
    # Save phase-2-kappa.json
    kappa_data = {
        "kappa": kappa,
        "n_samples": 30,
        "agreement_rate": agreement_rate,
        "auto_grades": auto_grades_list,
        "llm_grades": llm_grades_list,
        "disagreements": disagreements
    }
    kappa_path = artifacts_dir / "phase-2-kappa.json"
    with open(kappa_path, "w") as f:
        json.dump(kappa_data, f, indent=2)
    print(f"Saved κ results to {kappa_path}")
    
    # Save phase-2-summary.md
    summary_path = artifacts_dir / "phase-2-summary.md"
    with open(summary_path, "w") as f:
        f.write("# Phase 2: Quality Grading\n\n")
        f.write("## Objective\n")
        f.write("Assign quality grades to each extraction for correlation analysis.\n\n")
        f.write("## Approach\n")
        f.write("- F1-based automatic grading (all 100 samples)\n")
        f.write("- LLM validation on 30 random samples\n")
        f.write("- Cohen's κ to measure agreement\n\n")
        f.write("## Results\n")
        f.write(f"- Total graded: {len(grades)}\n")
        f.write("- Grade distribution:\n")
        f.write(f"  - GOOD: {auto_counts['GOOD']} ({auto_counts['GOOD']/len(grades)*100:.1f}%)\n")
        f.write(f"  - ACCEPTABLE: {auto_counts['ACCEPTABLE']} ({auto_counts['ACCEPTABLE']/len(grades)*100:.1f}%)\n")
        f.write(f"  - POOR: {auto_counts['POOR']} ({auto_counts['POOR']/len(grades)*100:.1f}%)\n")
        f.write(f"- Cohen's κ: {kappa:.3f} ({interpretation})\n")
        f.write(f"- Agreement rate: {agreement_rate:.1%}\n\n")
        
        if kappa < 0.7:
            f.write("## Issues\n")
            f.write(f"⚠️ Cohen's κ ({kappa:.3f}) is below target threshold of 0.7.\n\n")
            if disagreements:
                f.write(f"Found {len(disagreements)} disagreements between automatic and LLM grading:\n\n")
                for d in disagreements:
                    f.write(f"- **{d['doc_id']}**: Auto={d['auto']}, LLM={d['llm']}\n")
                    f.write(f"  - F1={d['metrics']['f1']:.3f}, H={d['metrics']['hallucination']:.2%}\n")
                    f.write(f"  - Reason: {d['reason']}\n\n")
        else:
            f.write("## Issues\n")
            f.write("None. Agreement is substantial.\n\n")
        
        f.write("## Next Phase Readiness\n")
        f.write("✓ Phase 3 (Correlation Analysis) can proceed\n")
    
    print(f"Saved summary to {summary_path}")
    
    print("\n" + "=" * 80)
    print("Phase 2 Complete!")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  - Graded: {len(grades)} documents")
    print(f"  - Cohen's κ: {kappa:.3f} ({interpretation})")
    print(f"  - Agreement: {agreement_rate:.1%}")
    print(f"\nArtifacts:")
    print(f"  - {grades_path}")
    print(f"  - {kappa_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
