#!/usr/bin/env python3
"""
Phase 4a: Signal Engineering - Add new signals and re-run correlation analysis.

New signals:
- technical_pattern_ratio: Ratio of terms matching technical patterns
- avg_term_length: Average character length of extracted terms
- text_grounding_score: Ratio of terms found verbatim in original text
"""

import json
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

from signals import (
    known_term_ratio,
    coverage_score,
    entity_density,
    section_type_mismatch,
    technical_pattern_ratio,
    avg_term_length,
    text_grounding_score,
)

warnings.filterwarnings('ignore', category=UserWarning)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

ORIGINAL_SIGNALS = ["known_term_ratio", "coverage", "entity_density", "section_type_mismatch"]
NEW_SIGNALS = ["technical_pattern_ratio", "avg_term_length", "text_grounding_score"]
ALL_SIGNALS = ORIGINAL_SIGNALS + NEW_SIGNALS


def load_data():
    with open(ARTIFACTS_DIR / "phase-1-dataset.json") as f:
        dataset = json.load(f)
    with open(ARTIFACTS_DIR / "phase-2-grades.json") as f:
        grades = json.load(f)
    
    grade_lookup = {g["doc_id"]: g for g in grades}
    
    merged = []
    for doc in dataset:
        doc_id = doc["doc_id"]
        if doc_id in grade_lookup:
            merged.append({
                "doc_id": doc_id,
                "text": doc.get("text", ""),
                "extracted_terms": doc.get("extracted_terms", []),
                "signals": doc["signals"],
                "grade": grade_lookup[doc_id]["grade"],
                "grade_numeric": grade_lookup[doc_id]["grade_numeric"],
            })
    return merged


def compute_new_signals(data):
    for doc in data:
        terms = doc["extracted_terms"]
        text = doc["text"]
        doc["signals"]["technical_pattern_ratio"] = technical_pattern_ratio(terms)
        doc["signals"]["avg_term_length"] = avg_term_length(terms)
        doc["signals"]["text_grounding_score"] = text_grounding_score(terms, text)
    return data


def compute_correlations(data, signal_names):
    grades = np.array([d["grade_numeric"] for d in data])
    
    correlations = {}
    for signal_name in signal_names:
        signal_values = np.array([d["signals"].get(signal_name, 0.0) for d in data])
        
        if np.std(signal_values) == 0:
            correlations[signal_name] = {"r": 0.0, "p": 1.0, "significant": False, "note": "No variance"}
            continue
        if np.std(grades) == 0:
            correlations[signal_name] = {"r": 0.0, "p": 1.0, "significant": False, "note": "No grade variance"}
            continue
        
        r, p = stats.spearmanr(signal_values, grades)
        if np.isnan(r):
            r, p = 0.0, 1.0
        
        correlations[signal_name] = {
            "r": float(r),
            "p": float(p),
            "significant": bool(p < 0.05)
        }
    
    max_signal = max(correlations.items(), key=lambda x: abs(x[1]["r"]))
    return {
        "signals": correlations,
        "max_correlation": {
            "signal": max_signal[0],
            "r": max_signal[1]["r"],
            "p": max_signal[1]["p"]
        }
    }


def train_and_evaluate(data, signal_names):
    X = np.array([[d["signals"].get(s, 0.0) for s in signal_names] for d in data])
    y = np.array([1 if d["grade_numeric"] > 0 else 0 for d in data])
    
    unique = np.unique(y)
    stratify = y if len(unique) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=42, stratify=stratify)
    
    if len(np.unique(y_train)) < 2:
        return {"error": "Only one class in training", "validation_accuracy": 0, "auc_roc": 0.5}
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    clf.fit(X_train_s, y_train)
    
    y_pred = clf.predict(X_val_s)
    y_proba = clf.predict_proba(X_val_s)[:, 1]
    
    if len(np.unique(y_val)) < 2:
        roc_auc = 0.5
        optimal_threshold = 0.5
    else:
        fpr, tpr, thresholds = roc_curve(y_val, y_proba)
        roc_auc = auc(fpr, tpr)
        youden_j = tpr - fpr
        optimal_idx = np.argmax(youden_j)
        optimal_threshold = thresholds[optimal_idx]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], color='red', s=100, zorder=5,
                   label=f'Threshold = {optimal_threshold:.3f}')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Phase 4a: ROC Curve (7 Signals)')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        fig.savefig(ARTIFACTS_DIR / "phase-4a-roc.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    cm = confusion_matrix(y_val, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    actual_poor = (y_val == 0)
    pred_poor = (y_pred == 0)
    poor_recall = np.sum(actual_poor & pred_poor) / np.sum(actual_poor) if np.sum(actual_poor) > 0 else 0
    slow_rate = np.sum(y_pred == 0) / len(y_pred)
    
    return {
        "train_accuracy": float(accuracy_score(y_train, clf.predict(X_train_s))),
        "validation_accuracy": float(accuracy_score(y_val, y_pred)),
        "auc_roc": float(roc_auc),
        "optimal_threshold": float(optimal_threshold),
        "poor_recall": float(poor_recall),
        "slow_route_rate": float(slow_rate),
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "train_distribution": {int(k): int(v) for k, v in Counter(y_train).items()},
        "val_distribution": {int(k): int(v) for k, v in Counter(y_val).items()}
    }


def determine_verdict(max_r):
    if max_r >= 0.6:
        return "SUCCESS", f"Max correlation r={max_r:.3f} >= 0.6"
    elif max_r >= 0.4:
        return "PARTIAL", f"Max correlation r={max_r:.3f} in [0.4, 0.6)"
    else:
        return "FAILURE", f"Max correlation r={max_r:.3f} < 0.4"


def main():
    print("Phase 4a: Signal Engineering - New Signals Analysis")
    print("=" * 60)
    
    print("\n[1/5] Loading data...")
    data = load_data()
    print(f"  Loaded {len(data)} documents")
    
    print("\n[2/5] Computing new signals...")
    data = compute_new_signals(data)
    print(f"  Added 3 new signals: {NEW_SIGNALS}")
    
    print("\n[3/5] Computing correlations (all 7 signals)...")
    correlations = compute_correlations(data, ALL_SIGNALS)
    
    print("\n  Signal Correlations:")
    for sig in ALL_SIGNALS:
        c = correlations["signals"][sig]
        new_marker = " [NEW]" if sig in NEW_SIGNALS else ""
        sig_marker = "*" if c.get("significant", False) else ""
        print(f"    {sig}: r={c['r']:.4f}, p={c['p']:.4f} {sig_marker}{new_marker}")
    
    max_c = correlations["max_correlation"]
    print(f"\n  Max: {max_c['signal']} (r={max_c['r']:.4f})")
    
    print("\n[4/5] Training classifier with all 7 signals...")
    eval_results = train_and_evaluate(data, ALL_SIGNALS)
    if "error" not in eval_results:
        print(f"  Train accuracy: {eval_results['train_accuracy']*100:.1f}%")
        print(f"  Val accuracy: {eval_results['validation_accuracy']*100:.1f}%")
        print(f"  AUC-ROC: {eval_results['auc_roc']:.4f}")
        print(f"  POOR recall: {eval_results['poor_recall']*100:.1f}%")
        print(f"  Slow route rate: {eval_results['slow_route_rate']*100:.1f}%")
    
    print("\n[5/5] Determining verdict...")
    verdict, reason = determine_verdict(abs(max_c["r"]))
    print(f"  Verdict: {verdict}")
    print(f"  Reason: {reason}")
    
    signals_json = {
        "new_signals": {
            "technical_pattern_ratio": "Ratio of terms matching technical patterns (CamelCase, snake_case, dot.notation)",
            "avg_term_length": "Average character length of extracted terms",
            "text_grounding_score": "Ratio of terms found verbatim in original text"
        },
        "all_signals": ALL_SIGNALS,
        "correlations": correlations,
        "classification": eval_results,
        "verdict": verdict,
        "reason": reason
    }
    
    with open(ARTIFACTS_DIR / "phase-4a-signals.json", "w") as f:
        json.dump(signals_json, f, indent=2)
    print("  Saved: phase-4a-signals.json")
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    corr_table = "| Signal | r | p | Significant | New? |\n|--------|---|---|-------------|------|\n"
    for sig in ALL_SIGNALS:
        c = correlations["signals"][sig]
        is_new = "Yes" if sig in NEW_SIGNALS else "No"
        is_sig = "Yes" if c.get("significant", False) else "No"
        corr_table += f"| {sig} | {c['r']:.4f} | {c['p']:.4f} | {is_sig} | {is_new} |\n"
    
    summary = f"""# Phase 4a: Signal Engineering Results

**Generated**: {timestamp}

## Objective
Design and implement new confidence signals to improve correlation with extraction quality.

## New Signals Implemented

1. **technical_pattern_ratio**: Ratio of terms matching technical naming patterns (CamelCase, PascalCase, snake_case, dot.notation, CONSTANT_CASE, file paths)

2. **avg_term_length**: Average character length of extracted terms. Technical terms typically 5-20 chars; very short or very long terms indicate noise.

3. **text_grounding_score**: Ratio of terms found verbatim in original text. Low grounding suggests hallucination.

## Correlation Analysis (All 7 Signals)

{corr_table}

**Maximum Correlation**: {max_c['signal']} with r={max_c['r']:.4f}, p={max_c['p']:.4f}

## Classification Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | {eval_results.get('train_accuracy', 0)*100:.1f}% |
| Validation Accuracy | {eval_results.get('validation_accuracy', 0)*100:.1f}% |
| AUC-ROC | {eval_results.get('auc_roc', 0.5):.4f} |
| Optimal Threshold | {eval_results.get('optimal_threshold', 0.5):.4f} |
| POOR Recall | {eval_results.get('poor_recall', 0)*100:.1f}% |
| Slow Route Rate | {eval_results.get('slow_route_rate', 0)*100:.1f}% |

## Verdict: **{verdict}**

{reason}

### Exit Criteria Check
- SUCCESS (r >= 0.6): {"Met" if verdict == "SUCCESS" else "Not met"}
- PARTIAL (r >= 0.4 but < 0.6): {"Met" if verdict == "PARTIAL" else "Not met"}  
- FAILURE (r < 0.4): {"Met" if verdict == "FAILURE" else "Not met"}

## Analysis

### New Signal Performance
"""
    
    for sig in NEW_SIGNALS:
        c = correlations["signals"][sig]
        sig_str = "significant" if c.get("significant") else "not significant"
        summary += f"- **{sig}**: r={c['r']:.4f} ({sig_str})\n"
    
    summary += f"""
### Comparison with Phase 3
- Phase 3 max correlation: coverage (r=0.276)
- Phase 4a max correlation: {max_c['signal']} (r={max_c['r']:.4f})
- Improvement: {'+' if max_c['r'] > 0.276 else ''}{(max_c['r'] - 0.276):.4f}

## Next Steps
"""
    
    if verdict == "SUCCESS":
        summary += "Signal engineering successful. Proceed with integration into production pipeline."
    elif verdict == "PARTIAL":
        summary += "Partial success. Proceed to Phase 4b (Ensemble Optimization) to combine signals."
    else:
        summary += "New signals did not achieve target correlation. Proceed to Phase 4b or re-evaluate approach."
    
    summary += """

## Artifacts Generated
- `phase-4a-signals.json`: Signal definitions and correlation results
- `phase-4a-roc.png`: ROC curve visualization
- `phase-4a-summary.md`: This summary document
"""
    
    with open(ARTIFACTS_DIR / "phase-4a-summary.md", "w") as f:
        f.write(summary)
    print("  Saved: phase-4a-summary.md")
    
    print("\n" + "=" * 60)
    print(f"Phase 4a complete. Verdict: {verdict}")
    print("=" * 60)
    
    return verdict


if __name__ == "__main__":
    main()
