#!/usr/bin/env python3
"""
Phase 3: Correlation Analysis and Threshold Determination

This script:
1. Loads phase-1 dataset (signals) and phase-2 grades
2. Computes Spearman correlation for each signal vs quality grade
3. Splits data 70/30 for train/validation (seed=42)
4. Fits logistic regression on combined signals
5. Generates ROC curve and saves as artifacts/phase-3-roc.png
6. Finds optimal threshold using Youden's J statistic
7. Evaluates on validation set
8. Determines PASS/PARTIAL/FAIL verdict based on success criteria
9. Saves all results to artifacts/phase-3-*.json and phase-3-summary.md
"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, auc, accuracy_score, confusion_matrix,
    classification_report, recall_score
)
from sklearn.preprocessing import StandardScaler

# Suppress sklearn convergence warnings for small datasets
warnings.filterwarnings('ignore', category=UserWarning)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
PHASE1_DATASET = ARTIFACTS_DIR / "phase-1-dataset.json"
PHASE2_GRADES = ARTIFACTS_DIR / "phase-2-grades.json"

SIGNAL_NAMES = ["known_term_ratio", "coverage", "entity_density", "section_type_mismatch"]


def load_data():
    """Load and merge phase-1 signals and phase-2 grades."""
    with open(PHASE1_DATASET) as f:
        dataset = json.load(f)
    
    with open(PHASE2_GRADES) as f:
        grades = json.load(f)
    
    # Create lookup for grades
    grade_lookup = {g["doc_id"]: g for g in grades}
    
    # Merge signals with grades
    merged = []
    for doc in dataset:
        doc_id = doc["doc_id"]
        if doc_id in grade_lookup:
            merged.append({
                "doc_id": doc_id,
                "signals": doc["signals"],
                "grade": grade_lookup[doc_id]["grade"],
                "grade_numeric": grade_lookup[doc_id]["grade_numeric"],
                "metrics": grade_lookup[doc_id]["metrics"]
            })
    
    return merged


def compute_correlations(data):
    """Compute Spearman correlation for each signal vs grade_numeric."""
    grades = np.array([d["grade_numeric"] for d in data])
    
    correlations = {}
    for signal_name in SIGNAL_NAMES:
        signal_values = np.array([d["signals"][signal_name] for d in data])
        
        # Handle edge case: no variance in signal
        if np.std(signal_values) == 0:
            correlations[signal_name] = {
                "r": 0.0,
                "p": 1.0,
                "significant": False,
                "note": "No variance in signal"
            }
            continue
        
        # Handle edge case: no variance in grades
        if np.std(grades) == 0:
            correlations[signal_name] = {
                "r": 0.0,
                "p": 1.0,
                "significant": False,
                "note": "No variance in grades (all same grade)"
            }
            continue
        
        r, p = stats.spearmanr(signal_values, grades)
        
        # Handle NaN values
        if np.isnan(r):
            r = 0.0
            p = 1.0
        
        correlations[signal_name] = {
            "r": float(r),
            "p": float(p),
            "significant": bool(p < 0.05)
        }
    
    # Find max correlation
    max_signal = max(correlations.items(), key=lambda x: abs(x[1]["r"]))
    max_correlation = {
        "signal": max_signal[0],
        "r": max_signal[1]["r"],
        "p": max_signal[1]["p"]
    }
    
    return {
        "signals": correlations,
        "max_correlation": max_correlation
    }


def prepare_features_labels(data):
    """Prepare feature matrix X and labels y for classification."""
    X = np.array([[d["signals"][s] for s in SIGNAL_NAMES] for d in data])
    
    # Binary classification: GOOD (grade_numeric=2) or ACCEPTABLE (1) vs POOR (0)
    # We want to identify POOR extractions (class 0) to route to slow system
    y = np.array([1 if d["grade_numeric"] > 0 else 0 for d in data])
    
    return X, y


def train_classifier(X_train, y_train, X_val, y_val):
    """Train logistic regression and evaluate."""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Handle extreme class imbalance
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        # Only one class in training set - can't train a classifier
        return None, scaler, {
            "error": "Only one class in training set",
            "train_class_distribution": {int(c): int(np.sum(y_train == c)) for c in unique_classes}
        }
    
    # Use class_weight='balanced' to handle imbalance
    clf = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    
    clf.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = clf.predict(X_train_scaled)
    y_val_pred = clf.predict(X_val_scaled)
    y_val_proba = clf.predict_proba(X_val_scaled)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    
    return clf, scaler, {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "y_val_pred": y_val_pred,
        "y_val_proba": y_val_proba,
        "train_class_distribution": {int(c): int(np.sum(y_train == c)) for c in np.unique(y_train)},
        "val_class_distribution": {int(c): int(np.sum(y_val == c)) for c in np.unique(y_val)}
    }


def compute_roc_and_threshold(y_true, y_proba, save_path):
    """Compute ROC curve, find optimal threshold using Youden's J, and save plot."""
    # Get probability of positive class (GOOD/ACCEPTABLE = 1)
    if y_proba.ndim == 2:
        proba = y_proba[:, 1]
    else:
        proba = y_proba
    
    # Handle edge case: only one class in y_true
    if len(np.unique(y_true)) < 2:
        # Save a simple plot indicating no ROC can be computed
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (Cannot compute - only one class in validation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return {
            "error": "Only one class in validation set",
            "auc_roc": 0.5,
            "optimal_threshold": 0.5,
            "method": "youden_j"
        }
    
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold using Youden's J statistic: max(TPR - FPR)
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], 
               color='red', s=100, zorder=5,
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve with Optimal Threshold (Youden\'s J)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return {
        "auc_roc": float(roc_auc),
        "optimal_threshold": float(optimal_threshold),
        "method": "youden_j",
        "fpr_at_optimal": float(fpr[optimal_idx]),
        "tpr_at_optimal": float(tpr[optimal_idx])
    }


def compute_routing_metrics(y_true, y_proba, threshold):
    """
    Compute routing metrics at the given threshold.
    
    We route to slow system when confidence < threshold (predict POOR).
    POOR recall = fraction of actual POOR extractions we correctly identify.
    Slow route rate = fraction of all samples routed to slow system.
    """
    if y_proba.ndim == 2:
        proba = y_proba[:, 1]  # Probability of being GOOD/ACCEPTABLE
    else:
        proba = y_proba
    
    # Predict POOR (class 0) when probability < threshold
    y_pred = (proba >= threshold).astype(int)
    
    # POOR recall: Of actual POOR (y_true=0), how many did we predict as POOR?
    actual_poor = (y_true == 0)
    if np.sum(actual_poor) > 0:
        predicted_poor = (y_pred == 0)
        poor_recall = np.sum(actual_poor & predicted_poor) / np.sum(actual_poor)
    else:
        poor_recall = 0.0
    
    # Slow route rate: What fraction goes to slow system (predicted as POOR)?
    slow_route_rate = np.sum(y_pred == 0) / len(y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Handle edge cases
        tn = fp = fn = tp = 0
        if cm.shape == (1, 1):
            if y_true[0] == 0:
                tn = cm[0, 0]
            else:
                tp = cm[0, 0]
    
    return {
        "poor_recall": float(poor_recall),
        "slow_route_rate": float(slow_route_rate),
        "confusion_matrix": {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn)
        }
    }


def determine_verdict(correlations, threshold_results, kappa_from_phase2=None):
    """
    Determine PASS/PARTIAL/FAIL based on success criteria.
    
    PASS (All Must Be True):
    - Max signal correlation: r >= 0.6, p < 0.05
    - Classification accuracy: >= 80% on validation
    - Grader reliability: kappa >= 0.7 (checked in Phase 2)
    - Threshold exists: POOR recall >= 90% AND slow rate <= 30%
    
    PARTIAL PASS:
    - Correlation r = 0.4-0.6: One signal shows moderate correlation
    - Accuracy 70-80%: Some predictive power
    
    FAIL:
    - All correlations r < 0.4
    - Accuracy < 70%
    """
    max_corr = correlations["max_correlation"]
    max_r = abs(max_corr["r"])
    max_p = max_corr["p"]
    
    val_accuracy = threshold_results.get("validation_accuracy", 0)
    poor_recall = threshold_results.get("poor_recall", 0)
    slow_route_rate = threshold_results.get("slow_route_rate", 1.0)
    
    # Check criteria
    criteria = {
        "correlation_strong": max_r >= 0.6 and max_p < 0.05,
        "correlation_moderate": 0.4 <= max_r < 0.6,
        "correlation_weak": max_r < 0.4,
        "accuracy_high": val_accuracy >= 0.80,
        "accuracy_moderate": 0.70 <= val_accuracy < 0.80,
        "accuracy_low": val_accuracy < 0.70,
        "routing_optimal": poor_recall >= 0.90 and slow_route_rate <= 0.30,
        "kappa_acceptable": kappa_from_phase2 is None or kappa_from_phase2 >= 0.7
    }
    
    # Determine verdict
    if (criteria["correlation_strong"] and 
        criteria["accuracy_high"] and 
        criteria["routing_optimal"] and
        criteria["kappa_acceptable"]):
        verdict = "PASS"
        explanation = (
            f"All criteria met: Max correlation r={max_r:.3f} >= 0.6 (p={max_p:.4f}), "
            f"validation accuracy {val_accuracy*100:.1f}% >= 80%, "
            f"POOR recall {poor_recall*100:.1f}% >= 90%, "
            f"slow route rate {slow_route_rate*100:.1f}% <= 30%."
        )
    elif criteria["correlation_moderate"] or criteria["accuracy_moderate"]:
        verdict = "PARTIAL"
        reasons = []
        if criteria["correlation_moderate"]:
            reasons.append(f"moderate correlation r={max_r:.3f} (0.4-0.6)")
        if criteria["accuracy_moderate"]:
            reasons.append(f"moderate accuracy {val_accuracy*100:.1f}% (70-80%)")
        
        failures = []
        if not criteria["correlation_strong"]:
            failures.append(f"correlation r={max_r:.3f} < 0.6")
        if not criteria["accuracy_high"]:
            failures.append(f"accuracy {val_accuracy*100:.1f}% < 80%")
        if not criteria["routing_optimal"]:
            failures.append(
                f"routing suboptimal: POOR recall={poor_recall*100:.1f}%, "
                f"slow rate={slow_route_rate*100:.1f}%"
            )
        
        explanation = (
            f"Partial success: {', '.join(reasons)}. "
            f"Failed criteria: {', '.join(failures)}. "
            "Signal engineering may improve results."
        )
    else:
        verdict = "FAIL"
        failures = []
        if criteria["correlation_weak"]:
            failures.append(f"all correlations weak (max r={max_r:.3f} < 0.4)")
        if criteria["accuracy_low"]:
            failures.append(f"low accuracy {val_accuracy*100:.1f}% < 70%")
        
        explanation = (
            f"Failed criteria: {', '.join(failures)}. "
            "Current signals do not correlate with extraction quality. "
            "Signal engineering required (Phase 4a)."
        )
    
    return {
        "verdict": verdict,
        "explanation": explanation,
        "criteria_checked": criteria
    }


def generate_summary(correlations, threshold_results, verdict_info):
    """Generate the phase-3-summary.md file."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format correlation results
    corr_lines = []
    for signal_name in SIGNAL_NAMES:
        c = correlations["signals"][signal_name]
        sig = "significant" if c.get("significant", False) else "not significant"
        note = c.get("note", "")
        if note:
            corr_lines.append(f"- {signal_name}: r={c['r']:.4f}, p={c['p']:.4f} ({sig}) - {note}")
        else:
            corr_lines.append(f"- {signal_name}: r={c['r']:.4f}, p={c['p']:.4f} ({sig})")
    
    max_c = correlations["max_correlation"]
    corr_lines.append(f"- **Max correlation**: {max_c['signal']} with r={max_c['r']:.4f}")
    
    # Handle errors in threshold_results
    if "error" in threshold_results:
        classification_section = f"""### Classification Performance
**Error**: {threshold_results['error']}

This indicates severe class imbalance in the dataset. The validation set may have insufficient samples of the minority class(es) to compute meaningful metrics.

- Train class distribution: {threshold_results.get('train_class_distribution', 'N/A')}
- Val class distribution: {threshold_results.get('val_class_distribution', 'N/A')}
"""
        routing_section = """### Routing Performance
Unable to compute routing metrics due to classification error.
"""
    else:
        cm = threshold_results.get("confusion_matrix", {"TP": 0, "TN": 0, "FP": 0, "FN": 0})
        classification_section = f"""### Classification Performance
- Training accuracy: {threshold_results.get('train_accuracy', 0)*100:.1f}%
- Validation accuracy: {threshold_results.get('validation_accuracy', 0)*100:.1f}%
- AUC-ROC: {threshold_results.get('auc_roc', 0):.4f}
- Optimal threshold: {threshold_results.get('optimal_threshold', 0):.4f}

**Confusion Matrix (Validation Set)**
|          | Pred POOR | Pred GOOD/ACC |
|----------|-----------|---------------|
| Act POOR | {cm['TN']:>9} | {cm['FP']:>13} |
| Act GOOD/ACC | {cm['FN']:>9} | {cm['TP']:>13} |
"""
        routing_section = f"""### Routing Performance
- POOR recall: {threshold_results.get('poor_recall', 0)*100:.1f}%
- Slow route rate: {threshold_results.get('slow_route_rate', 0)*100:.1f}%
"""

    # Check for dataset issues
    issues = []
    
    # Check class distribution from threshold results
    train_dist = threshold_results.get("train_class_distribution", {})
    val_dist = threshold_results.get("val_class_distribution", {})
    
    total_train = sum(train_dist.values()) if train_dist else 0
    total_val = sum(val_dist.values()) if val_dist else 0
    
    if train_dist:
        poor_train = train_dist.get(0, 0)
        if poor_train > 0.9 * total_train:
            issues.append(f"Severe class imbalance: {poor_train}/{total_train} training samples are POOR ({poor_train/total_train*100:.1f}%)")
    
    if val_dist:
        good_val = val_dist.get(1, 0)
        if good_val < 3:
            issues.append(f"Insufficient minority class in validation: only {good_val} GOOD/ACCEPTABLE samples")
    
    issues_section = "\n".join(f"- {issue}" for issue in issues) if issues else "None identified."
    
    # Determine next steps
    verdict = verdict_info["verdict"]
    if verdict == "PASS":
        next_steps = "Phase 3 complete. The correlation analysis validates the confidence signals. Proceed to implementation in production code."
    elif verdict == "PARTIAL":
        next_steps = "Proceed to Phase 4a (Signal Engineering). Current signals show promise but need improvement to meet all success criteria."
    else:
        next_steps = "Proceed to Phase 4a (Signal Engineering). Current signals do not correlate with extraction quality. Consider:\n  - Adding new signals (e.g., text length, term frequency)\n  - Engineering signal combinations\n  - Re-evaluating the grading criteria"
    
    summary = f"""# Phase 3: Correlation Analysis and Threshold Determination

**Generated**: {timestamp}

## Objective
Validate confidence signals correlate with extraction quality and determine optimal routing threshold.

## Approach
- Spearman correlation for each signal vs quality grade (GOOD=2, ACCEPTABLE=1, POOR=0)
- 70/30 train/validation split (seed=42)
- Logistic regression on combined signals (balanced class weights)
- ROC analysis with Youden's J statistic for threshold selection

## Results

### Correlation Analysis
{chr(10).join(corr_lines)}

{classification_section}

{routing_section}

## Verdict: **{verdict}**

{verdict_info["explanation"]}

### Criteria Checked
| Criterion | Result |
|-----------|--------|
| Correlation >= 0.6 | {verdict_info['criteria_checked']['correlation_strong']} |
| Accuracy >= 80% | {verdict_info['criteria_checked']['accuracy_high']} |
| Routing optimal | {verdict_info['criteria_checked']['routing_optimal']} |
| Correlation 0.4-0.6 (partial) | {verdict_info['criteria_checked']['correlation_moderate']} |
| Accuracy 70-80% (partial) | {verdict_info['criteria_checked']['accuracy_moderate']} |

## Issues
{issues_section}

## Dataset Statistics
- Total samples: {total_train + total_val}
- Training set: {total_train} samples
- Validation set: {total_val} samples
- Training distribution: {train_dist}
- Validation distribution: {val_dist}

## Next Steps
{next_steps}

## Artifacts Generated
- `phase-3-correlations.json`: Signal correlation results
- `phase-3-threshold.json`: Optimal threshold and classification metrics
- `phase-3-roc.png`: ROC curve visualization
- `phase-3-summary.md`: This summary document
"""
    
    return summary


def main():
    print("Phase 3: Correlation Analysis and Threshold Determination")
    print("=" * 60)
    
    # Load data
    print("\n[1/7] Loading phase-1 and phase-2 data...")
    data = load_data()
    print(f"  Loaded {len(data)} documents")
    
    # Check grade distribution
    from collections import Counter
    grade_dist = Counter(d["grade"] for d in data)
    print(f"  Grade distribution: {dict(grade_dist)}")
    
    # Compute correlations
    print("\n[2/7] Computing Spearman correlations...")
    correlations = compute_correlations(data)
    for signal in SIGNAL_NAMES:
        c = correlations["signals"][signal]
        sig = "*" if c.get("significant", False) else ""
        print(f"  {signal}: r={c['r']:.4f}, p={c['p']:.4f} {sig}")
    
    max_c = correlations["max_correlation"]
    print(f"  Max correlation: {max_c['signal']} (r={max_c['r']:.4f})")
    
    # Save correlations
    with open(ARTIFACTS_DIR / "phase-3-correlations.json", "w") as f:
        json.dump(correlations, f, indent=2)
    print("  Saved: phase-3-correlations.json")
    
    # Prepare features and labels
    print("\n[3/7] Preparing features and splitting data 70/30...")
    X, y = prepare_features_labels(data)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
    )
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Validation set: {len(X_val)} samples")
    print(f"  Train class distribution: {dict(Counter(y_train))}")
    print(f"  Val class distribution: {dict(Counter(y_val))}")
    
    # Train classifier
    print("\n[4/7] Training logistic regression...")
    clf, scaler, train_results = train_classifier(X_train, y_train, X_val, y_val)
    
    if clf is None:
        print(f"  Error: {train_results.get('error', 'Unknown error')}")
        threshold_results = train_results
        threshold_results["validation_accuracy"] = 0
        threshold_results["auc_roc"] = 0.5
        threshold_results["optimal_threshold"] = 0.5
        threshold_results["poor_recall"] = 0
        threshold_results["slow_route_rate"] = 1.0
        
        # Save basic ROC plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve (Cannot compute - classifier training failed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(ARTIFACTS_DIR / "phase-3-roc.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  Saved: phase-3-roc.png (placeholder)")
    else:
        print(f"  Training accuracy: {train_results['train_accuracy']*100:.1f}%")
        print(f"  Validation accuracy: {train_results['val_accuracy']*100:.1f}%")
        
        # Compute ROC and find optimal threshold
        print("\n[5/7] Computing ROC curve and optimal threshold...")
        roc_results = compute_roc_and_threshold(
            y_val, train_results["y_val_proba"], 
            ARTIFACTS_DIR / "phase-3-roc.png"
        )
        print(f"  AUC-ROC: {roc_results['auc_roc']:.4f}")
        print(f"  Optimal threshold: {roc_results['optimal_threshold']:.4f}")
        print("  Saved: phase-3-roc.png")
        
        # Compute routing metrics
        print("\n[6/7] Computing routing metrics at optimal threshold...")
        routing_results = compute_routing_metrics(
            y_val, train_results["y_val_proba"], 
            roc_results["optimal_threshold"]
        )
        print(f"  POOR recall: {routing_results['poor_recall']*100:.1f}%")
        print(f"  Slow route rate: {routing_results['slow_route_rate']*100:.1f}%")
        
        # Combine results
        threshold_results = {
            "optimal_threshold": roc_results["optimal_threshold"],
            "method": roc_results["method"],
            "train_accuracy": train_results["train_accuracy"],
            "validation_accuracy": train_results["val_accuracy"],
            "poor_recall": routing_results["poor_recall"],
            "slow_route_rate": routing_results["slow_route_rate"],
            "auc_roc": roc_results["auc_roc"],
            "confusion_matrix": routing_results["confusion_matrix"],
            "train_class_distribution": train_results["train_class_distribution"],
            "val_class_distribution": train_results["val_class_distribution"]
        }
    
    # Save threshold results
    with open(ARTIFACTS_DIR / "phase-3-threshold.json", "w") as f:
        json.dump(threshold_results, f, indent=2)
    print("  Saved: phase-3-threshold.json")
    
    # Determine verdict
    print("\n[7/7] Determining verdict...")
    verdict_info = determine_verdict(correlations, threshold_results)
    print(f"  Verdict: {verdict_info['verdict']}")
    print(f"  {verdict_info['explanation']}")
    
    # Generate and save summary
    summary = generate_summary(correlations, threshold_results, verdict_info)
    with open(ARTIFACTS_DIR / "phase-3-summary.md", "w") as f:
        f.write(summary)
    print("  Saved: phase-3-summary.md")
    
    print("\n" + "=" * 60)
    print(f"Phase 3 complete. Verdict: {verdict_info['verdict']}")
    print("=" * 60)
    
    return verdict_info["verdict"]


if __name__ == "__main__":
    main()
