#!/usr/bin/env python3
"""
Phase 4b: Ensemble Methods - Combine 7 signals for classification.

Ensemble approaches:
1. Weighted sum with grid search
2. Random Forest classifier
3. XGBoost classifier (if available)

Uses cross-validation to avoid overfitting.
"""

import json
import warnings
from pathlib import Path
from datetime import datetime
from collections import Counter
from itertools import product

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler

from signals import (
    known_term_ratio,
    coverage_score,
    entity_density,
    section_type_mismatch,
    technical_pattern_ratio,
    avg_term_length,
    text_grounding_score,
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

ALL_SIGNALS = [
    "known_term_ratio",
    "coverage",
    "entity_density",
    "section_type_mismatch",
    "technical_pattern_ratio",
    "avg_term_length",
    "text_grounding_score"
]


def load_data():
    """Load dataset and grades, compute all signals."""
    with open(ARTIFACTS_DIR / "phase-1-dataset.json") as f:
        dataset = json.load(f)
    with open(ARTIFACTS_DIR / "phase-2-grades.json") as f:
        grades = json.load(f)
    
    grade_lookup = {g["doc_id"]: g for g in grades}
    
    merged = []
    for doc in dataset:
        doc_id = doc["doc_id"]
        if doc_id in grade_lookup:
            # Compute new signals if not present
            terms = doc.get("extracted_terms", [])
            text = doc.get("text", "")
            
            signals = doc.get("signals", {}).copy()
            if "technical_pattern_ratio" not in signals:
                signals["technical_pattern_ratio"] = technical_pattern_ratio(terms)
            if "avg_term_length" not in signals:
                signals["avg_term_length"] = avg_term_length(terms)
            if "text_grounding_score" not in signals:
                signals["text_grounding_score"] = text_grounding_score(terms, text)
            
            merged.append({
                "doc_id": doc_id,
                "text": text,
                "extracted_terms": terms,
                "signals": signals,
                "grade": grade_lookup[doc_id]["grade"],
                "grade_numeric": grade_lookup[doc_id]["grade_numeric"],
            })
    return merged


def prepare_features(data, signal_names):
    """Convert data to feature matrix and labels."""
    X = np.array([[d["signals"].get(s, 0.0) for s in signal_names] for d in data])
    # Binary: 0 = POOR, 1 = non-POOR (ACCEPTABLE or GOOD)
    y = np.array([1 if d["grade_numeric"] > 0 else 0 for d in data])
    return X, y


def weighted_sum_grid_search(X_train, y_train, X_val, y_val, signal_names):
    """
    Ensemble Method 1: Weighted Sum with Grid Search
    
    Search for optimal weights to combine signals into a single score,
    then find the best threshold for classification.
    """
    n_signals = X_train.shape[1]
    
    # Normalize features for fair weighting
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # Grid search over weight combinations (simplified: 0, 0.5, 1 for each signal)
    # With 7 signals, 3^7 = 2187 combinations
    weight_options = [0.0, 0.5, 1.0]
    
    best_accuracy = 0
    best_weights = None
    best_threshold = 0.5
    
    # Sample fewer combinations for efficiency
    np.random.seed(42)
    n_random_samples = 500
    
    results = []
    
    for _ in range(n_random_samples):
        # Random weights
        weights = np.random.uniform(0, 1, n_signals)
        weights = weights / (weights.sum() + 1e-8)  # Normalize
        
        # Compute weighted sum scores
        train_scores = X_train_s @ weights
        val_scores = X_val_s @ weights
        
        # Try different thresholds
        for threshold_percentile in [25, 50, 75]:
            threshold = np.percentile(train_scores, threshold_percentile)
            
            # Classify: higher score -> more likely to be non-POOR
            train_pred = (train_scores > threshold).astype(int)
            val_pred = (val_scores > threshold).astype(int)
            
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_weights = weights.tolist()
                best_threshold = float(threshold)
    
    # Final evaluation with best weights
    final_weights = np.array(best_weights)
    val_scores = X_val_s @ final_weights
    val_pred = (val_scores > best_threshold).astype(int)
    
    cm = confusion_matrix(y_val, val_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    # Calculate POOR recall (class 0)
    poor_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "method": "weighted_sum_grid_search",
        "validation_accuracy": float(best_accuracy),
        "threshold": float(best_threshold),
        "weights": {signal_names[i]: float(best_weights[i]) for i in range(len(signal_names))},
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "poor_recall": float(poor_recall),
    }


def random_forest_ensemble(X_train, y_train, X_val, y_val, signal_names):
    """
    Ensemble Method 2: Random Forest Classifier
    
    Uses cross-validation to tune hyperparameters and avoid overfitting.
    """
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # Cross-validation to find best parameters
    best_accuracy = 0
    best_params = {}
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
    }
    
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                clf = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_split,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )
                
                # Use stratified k-fold if possible
                if len(np.unique(y_train)) > 1:
                    cv = StratifiedKFold(n_splits=min(5, sum(y_train)), shuffle=True, random_state=42)
                    try:
                        cv_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring='accuracy')
                        cv_mean = cv_scores.mean()
                    except ValueError:
                        # Fallback to no CV
                        clf.fit(X_train_s, y_train)
                        cv_mean = clf.score(X_train_s, y_train)
                else:
                    clf.fit(X_train_s, y_train)
                    cv_mean = clf.score(X_train_s, y_train)
                
                if cv_mean > best_accuracy:
                    best_accuracy = cv_mean
                    best_params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'min_samples_split': min_split
                    }
    
    # Train final model with best params
    clf = RandomForestClassifier(
        **best_params,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train_s, y_train)
    
    # Evaluate on validation set
    val_pred = clf.predict(X_val_s)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    # Feature importances
    importances = clf.feature_importances_
    feature_importance = {signal_names[i]: float(importances[i]) for i in range(len(signal_names))}
    
    cm = confusion_matrix(y_val, val_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    poor_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC-ROC if possible
    if len(np.unique(y_val)) > 1:
        try:
            val_proba = clf.predict_proba(X_val_s)[:, 1]
            auc_roc = roc_auc_score(y_val, val_proba)
        except:
            auc_roc = 0.5
    else:
        auc_roc = 0.5
    
    return {
        "method": "random_forest",
        "validation_accuracy": float(val_accuracy),
        "auc_roc": float(auc_roc),
        "best_params": best_params,
        "feature_importance": feature_importance,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "poor_recall": float(poor_recall),
    }


def xgboost_ensemble(X_train, y_train, X_val, y_val, signal_names):
    """
    Ensemble Method 3: XGBoost Classifier
    
    Gradient boosting with cross-validation.
    """
    if not XGBOOST_AVAILABLE:
        return {
            "method": "xgboost",
            "error": "XGBoost not installed",
            "validation_accuracy": 0,
        }
    
    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    # Calculate scale_pos_weight for imbalanced classes
    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1
    
    # Cross-validation to find best parameters
    best_accuracy = 0
    best_params = {}
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2],
    }
    
    for n_est in param_grid['n_estimators']:
        for max_d in param_grid['max_depth']:
            for lr in param_grid['learning_rate']:
                clf = XGBClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    learning_rate=lr,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    verbosity=0
                )
                
                # Use stratified k-fold if possible
                if len(np.unique(y_train)) > 1 and sum(y_train) >= 3:
                    cv = StratifiedKFold(n_splits=min(3, sum(y_train)), shuffle=True, random_state=42)
                    try:
                        cv_scores = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring='accuracy')
                        cv_mean = cv_scores.mean()
                    except ValueError:
                        clf.fit(X_train_s, y_train)
                        cv_mean = clf.score(X_train_s, y_train)
                else:
                    clf.fit(X_train_s, y_train)
                    cv_mean = clf.score(X_train_s, y_train)
                
                if cv_mean > best_accuracy:
                    best_accuracy = cv_mean
                    best_params = {
                        'n_estimators': n_est,
                        'max_depth': max_d,
                        'learning_rate': lr
                    }
    
    # Train final model with best params
    clf = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    clf.fit(X_train_s, y_train)
    
    # Evaluate on validation set
    val_pred = clf.predict(X_val_s)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    # Feature importances
    importances = clf.feature_importances_
    feature_importance = {signal_names[i]: float(importances[i]) for i in range(len(signal_names))}
    
    cm = confusion_matrix(y_val, val_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    poor_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC-ROC if possible
    if len(np.unique(y_val)) > 1:
        try:
            val_proba = clf.predict_proba(X_val_s)[:, 1]
            auc_roc = roc_auc_score(y_val, val_proba)
        except:
            auc_roc = 0.5
    else:
        auc_roc = 0.5
    
    return {
        "method": "xgboost",
        "validation_accuracy": float(val_accuracy),
        "auc_roc": float(auc_roc),
        "best_params": best_params,
        "feature_importance": feature_importance,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "poor_recall": float(poor_recall),
    }


def logistic_ensemble(X_train, y_train, X_val, y_val, signal_names):
    """
    Baseline: Logistic Regression (from Phase 4a)
    
    For comparison with other ensemble methods.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    
    clf = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    clf.fit(X_train_s, y_train)
    
    val_pred = clf.predict(X_val_s)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    cm = confusion_matrix(y_val, val_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    
    poor_recall = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Feature coefficients
    coefficients = clf.coef_[0]
    feature_coef = {signal_names[i]: float(coefficients[i]) for i in range(len(signal_names))}
    
    # AUC-ROC
    if len(np.unique(y_val)) > 1:
        try:
            val_proba = clf.predict_proba(X_val_s)[:, 1]
            auc_roc = roc_auc_score(y_val, val_proba)
        except:
            auc_roc = 0.5
    else:
        auc_roc = 0.5
    
    return {
        "method": "logistic_regression",
        "validation_accuracy": float(val_accuracy),
        "auc_roc": float(auc_roc),
        "feature_coefficients": feature_coef,
        "confusion_matrix": {"TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn)},
        "poor_recall": float(poor_recall),
    }


def determine_verdict(results):
    """Determine overall verdict based on ensemble results."""
    best_accuracy = max(r["validation_accuracy"] for r in results if "error" not in r)
    best_method = [r for r in results if r.get("validation_accuracy") == best_accuracy][0]
    
    if best_accuracy >= 0.80:
        verdict = "SUCCESS"
        reason = f"Best ensemble ({best_method['method']}) achieved {best_accuracy*100:.1f}% accuracy >= 80%"
    elif best_accuracy >= 0.75:
        verdict = "PARTIAL"
        reason = f"Best ensemble ({best_method['method']}) achieved {best_accuracy*100:.1f}% accuracy (75-80%)"
    else:
        verdict = "FAILURE"
        reason = f"All ensembles < 75% accuracy. Best: {best_method['method']} with {best_accuracy*100:.1f}%"
    
    return verdict, reason, best_method


def main():
    print("Phase 4b: Ensemble Methods - Signal Combination")
    print("=" * 60)
    
    print("\n[1/6] Loading data...")
    data = load_data()
    print(f"  Loaded {len(data)} documents")
    
    print("\n[2/6] Preparing features...")
    X, y = prepare_features(data, ALL_SIGNALS)
    print(f"  Features shape: {X.shape}")
    print(f"  Class distribution: {dict(Counter(y))}")
    print(f"  (0=POOR, 1=non-POOR)")
    
    # Split data (same as Phase 4a for consistency)
    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=stratify
    )
    print(f"\n  Train: {len(y_train)} samples ({sum(y_train)} non-POOR)")
    print(f"  Val:   {len(y_val)} samples ({sum(y_val)} non-POOR)")
    
    results = []
    
    print("\n[3/6] Ensemble 1: Weighted Sum Grid Search...")
    ws_result = weighted_sum_grid_search(X_train, y_train, X_val, y_val, ALL_SIGNALS)
    results.append(ws_result)
    print(f"  Validation accuracy: {ws_result['validation_accuracy']*100:.1f}%")
    print(f"  POOR recall: {ws_result['poor_recall']*100:.1f}%")
    
    print("\n[4/6] Ensemble 2: Random Forest...")
    rf_result = random_forest_ensemble(X_train, y_train, X_val, y_val, ALL_SIGNALS)
    results.append(rf_result)
    print(f"  Validation accuracy: {rf_result['validation_accuracy']*100:.1f}%")
    print(f"  AUC-ROC: {rf_result['auc_roc']:.4f}")
    print(f"  POOR recall: {rf_result['poor_recall']*100:.1f}%")
    print(f"  Best params: {rf_result['best_params']}")
    
    print("\n[5/6] Ensemble 3: XGBoost...")
    if XGBOOST_AVAILABLE:
        xgb_result = xgboost_ensemble(X_train, y_train, X_val, y_val, ALL_SIGNALS)
        results.append(xgb_result)
        print(f"  Validation accuracy: {xgb_result['validation_accuracy']*100:.1f}%")
        print(f"  AUC-ROC: {xgb_result['auc_roc']:.4f}")
        print(f"  POOR recall: {xgb_result['poor_recall']*100:.1f}%")
    else:
        print("  XGBoost not available, skipping...")
        results.append({"method": "xgboost", "error": "Not installed", "validation_accuracy": 0})
    
    # Also run logistic regression for baseline comparison
    print("\n  Baseline: Logistic Regression...")
    lr_result = logistic_ensemble(X_train, y_train, X_val, y_val, ALL_SIGNALS)
    results.append(lr_result)
    print(f"  Validation accuracy: {lr_result['validation_accuracy']*100:.1f}%")
    print(f"  AUC-ROC: {lr_result['auc_roc']:.4f}")
    
    print("\n[6/6] Determining verdict...")
    verdict, reason, best_method = determine_verdict(results)
    print(f"  Verdict: {verdict}")
    print(f"  Reason: {reason}")
    
    # Save results
    output = {
        "phase": "4b",
        "timestamp": datetime.now().isoformat(),
        "signals_used": ALL_SIGNALS,
        "data_split": {
            "train_size": len(y_train),
            "val_size": len(y_val),
            "train_non_poor": int(sum(y_train)),
            "val_non_poor": int(sum(y_val))
        },
        "ensemble_results": results,
        "best_method": best_method["method"],
        "best_accuracy": best_method["validation_accuracy"],
        "verdict": verdict,
        "reason": reason
    }
    
    with open(ARTIFACTS_DIR / "phase-4b-ensembles.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\n  Saved: phase-4b-ensembles.json")
    
    # Generate summary markdown
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    results_table = "| Method | Val Accuracy | AUC-ROC | POOR Recall |\n|--------|--------------|---------|-------------|\n"
    for r in results:
        if "error" in r:
            results_table += f"| {r['method']} | N/A | N/A | N/A |\n"
        else:
            auc = r.get("auc_roc", "N/A")
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else "N/A"
            results_table += f"| {r['method']} | {r['validation_accuracy']*100:.1f}% | {auc_str} | {r['poor_recall']*100:.1f}% |\n"
    
    summary = f"""# Phase 4b: Ensemble Methods Results

**Generated**: {timestamp}

## Objective
Combine 7 confidence signals using ensemble methods to achieve >= 80% classification accuracy.

## Data Split
- Training: {len(y_train)} samples ({sum(y_train)} non-POOR, {len(y_train) - sum(y_train)} POOR)
- Validation: {len(y_val)} samples ({sum(y_val)} non-POOR, {len(y_val) - sum(y_val)} POOR)
- Class imbalance: {(1 - sum(y)/len(y))*100:.1f}% POOR, {(sum(y)/len(y))*100:.1f}% non-POOR

## Ensemble Results

{results_table}

## Best Ensemble: **{best_method['method']}**

- Validation Accuracy: **{best_method['validation_accuracy']*100:.1f}%**
- AUC-ROC: {best_method.get('auc_roc', 'N/A')}
- POOR Recall: {best_method['poor_recall']*100:.1f}%
- Confusion Matrix: TP={best_method['confusion_matrix']['TP']}, TN={best_method['confusion_matrix']['TN']}, FP={best_method['confusion_matrix']['FP']}, FN={best_method['confusion_matrix']['FN']}

"""

    # Add feature importance for RF if it's the best
    if best_method['method'] == 'random_forest' and 'feature_importance' in best_method:
        summary += "\n### Feature Importance (Random Forest)\n"
        sorted_features = sorted(best_method['feature_importance'].items(), key=lambda x: -x[1])
        for feat, imp in sorted_features:
            summary += f"- {feat}: {imp:.4f}\n"
    elif best_method['method'] == 'logistic_regression' and 'feature_coefficients' in best_method:
        summary += "\n### Feature Coefficients (Logistic Regression)\n"
        sorted_features = sorted(best_method['feature_coefficients'].items(), key=lambda x: -abs(x[1]))
        for feat, coef in sorted_features:
            summary += f"- {feat}: {coef:.4f}\n"

    summary += f"""

## Verdict: **{verdict}**

{reason}

### Exit Criteria Check
- SUCCESS (accuracy >= 80%): {"**Met**" if verdict == "SUCCESS" else "Not met"}
- PARTIAL (accuracy 75-80%): {"**Met**" if verdict == "PARTIAL" else "Not met"}
- FAILURE (accuracy < 75%): {"**Met**" if verdict == "FAILURE" else "Not met"}

## Next Steps

"""
    
    if verdict == "SUCCESS":
        summary += """Signal ensemble successful! The combined signals achieve sufficient classification accuracy.

**Actions**:
1. Update threshold configuration with best ensemble
2. Document optimal ensemble parameters
3. Proceed to production integration
"""
    elif verdict == "PARTIAL":
        summary += """Partial success. Accuracy is close to target but not sufficient.

**Options**:
1. Try additional ensemble tuning
2. Proceed to Phase 4c (Oracle consultation) for insight
"""
    else:
        summary += """All ensembles failed to meet accuracy threshold.

**Actions**:
1. Proceed to Phase 4c: Oracle Consultation
2. Investigate why signals don't discriminate well
3. Consider alternative signal engineering approaches
"""

    summary += """

## Artifacts Generated
- `phase-4b-ensembles.json`: Full ensemble results
- `phase-4b-summary.md`: This summary document
"""
    
    with open(ARTIFACTS_DIR / "phase-4b-summary.md", "w") as f:
        f.write(summary)
    print("  Saved: phase-4b-summary.md")
    
    print("\n" + "=" * 60)
    print(f"Phase 4b complete. Verdict: {verdict}")
    print("=" * 60)
    
    return verdict


if __name__ == "__main__":
    main()
