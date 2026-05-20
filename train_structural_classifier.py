"""
train_structural_classifier.py — train logistic regression and random forest
classifiers on hand-crafted structural MIDI features.

Input:  data/features.npz  (produced by build_feature_dataset.py)
Output:
    classifier/structural_logreg.pkl          — sklearn Pipeline (scaler + LR)
    classifier/structural_rf.pkl              — sklearn Pipeline (scaler + RF)
    classifier/structural_feature_importance.png

Usage:
    python train_structural_classifier.py
"""

import os
import sys
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score)

from extract_features import FEATURE_NAMES

FEATURES_PATH = os.path.join("data", "features.npz")
OUT_DIR       = "classifier"
SEED          = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def report_metrics(name: str, y_true, y_pred) -> None:
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1  = f1_score(y_true, y_pred, zero_division=0)
    print(f"  Accuracy : {acc:.3f}")
    print(f"  Precision: {pre:.3f}   Recall: {rec:.3f}   F1: {f1:.3f}")


def save_model(pipeline: Pipeline, filename: str) -> None:
    path = os.path.join(OUT_DIR, filename)
    with open(path, "wb") as fh:
        pickle.dump(pipeline, fh)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Feature importance plot
# ---------------------------------------------------------------------------

def plot_importance(lr_pipeline: Pipeline, rf_pipeline: Pipeline) -> None:
    """
    Two-panel plot:
    - Left:  Logistic regression coefficients (signed; positive = AI signal).
             Sorted by absolute value so the most discriminating features
             appear at the top.
    - Right: Random forest mean decrease in impurity (always positive).

    Comparing the two panels reveals whether non-linear feature interactions
    (captured only by RF) change which features appear important.
    """
    lr_coef = lr_pipeline.named_steps["lr"].coef_[0]
    rf_imp  = rf_pipeline.named_steps["rf"].feature_importances_

    # Sort logistic regression by absolute coefficient.
    lr_order = np.argsort(np.abs(lr_coef))
    rf_order = np.argsort(rf_imp)

    fig, (ax_lr, ax_rf) = plt.subplots(1, 2, figsize=(14, 5))

    # — Logistic Regression panel —
    names_lr = [FEATURE_NAMES[i] for i in lr_order]
    coefs_lr = [lr_coef[i] for i in lr_order]
    colors_lr = ["#e74c3c" if c > 0 else "#3498db" for c in coefs_lr]
    ax_lr.barh(names_lr, coefs_lr, color=colors_lr, edgecolor="none")
    ax_lr.axvline(0, color="#475569", linewidth=0.8)
    ax_lr.set_title("Logistic Regression coefficients\n"
                    "(red = AI signal, blue = human signal)", fontsize=10)
    ax_lr.set_xlabel("Coefficient value")
    ax_lr.tick_params(labelsize=8)
    ax_lr.grid(axis="x", alpha=0.25)

    # — Random Forest panel —
    names_rf = [FEATURE_NAMES[i] for i in rf_order]
    imps_rf  = [rf_imp[i] for i in rf_order]
    ax_rf.barh(names_rf, imps_rf, color="#6366f1", edgecolor="none")
    ax_rf.set_title("Random Forest feature importance\n"
                    "(mean decrease in impurity)", fontsize=10)
    ax_rf.set_xlabel("Importance")
    ax_rf.tick_params(labelsize=8)
    ax_rf.grid(axis="x", alpha=0.25)

    plt.suptitle("Structural classifier — feature importance", fontsize=12,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "structural_feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(FEATURES_PATH):
        print(f"Feature matrix not found at {FEATURES_PATH}.\n"
              "Run build_feature_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    data = np.load(FEATURES_PATH, allow_pickle=True)
    X, y = data["X"].astype(np.float32), data["y"].astype(int)

    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features  "
          f"({int(y.sum())} AI, {int((y == 0).sum())} human)\n")

    # 80/20 stratified split so class balance is preserved in both sets.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"Train: {len(y_tr)}   Val: {len(y_val)}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Model 1 — Logistic Regression
    # Logistic regression is interpretable: each feature has a signed coefficient
    # that directly shows its direction and magnitude of contribution.
    # We scale features first because LR is sensitive to feature scale.
    # C=1 is the default inverse-regularisation strength; adjust if underfitting.
    # -------------------------------------------------------------------------
    print("─" * 50)
    print("Logistic Regression")
    print("─" * 50)

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    lr_pipe.fit(X_tr, y_tr)

    lr_pred      = lr_pipe.predict(X_val)
    lr_pred_prob = lr_pipe.predict_proba(X_val)[:, 1]

    report_metrics("LR", y_val, lr_pred)

    # Print coefficients ranked by absolute value — the most informative view
    # for understanding which structural features drive the decision.
    lr_coef = lr_pipe.named_steps["lr"].coef_[0]
    order   = np.argsort(np.abs(lr_coef))[::-1]
    print("\n  Feature coefficients (ranked by |coef|):")
    print(f"  {'Feature':<22}  {'Coef':>8}  Direction")
    print("  " + "-" * 42)
    for i in order:
        direction = "→ AI" if lr_coef[i] > 0 else "→ human"
        print(f"  {FEATURE_NAMES[i]:<22}  {lr_coef[i]:>8.4f}  {direction}")

    save_model(lr_pipe, "structural_logreg.pkl")

    # -------------------------------------------------------------------------
    # Model 2 — Random Forest
    # RF captures non-linear interactions between features (e.g. high stepwise
    # ratio AND low rhythmic CV together may be more diagnostic than either
    # alone).  It also provides feature importance via mean decrease in impurity.
    # We still run it inside a Pipeline with StandardScaler for API consistency
    # with predict_structural.py, even though RF itself is scale-invariant.
    # -------------------------------------------------------------------------
    print("\n" + "─" * 50)
    print("Random Forest")
    print("─" * 50)

    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(
            n_estimators=200,
            max_depth=None,          # let trees grow fully; forest-level averaging handles overfitting
            min_samples_leaf=2,      # prevent single-sample leaves on our small dataset
            random_state=SEED,
        )),
    ])
    rf_pipe.fit(X_tr, y_tr)

    rf_pred = rf_pipe.predict(X_val)
    report_metrics("RF", y_val, rf_pred)

    save_model(rf_pipe, "structural_rf.pkl")

    # -------------------------------------------------------------------------
    # Feature importance plot
    # -------------------------------------------------------------------------
    print()
    plot_importance(lr_pipe, rf_pipe)

    print("\nDone.")


if __name__ == "__main__":
    main()
