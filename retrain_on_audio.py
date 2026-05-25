"""
retrain_on_audio.py — train structural classifiers on FMA-derived MIDI data.

Run after prepare_fma_training.py has populated data/human_audio/:
    python retrain_on_audio.py

Reads:
    data/ai/          — AI-generated MIDI files    (label 1)
    data/human_audio/ — FMA-transcribed MIDI files (label 0)

Writes:
    data/features_audio.npz
    classifier/structural_logreg_audio.pkl
    classifier/structural_rf_audio.pkl

Prints accuracy, precision, recall, F1, and LR feature coefficients.
"""

import datetime
import json
import os
import sys
import pickle

import numpy as np
from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (accuracy_score, f1_score,
                                     precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

from extract_features import FEATURE_NAMES, extract_features

# ── paths ────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(__file__)
AI_DIR    = os.path.join(_HERE, "data", "ai")
HUMAN_DIR = os.path.join(_HERE, "data", "human_audio")
NPZ_PATH  = os.path.join(_HERE, "data", "features_audio.npz")
CLF_DIR   = os.path.join(_HERE, "classifier")
LR_PATH   = os.path.join(CLF_DIR, "structural_logreg_audio.pkl")
RF_PATH   = os.path.join(CLF_DIR, "structural_rf_audio.pkl")

# ── config ───────────────────────────────────────────────────────────────────
SEED      = 42
TEST_SIZE = 0.2


# ── helpers ──────────────────────────────────────────────────────────────────

def collect_features(directory: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from every .mid in directory; return (X, y)."""
    mid_files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".mid"))
    vecs, labels, failed = [], [], 0

    for fname in mid_files:
        path = os.path.join(directory, fname)
        try:
            vec = extract_features(path)
            if np.any(vec != 0):
                vecs.append(vec)
                labels.append(label)
            else:
                failed += 1
        except Exception:
            failed += 1

    if failed:
        print(f"    ({failed} files skipped — parse failure or all-zero vector)")

    X = np.array(vecs,   dtype=np.float32) if vecs   else np.empty((0, len(FEATURE_NAMES)), np.float32)
    y = np.array(labels, dtype=np.int32)   if labels else np.empty(0, np.int32)
    return X, y


def print_metrics(name: str, y_true, y_pred) -> None:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1        : {f1:.4f}")


def print_lr_coefficients(lr_pipe: Pipeline) -> None:
    scaler = lr_pipe.named_steps["scaler"]
    lr     = lr_pipe.named_steps["lr"]
    coefs  = lr.coef_[0]

    print()
    print("  Feature Coefficients (positive = AI signal, negative = human signal)")
    print(f"  {'Feature':<25} {'Coef':>9}  Direction   Bar")
    print("  " + "-" * 65)
    order = np.argsort(np.abs(coefs))[::-1]
    for fi in order:
        coef      = float(coefs[fi])
        direction = "→ AI   " if coef > 0 else "→ human"
        fill      = "█" if coef > 0 else "░"
        bar       = fill * min(int(abs(coef) * 5), 24)
        print(f"  {FEATURE_NAMES[fi]:<25} {coef:>+9.4f}  {direction}   {bar}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── validate ─────────────────────────────────────────────────────────────
    for d in [AI_DIR, HUMAN_DIR]:
        if not os.path.isdir(d):
            print(f"ERROR: {d} not found.", file=sys.stderr)
            if d == HUMAN_DIR:
                print("       Run prepare_fma_training.py first.", file=sys.stderr)
            sys.exit(1)

    # ── extract features ─────────────────────────────────────────────────────
    SEP = "=" * 65

    print(SEP)
    print("FEATURE EXTRACTION")
    print(SEP)

    print(f"  AI    ({AI_DIR})")
    X_ai,    y_ai    = collect_features(AI_DIR,    label=1)
    print(f"    → {len(X_ai)} usable files")

    print(f"  Human ({HUMAN_DIR})")
    X_human, y_human = collect_features(HUMAN_DIR, label=0)
    print(f"    → {len(X_human)} usable files")

    if len(X_ai) == 0 or len(X_human) == 0:
        print("ERROR: need samples from both classes.", file=sys.stderr)
        sys.exit(1)

    X = np.vstack([X_ai, X_human])
    y = np.concatenate([y_ai, y_human])
    print(f"\n  Dataset : {len(X)} total  ({int(y.sum())} AI, {int((y == 0).sum())} human)")

    np.savez(NPZ_PATH, X=X, y=y, feature_names=np.array(FEATURE_NAMES))
    print(f"  Saved   : {NPZ_PATH}")

    # ── split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    print(f"\n  Train   : {len(X_train)}   Test : {len(X_test)}")

    # ── logistic regression ───────────────────────────────────────────────────
    print()
    print(SEP)
    print("LOGISTIC REGRESSION")
    print(SEP)

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
    ])
    lr_pipe.fit(X_train, y_train)
    print("  Test-set metrics:")
    print_metrics("lr", y_test, lr_pipe.predict(X_test))
    print_lr_coefficients(lr_pipe)

    # ── random forest ─────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("RANDOM FOREST (200 trees)")
    print(SEP)

    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)),
    ])
    rf_pipe.fit(X_train, y_train)
    print("  Test-set metrics:")
    print_metrics("rf", y_test, rf_pipe.predict(X_test))

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(CLF_DIR, exist_ok=True)
    with open(LR_PATH, "wb") as fh:
        pickle.dump(lr_pipe, fh)
    with open(RF_PATH, "wb") as fh:
        pickle.dump(rf_pipe, fh)

    print()
    print(SEP)
    print(f"  Saved : {LR_PATH}")
    print(f"  Saved : {RF_PATH}")
    print(SEP)
    print("\nDone. Restart app.py to pick up the new models automatically.")

    # ── update model registry ─────────────────────────────────────────────────
    _update_registry(CLF_DIR, len(X_ai), len(X_human), X_train, X_test, y_train, y_test, lr_pipe, rf_pipe)


def _update_registry(clf_dir, n_ai, n_human, X_train, X_test, y_train, y_test, lr_pipe, rf_pipe):
    from sklearn.metrics import accuracy_score, f1_score
    registry_path = os.path.join(clf_dir, "model_registry.json")
    if not os.path.isfile(registry_path):
        return
    with open(registry_path) as fh:
        reg = json.load(fh)
    today = datetime.date.today().isoformat()
    for key, pipe in [("structural_logreg_audio", lr_pipe), ("structural_rf_audio", rf_pipe)]:
        if key in reg["models"]:
            y_pred = pipe.predict(X_test)
            reg["models"][key]["ai_samples"] = int(n_ai)
            reg["models"][key]["human_samples"] = int(n_human)
            reg["models"][key]["total_training_samples"] = int(n_ai + n_human)
            reg["models"][key]["reported_test_accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            reg["models"][key]["reported_test_f1"] = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
            reg["models"][key]["last_trained"] = today
    with open(registry_path, "w") as fh:
        json.dump(reg, fh, indent=2)


if __name__ == "__main__":
    main()
