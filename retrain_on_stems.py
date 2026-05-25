"""
retrain_on_stems.py — train 28-feature structural classifiers on stem-transcribed AI MIDIs.

Run after transcribe_stems.py has rebuilt data/ai_stems/ with GM instrument labels:
    python retrain_on_stems.py

Reads:
    data/ai_stems/    — per-stem Basic Pitch transcriptions of Suno audio  (label 1)
    data/human/midi/  — 200 sampled Lakh MIDI files                        (label 0)

Writes:
    data/features_stems.npz
    classifier/structural_logreg_stems_v2.pkl    — LR with L2 regularisation (28 features)
    classifier/structural_logreg_stems_lasso.pkl — LR with L1 regularisation (28 features, sparse)

28 features = 14 melody-only (mel_*) + 14 all-notes (all_*).
Melody part = vocal part by name/instrument, else highest-mean-pitch part.

Both models use class_weight='balanced' to compensate for ~1:5 AI:human imbalance.
Random Forest is intentionally omitted: with only ~42 AI samples it reaches perfect
precision but only 33% recall, which is not useful for reliable detection.

Prints:
  • Accuracy / precision / recall / F1 for both models
  • Full feature coefficient tables (mel/all annotated, zeroed features listed)
  • Side-by-side non-zero L1 vs L2 comparison showing which features survive L1
"""

import datetime
import json
import os
import pickle
import sys

import numpy as np
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler

from extract_features import FEATURE_NAMES_28, N_FEATURES_28, extract_features_28

# ── paths ────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(__file__)
AI_DIR      = os.path.join(_HERE, "data", "ai_stems")
HUMAN_DIR   = os.path.join(_HERE, "data", "human", "midi")
NPZ_PATH    = os.path.join(_HERE, "data", "features_stems.npz")
CLF_DIR     = os.path.join(_HERE, "classifier")
LR_V2_PATH  = os.path.join(CLF_DIR, "structural_logreg_stems_v2.pkl")
LASSO_PATH  = os.path.join(CLF_DIR, "structural_logreg_stems_lasso.pkl")

# ── config ───────────────────────────────────────────────────────────────────
SEED      = 42
TEST_SIZE = 0.2
N_MEL     = 14  # first 14 of 28 features are melody-only


# ── helpers ──────────────────────────────────────────────────────────────────

def feature_type(i: int) -> str:
    return "mel" if i < N_MEL else "all"


def collect_features(directory: str, label: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract 28 features from every .mid in directory; return (X, y)."""
    mid_files = sorted(f for f in os.listdir(directory) if f.lower().endswith(".mid"))
    vecs, labels, failed = [], [], 0

    for fname in mid_files:
        path = os.path.join(directory, fname)
        try:
            vec = extract_features_28(path)
            if np.any(vec != 0):
                vecs.append(vec)
                labels.append(label)
            else:
                failed += 1
        except Exception:
            failed += 1

    if failed:
        print(f"    ({failed} files skipped — parse failure or all-zero vector)")

    X = np.array(vecs,   dtype=np.float32) if vecs   else np.empty((0, N_FEATURES_28), np.float32)
    y = np.array(labels, dtype=np.int32)   if labels else np.empty(0, np.int32)
    return X, y


def print_metrics(y_true, y_pred) -> None:
    print(f"    Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"    Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    F1        : {f1_score(y_true, y_pred, zero_division=0):.4f}")


def print_lr_coefficients(lr_pipe: Pipeline, label: str) -> None:
    coefs = lr_pipe.named_steps["lr"].coef_[0]
    print()
    print(f"  {label} — Feature Coefficients (positive = AI signal, negative = human signal)")
    print(f"  {'Feature':<30} {'Type':<6} {'Coef':>9}  Direction   Bar")
    print("  " + "-" * 72)
    for fi in np.argsort(np.abs(coefs))[::-1]:
        coef = float(coefs[fi])
        if coef == 0.0:
            continue
        direction = "→ AI   " if coef > 0 else "→ human"
        fill      = "█" if coef > 0 else "░"
        bar       = fill * min(int(abs(coef) * 5), 24)
        print(f"  {FEATURE_NAMES_28[fi]:<30} {feature_type(fi):<6} {coef:>+9.4f}  {direction}   {bar}")
    zeroed = [FEATURE_NAMES_28[fi] for fi in range(N_FEATURES_28) if coefs[fi] == 0.0]
    if zeroed:
        print(f"\n  Zeroed ({len(zeroed)} features):")
        for i in range(0, len(zeroed), 4):
            print("    " + "  ".join(zeroed[i:i + 4]))


def print_l1_comparison(lr_v2_pipe: Pipeline, lasso_pipe: Pipeline) -> None:
    """Side-by-side: non-zero L1 coefs vs their L2 equivalents."""
    c_l2 = lr_v2_pipe.named_steps["lr"].coef_[0]
    c_l1 = lasso_pipe.named_steps["lr"].coef_[0]

    surviving = sorted(
        [i for i in range(N_FEATURES_28) if c_l1[i] != 0.0],
        key=lambda i: abs(c_l1[i]), reverse=True,
    )
    zeroed = [FEATURE_NAMES_28[i] for i in range(N_FEATURES_28) if c_l1[i] == 0.0]

    print()
    print(f"  {'Feature':<30} {'Type':<6} {'LR v2 (L2)':>12} {'Lasso (L1)':>12}  Direction")
    print("  " + "-" * 72)
    for fi in surviving:
        l2   = float(c_l2[fi])
        l1   = float(c_l1[fi])
        dir_ = "→ AI   " if l1 > 0 else "→ human"
        print(f"  {FEATURE_NAMES_28[fi]:<30} {feature_type(fi):<6} {l2:>+12.4f} {l1:>+12.4f}  {dir_}")

    print(f"\n  Zeroed out by L1 ({len(zeroed)} / {N_FEATURES_28} features):")
    for i in range(0, len(zeroed), 4):
        print("    " + "  ".join(zeroed[i:i + 4]))


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    for label, d in [("AI stems", AI_DIR), ("human MIDI", HUMAN_DIR)]:
        if not os.path.isdir(d):
            print(f"ERROR: {d} not found.", file=sys.stderr)
            if label == "AI stems":
                print("       Run transcribe_stems.py first.", file=sys.stderr)
            sys.exit(1)

    SEP = "=" * 68
    print(SEP)
    print("NOTE: AI training data produced by Basic Pitch on pre-separated Suno stems.")
    print("      Percussion stems excluded. GM instruments assigned per stem name.")
    print("      28 features: 14 melody-only (mel_) + 14 all-notes (all_).")
    print("      class_weight='balanced' compensates for class imbalance.")
    print(SEP)
    print()

    # ── feature extraction ────────────────────────────────────────────────────
    print(SEP)
    print("FEATURE EXTRACTION (28 features per file)")
    print(SEP)
    print(f"  AI stems ({AI_DIR})")
    X_ai, y_ai = collect_features(AI_DIR, label=1)
    print(f"    → {len(X_ai)} usable files")

    print(f"  Human ({HUMAN_DIR})")
    X_human, y_human = collect_features(HUMAN_DIR, label=0)
    print(f"    → {len(X_human)} usable files")

    if len(X_ai) == 0 or len(X_human) == 0:
        print("ERROR: need samples from both classes.", file=sys.stderr)
        sys.exit(1)

    X = np.vstack([X_ai, X_human])
    y = np.concatenate([y_ai, y_human])

    # ── dataset summary ───────────────────────────────────────────────────────
    n_ai    = int(y.sum())
    n_human = int((y == 0).sum())
    ratio   = n_human / n_ai if n_ai > 0 else float("inf")
    print()
    print(SEP)
    print("DATASET SUMMARY")
    print(SEP)
    print(f"  AI samples    : {n_ai}  (data/ai_stems/)")
    print(f"  Human samples : {n_human}  (data/human/midi/)")
    print(f"  Total         : {n_ai + n_human}")
    print(f"  Balance ratio : {ratio:.1f}:1 human:AI  ← imbalanced")
    print(f"  Remedy        : class_weight='balanced' (AI weight = {ratio:.1f}x)")
    print(f"  Features      : {N_FEATURES_28}  ({N_MEL} melody-only + {N_FEATURES_28 - N_MEL} all-notes)")
    np.savez(NPZ_PATH, X=X, y=y, feature_names=np.array(FEATURE_NAMES_28))
    print(f"  Saved         : {NPZ_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    print(f"\n  Train : {len(X_train)}   Test : {len(X_test)}")
    print(f"  Test  : {int(y_test.sum())} AI, {int((y_test == 0).sum())} human")

    # ── LR v2  (L2 regularisation) ────────────────────────────────────────────
    print()
    print(SEP)
    print("LOGISTIC REGRESSION v2  (L2, class_weight='balanced', 28 features)")
    print(SEP)

    lr_v2_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
            C=1.0, max_iter=1000, random_state=SEED, class_weight="balanced",
        )),
    ])
    lr_v2_pipe.fit(X_train, y_train)
    print("  Test-set metrics:")
    print_metrics(y_test, lr_v2_pipe.predict(X_test))
    print_lr_coefficients(lr_v2_pipe, "LR v2 (L2)")

    # ── LR lasso  (L1 regularisation) ────────────────────────────────────────
    print()
    print(SEP)
    print("LOGISTIC REGRESSION LASSO  (L1, liblinear, class_weight='balanced', 28 features)")
    print(SEP)

    lasso_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(
            C=1.0, penalty="l1", solver="liblinear",
            max_iter=1000, random_state=SEED, class_weight="balanced",
        )),
    ])
    lasso_pipe.fit(X_train, y_train)
    print("  Test-set metrics:")
    print_metrics(y_test, lasso_pipe.predict(X_test))
    print_lr_coefficients(lasso_pipe, "LR Lasso (L1)")

    # ── L1 vs L2 comparison ───────────────────────────────────────────────────
    print()
    print(SEP)
    print("FEATURES SURVIVING L1 REGULARISATION")
    print("(non-zero L1 coefs vs L2 equivalents — these are the most robust signals)")
    print(SEP)
    print_l1_comparison(lr_v2_pipe, lasso_pipe)

    # ── save ──────────────────────────────────────────────────────────────────
    os.makedirs(CLF_DIR, exist_ok=True)
    with open(LR_V2_PATH, "wb") as fh:
        pickle.dump(lr_v2_pipe, fh)
    with open(LASSO_PATH, "wb") as fh:
        pickle.dump(lasso_pipe, fh)

    print()
    print(SEP)
    print(f"  Saved : {LR_V2_PATH}")
    print(f"  Saved : {LASSO_PATH}")
    print(SEP)
    print("\nDone.")

    _update_registry(CLF_DIR, n_ai, n_human, X_test, y_test, lr_v2_pipe, lasso_pipe)


def _update_registry(clf_dir, n_ai, n_human, X_test, y_test, lr_v2_pipe, lasso_pipe):
    registry_path = os.path.join(clf_dir, "model_registry.json")
    if not os.path.isfile(registry_path):
        return
    with open(registry_path) as fh:
        reg = json.load(fh)
    today = datetime.date.today().isoformat()
    for key, pipe in [
        ("structural_logreg_stems_v2",    lr_v2_pipe),
        ("structural_logreg_stems_lasso", lasso_pipe),
    ]:
        if key in reg["models"]:
            y_pred = pipe.predict(X_test)
            reg["models"][key]["ai_samples"]             = int(n_ai)
            reg["models"][key]["human_samples"]          = int(n_human)
            reg["models"][key]["total_training_samples"] = int(n_ai + n_human)
            reg["models"][key]["reported_test_accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
            reg["models"][key]["reported_test_f1"]       = round(float(f1_score(y_test, y_pred, zero_division=0)), 4)
            reg["models"][key]["last_trained"]           = today
    with open(registry_path, "w") as fh:
        json.dump(reg, fh, indent=2)


if __name__ == "__main__":
    main()
