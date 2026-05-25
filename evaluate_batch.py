"""
evaluate_batch.py — batch evaluation of all three classifiers on a folder of MIDI files.

Usage:
    python evaluate_batch.py <midi_folder> <label>

    label: 0 = human, 1 = AI

Examples:
    python evaluate_batch.py data/human/midi/ 0
    python evaluate_batch.py data/ai/ 1
    python evaluate_batch.py data/human_audio/ 0

Outputs:
    • Per-file scores printed to stdout
    • Confusion matrix per classifier
    • Disagreement flags for files where classifier range > 0.25
    • evaluation/batch_results.csv

The CSV has one row per file:
    filename, true_label, cnn, logreg, rf, mean_score, range, disagree
"""

import csv
import os
import sys

import numpy as np

from predict import classify_midi
from predict_structural import classify_structural
from app import _classify_with_structural_models

DISAGREE_THRESHOLD = 0.25
OUT_DIR  = os.path.join(os.path.dirname(__file__), "evaluation")
CSV_PATH = os.path.join(OUT_DIR, "batch_results.csv")


def run_classifiers(midi_path: str) -> dict:
    """Run all three classifiers. Returns None for any unavailable model."""
    result = {"cnn": None, "logreg": None, "rf": None}

    try:
        result["cnn"] = classify_midi(midi_path)
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"      CNN error: {exc}", file=sys.stderr)

    try:
        lr, rf, lasso, _ = _classify_with_structural_models(midi_path)
        result["logreg"] = lr
        result["rf"]     = rf if rf is not None else lasso
    except FileNotFoundError:
        try:
            lr, rf = classify_structural(midi_path)
            result["logreg"] = lr
            result["rf"]     = rf
        except FileNotFoundError:
            pass
    except Exception as exc:
        print(f"      Structural error: {exc}", file=sys.stderr)

    return result


def confusion_stats(y_true: list[int], y_pred: list[float | None], threshold: float = 0.5):
    """
    Compute TP, FP, TN, FN from binary true labels and continuous predictions.
    Skips entries where y_pred is None.
    """
    tp = fp = tn = fn = skipped = 0
    for yt, yp in zip(y_true, y_pred):
        if yp is None:
            skipped += 1
            continue
        pred = 1 if yp >= threshold else 0
        if yt == 1 and pred == 1: tp += 1
        elif yt == 0 and pred == 1: fp += 1
        elif yt == 0 and pred == 0: tn += 1
        else:                       fn += 1
    return tp, fp, tn, fn, skipped


def print_confusion(name: str, tp: int, fp: int, tn: int, fn: int, skipped: int) -> None:
    total = tp + fp + tn + fn
    if total == 0:
        print(f"  {name}: no predictions available")
        return
    acc   = (tp + tn) / total
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    print(f"\n  {name}")
    print(f"    {'':12} {'Pred AI':>8} {'Pred Human':>10}")
    print(f"    {'True AI':12} {tp:>8} {fn:>10}")
    print(f"    {'True Human':12} {fp:>8} {tn:>10}")
    print(f"    Accuracy: {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")
    if skipped:
        print(f"    ({skipped} files skipped — model unavailable or error)")


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python evaluate_batch.py <midi_folder> <label>", file=sys.stderr)
        print("       label: 0 = human, 1 = AI", file=sys.stderr)
        sys.exit(1)

    folder     = sys.argv[1]
    true_label = int(sys.argv[2])
    if true_label not in (0, 1):
        print("ERROR: label must be 0 (human) or 1 (AI)", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(folder):
        print(f"ERROR: folder not found: {folder}", file=sys.stderr)
        sys.exit(1)

    mid_files = sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith((".mid", ".midi"))
    )
    if not mid_files:
        print(f"No MIDI files found in {folder}", file=sys.stderr)
        sys.exit(1)

    label_str = "AI" if true_label == 1 else "human"
    SEP = "=" * 70

    print(SEP)
    print(f"BATCH EVALUATION — {len(mid_files)} files — true label: {label_str}")
    print(f"Folder: {folder}")
    print(SEP)

    rows: list[dict] = []
    cnn_preds:    list[float | None] = []
    logreg_preds: list[float | None] = []
    rf_preds:     list[float | None] = []
    true_labels:  list[int]          = []
    disagree_files: list[str]        = []

    for i, fname in enumerate(mid_files, 1):
        midi_path = os.path.join(folder, fname)
        scores    = run_classifiers(midi_path)

        available = [v for v in scores.values() if v is not None]
        mean_score = float(np.mean(available)) if available else None
        score_range = (max(available) - min(available)) if len(available) >= 2 else None
        disagree = score_range is not None and score_range > DISAGREE_THRESHOLD

        tag = " *** DISAGREE" if disagree else ""
        cnn_str    = f"{scores['cnn']:.3f}"    if scores['cnn']    is not None else "  N/A"
        logreg_str = f"{scores['logreg']:.3f}" if scores['logreg'] is not None else "  N/A"
        rf_str     = f"{scores['rf']:.3f}"     if scores['rf']     is not None else "  N/A"
        mean_str   = f"{mean_score:.3f}"        if mean_score       is not None else "  N/A"

        print(f"  [{i:>4}/{len(mid_files)}] {fname:<40}  "
              f"CNN={cnn_str}  LR={logreg_str}  RF={rf_str}  "
              f"mean={mean_str}{tag}")

        if disagree:
            disagree_files.append(fname)

        cnn_preds.append(scores["cnn"])
        logreg_preds.append(scores["logreg"])
        rf_preds.append(scores["rf"])
        true_labels.append(true_label)

        rows.append({
            "filename":   fname,
            "true_label": true_label,
            "cnn":        scores["cnn"]    if scores["cnn"]    is not None else "",
            "logreg":     scores["logreg"] if scores["logreg"] is not None else "",
            "rf":         scores["rf"]     if scores["rf"]     is not None else "",
            "mean_score": mean_score if mean_score is not None else "",
            "range":      round(score_range, 4) if score_range is not None else "",
            "disagree":   int(disagree),
        })

    # ── confusion matrices ───────────────────────────────────────────────────
    print()
    print(SEP)
    print("CONFUSION MATRICES (threshold = 0.5)")
    print(SEP)
    for name, preds in [("CNN", cnn_preds), ("Logistic Regression", logreg_preds), ("Random Forest", rf_preds)]:
        tp, fp, tn, fn, sk = confusion_stats(true_labels, preds)
        print_confusion(name, tp, fp, tn, fn, sk)

    # ── disagreement summary ─────────────────────────────────────────────────
    print()
    print(SEP)
    print(f"CLASSIFIER DISAGREEMENT (range > {DISAGREE_THRESHOLD})")
    print(SEP)
    if disagree_files:
        print(f"  {len(disagree_files)} / {len(mid_files)} files flagged:\n")
        for f in disagree_files:
            print(f"    {f}")
    else:
        print(f"  No files flagged — all classifiers agreed within {DISAGREE_THRESHOLD} range.")

    # ── save CSV ─────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    fieldnames = ["filename", "true_label", "cnn", "logreg", "rf", "mean_score", "range", "disagree"]
    with open(CSV_PATH, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(SEP)
    print(f"  Results saved → {CSV_PATH}")
    print(f"  Files evaluated: {len(rows)}")
    print(f"  Disagreements:   {len(disagree_files)}")
    print(SEP)


if __name__ == "__main__":
    main()
