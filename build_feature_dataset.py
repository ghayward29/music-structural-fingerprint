"""
build_feature_dataset.py — extract structural features from all training MIDIs
and save a labelled feature matrix to data/features.npz.

Labels: 0 = human-composed (data/human/midi/), 1 = AI-generated (data/ai/)

Also prints a side-by-side mean comparison of every feature for AI vs human
so you can immediately see which features differ most between classes.

Usage:
    python build_feature_dataset.py
"""

import glob
import os
import sys

import numpy as np

from extract_features import extract_features, FEATURE_NAMES

AI_DIR    = os.path.join("data", "ai")
HUMAN_DIR = os.path.join("data", "human", "midi")
OUT_PATH  = os.path.join("data", "features.npz")


def collect_midis(folder: str) -> list[str]:
    """Return sorted list of .mid files directly inside folder (not recursive).
    We intentionally stay non-recursive here because data/ai/ and data/human/midi/
    are both flat directories created by earlier pipeline steps."""
    return sorted(
        p for p in glob.glob(os.path.join(folder, "*.mid"))
    )


def build_dataset() -> None:
    ai_files    = collect_midis(AI_DIR)
    human_files = collect_midis(HUMAN_DIR)

    if not ai_files:
        print(f"No .mid files found in {AI_DIR}/", file=sys.stderr)
        sys.exit(1)
    if not human_files:
        print(f"No .mid files found in {HUMAN_DIR}/ — run sample_human.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"AI files    : {len(ai_files)}")
    print(f"Human files : {len(human_files)}")
    print(f"Extracting features (this may take a few minutes)…\n")

    rows:   list[np.ndarray] = []
    labels: list[int]        = []

    for label, files, tag in [(1, ai_files, "AI"), (0, human_files, "Human")]:
        ok = fail = 0
        for path in files:
            vec = extract_features(path)
            rows.append(vec)
            labels.append(label)
            # A zero vector means extraction failed — count it but warn.
            if np.all(vec == 0):
                print(f"  ✗  [{tag}] {os.path.basename(path)} — zero vector (parse failed)",
                      file=sys.stderr)
                fail += 1
            else:
                ok += 1
        print(f"[{tag}] {ok} succeeded, {fail} failed")

    X = np.stack(rows)           # shape (N, 14)
    y = np.array(labels)         # shape (N,)

    # Save — npz lets us store X, y, and feature names in one file.
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez(OUT_PATH, X=X, y=y, feature_names=np.array(FEATURE_NAMES))
    print(f"\nSaved → {OUT_PATH}  (shape {X.shape})")

    # -------------------------------------------------------------------------
    # Side-by-side mean comparison — shows which features differ most
    # -------------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Feature':<22}  {'Human mean':>12}  {'AI mean':>10}  {'Δ (AI-Human)':>13}")
    print("-" * 72)

    human_X = X[y == 0]
    ai_X    = X[y == 1]

    for i, name in enumerate(FEATURE_NAMES):
        h_mean = float(np.mean(human_X[:, i]))
        a_mean = float(np.mean(ai_X[:, i]))
        delta  = a_mean - h_mean
        arrow  = "↑" if delta > 0 else "↓"
        print(f"  {name:<22}  {h_mean:>12.4f}  {a_mean:>10.4f}  {arrow} {abs(delta):>10.4f}")

    print("=" * 72)
    print("\nPositive Δ means AI scores higher; negative Δ means human scores higher.")


if __name__ == "__main__":
    build_dataset()
