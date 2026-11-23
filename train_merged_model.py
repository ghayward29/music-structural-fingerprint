"""
train_merged_model.py — Train the StructuralPrint v6 merged 38-feature model.

Combines two orthogonal feature views of the same 964 SONICS songs:
  * 28 note-level features   (data/features_sonics.npz / Downloads copy)
  * 10 macro-structural audio features (experiments/{real,fake}_features_v3.csv)

The merged 38-feature HistGradientBoosting model reaches ~0.82 accuracy /
0.91 AUC in 5-fold CV — a real jump over the shipped note-level-only model
(~0.73 / 0.80). This script:

  1. Builds the merged dataframe.
  2. Reports 5-fold CV for note-only / macro-only / merged.
  3. Fits a held-out 80/20 split and records honest out-of-sample verdicts
     for every song (used by the demo gallery so nothing is shown in-sample).
  4. Fits a final model on ALL data and pickles it for inference.
  5. Writes classifier/merged38_meta.json with metrics + feature metadata.

Run:  python train_merged_model.py
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
NPZ_CANDIDATES = [
    ROOT / "data" / "features_sonics.npz",
    Path.home() / "Downloads" / "features_sonics.npz",
]

NOTE_NAMES = (
    ["mel_interval_mean", "mel_interval_std", "mel_interval_unison", "mel_interval_step",
     "mel_interval_skip", "mel_interval_leap", "mel_interval_large_leap", "mel_harm_rhythm_mean",
     "mel_harm_rhythm_std", "mel_note_density", "mel_rhythmic_cv", "mel_pitch_range",
     "mel_stepwise_ratio", "mel_cadence_count"]
    + [n.replace("mel_", "all_", 1) for n in
       ["mel_interval_mean", "mel_interval_std", "mel_interval_unison", "mel_interval_step",
        "mel_interval_skip", "mel_interval_leap", "mel_interval_large_leap", "mel_harm_rhythm_mean",
        "mel_harm_rhythm_std", "mel_note_density", "mel_rhythmic_cv", "mel_pitch_range",
        "mel_stepwise_ratio", "mel_cadence_count"]]
)

MACRO_NAMES = [
    "chroma_ssm_symmetry_index", "chroma_ssm_novelty_decay", "chroma_ssm_avg_similarity",
    "chroma_ssm_diagonal_dominance", "chord_compression_ratio", "phrase_repetition_fidelity",
    "chorus_edit_distance_audio", "beat_periodicity_entropy", "detected_harmonic_cadences",
    "tempo_stability_variance",
]

ALL_NAMES = NOTE_NAMES + MACRO_NAMES


def load_merged() -> pd.DataFrame:
    npz_path = next((p for p in NPZ_CANDIDATES if p.exists()), None)
    if npz_path is None:
        raise FileNotFoundError(
            "features_sonics.npz not found in data/ or ~/Downloads. "
            "It holds the 28 note-level features for the 964 SONICS songs."
        )
    d = np.load(npz_path, allow_pickle=True)
    note = pd.DataFrame(d["X"], columns=NOTE_NAMES)
    note["filename"] = d["filenames"]
    note["label"] = d["y"].astype(int)

    macro = pd.concat(
        [pd.read_csv(ROOT / "experiments" / "real_features_v3.csv"),
         pd.read_csv(ROOT / "experiments" / "fake_features_v3.csv")],
        ignore_index=True,
    )
    merged = note.merge(macro.drop(columns=["label"]), on="filename", how="inner")
    return merged


def clean_X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    return X.values


def make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(random_state=42, max_iter=400)


def cv_report(m: pd.DataFrame) -> dict:
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    y = m["label"].values
    out = {}
    for tag, cols in [("note28", NOTE_NAMES), ("macro10", MACRO_NAMES), ("merged38", ALL_NAMES)]:
        X = clean_X(m, cols)
        res = {}
        for name, mdl in [
            ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
            ("hgb", make_hgb()),
        ]:
            acc = cross_val_score(mdl, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
            auc = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
            res[name] = {"acc": round(float(acc.mean()), 4), "acc_std": round(float(acc.std()), 4),
                         "auc": round(float(auc.mean()), 4)}
        out[tag] = res
        print(f"  {tag:10s} logreg acc={out[tag]['logreg']['acc']:.3f} auc={out[tag]['logreg']['auc']:.3f} | "
              f"hgb acc={out[tag]['hgb']['acc']:.3f} auc={out[tag]['hgb']['auc']:.3f}")
    return out


def main() -> None:
    m = load_merged()
    print(f"Merged dataset: {len(m)} songs  balance={m['label'].value_counts().to_dict()}")

    print("\n5-fold cross-validation:")
    cv = cv_report(m)

    # ---- Honest held-out split: out-of-sample verdict for every song ----
    idx = np.arange(len(m))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=m["label"].values, random_state=42)
    Xall = clean_X(m, ALL_NAMES)
    y = m["label"].values

    heldout = make_hgb().fit(Xall[tr_idx], y[tr_idx])
    te_prob = heldout.predict_proba(Xall[te_idx])[:, 1]
    te_pred = (te_prob >= 0.5).astype(int)
    ho_acc = accuracy_score(y[te_idx], te_pred)
    ho_auc = roc_auc_score(y[te_idx], te_prob)
    cm = confusion_matrix(y[te_idx], te_pred).tolist()  # [[TN,FP],[FN,TP]]
    print(f"\nHeld-out (20%, n={len(te_idx)}): acc={ho_acc:.3f} auc={ho_auc:.3f} confusion(TN,FP/FN,TP)={cm}")

    # Record out-of-sample probability for every TEST song (safe to display).
    oos = {m["filename"].iloc[i]: round(float(p), 4) for i, p in zip(te_idx, te_prob)}

    # ---- Final model on ALL data for live inference ----
    final = make_hgb().fit(Xall, y)
    model_path = ROOT / "classifier" / "structural_merged38.pkl"
    with open(model_path, "wb") as fh:
        pickle.dump({"model": final, "feature_names": ALL_NAMES,
                     "note_names": NOTE_NAMES, "macro_names": MACRO_NAMES,
                     "median": pd.DataFrame(Xall, columns=ALL_NAMES).median().to_dict()}, fh)
    print(f"\nSaved final model -> {model_path}")

    # ---- Class means (for per-song z-score vs class) ----
    ai = m[m.label == 1]
    hu = m[m.label == 0]
    feat_stats = {}
    for name in ALL_NAMES:
        col = m[name].replace([np.inf, -np.inf], np.nan)
        feat_stats[name] = {
            "ai_mean": round(float(ai[name].replace([np.inf, -np.inf], np.nan).mean()), 4),
            "human_mean": round(float(hu[name].replace([np.inf, -np.inf], np.nan).mean()), 4),
            "std": round(float(col.std()), 4) or 1.0,
            "overall_mean": round(float(col.mean()), 4),
        }

    meta = {
        "version": "v6-merged38",
        "n_songs": int(len(m)),
        "n_ai": int((m.label == 1).sum()),
        "n_human": int((m.label == 0).sum()),
        "cv": cv,
        "heldout": {"n": int(len(te_idx)), "acc": round(float(ho_acc), 4),
                    "auc": round(float(ho_auc), 4), "confusion": cm},
        "note_names": NOTE_NAMES,
        "macro_names": MACRO_NAMES,
        "all_names": ALL_NAMES,
        "feature_stats": feat_stats,
        "test_indices": te_idx.tolist(),
        "oos_prob": oos,
    }
    meta_path = ROOT / "classifier" / "merged38_meta.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved metadata     -> {meta_path}")


if __name__ == "__main__":
    main()
