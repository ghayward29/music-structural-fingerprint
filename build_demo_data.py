"""
build_demo_data.py — Precompute the StructuralPrint demo gallery.

Reads the trained merged-38 model metadata and the merged feature table,
then writes demo/gallery.json containing everything the static demo UI needs:

  * headline finding + model comparison (note / macro / merged)
  * held-out metrics + confusion matrix
  * a curated gallery of OUT-OF-SAMPLE songs (from the held-out 20% test set,
    so every displayed verdict is honest) with:
      - human-readable source label (Suno / Udio / Human) + optional YouTube link
      - ground-truth label, model AI-probability (out-of-sample)
      - per-feature values and Gaussian per-feature "evidence" (which structural
        traits made it look AI vs human) for a faithful, model-agnostic "why"
  * feature dictionary (plain-English names, grouping, class means)

Everything is precomputed so the demo loads instantly and never needs the
heavy audio pipeline (Demucs / Basic Pitch) at runtime.

Run:  python build_demo_data.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from train_merged_model import load_merged, NOTE_NAMES, MACRO_NAMES, ALL_NAMES

ROOT = Path(__file__).resolve().parent

# ---- plain-English feature dictionary ----
BASE_PLAIN = {
    "interval_mean": "Average melodic interval size",
    "interval_std": "Spread of interval sizes",
    "interval_unison": "Repeated pitches",
    "interval_step": "Stepwise motion (1-2 semitones)",
    "interval_skip": "Skips (3-4 semitones)",
    "interval_leap": "Leaps (5-7 semitones)",
    "interval_large_leap": "Large leaps (8+ semitones)",
    "harm_rhythm_mean": "Rate of harmonic change",
    "harm_rhythm_std": "Consistency of harmonic rhythm",
    "note_density": "Notes per second",
    "rhythmic_cv": "Variability in note timing",
    "pitch_range": "Lowest-to-highest pitch span",
    "stepwise_ratio": "Share of smooth, stepwise motion",
    "cadence_count": "Number of phrase-ending cadences",
}
MACRO_PLAIN = {
    "chroma_ssm_symmetry_index": "Self-similarity matrix symmetry",
    "chroma_ssm_novelty_decay": "How fast musical novelty fades",
    "chroma_ssm_avg_similarity": "Overall internal repetition",
    "chroma_ssm_diagonal_dominance": "Strength of repeated sections",
    "chord_compression_ratio": "How compressible the chord sequence is",
    "phrase_repetition_fidelity": "How exactly phrases repeat",
    "chorus_edit_distance_audio": "Chorus-to-chorus variation",
    "beat_periodicity_entropy": "Rhythmic-grid regularity",
    "detected_harmonic_cadences": "Textbook harmonic cadences found",
    "tempo_stability_variance": "Tempo drift over the song",
}


def plain(name: str) -> str:
    if name in MACRO_PLAIN:
        return MACRO_PLAIN[name]
    for pre, tag in (("mel_", "Melody: "), ("all_", "All notes: ")):
        if name.startswith(pre):
            base = name[len(pre):]
            if base in BASE_PLAIN:
                return tag + BASE_PLAIN[base][0].lower() + BASE_PLAIN[base][1:]
    return name


def group_of(name: str) -> str:
    if name in MACRO_NAMES:
        return "macro"
    return "note"


def source_label(fn: str) -> dict:
    """Human-readable source + optional playback link."""
    low = fn.lower()
    if "suno" in low:
        return {"kind": "ai", "name": "Suno", "sub": "AI generator", "link": None}
    if "udio" in low:
        return {"kind": "ai", "name": "Udio", "sub": "AI generator", "link": None}
    # human = YouTube ID
    return {"kind": "human", "name": "Human song",
            "sub": "commercial recording",
            "link": f"https://www.youtube.com/watch?v={fn}"}


def main() -> None:
    meta = json.loads((ROOT / "classifier" / "merged38_meta.json").read_text())
    m = load_merged().reset_index(drop=True)

    fstats = meta["feature_stats"]
    oos = meta["oos_prob"]  # filename -> out-of-sample prob (test songs only)

    def evidence(row) -> list[dict]:
        """Per-feature Gaussian evidence: positive => looks more AI.
        evidence_i = |z to human mean| - |z to AI mean|  (in std units)."""
        items = []
        for name in ALL_NAMES:
            v = row[name]
            if v is None or (isinstance(v, float) and (math.isinf(v) or math.isnan(v))):
                continue
            st = fstats[name]
            sd = st["std"] or 1.0
            z_h = abs((v - st["human_mean"]) / sd)
            z_a = abs((v - st["ai_mean"]) / sd)
            items.append({
                "feature": name, "plain": plain(name), "group": group_of(name),
                "value": round(float(v), 3),
                "ai_mean": st["ai_mean"], "human_mean": st["human_mean"],
                "evidence": round(float(z_h - z_a), 3),  # + => AI-like
            })
        items.sort(key=lambda x: abs(x["evidence"]), reverse=True)
        return items

    # ---- Build gallery from the held-out TEST set (honest, out-of-sample) ----
    test_rows = m[m["filename"].isin(oos.keys())].copy()
    test_rows["prob"] = test_rows["filename"].map(oos)
    test_rows["pred"] = (test_rows["prob"] >= 0.5).astype(int)
    test_rows["correct"] = test_rows["pred"] == test_rows["label"]

    def pick(df, n):
        return df["filename"].tolist()[:n]

    ai_conf = test_rows[(test_rows.label == 1) & test_rows.correct].sort_values("prob", ascending=False)
    hu_conf = test_rows[(test_rows.label == 0) & test_rows.correct].sort_values("prob", ascending=True)
    false_pos = test_rows[(test_rows.label == 0) & ~test_rows.correct].sort_values("prob", ascending=False)
    false_neg = test_rows[(test_rows.label == 1) & ~test_rows.correct].sort_values("prob", ascending=True)
    borderline = test_rows.assign(d=(test_rows.prob - 0.5).abs()).sort_values("d")

    chosen: list[str] = []
    chosen += pick(ai_conf, 5)
    chosen += pick(hu_conf, 5)
    chosen += pick(false_pos, 1)
    chosen += pick(false_neg, 1)
    for fn in borderline["filename"].tolist():
        if fn not in chosen:
            chosen.append(fn)
        if len(chosen) >= 16:
            break

    gallery = []
    for fn in chosen:
        row = test_rows[test_rows.filename == fn].iloc[0]
        ev = evidence(row)
        src = source_label(fn)
        gallery.append({
            "id": fn,
            "source": src,
            "truth": int(row["label"]),          # 1 = AI
            "prob": round(float(row["prob"]), 4),  # model P(AI), out-of-sample
            "pred": int(row["pred"]),
            "correct": bool(row["correct"]),
            "top_ai": [e for e in ev if e["evidence"] > 0][:4],
            "top_human": [e for e in ev if e["evidence"] < 0][:4],
            "features": {e["feature"]: e for e in ev},
        })

    out = {
        "finding": {
            "headline": "At the note level, AI and human music are nearly identical. "
                        "Add long-range structure and the gap reappears.",
            "note_acc": meta["cv"]["note28"]["hgb"]["acc"],
            "note_auc": meta["cv"]["note28"]["hgb"]["auc"],
            "macro_acc": meta["cv"]["macro10"]["hgb"]["acc"],
            "macro_auc": meta["cv"]["macro10"]["hgb"]["auc"],
            "merged_acc": meta["cv"]["merged38"]["hgb"]["acc"],
            "merged_auc": meta["cv"]["merged38"]["hgb"]["auc"],
        },
        "heldout": meta["heldout"],
        "dataset": {"n": meta["n_songs"], "n_ai": meta["n_ai"], "n_human": meta["n_human"]},
        "features": [
            {"feature": n, "plain": plain(n), "group": group_of(n),
             "ai_mean": fstats[n]["ai_mean"], "human_mean": fstats[n]["human_mean"]}
            for n in ALL_NAMES
        ],
        "gallery": gallery,
    }

    demo_dir = ROOT / "demo"
    demo_dir.mkdir(exist_ok=True)
    (demo_dir / "gallery.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {demo_dir/'gallery.json'}")
    print(f"  gallery tracks: {len(gallery)} "
          f"(AI={sum(1 for g in gallery if g['truth']==1)}, "
          f"human={sum(1 for g in gallery if g['truth']==0)}, "
          f"errors={sum(1 for g in gallery if not g['correct'])})")


if __name__ == "__main__":
    main()
