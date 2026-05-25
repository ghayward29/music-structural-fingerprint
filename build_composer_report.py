"""
build_composer_report.py — load fingerprints/artist_fingerprints.npz and
print a feature comparison table across all artists.

Reports:
  1. Per-feature variance ranking (which features differ most across artists)
  2. Full fingerprint table (artists × all 14 features)
  3. Spotlight: top-3 most-variable features with per-artist values
"""

import os
import sys

import numpy as np

NPZ_PATH = os.path.join(os.path.dirname(__file__), "fingerprints", "artist_fingerprints.npz")

DISPLAY_NAMES = {
    "Bach Johann Sebastian": "Johann Sebastian Bach",
    "John Elton":            "Elton John",
    "Jackson Michael":       "Michael Jackson",
}

# Short labels for the wide table
SHORT = {
    "interval_mean":       "int_mean",
    "interval_std":        "int_std",
    "interval_unison":     "unison",
    "interval_step":       "step",
    "interval_skip":       "skip",
    "interval_leap":       "leap",
    "interval_large_leap": "lg_leap",
    "harm_rhythm_mean":    "hr_mean",
    "harm_rhythm_std":     "hr_std",
    "note_density":        "density",
    "rhythmic_cv":         "rhy_cv",
    "pitch_range":         "p_range",
    "stepwise_ratio":      "sw_ratio",
    "cadence_count":       "cadence",
}


def main():
    if not os.path.isfile(NPZ_PATH):
        print(f"ERROR: {NPZ_PATH} not found — run build_fingerprints.py first.", file=sys.stderr)
        sys.exit(1)

    data          = np.load(NPZ_PATH, allow_pickle=True)
    artists_raw   = data["artists"].tolist()
    fingerprints  = data["fingerprints"]        # (N, 14)
    feature_names = data["feature_names"].tolist()
    n_files       = data["n_files"].tolist()

    artists = [DISPLAY_NAMES.get(a, a) for a in artists_raw]
    N, F    = fingerprints.shape

    # ── 1. Feature variance ranking ──────────────────────────────────────────
    feat_std  = fingerprints.std(axis=0)
    feat_mean = fingerprints.mean(axis=0)
    feat_cv   = np.where(feat_mean != 0, feat_std / np.abs(feat_mean), 0.0)
    feat_min  = fingerprints.min(axis=0)
    feat_max  = fingerprints.max(axis=0)
    ranked    = np.argsort(feat_std)[::-1]

    SEP = "=" * 78
    print(SEP)
    print("FEATURE VARIANCE RANKING  (cross-artist std, descending)")
    print(SEP)
    print(f"{'Rank':<5} {'Feature':<22} {'Mean':>8} {'Std':>8} {'CV':>7} {'Min':>8} {'Max':>8}")
    print("-" * 78)
    for rank, fi in enumerate(ranked, 1):
        print(
            f"{rank:<5} {feature_names[fi]:<22} "
            f"{feat_mean[fi]:>8.4f} {feat_std[fi]:>8.4f} {feat_cv[fi]:>7.3f} "
            f"{feat_min[fi]:>8.4f} {feat_max[fi]:>8.4f}"
        )

    # ── 2. Full fingerprint table ─────────────────────────────────────────────
    col_w    = 8
    name_w   = 26
    headers  = [SHORT.get(fn, fn[:col_w]) for fn in feature_names]
    hdr_line = f"{'Artist':<{name_w}}" + "".join(f"{h:>{col_w}}" for h in headers)

    print()
    print(SEP)
    print("FULL FINGERPRINT TABLE  (mean feature value per artist)")
    print(SEP)
    print(hdr_line)
    print("-" * len(hdr_line))
    for i, (artist, fp) in enumerate(zip(artists, fingerprints)):
        row = f"{artist:<{name_w}}" + "".join(f"{fp[j]:>{col_w}.3f}" for j in range(F))
        print(row)
    print("-" * len(hdr_line))
    std_row = f"{'[cross-artist std]':<{name_w}}" + "".join(f"{feat_std[j]:>{col_w}.3f}" for j in range(F))
    print(std_row)

    # ── 3. Spotlight: top 5 most-variable features ────────────────────────────
    top5 = ranked[:5]
    name_w2 = 26
    col_w2  = 12

    print()
    print(SEP)
    print("SPOTLIGHT: TOP 5 MOST VARIABLE FEATURES  (by cross-artist std)")
    print(SEP)

    for fi in top5:
        fname = feature_names[fi]
        print(f"\n  {fname}  (std={feat_std[fi]:.4f}, cv={feat_cv[fi]:.3f})")
        # sort artists by this feature value
        order = np.argsort(fingerprints[:, fi])[::-1]
        print(f"  {'Artist':<{name_w2}} {'Value':>10}  bar")
        print("  " + "-" * 55)
        vals = fingerprints[:, fi]
        v_min, v_max = vals.min(), vals.max()
        v_range = v_max - v_min if v_max != v_min else 1.0
        for idx in order:
            bar_len = int((vals[idx] - v_min) / v_range * 30)
            bar = "█" * bar_len
            print(f"  {artists[idx]:<{name_w2}} {vals[idx]:>10.4f}  {bar}")

    print()
    print(SEP)
    print(f"Artists: {N}   Features: {F}   Files per artist: "
          f"min={min(n_files)}  max={max(n_files)}  "
          f"mean={sum(n_files)/len(n_files):.1f}")
    print(SEP)


if __name__ == "__main__":
    main()
