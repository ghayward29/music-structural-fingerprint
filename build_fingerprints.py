"""
build_fingerprints.py — extract structural feature fingerprints for each
artist in data/artists/ and save them to fingerprints/artist_fingerprints.npz.

For each artist folder, every .mid file is processed with extract_features().
The per-artist fingerprint is the mean feature vector across all files where
extraction succeeded (non-zero vector).

Output: fingerprints/artist_fingerprints.npz with arrays:
    artists      — shape (N,)    str   — folder names
    fingerprints — shape (N, 14) float32 — mean feature vectors
    feature_names — shape (14,)  str
    n_files       — shape (N,)   int   — files that contributed to each mean
"""

import os
import sys

import numpy as np

from extract_features import extract_features, FEATURE_NAMES

# Display names used by the report; does not change folder names.
DISPLAY_NAMES = {
    "Bach Johann Sebastian": "Johann Sebastian Bach",
    "John Elton":            "Elton John",
    "Jackson Michael":       "Michael Jackson",
}

ARTISTS_DIR   = os.path.join(os.path.dirname(__file__), "data", "artists")
OUT_DIR       = os.path.join(os.path.dirname(__file__), "fingerprints")
OUT_PATH      = os.path.join(OUT_DIR, "artist_fingerprints.npz")


def main():
    if not os.path.isdir(ARTISTS_DIR):
        print(f"ERROR: {ARTISTS_DIR} not found — run setup_artists.py first.", file=sys.stderr)
        sys.exit(1)

    artist_dirs = sorted(
        d for d in os.listdir(ARTISTS_DIR)
        if os.path.isdir(os.path.join(ARTISTS_DIR, d))
    )
    if not artist_dirs:
        print("ERROR: no artist subdirectories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(artist_dirs)} artist folder(s) in {ARTISTS_DIR}\n")

    artists_out      = []
    fingerprints_out = []
    n_files_out      = []

    for artist in artist_dirs:
        display = DISPLAY_NAMES.get(artist, artist)
        mid_dir = os.path.join(ARTISTS_DIR, artist)
        mid_files = sorted(f for f in os.listdir(mid_dir) if f.lower().endswith(".mid"))

        vecs = []
        errors = 0
        for fname in mid_files:
            path = os.path.join(mid_dir, fname)
            try:
                v = extract_features(path)
                # skip all-zero vectors (parse failure)
                if np.any(v != 0):
                    vecs.append(v)
                else:
                    errors += 1
            except Exception:
                errors += 1

        if not vecs:
            print(f"  {display:<35} SKIPPED — no usable files")
            continue

        fingerprint = np.mean(vecs, axis=0).astype(np.float32)
        artists_out.append(artist)
        fingerprints_out.append(fingerprint)
        n_files_out.append(len(vecs))

        status = f"({errors} failed)" if errors else ""
        print(f"  {display:<35} {len(vecs):>3} files  {status}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        OUT_PATH,
        artists       = np.array(artists_out),
        fingerprints  = np.array(fingerprints_out, dtype=np.float32),
        feature_names = np.array(FEATURE_NAMES),
        n_files       = np.array(n_files_out, dtype=np.int32),
    )
    print(f"\nSaved {len(artists_out)} fingerprints → {OUT_PATH}")


if __name__ == "__main__":
    main()
