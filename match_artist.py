"""
match_artist.py — compare a MIDI file's structural features against artist
fingerprints and return cosine-similarity scores.

Usage:
    python match_artist.py path/to/file.mid

Importable function:
    from match_artist import match_artist
    results = match_artist(midi_path)   # list of dicts, top 5 by default
"""

import os
import sys

import numpy as np

from extract_features import extract_features

NPZ_PATH = os.path.join(os.path.dirname(__file__), "fingerprints", "artist_fingerprints.npz")

DISPLAY_NAMES = {
    "Bach Johann Sebastian": "Johann Sebastian Bach",
    "John Elton":            "Elton John",
    "Jackson Michael":       "Michael Jackson",
}

# A piece is marked "within normal range" for an artist when its z-score
# normalised cosine similarity (scaled to [0,1]) meets this threshold.
IN_RANGE_THRESHOLD = 0.75


def match_artist(midi_path: str, top_n: int = 5) -> list[dict]:
    """
    Compare a MIDI file's structural feature vector against all artist
    fingerprints and return the top_n most similar artists.

    Each result dict:
        artist      — display name string
        similarity  — float in [0, 1]  (1 = identical feature profile)
        in_range    — True if similarity >= IN_RANGE_THRESHOLD

    Raises FileNotFoundError if fingerprints/artist_fingerprints.npz is missing.
    """
    if not os.path.isfile(NPZ_PATH):
        raise FileNotFoundError(
            f"Artist fingerprints not found at {NPZ_PATH}. "
            "Run setup_artists.py then build_fingerprints.py first."
        )

    data         = np.load(NPZ_PATH, allow_pickle=True)
    artists_raw  = data["artists"]
    fingerprints = data["fingerprints"].astype(np.float64)   # (N, 14)

    query_vec = extract_features(midi_path).astype(np.float64)  # (14,)

    # Z-score normalise across all artist fingerprints so every feature
    # contributes equally regardless of magnitude.
    cross_mean = fingerprints.mean(axis=0)
    cross_std  = fingerprints.std(axis=0)
    cross_std  = np.where(cross_std < 1e-8, 1.0, cross_std)

    normed_fps   = (fingerprints - cross_mean) / cross_std   # (N, 14)
    normed_query = (query_vec    - cross_mean) / cross_std   # (14,)

    norm_q = np.linalg.norm(normed_query)

    results = []
    for i, artist_raw in enumerate(artists_raw):
        fp     = normed_fps[i]
        norm_f = np.linalg.norm(fp)

        if norm_q < 1e-8 or norm_f < 1e-8:
            cosine_sim = 0.0
        else:
            cosine_sim = float(np.dot(normed_query, fp) / (norm_q * norm_f))

        # Scale cosine similarity from [-1, 1] → [0, 1]
        sim_01 = round((cosine_sim + 1.0) / 2.0, 4)

        display = DISPLAY_NAMES.get(str(artist_raw), str(artist_raw))
        results.append({
            "artist":     display,
            "similarity": sim_01,
            "in_range":   sim_01 >= IN_RANGE_THRESHOLD,
        })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python match_artist.py path/to/file.mid", file=sys.stderr)
        sys.exit(1)

    midi_path = sys.argv[1]
    if not os.path.isfile(midi_path):
        print(f"Error: file not found: {midi_path}", file=sys.stderr)
        sys.exit(1)

    try:
        matches = match_artist(midi_path)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\nArtist structural similarity for: {os.path.basename(midi_path)}\n")
    print(f"{'Rank':<5} {'Artist':<30} {'Similarity':>12}  In range")
    print("-" * 60)
    for i, m in enumerate(matches, 1):
        badge = "✓ within normal range" if m["in_range"] else ""
        print(f"{i:<5} {m['artist']:<30} {m['similarity']:>11.1%}  {badge}")


if __name__ == "__main__":
    main()
