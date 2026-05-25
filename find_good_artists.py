"""
find_good_artists.py — diagnostic only, no files are moved or modified.

Walks clean_midi/, counts .mid files per artist subfolder,
and reports artists with at least 15 files.
"""

import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "clean_midi")
MIN_FILES = 15
TOP_N = 50


def main():
    if not os.path.isdir(ROOT):
        print(f"ERROR: directory not found: {ROOT}", file=sys.stderr)
        sys.exit(1)

    counts = {}
    for artist in os.listdir(ROOT):
        artist_dir = os.path.join(ROOT, artist)
        if not os.path.isdir(artist_dir):
            continue
        mid_count = sum(
            1 for f in os.listdir(artist_dir)
            if f.lower().endswith(".mid")
        )
        counts[artist] = mid_count

    qualified = {a: c for a, c in counts.items() if c >= MIN_FILES}
    ranked = sorted(qualified.items(), key=lambda x: x[1], reverse=True)

    print(f"Total artist folders scanned : {len(counts):,}")
    print(f"Artists with >= {MIN_FILES} MIDI files : {len(qualified):,}")
    print()
    print(f"{'Rank':<6} {'Files':<8} Artist")
    print("-" * 60)
    for i, (artist, count) in enumerate(ranked[:TOP_N], 1):
        print(f"{i:<6} {count:<8} {artist}")


if __name__ == "__main__":
    main()
