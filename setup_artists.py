"""
setup_artists.py — copies a random sample of MIDI files from clean_midi/
into data/artists/ for the 15 selected artists.

Does NOT move, delete, or modify any file in clean_midi/.
Safe to re-run: existing destination files are skipped.
"""

import os
import random
import shutil
import sys

ARTISTS = [
    "The Beatles",
    "Queen",
    "Metallica",
    "Nirvana",
    "Led Zeppelin",
    "Elvis Presley",
    "Billy Joel",
    "U2",
    "The Rolling Stones",
    "Pink Floyd",
    "Green Day",
    "Eric Clapton",
    "Bach Johann Sebastian",
    "Prince",
    "Fleetwood Mac",
]

MAX_PER_ARTIST = 30
SEED = 42

SRC_ROOT = os.path.join(os.path.dirname(__file__), "clean_midi")
DST_ROOT = os.path.join(os.path.dirname(__file__), "data", "artists")


def main():
    if not os.path.isdir(SRC_ROOT):
        print(f"ERROR: clean_midi/ not found at {SRC_ROOT}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(DST_ROOT, exist_ok=True)
    print(f"Destination: {DST_ROOT}\n")

    rng = random.Random(SEED)
    total_copied = 0

    for artist in ARTISTS:
        src_dir = os.path.join(SRC_ROOT, artist)
        if not os.path.isdir(src_dir):
            print(f"  WARNING: folder not found — {src_dir}")
            continue

        mid_files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".mid"))
        sample = rng.sample(mid_files, min(MAX_PER_ARTIST, len(mid_files)))

        dst_dir = os.path.join(DST_ROOT, artist)
        os.makedirs(dst_dir, exist_ok=True)

        copied = skipped = 0
        for fname in sample:
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, fname)
            if os.path.exists(dst):
                skipped += 1
            else:
                shutil.copy2(src, dst)
                copied += 1

        total_copied += copied
        print(
            f"  {artist:<30} {len(mid_files):>4} available → "
            f"{len(sample):>2} sampled  ({copied} copied, {skipped} already present)"
        )

    print(f"\nDone. Total files copied this run: {total_copied}")


if __name__ == "__main__":
    main()
