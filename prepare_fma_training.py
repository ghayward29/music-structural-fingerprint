"""
prepare_fma_training.py — sample 20 FMA tracks per genre, transcribe
MP3s to MIDI via Basic Pitch, and save results to data/human_audio/.

Run after extracting fma_small:
    python prepare_fma_training.py

Actual layout on disk:
    fma_small/
        fma_metadata/
            tracks.csv
        fma_small/          ← audio lives here (nested)
            000/000002.mp3
            001/001234.mp3
            ...

Outputs MIDI files to data/human_audio/fma_<track_id>.mid.
Safe to re-run: existing output files are skipped.
"""

import os
import sys
import random
import shutil
import tempfile
from collections import Counter

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required.  Run: pip install pandas", file=sys.stderr)
    sys.exit(1)

from pipeline import transcribe_audio

# ── paths ────────────────────────────────────────────────────────────────────
FMA_ROOT       = os.path.join(os.path.dirname(__file__), "fma_small")
FMA_AUDIO_ROOT = os.path.join(FMA_ROOT, "fma_small")   # audio is nested one level deeper
META_CSV       = os.path.join(FMA_ROOT, "fma_metadata", "tracks.csv")
OUT_DIR        = os.path.join(os.path.dirname(__file__), "data", "human_audio")

# ── config ───────────────────────────────────────────────────────────────────
SEED             = 42
PER_GENRE        = 20    # tracks sampled per genre (8 genres → ~160 total)


def track_mp3(track_id: int) -> str:
    """Return expected MP3 path for a track ID.
    FMA layout: fma_small/fma_small/NNN/NNNNNN.mp3 where NNN = track_id // 1000.
    """
    return os.path.join(FMA_AUDIO_ROOT, f"{track_id // 1000:03d}", f"{track_id:06d}.mp3")


def main() -> None:
    # ── validate prerequisites ────────────────────────────────────────────────
    if not os.path.isdir(FMA_AUDIO_ROOT):
        print(f"ERROR: audio directory not found at {FMA_AUDIO_ROOT}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(META_CSV):
        print(f"ERROR: metadata not found at {META_CSV}", file=sys.stderr)
        print("       Download fma_metadata.zip from the FMA repo and unzip into fma_small/.",
              file=sys.stderr)
        sys.exit(1)

    # ── load metadata ─────────────────────────────────────────────────────────
    print(f"Loading {META_CSV} ...")
    # tracks.csv has two header rows (category, field) and track_id as index.
    tracks = pd.read_csv(META_CSV, index_col=0, header=[0, 1])

    # Filter to the fma_small subset and drop tracks missing a top genre.
    subset_col = ("set",   "subset")
    genre_col  = ("track", "genre_top")
    small = tracks[tracks[subset_col] == "small"]
    genres = small[genre_col].dropna()

    print(f"  {len(small):,} tracks in fma_small subset, "
          f"{len(genres):,} with a top-level genre label")

    # ── sample evenly across genres ───────────────────────────────────────────
    rng = random.Random(SEED)

    genre_groups: dict[str, list[int]] = {}
    for track_id, genre in genres.items():
        genre_groups.setdefault(genre, []).append(int(track_id))

    unique_genres = sorted(genre_groups)
    n_genres = len(unique_genres)

    sampled: list[tuple[int, str]] = []
    for genre in unique_genres:
        ids  = sorted(genre_groups[genre])
        take = rng.sample(ids, min(PER_GENRE, len(ids)))
        sampled.extend((tid, genre) for tid in take)

    # Print sampling plan
    genre_counts = Counter(g for _, g in sampled)
    print(f"\nSampled {len(sampled)} tracks across {n_genres} genres (seed={SEED}):")
    print(f"  {'Genre':<22} {'Tracks':>7}")
    print("  " + "-" * 30)
    for genre in sorted(genre_counts):
        print(f"  {genre:<22} {genre_counts[genre]:>7}")

    # ── transcribe ────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    succeeded = failed = skipped = 0
    succeeded_by_genre: Counter = Counter()

    print(f"\nTranscribing {len(sampled)} tracks → {OUT_DIR}\n")
    for i, (track_id, genre) in enumerate(sampled, 1):
        mp3_path = track_mp3(track_id)
        dst_path = os.path.join(OUT_DIR, f"fma_{track_id:06d}.mid")
        tag      = f"[{i:>3}/{len(sampled)}]"

        if os.path.isfile(dst_path):
            print(f"  {tag} SKIP  fma_{track_id:06d}.mid (already exists)")
            skipped += 1
            continue

        if not os.path.isfile(mp3_path):
            print(f"  {tag} MISS  {mp3_path}")
            failed += 1
            continue

        try:
            with tempfile.TemporaryDirectory() as tmp:
                midi_path, _ = transcribe_audio(mp3_path, tmp)
                shutil.copy2(midi_path, dst_path)
            print(f"  {tag} OK    fma_{track_id:06d}.mid  ({genre})")
            succeeded += 1
            succeeded_by_genre[genre] += 1
        except Exception as exc:
            print(f"  {tag} FAIL  track {track_id}: {exc}")
            failed += 1

    # ── summary ───────────────────────────────────────────────────────────────
    total = succeeded + skipped
    print(f"\n{'='*55}")
    print(f"Transcription summary:")
    print(f"  Succeeded this run : {succeeded}")
    print(f"  Skipped (existed)  : {skipped}")
    print(f"  Failed / missing   : {failed}")
    print(f"  Total usable files : {total}")
    if succeeded_by_genre:
        print(f"\n  New files per genre:")
        for genre in sorted(succeeded_by_genre):
            print(f"    {genre:<22} {succeeded_by_genre[genre]:>3}")
    print(f"\n  MIDI files in {OUT_DIR}: {total}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
