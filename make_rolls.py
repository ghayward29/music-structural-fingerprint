"""
make_rolls.py — convert MIDI files to piano-roll PNGs for classifier training.

Processes both class folders in one run:
    data/ai/          → data/rolls/ai/      (only .mid files — .mp3s are ignored)
    data/human/midi/  → data/rolls/human/

Each PNG is 512×128 pixels, grayscale, with time on the x-axis and pitch on
the y-axis.  The time axis is normalised so every image is exactly 512 pixels
wide regardless of piece duration — this makes the CNN invariant to tempo and
lets it focus on compositional patterns rather than absolute timing.

Image format details live in roll_utils.py; any change there automatically
applies to both training-data generation and inference in predict.py.

Usage:
    python make_rolls.py
"""

import glob
import os
import sys

from roll_utils import midi_to_roll

# ---------------------------------------------------------------------------
# Source / destination pairs
# ---------------------------------------------------------------------------

# data/ai/ contains both .mid and .mp3 files (Suno exports audio alongside
# MIDI).  We explicitly list only .mid here because .mp3 files cannot be
# parsed as MIDI and would cause errors downstream.
AI_SRC  = os.path.join("data", "ai")
AI_DST  = os.path.join("data", "rolls", "ai")

# data/human/midi/ is the flat folder of 200 sampled Lakh MIDI files
# produced by sample_human.py.  It contains .mid files only.
HUMAN_SRC = os.path.join("data", "human", "midi")
HUMAN_DST = os.path.join("data", "rolls", "human")

JOBS = [
    (AI_SRC,    AI_DST,    [".mid"]),          # AI: .mid only, not .mp3
    (HUMAN_SRC, HUMAN_DST, [".mid", ".midi"]), # Human: standard MIDI extensions
]


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert_folder(src: str, dst: str, extensions: list[str]) -> tuple[int, int]:
    """
    Render every MIDI file in src (matching extensions) to a piano-roll PNG
    in dst.  Returns (n_ok, n_failed).

    Files are collected non-recursively from src because:
    - data/ai/ is a flat folder (Suno exports are not nested)
    - data/human/midi/ is also flat (sample_human.py copies into a flat dir)
    Recursive glob is available via roll_utils if needed in future.
    """
    os.makedirs(dst, exist_ok=True)

    # Collect files matching the allowed extensions (case-insensitive).
    candidates = []
    for fname in sorted(os.listdir(src)):
        if os.path.splitext(fname)[1].lower() in extensions:
            candidates.append(os.path.join(src, fname))

    if not candidates:
        print(f"  No files with extensions {extensions} found in {src}",
              file=sys.stderr)
        return 0, 0

    ok = failed = 0
    for midi_path in candidates:
        stem    = os.path.splitext(os.path.basename(midi_path))[0]
        out     = os.path.join(dst, f"{stem}.png")
        try:
            img = midi_to_roll(midi_path)
            img.save(out)
            print(f"  ✓  {os.path.basename(midi_path)}")
            ok += 1
        except Exception as exc:
            # Log the failure and continue — one bad file should not abort the
            # whole batch; we'll train on whatever rendered successfully.
            print(f"  ✗  {os.path.basename(midi_path)}: {exc}", file=sys.stderr)
            failed += 1

    return ok, failed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    total_ok = total_fail = 0

    for src, dst, exts in JOBS:
        if not os.path.isdir(src):
            print(f"\nSkipping {src}/ — directory not found.", file=sys.stderr)
            print(f"  For the human set, run sample_human.py first.")
            continue

        n_src = sum(
            1 for f in os.listdir(src)
            if os.path.splitext(f)[1].lower() in exts
        )
        print(f"\n[{src}/ → {dst}/]  {n_src} file(s) to convert")

        ok, fail = convert_folder(src, dst, exts)
        total_ok   += ok
        total_fail += fail
        print(f"  → {ok} succeeded, {fail} failed")

    print(f"\nAll done — {total_ok} PNGs written, {total_fail} failures.")
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
