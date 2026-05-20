"""
sample_human.py — randomly sample 200 MIDI files from the Lakh MIDI Dataset
and copy them into data/human/midi/ for classifier training.

Why sample instead of using all 45,000+ files?
The AI side has ~100 files.  Training with a 450:1 class imbalance would cause
the classifier to predict "human" for nearly everything and still achieve high
accuracy — a useless model.  Sampling 200 human files keeps both classes at a
similar scale so the CNN learns genuine signal rather than the prior.

We use a fixed random seed (42) so the sample is reproducible — every run
produces the exact same 200 files.
"""

import glob
import os
import random
import shutil
import sys

SEED       = 42
N_SAMPLES  = 200
SOURCE_DIR = os.path.join("data", "human", "lmd_matched")
DEST_DIR   = os.path.join("data", "human", "midi")


def main() -> None:
    random.seed(SEED)

    # Walk the full Lakh directory tree and collect every .mid file.
    # We only want .mid here — the LMD ships MIDIs only, so this is safe,
    # but being explicit avoids surprises if other file types appear.
    all_midis = glob.glob(os.path.join(SOURCE_DIR, "**", "*.mid"), recursive=True)
    all_midis.sort()   # sort before sampling so seed produces the same result
                       # regardless of filesystem traversal order

    if not all_midis:
        print(f"Error: no .mid files found in {SOURCE_DIR}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_midis):,} MIDI files in {SOURCE_DIR}/")

    # Sample without replacement — random.sample guarantees no duplicates.
    n = min(N_SAMPLES, len(all_midis))
    sampled = random.sample(all_midis, n)

    os.makedirs(DEST_DIR, exist_ok=True)

    print(f"Copying {n} files → {DEST_DIR}/")

    copied = 0
    skipped = 0
    for src in sampled:
        stem = os.path.splitext(os.path.basename(src))[0]
        dest = os.path.join(DEST_DIR, f"{stem}.mid")

        # The LMD uses hash-based filenames (e.g. TRAAAAW128F429D538.mid) so
        # collisions are extremely rare, but we handle them safely by appending
        # a counter rather than silently overwriting.
        if os.path.exists(dest):
            counter = 1
            while os.path.exists(dest):
                dest = os.path.join(DEST_DIR, f"{stem}_{counter}.mid")
                counter += 1

        try:
            shutil.copy2(src, dest)
            copied += 1
        except OSError as exc:
            print(f"  ✗  {src}: {exc}", file=sys.stderr)
            skipped += 1

    print(f"\nDone — {copied} copied, {skipped} skipped → {DEST_DIR}/")


if __name__ == "__main__":
    main()
