"""
make_rolls.py — batch-convert a folder of MIDI files to piano-roll PNGs.

Run this twice: once for human-composed MIDIs, once for AI-generated MIDIs.

Usage:
    python make_rolls.py --input data/human/ --output data/rolls/human/
    python make_rolls.py --input data/ai/    --output data/rolls/ai/

Each PNG is 512×128 pixels, grayscale, time-normalised — exactly the format
the CNN classifier expects.  See roll_utils.py for rendering details.
"""

import argparse
import glob
import os
import sys

from roll_utils import midi_to_roll


def convert_folder(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Collect .mid and .midi files, recursively so subfolders work too.
    patterns = ["**/*.mid", "**/*.midi"]
    midi_files = []
    for pat in patterns:
        midi_files.extend(glob.glob(os.path.join(input_dir, pat), recursive=True))

    if not midi_files:
        print(f"No MIDI files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Converting {len(midi_files)} MIDI file(s) → {output_dir}")

    ok = 0
    for midi_path in sorted(midi_files):
        try:
            img  = midi_to_roll(midi_path)
            stem = os.path.splitext(os.path.basename(midi_path))[0]
            out  = os.path.join(output_dir, f"{stem}.png")
            img.save(out)
            print(f"  ✓  {os.path.relpath(midi_path)} → {os.path.relpath(out)}")
            ok += 1
        except Exception as exc:
            # Skip bad files rather than aborting the whole batch.
            print(f"  ✗  {midi_path}: {exc}", file=sys.stderr)

    print(f"\nDone — {ok}/{len(midi_files)} converted successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MIDI files to 512×128 piano-roll PNGs for classifier training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",  required=True, help="Folder of MIDI files to convert")
    parser.add_argument("--output", required=True, help="Destination folder for PNG files")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Error: input folder not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    convert_folder(args.input, args.output)


if __name__ == "__main__":
    main()
