"""
transcribe_stems.py — stem-aware batch MIDI transcription for Suno stem exports.

Usage:
    python transcribe_stems.py [stems_root]

Default stems_root: stems/

Expected input layout:
    stems/
        Suncream_Confetti/
            Lead_Vocals.mp3
            Instrumental.mp3
            Bass.mp3
            Drums.mp3
        Another_Track/
            ...

For each track subfolder:
  - Percussion stems are identified and skipped entirely.
    Percussion stems (drums, kick, snare, etc.) contain no stable pitch
    information; Basic Pitch interprets transient attack noise as random
    pitches, producing note clusters that corrupt compositional feature
    analysis. Skipping them keeps the merged MIDI clean and pitch-meaningful.
  - All remaining stems are transcribed with Basic Pitch.
  - The resulting per-stem MIDIs are merged into a single multi-track MIDI
    using music21, with each stem as a named Part.

Output: data/ai_stems/<track_name>.mid
Safe to re-run: existing output files are skipped.
"""

import copy
import os
import re
import shutil
import sys
import tempfile

import music21.converter
import music21.instrument as _m21_instr
import music21.note
import music21.stream

from separate_stems import transcribe_with_basic_pitch

# ── General MIDI instrument assignment ───────────────────────────────────────
# Maps keywords found in stem filenames to music21 instrument classes.
# Suno stem names are like "0 Bass.mp3", "3 Lead Vocals.mp3", etc.
_STEM_GM_MAP: list[tuple[str, type]] = [
    ("vocal",    _m21_instr.Vocalist),
    ("voice",    _m21_instr.Vocalist),
    ("lead",     _m21_instr.Vocalist),
    ("backing",  _m21_instr.Vocalist),
    ("guitar",   _m21_instr.Guitar),
    ("keyboard", _m21_instr.Piano),
    ("piano",    _m21_instr.Piano),
    ("violin",   _m21_instr.Violin),
    ("strings",  _m21_instr.Violin),
    ("trumpet",  _m21_instr.Trumpet),
    ("brass",    _m21_instr.Trumpet),
    ("woodwind", _m21_instr.Flute),
    ("flute",    _m21_instr.Flute),
]
_PREFIX_RE = re.compile(r"^\d+\s+")


def stem_name_to_instrument(stem_name: str) -> _m21_instr.Instrument:
    """Return a General MIDI instrument for the given stem filename (no extension)."""
    clean = _PREFIX_RE.sub("", stem_name).lower()
    for kw, cls in _STEM_GM_MAP:
        if kw in clean:
            try:
                return cls()
            except Exception:
                pass
    # Bass and synth handled with explicit MIDI program numbers
    if "bass" in clean:
        instr = _m21_instr.Instrument()
        instr.instrumentName = "Electric Bass"
        instr.midiProgram = 33
        return instr
    if "synth" in clean:
        instr = _m21_instr.Instrument()
        instr.instrumentName = "Synth Lead"
        instr.midiProgram = 80
        return instr
    return _m21_instr.Piano()

# ── paths ────────────────────────────────────────────────────────────────────
DEFAULT_STEMS_ROOT = os.path.join(os.path.dirname(__file__), "stems")
OUT_DIR            = os.path.join(os.path.dirname(__file__), "data", "ai_stems")
AI_DIR             = os.path.join(os.path.dirname(__file__), "data", "ai")

# ── percussion filter ─────────────────────────────────────────────────────────
# Percussion stems have no stable pitch information. Basic Pitch transcription
# of drum/percussion stems treats transient noise as pitched events, producing
# random note clusters that would corrupt the compositional feature analysis.
# We skip any stem whose name contains one of these tokens (case-insensitive).
PERCUSSION_KEYWORDS = {"drum", "perc", "kick", "snare", "hat", "cymbal", "rhythm"}

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aiff", ".ogg"}


def is_percussion(stem_filename: str) -> bool:
    stem = os.path.splitext(stem_filename)[0].lower()
    return any(kw in stem for kw in PERCUSSION_KEYWORDS)


def stem_to_part(midi_path: str, part_name: str) -> tuple[music21.stream.Part, int]:
    """
    Parse a Basic Pitch MIDI and return a named music21 Part plus note count.

    Flattens the parsed MIDI so all tracks/voices collapse into one timeline,
    then deep-copies each note/chord into a fresh Part at its absolute offset.
    Deep-copying is required because music21 elements may only live in one
    stream at a time; reusing the original would move it out of the parsed tree.
    """
    parsed = music21.converter.parse(midi_path)
    flat   = parsed.flatten()

    part = music21.stream.Part()
    part.id       = part_name
    part.partName = part_name
    part.insert(0, stem_name_to_instrument(part_name))

    n_notes = 0
    for element in flat.notes:                       # Notes and Chords, no Rests
        part.insert(element.offset, copy.deepcopy(element))
        # A Chord counts as one element but may contain several pitches.
        n_notes += len(element.pitches) if hasattr(element, "pitches") else 1

    return part, n_notes


def count_notes_midi(midi_path: str) -> int:
    """Return total pitched note-events in a MIDI file (for comparison stats)."""
    try:
        parsed = music21.converter.parse(midi_path)
        total = 0
        for el in parsed.flatten().notes:
            total += len(el.pitches) if hasattr(el, "pitches") else 1
        return total
    except Exception:
        return 0


def process_track(track_dir: str, track_name: str) -> int | None:
    """
    Transcribe non-percussion stems in track_dir and merge into a single MIDI.

    Returns total note count in the merged MIDI, or None if the track was
    skipped (no usable stems, or output already exists).
    """
    out_path = os.path.join(OUT_DIR, f"{track_name}.mid")

    if os.path.isfile(out_path):
        print(f"  SKIP  {track_name} (already exists)")
        return count_notes_midi(out_path)

    audio_files = sorted(
        f for f in os.listdir(track_dir)
        if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"  SKIP  {track_name} (no audio files found)")
        return None

    included: list[str] = []
    skipped:  list[str] = []
    for fname in audio_files:
        if is_percussion(fname):
            skipped.append(fname)
        else:
            included.append(fname)

    print(f"\n  Track: {track_name}")
    if skipped:
        print(f"    Skipped (percussion): {', '.join(skipped)}")
    if not included:
        print(f"    No usable stems — skipping track entirely.")
        return None

    # Transcribe each non-percussion stem
    parts: list[tuple[music21.stream.Part, int]] = []
    for stem_fname in included:
        stem_path = os.path.join(track_dir, stem_fname)
        stem_name = os.path.splitext(stem_fname)[0]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                midi_path = transcribe_with_basic_pitch(stem_path, tmp)
                part, n   = stem_to_part(midi_path, stem_name)
            parts.append((part, n))
            print(f"    OK  {stem_fname:<35} {n:>5} notes")
        except Exception as exc:
            print(f"    FAIL {stem_fname}: {exc}")

    if not parts:
        print(f"    All stems failed — skipping track.")
        return None

    # Merge into a multi-track Score and write
    score = music21.stream.Score()
    total_notes = 0
    for part, n in parts:
        score.append(part)
        total_notes += n

    os.makedirs(OUT_DIR, exist_ok=True)
    score.write("midi", fp=out_path)
    print(f"    Merged → {out_path}  ({total_notes} notes total)")
    return total_notes


def main() -> None:
    stems_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEMS_ROOT

    if not os.path.isdir(stems_root):
        print(f"ERROR: stems root not found: {stems_root}", file=sys.stderr)
        sys.exit(1)

    track_dirs = sorted(
        d for d in os.listdir(stems_root)
        if os.path.isdir(os.path.join(stems_root, d))
    )
    if not track_dirs:
        print(f"No track subfolders found in {stems_root}")
        sys.exit(0)

    print(f"Found {len(track_dirs)} track folder(s) in {stems_root}\n")

    # ── process tracks ────────────────────────────────────────────────────────
    processed_notes: dict[str, int] = {}
    for track_name in track_dirs:
        track_dir = os.path.join(stems_root, track_name)
        result = process_track(track_dir, track_name)
        if result is not None:
            processed_notes[track_name] = result

    # ── per-track percussion skip count ───────────────────────────────────────
    total_perc_skipped = 0
    for track_name in track_dirs:
        track_dir = os.path.join(stems_root, track_name)
        audio_files = [
            f for f in os.listdir(track_dir)
            if os.path.splitext(f)[1].lower() in AUDIO_EXTENSIONS
        ]
        total_perc_skipped += sum(1 for f in audio_files if is_percussion(f))

    # ── compare with data/ai/ baseline ────────────────────────────────────────
    ai_notes: list[int] = []
    if os.path.isdir(AI_DIR):
        for fname in os.listdir(AI_DIR):
            if fname.lower().endswith(".mid"):
                n = count_notes_midi(os.path.join(AI_DIR, fname))
                if n > 0:
                    ai_notes.append(n)

    # ── final summary ─────────────────────────────────────────────────────────
    SEP = "=" * 60
    print(f"\n{SEP}")
    print("FINAL SUMMARY")
    print(SEP)
    print(f"  Tracks processed        : {len(processed_notes)}")
    print(f"  Tracks skipped/failed   : {len(track_dirs) - len(processed_notes)}")
    print(f"  Percussion stems skipped: {total_perc_skipped}")

    if processed_notes:
        stem_note_counts = list(processed_notes.values())
        avg_stem = sum(stem_note_counts) / len(stem_note_counts)
        total    = sum(stem_note_counts)
        print(f"  Total notes (stem MIDIs): {total:,}")
        print(f"  Avg notes / stem track  : {avg_stem:.0f}")

        if ai_notes:
            avg_ai = sum(ai_notes) / len(ai_notes)
            delta  = avg_stem - avg_ai
            direction = "richer" if delta > 0 else "sparser"
            print(f"\n  Comparison vs data/ai/ ({len(ai_notes)} files):")
            print(f"    Avg notes data/ai/    : {avg_ai:.0f}")
            print(f"    Avg notes ai_stems/   : {avg_stem:.0f}")
            print(f"    Difference            : {delta:+.0f}  ({direction})")
        else:
            print(f"\n  (data/ai/ not found — skipping comparison)")

    print(f"\n  Output directory: {OUT_DIR}")
    print(SEP)


if __name__ == "__main__":
    main()
