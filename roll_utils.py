"""
roll_utils.py — shared piano-roll rendering used by make_rolls.py and predict.py.

Isolating this here guarantees the image format is byte-for-byte identical
between training-data generation and inference.  Any change here affects both
paths simultaneously, preventing silent train/serve skew.
"""

import numpy as np
from PIL import Image
import music21
import music21.tempo
from torchvision import transforms

# Canonical image dimensions — must match across make_rolls, train, and predict.
IMG_WIDTH  = 512   # pixels — time axis
IMG_HEIGHT = 128   # pixels — pitch axis (one row per MIDI pitch 0–127)

# Single normalisation transform shared between training and inference.
# ToTensor converts a PIL 'L' image (H×W, uint8) to a (1, H, W) float tensor
# in [0, 1], then Normalize maps it to [-1, 1].  We use the same mean/std as
# a zero-centred grayscale convention.
ROLL_TRANSFORM = transforms.Compose([
    transforms.Grayscale(),               # enforce 1-channel even if loaded as RGB
    transforms.ToTensor(),                # (H, W) → (1, H, W), values [0, 1]
    transforms.Normalize((0.5,), (0.5,)), # → values [-1, 1]
])


def midi_to_roll(midi_path: str) -> Image.Image:
    """
    Parse a MIDI file and render it as a 512×128 grayscale piano-roll image.

    Layout:
        x-axis — time, normalised so all pieces span the full 512 pixels
                 regardless of duration.  This makes the CNN invariant to
                 absolute tempo, which is desirable when comparing pieces of
                 different lengths.
        y-axis — MIDI pitch 0 at the bottom row, 127 at the top row.
                 Each row is exactly one semitone.

    Active note pixels are white (255); silence is black (0).
    Returns a PIL Image in mode 'L' (8-bit grayscale).
    """
    score = music21.converter.parse(midi_path)

    # Collect every sounding pitch as (midi_pitch, onset_ql, offset_ql).
    events: list[tuple[int, float, float]] = []
    for part in score.parts:
        for el in part.flatten().notesAndRests:
            if isinstance(el, music21.note.Rest):
                continue
            onset  = float(el.offset)
            offset = onset + float(el.duration.quarterLength)
            if isinstance(el, music21.note.Note):
                events.append((el.pitch.midi, onset, offset))
            elif isinstance(el, music21.chord.Chord):
                for n in el.notes:
                    events.append((n.pitch.midi, onset, offset))

    if not events:
        # Return a blank image rather than crashing on an empty MIDI file.
        return Image.new("L", (IMG_WIDTH, IMG_HEIGHT), 0)

    # Normalise the time axis to [0, 1] using the piece's actual time span.
    min_onset  = min(e[1] for e in events)
    max_offset = max(e[2] for e in events)
    span       = max_offset - min_onset or 1.0  # guard against zero-length

    # Build the pixel grid: shape (height, width) for PIL compatibility.
    pixels = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    for pitch, onset, offset in events:
        if not (0 <= pitch <= 127):
            continue

        # Flip y so low pitches are at the bottom (row IMG_HEIGHT-1).
        y = IMG_HEIGHT - 1 - pitch

        x_start = int((onset  - min_onset) / span * (IMG_WIDTH - 1))
        x_end   = int((offset - min_onset) / span * (IMG_WIDTH - 1))
        x_end   = max(x_end, x_start + 1)   # every note is at least 1 px wide
        x_end   = min(x_end, IMG_WIDTH)      # clamp to image boundary

        pixels[y, x_start:x_end] = 255

    return Image.fromarray(pixels, mode="L")
