"""
extract_features.py — hand-crafted musical features extracted from a MIDI file.

These 14 features target deep compositional structure rather than surface
appearance.  The hypothesis is that structural patterns (voice-leading logic,
harmonic rhythm, phrase architecture) are harder for AI models to imitate
consistently than pixel-level patterns in a piano roll.

Each feature is computed from music21 note data and returned as a float32
numpy array in the order given by FEATURE_NAMES.

Can also be run standalone:
    python extract_features.py path/to/file.mid
"""

from collections import Counter
from typing import Optional

import numpy as np
import music21
import music21.tempo

# ---------------------------------------------------------------------------
# Feature registry — order here defines the vector layout everywhere
# ---------------------------------------------------------------------------

FEATURE_NAMES: list[str] = [
    "interval_mean",        # mean signed interval (semitones) per voice
    "interval_std",         # spread of intervals — high spread = disjunct writing
    "interval_unison",      # fraction of moves that repeat the same pitch
    "interval_step",        # fraction of moves by 1–2 semitones (stepwise)
    "interval_skip",        # fraction of moves by 3–4 semitones
    "interval_leap",        # fraction of moves by 5–7 semitones
    "interval_large_leap",  # fraction of moves by ≥8 semitones
    "harm_rhythm_mean",     # mean fraction of 1-second windows where harmony changes
    "harm_rhythm_std",      # std of the per-window harmonic-change signal
    "note_density",         # notes per second (baseline density feature)
    "rhythmic_cv",          # coefficient of variation of inter-onset intervals
    "pitch_range",          # semitone span from lowest to highest note
    "stepwise_ratio",       # fraction of consecutive pairs moving by 1–2 semitones
    "cadence_count",        # phrase-end cadences on stable scale degrees per 30 s
]

N_FEATURES = len(FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_bpm(score: music21.stream.Score) -> float:
    """Return the first MetronomeMark BPM in the score, else fall back to 120."""
    for el in score.flatten().getElementsByClass(music21.tempo.MetronomeMark):
        return float(el.number)
    return 120.0


def _collect_intervals(score: music21.stream.Score) -> list[int]:
    """
    Signed chromatic intervals (semitones) between consecutive notes within
    each part.  Per-part iteration approximates voice-level analysis without
    needing explicit voice separation — sufficient for these coarse features.

    Chords are represented by their soprano (highest) note so that the
    interval reflects the melodic surface rather than arbitrary bass motion.
    """
    intervals: list[int] = []
    for part in score.parts:
        notes: list[tuple[float, int]] = []
        for el in part.flatten().notesAndRests:
            if isinstance(el, music21.note.Rest):
                continue
            onset = float(el.offset)
            if isinstance(el, music21.note.Note):
                notes.append((onset, el.pitch.midi))
            elif isinstance(el, music21.chord.Chord):
                notes.append((onset, max(n.pitch.midi for n in el.notes)))
        notes.sort(key=lambda x: x[0])
        for i in range(1, len(notes)):
            intervals.append(notes[i][1] - notes[i - 1][1])
    return intervals


def _interval_histogram(abs_intervals: list[int]) -> tuple[float, float, float, float, float]:
    """
    Bin absolute intervals into five Schenkerian motion categories.
    Returns (unison, step, skip, leap, large_leap) as fractions summing to ~1.

    These categories come from tonal voice-leading pedagogy: stepwise motion
    is the normative baseline; leaps and skips require resolution by step.
    AI models that generate intervals from statistical models often produce
    unnatural distributions (too many large leaps or too rigid stepwise motion).
    """
    n = max(len(abs_intervals), 1)
    unison     = sum(1 for i in abs_intervals if i == 0) / n
    step       = sum(1 for i in abs_intervals if 1 <= i <= 2) / n
    skip       = sum(1 for i in abs_intervals if 3 <= i <= 4) / n
    leap       = sum(1 for i in abs_intervals if 5 <= i <= 7) / n
    large_leap = sum(1 for i in abs_intervals if i >= 8) / n
    return unison, step, skip, leap, large_leap


def _harmonic_rhythm(
    score: music21.stream.Score, bpm: float
) -> tuple[float, float]:
    """
    Divide the piece into 1-second windows and check whether the active
    pitch-class set changes between adjacent windows.  Return mean and std
    of this binary change signal.

    Harmonic rhythm is a structural marker: human composers vary the pace
    of chord changes (fast during sequences, slow on cadences).  AI models
    trained on token-level generation can produce artificially uniform
    harmonic rhythms with no sense of phrase architecture.
    """
    events: list[tuple[float, float, int]] = []
    for part in score.parts:
        for el in part.flatten().notesAndRests:
            if isinstance(el, music21.note.Rest):
                continue
            onset  = float(el.offset) * 60.0 / bpm
            offset = onset + float(el.duration.quarterLength) * 60.0 / bpm
            if isinstance(el, music21.note.Note):
                events.append((onset, offset, el.pitch.midi % 12))
            elif isinstance(el, music21.chord.Chord):
                for n in el.notes:
                    events.append((onset, offset, n.pitch.midi % 12))

    if not events:
        return 0.0, 0.0

    max_time  = max(o for _, o, _ in events)
    n_windows = max(int(max_time) + 1, 2)

    # Build pitch-class set for each 1-second window.
    window_pcs = []
    for w in range(n_windows):
        t0, t1 = float(w), float(w + 1)
        active = frozenset(
            pc for onset, offset, pc in events
            if onset < t1 and offset > t0
        )
        window_pcs.append(active)

    changes = [
        1.0 if window_pcs[i] != window_pcs[i - 1] else 0.0
        for i in range(1, len(window_pcs))
    ]
    return float(np.mean(changes)), float(np.std(changes))


def _note_density(score: music21.stream.Score, bpm: float) -> float:
    """
    Notes per second.  Kept as a baseline feature so later ablation studies
    can measure how much the structural features add beyond density alone.
    AI models sometimes produce unnaturally dense or sparse textures.
    """
    total = sum(
        1 for part in score.parts
        for el in part.flatten().notesAndRests
        if isinstance(el, (music21.note.Note, music21.chord.Chord))
    )
    dur_ql  = float(score.duration.quarterLength)
    dur_sec = dur_ql * 60.0 / bpm if dur_ql > 0 else 1.0
    return total / dur_sec


def _rhythmic_cv(score: music21.stream.Score, bpm: float) -> float:
    """
    Coefficient of variation (std / mean) of inter-onset intervals.

    Low CV ≈ rigid quantization — a suspected AI signature because language-
    model-style MIDI generation often lacks the micro-timing variability that
    human performers introduce even when the written notation is regular.
    High CV ≈ highly irregular pacing (rubato, polyrhythm).
    """
    onsets = sorted(
        float(el.offset) * 60.0 / bpm
        for part in score.parts
        for el in part.flatten().notesAndRests
        if isinstance(el, (music21.note.Note, music21.chord.Chord))
    )
    iois = [b - a for a, b in zip(onsets, onsets[1:]) if b > a]
    if len(iois) < 2:
        return 0.0
    mean = float(np.mean(iois))
    return float(np.std(iois) / mean) if mean > 1e-6 else 0.0


def _pitch_range(score: music21.stream.Score) -> float:
    """
    Semitone distance between the highest and lowest note.
    Some AI models collapse to a limited register (narrow range) or produce
    physically unplayable ranges (very wide); either can be a discriminating
    feature with sufficient training data.
    """
    pitches: list[int] = []
    for part in score.parts:
        for el in part.flatten().notesAndRests:
            if isinstance(el, music21.note.Note):
                pitches.append(el.pitch.midi)
            elif isinstance(el, music21.chord.Chord):
                pitches.extend(n.pitch.midi for n in el.notes)
    return float(max(pitches) - min(pitches)) if pitches else 0.0


def _stepwise_ratio(intervals: list[int]) -> float:
    """
    Proportion of consecutive note pairs that move by 1–2 semitones.

    Schenkerian theory identifies stepwise motion as the primary carrier of
    structural voice leading at every level of the tonal hierarchy.  Smooth
    voice leading is a hallmark of trained compositional style; AI models
    may over- or under-produce stepwise motion relative to human norms.
    """
    if not intervals:
        return 0.0
    return sum(1 for i in intervals if 1 <= abs(i) <= 2) / len(intervals)


def _cadence_count(score: music21.stream.Score, bpm: float) -> float:
    """
    Normalised count of approximate phrase-ending cadences.

    We approximate a cadence as a moment where:
    (a) no new note is attacked for ≥ 0.5 seconds (phrase boundary), and
    (b) the most recently attacked note lands on scale degree 1, 3, or 5
        relative to the estimated tonic.

    Tonic is estimated as the most common pitch class (crude but fast).
    The count is normalised per 30 seconds of music so pieces of different
    lengths are comparable.

    Human composers typically articulate cadences with clear phrase endings;
    some AI models generate textures that never fully cadence, creating a
    continuous stream without structural articulation.
    """
    # Collect all note attacks: (onset_sec, pitch_class).
    events: list[tuple[float, int]] = []
    for part in score.parts:
        for el in part.flatten().notesAndRests:
            if isinstance(el, music21.note.Rest):
                continue
            onset_sec = float(el.offset) * 60.0 / bpm
            if isinstance(el, music21.note.Note):
                events.append((onset_sec, el.pitch.midi % 12))
            elif isinstance(el, music21.chord.Chord):
                # Use the bass note as the harmonic representative.
                bass_pc = min(n.pitch.midi for n in el.notes) % 12
                events.append((onset_sec, bass_pc))

    if len(events) < 2:
        return 0.0

    # Estimate tonic as the most common pitch class in the piece.
    pc_counts: Counter = Counter(pc for _, pc in events)
    tonic = pc_counts.most_common(1)[0][0]
    # Scale degrees 1, 3, 5 in relative semitones from tonic.
    stable_pcs = {tonic % 12, (tonic + 4) % 12, (tonic + 7) % 12}

    events.sort(key=lambda e: e[0])
    onset_times = [e[0] for e in events]

    cadences = 0
    for i in range(len(events) - 1):
        gap = onset_times[i + 1] - onset_times[i]
        if gap >= 0.5 and events[i][1] in stable_pcs:
            cadences += 1

    # Normalise: cadences per 30 seconds of music.
    duration_sec = onset_times[-1] if onset_times else 1.0
    return cadences / max(duration_sec / 30.0, 1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_features(midi_path: str) -> np.ndarray:
    """
    Parse a MIDI file and return a float32 feature vector of length N_FEATURES.
    Feature order matches FEATURE_NAMES.
    Returns a zero vector (not NaN) if the file cannot be parsed, so downstream
    models can train/predict without special-casing missing data.
    """
    try:
        score = music21.converter.parse(midi_path)
        bpm   = _get_bpm(score)

        intervals     = _collect_intervals(score)
        abs_intervals = [abs(i) for i in intervals]
        unison, step, skip, leap, large_leap = _interval_histogram(abs_intervals)
        hr_mean, hr_std = _harmonic_rhythm(score, bpm)

        return np.array([
            float(np.mean(intervals)) if intervals else 0.0,  # interval_mean
            float(np.std(intervals))  if intervals else 0.0,  # interval_std
            unison,                                            # interval_unison
            step,                                              # interval_step
            skip,                                              # interval_skip
            leap,                                              # interval_leap
            large_leap,                                        # interval_large_leap
            hr_mean,                                           # harm_rhythm_mean
            hr_std,                                            # harm_rhythm_std
            _note_density(score, bpm),                        # note_density
            _rhythmic_cv(score, bpm),                         # rhythmic_cv
            _pitch_range(score),                               # pitch_range
            _stepwise_ratio(intervals),                        # stepwise_ratio
            _cadence_count(score, bpm),                       # cadence_count
        ], dtype=np.float32)

    except Exception:
        # Return zeros rather than crashing — bad files get treated as "featureless"
        # and the model can down-weight them naturally through training.
        return np.zeros(N_FEATURES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_features.py path/to/file.mid")
        sys.exit(1)

    vec = extract_features(sys.argv[1])
    print(f"\nFeatures — {sys.argv[1]}")
    print("-" * 42)
    for name, val in zip(FEATURE_NAMES, vec):
        print(f"  {name:<22} {val:>10.4f}")
