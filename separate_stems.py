"""
separate_stems.py — Demucs stem separation + Basic Pitch transcription helpers.

Two public functions:

    separate_audio(audio_path, output_dir) -> list[str] | None
        Run Demucs htdemucs on a full-mix audio file and return paths to the
        non-percussion stems (vocals, bass, other — never drums).
        Returns None on failure.

    transcribe_with_basic_pitch(audio_path, output_dir) -> str
        Transcribe a single audio file to MIDI with the Basic Pitch ONNX model.
        Returns the path to the written .mid file.
"""

import logging
import os
import subprocess

from basic_pitch.inference import predict, Model
from basic_pitch import build_icassp_2022_model_path, FilenameSuffix

_ONNX_MODEL_PATH = build_icassp_2022_model_path(FilenameSuffix.onnx)

# htdemucs is the highest-quality 4-stem model with the broadest genre coverage.
# It outperforms mdx models on genre diversity at the cost of slightly slower runtime.
_DEMUCS_MODEL = "htdemucs"

# Drums have no stable pitch information.  Basic Pitch on drum stems interprets
# transient noise as random pitches, corrupting compositional feature analysis.
_PERCUSSION_STEMS = {"drums"}

# Lazy-loaded singleton avoids reloading the ONNX model for each stem.
_bp_model: "Model | None" = None


def _get_model() -> Model:
    global _bp_model
    if _bp_model is None:
        _bp_model = Model(_ONNX_MODEL_PATH)
    return _bp_model


def separate_audio(audio_path: str, output_dir: str) -> list[str] | None:
    """
    Run Demucs htdemucs on audio_path, saving stems under output_dir.

    Demucs output layout:
        output_dir/htdemucs/<track_name>/{vocals,drums,bass,other}.wav

    Returns [vocals.wav, bass.wav, other.wav] absolute paths (drums excluded).
    Returns None if separation fails for any reason.
    """
    track_name = os.path.splitext(os.path.basename(audio_path))[0]
    try:
        # Resolve the demucs binary relative to this file's venv to avoid
        # relying on PATH when invoked from a subprocess or different shell.
        _here = os.path.dirname(os.path.abspath(__file__))
        _demucs_bin = os.path.join(_here, "venv", "bin", "demucs")
        if not os.path.isfile(_demucs_bin):
            _demucs_bin = "demucs"  # fall back to PATH
        subprocess.run(
            [_demucs_bin, "-n", _DEMUCS_MODEL, "-o", output_dir, audio_path],
            capture_output=True,
            text=True,
            check=True,
        )
        stem_dir = os.path.join(output_dir, _DEMUCS_MODEL, track_name)
        non_perc = [s for s in ("vocals.wav", "bass.wav", "other.wav")
                    if s.split(".")[0] not in _PERCUSSION_STEMS]
        paths = [os.path.join(stem_dir, s) for s in non_perc]
        found = [p for p in paths if os.path.isfile(p)]
        return found if found else None
    except Exception as exc:
        logging.warning("Demucs separation failed for %s: %s", audio_path, exc)
        return None


def transcribe_with_basic_pitch(audio_path: str, output_dir: str) -> str:
    """
    Transcribe a single audio file to MIDI using the Basic Pitch ONNX model.

    Returns the path to the written .mid file.
    """
    model = _get_model()
    _out, midi_data, _events = predict(audio_path, model)
    base = os.path.splitext(os.path.basename(audio_path))[0]
    midi_path = os.path.join(output_dir, f"{base}.mid")
    midi_data.write(midi_path)
    return midi_path
