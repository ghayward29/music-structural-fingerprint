"""
app.py — Flask web interface for the music structural analysis pipeline.

Routes:
    GET  /                    — serves the upload UI (templates/index.html)
    POST /analyze             — accepts an MP3/WAV, runs full pipeline + all classifiers
    POST /classify            — accepts a MIDI file, returns piano-roll CNN probability
    POST /classify_structural — accepts a MIDI file, returns structural classifier scores
    GET  /health              — liveness check

Uploaded files are written to uploads/, processed, then deleted.
All pipeline logic lives in pipeline.py and is imported directly.
Piano-roll classifier logic lives in predict.py and is imported directly.
Structural classifier logic lives in predict_structural.py and is imported directly.
"""

import os
import base64
import pickle
import tempfile
import traceback

import music21.converter
import music21.tempo

from flask import Flask, request, jsonify, render_template

# Import pipeline steps directly — no logic is duplicated here.
from pipeline import (
    transcribe_audio,
    parse_midi,
    build_note_graph,
    visualize_graph,
    run_structural_analysis,
)

# Import classifier — classify_midi raises FileNotFoundError if the model
# has not been trained yet; /classify handles that case gracefully.
from predict import classify_midi

# Import structural classifier — raises FileNotFoundError if either model pkl
# is missing; /classify_structural surfaces this as a 503.
from predict_structural import classify_structural

# Import structural feature extraction for the key-features table.
from extract_features import extract_features as _extract_features, FEATURE_NAMES

# Import artist matcher — raises FileNotFoundError if fingerprints not built.
from match_artist import match_artist as _match_artist

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"mp3", "wav"}

_LOGREG_PATH        = os.path.join(os.path.dirname(__file__), "classifier", "archive", "v1_lakh_logreg.pkl")
_AUDIO_LOGREG_PATH  = os.path.join(os.path.dirname(__file__), "classifier", "archive", "v2_fma_logreg.pkl")
_AUDIO_RF_PATH      = os.path.join(os.path.dirname(__file__), "classifier", "archive", "v2_fma_rf.pkl")
_STEMS_LOGREG_PATH  = os.path.join(os.path.dirname(__file__), "classifier", "archive", "v3_stems_logreg.pkl")
_STEMS_RF_PATH      = os.path.join(os.path.dirname(__file__), "classifier", "archive", "v3_stems_rf.pkl")
_STEMS_V2_LR_PATH   = os.path.join(os.path.dirname(__file__), "classifier", "archive", "v4_stems_v2_logreg.pkl")
_STEMS_LASSO_PATH   = os.path.join(os.path.dirname(__file__), "classifier", "archive", "v4_stems_lasso.pkl")
_SONICS_LR_PATH     = os.path.join(os.path.dirname(__file__), "classifier", "structural_logreg_sonics.pkl")
_SONICS_LASSO_PATH  = os.path.join(os.path.dirname(__file__), "classifier", "structural_logreg_sonics_lasso.pkl")


def _sonics_models_available() -> bool:
    return os.path.isfile(_SONICS_LR_PATH) and os.path.isfile(_SONICS_LASSO_PATH)


def _stems_v2_models_available() -> bool:
    return os.path.isfile(_STEMS_V2_LR_PATH) and os.path.isfile(_STEMS_LASSO_PATH)


def _stems_models_available() -> bool:
    return os.path.isfile(_STEMS_LOGREG_PATH) and os.path.isfile(_STEMS_RF_PATH)


def _audio_models_available() -> bool:
    return os.path.isfile(_AUDIO_LOGREG_PATH) and os.path.isfile(_AUDIO_RF_PATH)


def _classify_with_structural_models(
    midi_path: str,
) -> tuple[float, "float | None", "float | None", str]:
    """
    Run the best available structural models on midi_path.

    Priority: SONICS (28-feat) → stems v2 (28-feat) → FMA audio (14-feat) → error.
    Returns (lr_prob, rf_prob_or_None, lasso_prob_or_None, source).
    Exactly one of rf_prob / lasso_prob will be non-None.
    """
    if _sonics_models_available():
        from extract_features import extract_features_28
        vec = extract_features_28(midi_path).reshape(1, -1)
        with open(_SONICS_LR_PATH, "rb") as fh:
            lr_pipe = pickle.load(fh)
        with open(_SONICS_LASSO_PATH, "rb") as fh:
            lasso_pipe = pickle.load(fh)
        source     = "SONICS dataset (500 Suno/Udio vs 464 real songs, matched Demucs+BasicPitch pipeline)"
        lr_prob    = round(float(lr_pipe.predict_proba(vec)[0, 1]), 4)
        lasso_prob = round(float(lasso_pipe.predict_proba(vec)[0, 1]), 4)
        return lr_prob, None, lasso_prob, source
    elif _stems_v2_models_available():
        from extract_features import extract_features_28
        vec = extract_features_28(midi_path).reshape(1, -1)
        with open(_STEMS_V2_LR_PATH, "rb") as fh:
            lr_pipe = pickle.load(fh)
        with open(_STEMS_LASSO_PATH, "rb") as fh:
            lasso_pipe = pickle.load(fh)
        source     = "Suno stem transcriptions vs Lakh MIDI (28-feat melody-aware)"
        lr_prob    = round(float(lr_pipe.predict_proba(vec)[0, 1]), 4)
        lasso_prob = round(float(lasso_pipe.predict_proba(vec)[0, 1]), 4)
        return lr_prob, None, lasso_prob, source
    elif _audio_models_available():
        vec = _extract_features(midi_path).reshape(1, -1)
        with open(_AUDIO_LOGREG_PATH, "rb") as fh:
            lr_pipe = pickle.load(fh)
        with open(_AUDIO_RF_PATH, "rb") as fh:
            rf_pipe = pickle.load(fh)
        source  = "FMA audio transcriptions"
        lr_prob = round(float(lr_pipe.predict_proba(vec)[0, 1]), 4)
        rf_prob = round(float(rf_pipe.predict_proba(vec)[0, 1]), 4)
        return lr_prob, rf_prob, None, source
    else:
        raise FileNotFoundError("No structural models found.")



FEATURE_PLAIN_ENGLISH = {
    "stepwise_ratio":      "Notes move by step (smooth voice leading)",
    "interval_step":       "Step-size intervals between notes",
    "interval_mean":       "Average melodic interval size",
    "interval_large_leap": "Large leaps of 8+ semitones between notes",
    "harm_rhythm_mean":    "Rate of harmonic change per second",
    "rhythmic_cv":         "Variability in note timing",
    "pitch_range":         "Span from lowest to highest note",
    "note_density":        "Number of notes per second",
    "harm_rhythm_std":     "Consistency of harmonic rhythm",
    "interval_std":        "Spread of interval sizes used",
    "interval_skip":       "Skip-size intervals (3-4 semitones)",
    "interval_unison":     "Repeated pitches",
    "interval_leap":       "Leap-size intervals (5-7 semitones)",
    "cadence_count":       "Number of phrase-ending cadences",
}


def _lr_attribution(vec, lr_pipe, feature_names=None):
    """Return (top_ai_features, top_human_features) — 3 items each.

    Each item: {feature_name, contribution, plain_english}.
    Contribution = scaled_value * coef, giving the signed push toward AI (positive)
    or human (negative) in log-odds space.
    Accepts optional feature_names for 28-feature models (mel_* / all_* prefixes).
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    def get_plain(fname):
        if fname in FEATURE_PLAIN_ENGLISH:
            return FEATURE_PLAIN_ENGLISH[fname]
        for prefix, tag in [("mel_", "[Melody] "), ("all_", "[All notes] ")]:
            if fname.startswith(prefix):
                base = fname[len(prefix):]
                if base in FEATURE_PLAIN_ENGLISH:
                    return tag + FEATURE_PLAIN_ENGLISH[base].lower()
        return fname

    scaler   = lr_pipe.named_steps["scaler"]
    lr       = lr_pipe.named_steps["lr"]
    scaled   = (vec - scaler.mean_) / scaler.scale_
    contribs = scaled * lr.coef_[0]

    items = sorted(
        [
            {
                "feature_name":  feature_names[i],
                "contribution":  round(float(contribs[i]), 4),
                "plain_english": get_plain(feature_names[i]),
            }
            for i in range(len(feature_names))
        ],
        key=lambda x: x["contribution"],
    )
    top_human = items[:3]
    top_ai    = list(reversed(items))[:3]
    return top_ai, top_human


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/model_registry")
def model_registry():
    path = os.path.join(os.path.dirname(__file__), "classifier", "model_registry.json")
    if not os.path.isfile(path):
        return jsonify({"error": "model_registry.json not found"}), 404
    with open(path) as fh:
        import json
        return jsonify(json.load(fh))


def _midi_bpm(midi_path: str) -> float:
    """Return the first MetronomeMark BPM found in the MIDI, or 120.0."""
    try:
        score = music21.converter.parse(midi_path)
        for el in score.flatten().getElementsByClass(music21.tempo.MetronomeMark):
            return float(el.number)
    except Exception:
        pass
    return 120.0


@app.route("/analyze", methods=["POST"])
def analyze():
    # --- validate upload ---
    if "audio" not in request.files:
        return jsonify({"error": "No file field named 'audio' in request."}), 400

    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": "Only MP3 and WAV files are supported."}), 400

    suffix = os.path.splitext(file.filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(
        dir=UPLOAD_FOLDER, suffix=suffix, delete=False
    )
    audio_path = tmp.name
    try:
        file.save(audio_path)
        tmp.close()

        with tempfile.TemporaryDirectory() as work_dir:
            midi_path, stem_separated = transcribe_audio(audio_path, work_dir)
            notes     = parse_midi(midi_path)

            if not notes:
                return jsonify({"error": "No notes detected in audio."}), 422

            G = build_note_graph(notes)

            # Piano-roll PNG → base64.
            png_path = os.path.join(work_dir, "graph.png")
            visualize_graph(G, png_path)
            with open(png_path, "rb") as fh:
                png_b64 = base64.b64encode(fh.read()).decode("utf-8")

            # --- Classifiers (null if model files are missing) ---
            cnn_prob = None
            try:
                cnn_prob = classify_midi(midi_path)
            except FileNotFoundError:
                pass

            lr_prob = rf_prob = lasso_prob = None
            classifier_source = None
            try:
                lr_prob, rf_prob, lasso_prob, classifier_source = \
                    _classify_with_structural_models(midi_path)
            except FileNotFoundError:
                try:
                    lr_prob, rf_prob = classify_structural(midi_path)
                    lasso_prob       = None
                    classifier_source = "Lakh MIDI files"
                except FileNotFoundError:
                    pass

            # --- Key structural features + LR attribution ---
            # If a 28-feature model is active, extract_features_28 provides both
            # melody and all-notes features; use the all-notes half for display.
            vec = None
            vec28 = None
            features = None
            try:
                if _sonics_models_available() or _stems_v2_models_available():
                    from extract_features import extract_features_28, FEATURE_NAMES_28
                    vec28 = extract_features_28(midi_path)
                    vec   = vec28[14:]   # all-notes features are the second 14
                else:
                    vec = _extract_features(midi_path)
                features = {
                    name: round(float(vec[i]), 4)
                    for i, name in enumerate(FEATURE_NAMES)
                }
            except Exception:
                pass

            lr_attribution = None
            if vec is not None:
                try:
                    if _sonics_models_available():
                        from extract_features import FEATURE_NAMES_28
                        attr_vec   = vec28 if vec28 is not None else vec
                        feat_names = FEATURE_NAMES_28
                        with open(_SONICS_LR_PATH, "rb") as fh:
                            lr_pipe_attr = pickle.load(fh)
                    elif _stems_v2_models_available():
                        from extract_features import FEATURE_NAMES_28
                        attr_vec   = vec28 if vec28 is not None else vec
                        feat_names = FEATURE_NAMES_28
                        with open(_STEMS_V2_LR_PATH, "rb") as fh:
                            lr_pipe_attr = pickle.load(fh)
                    elif _audio_models_available():
                        attr_vec   = vec
                        feat_names = FEATURE_NAMES
                        with open(_AUDIO_LOGREG_PATH, "rb") as fh:
                            lr_pipe_attr = pickle.load(fh)
                    else:
                        attr_vec   = vec
                        feat_names = FEATURE_NAMES
                        with open(_LOGREG_PATH, "rb") as fh:
                            lr_pipe_attr = pickle.load(fh)
                    top_ai, top_human = _lr_attribution(attr_vec, lr_pipe_attr, feat_names)
                    lr_attribution = {
                        "top_ai_features":    top_ai,
                        "top_human_features": top_human,
                    }
                except Exception:
                    pass

            tempo = _midi_bpm(midi_path)

            # --- Artist fingerprint matching (null if npz missing) ---
            artist_matches = None
            try:
                artist_matches = _match_artist(midi_path)
            except FileNotFoundError:
                pass

            # --- Classifier disagreement flag ---
            secondary = lasso_prob if lasso_prob is not None else rf_prob
            available = [p for p in [cnn_prob, lr_prob, secondary] if p is not None]
            if len(available) >= 2:
                score_range = max(available) - min(available)
                uncertain = score_range > 0.25
                if uncertain:
                    named = {}
                    if cnn_prob is not None:   named["CNN"] = cnn_prob
                    if lr_prob  is not None:   named["Logistic Regression"] = lr_prob
                    if lasso_prob is not None: named["Sparse LR (L1)"] = lasso_prob
                    elif rf_prob is not None:  named["Random Forest"] = rf_prob
                    hi = max(named, key=named.get)
                    lo = min(named, key=named.get)
                    disagreement_note = (
                        f"{hi} ({round(named[hi]*100)}% AI) and "
                        f"{lo} ({round(named[lo]*100)}% AI) "
                        f"disagree by {round(score_range*100)}%."
                    )
                else:
                    disagreement_note = None
            else:
                score_range = None
                uncertain = False
                disagreement_note = None

        return jsonify({
            "piano_roll_image":       png_b64,
            "piano_roll_probability": cnn_prob,
            "logreg_probability":     lr_prob,
            "rf_probability":         rf_prob,
            "lasso_probability":      lasso_prob,
            "note_count":             len(notes),
            "tempo":                  tempo,
            "features":               features,
            "uncertain":              uncertain,
            "score_range":            round(score_range, 4) if score_range is not None else None,
            "disagreement_note":      disagreement_note,
            "lr_attribution":         lr_attribution,
            "artist_matches":         artist_matches,
            "classifier_source":      classifier_source,
            "stem_separated":         stem_separated,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.route("/classify", methods=["POST"])
def classify():
    """
    Accept a MIDI file upload, run the piano-roll classifier, and return
    the AI-generated probability as JSON: {"ai_probability": 0.87}
    """
    if "midi" not in request.files:
        return jsonify({"error": "No file field named 'midi' in request."}), 400

    file = request.files["midi"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".mid", ".midi"}:
        return jsonify({"error": "Only .mid / .midi files are accepted."}), 400

    tmp = tempfile.NamedTemporaryFile(
        dir=UPLOAD_FOLDER, suffix=ext, delete=False
    )
    midi_path = tmp.name
    try:
        file.save(midi_path)
        tmp.close()

        prob = classify_midi(midi_path)
        return jsonify({"ai_probability": prob})

    except FileNotFoundError as exc:
        # Model weights not yet trained.
        return jsonify({"error": str(exc)}), 503

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)


@app.route("/classify_structural", methods=["POST"])
def classify_structural_route():
    """
    Accept a MIDI file, run both structural classifiers, and return:
        {"logreg_probability": 0.84, "rf_probability": 0.79}
    Returns 503 if the models have not been trained yet.
    """
    if "midi" not in request.files:
        return jsonify({"error": "No file field named 'midi' in request."}), 400

    file = request.files["midi"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".mid", ".midi"}:
        return jsonify({"error": "Only .mid / .midi files are accepted."}), 400

    tmp = tempfile.NamedTemporaryFile(
        dir=UPLOAD_FOLDER, suffix=ext, delete=False
    )
    midi_path = tmp.name
    try:
        file.save(midi_path)
        tmp.close()

        lr_prob, rf_prob = classify_structural(midi_path)
        return jsonify({"logreg_probability": lr_prob, "rf_probability": rf_prob})

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

    finally:
        if os.path.exists(midi_path):
            os.remove(midi_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5001)
