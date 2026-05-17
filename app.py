"""
app.py — Flask web interface for the music structural analysis pipeline.

Routes:
    GET  /        — serves the upload UI (templates/index.html)
    POST /analyze  — accepts an audio file, runs the pipeline, returns JSON
    POST /classify — accepts a MIDI file, returns AI-generated probability
    GET  /health  — liveness check

Uploaded files are written to uploads/, processed, then deleted.
All pipeline logic lives in pipeline.py and is imported directly.
Classifier logic lives in predict.py and is imported directly.
"""

import os
import base64
import tempfile
import traceback

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

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"mp3", "wav"}


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

    # --- save upload to a temp file inside uploads/ ---
    # We use a NamedTemporaryFile so the path is unique even under concurrency,
    # and we can hand the path to pipeline functions that expect a file path.
    suffix = os.path.splitext(file.filename)[1].lower()
    tmp = tempfile.NamedTemporaryFile(
        dir=UPLOAD_FOLDER, suffix=suffix, delete=False
    )
    audio_path = tmp.name
    try:
        file.save(audio_path)
        tmp.close()

        # Run the pipeline inside a dedicated temp dir for intermediate files
        # (MIDI, PNG) so we never touch the repo's out/ folder.
        with tempfile.TemporaryDirectory() as work_dir:
            midi_path = transcribe_audio(audio_path, work_dir)
            notes     = parse_midi(midi_path)

            if not notes:
                return jsonify({"error": "No notes detected in audio."}), 422

            G = build_note_graph(notes)

            # Render the piano-roll PNG and encode it for the browser.
            png_path = os.path.join(work_dir, "graph.png")
            visualize_graph(G, png_path)
            with open(png_path, "rb") as fh:
                png_b64 = base64.b64encode(fh.read()).decode("utf-8")

            # Build the graph payload the frontend will display as JSON.
            graph_data = {
                "nodes": [
                    {"id": nid, **attrs}
                    for nid, attrs in G.nodes(data=True)
                ],
                "edges": [
                    {"source": u, "target": v, "edge_type": d["edge_type"]}
                    for u, v, d in G.edges(data=True)
                ],
            }

            depth_scores = run_structural_analysis(G)

        return jsonify({
            "graph":        graph_data,
            "depth_scores": depth_scores,
            "image_b64":    png_b64,
            "stats": {
                "notes": len(notes),
                "edges": G.number_of_edges(),
            },
        })

    except Exception:
        # Surface the full traceback to the caller so errors are debuggable
        # without needing to tail server logs.
        return jsonify({"error": traceback.format_exc()}), 500

    finally:
        # Always clean up the uploaded file, even if the pipeline failed.
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5001)
