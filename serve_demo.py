"""
serve_demo.py — StructuralPrint demo server (fast, pre-baked, always-works).

Serves a single-page interactive demo built entirely from precomputed data
(classifier/merged38_meta.json + demo/gallery.json). No Demucs / Basic Pitch /
torch needed at runtime, so it loads instantly and never stalls on CPU audio
work. This is the "demoable product" surface; app.py remains the full live
audio pipeline.

Build the data first (once):
    python train_merged_model.py
    python build_demo_data.py

Then run:
    python serve_demo.py
    open http://localhost:5002
"""
from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, render_template, jsonify, abort

ROOT = Path(__file__).resolve().parent
GALLERY = ROOT / "demo" / "gallery.json"

app = Flask(__name__)


def _load_gallery() -> dict:
    if not GALLERY.exists():
        abort(503, "demo/gallery.json missing — run build_demo_data.py first.")
    return json.loads(GALLERY.read_text())


@app.route("/")
def index():
    data = _load_gallery()
    return render_template("demo.html", data_json=json.dumps(data))


@app.route("/api/gallery")
def api_gallery():
    return jsonify(_load_gallery())


@app.route("/health")
def health():
    return jsonify({"status": "ok", "gallery": GALLERY.exists()})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5002)
