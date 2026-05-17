# music-structural-fingerprint

A Python prototype for music structural analysis. Transcribes audio to MIDI,
extracts notes, builds a directed note graph, and exports it as JSON + a
piano-roll visualisation. Designed as the data-preparation stage for
AutoSchA-style Schenkerian depth assignment via GNN.

## Pipeline overview

```
Audio (MP3/WAV)
  └─ Basic Pitch ──────────────── polyphonic transcription → MIDI
       └─ music21 ─────────────── note parsing (pitch, onset, offset, velocity)
            └─ networkx ────────── directed note graph
                 ├─ JSON export ── {nodes, edges} for downstream GNN input
                 └─ matplotlib ─── piano-roll visualisation (PNG)
```

Three edge types mirror the AutoSchA paper's graph representation:

| Edge type | Colour | Meaning |
|-----------|--------|---------|
| `onset`   | red    | Notes overlap in time — harmonic simultaneity |
| `voice`   | green  | Smooth melodic succession (gap < 100 ms, pitch within one octave) |
| `rest`    | blue   | Same-voice succession across a perceivable silence |

## Setup

Requires Python 3.10+.

```bash
git clone <repo>
cd music-structural-fingerprint

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** Basic Pitch requires at least one inference backend.
> The default install uses the bundled ONNX/TFLite model.
> If you see backend warnings at runtime they are harmless — the pipeline
> falls back automatically.

## Usage

```bash
python pipeline.py path/to/audio.mp3
```

By default all outputs land in the current directory. Use `--output-dir` to
redirect them:

```bash
python pipeline.py path/to/audio.wav --output-dir out/
```

This produces three files:

| File | Description |
|------|-------------|
| `<name>.mid` | MIDI transcription from Basic Pitch |
| `<name>_graph.json` | Note graph — nodes with musical features, edges with type labels |
| `<name>_graph.png` | Piano-roll visualisation of the note graph |

## JSON format

```json
{
  "nodes": [
    {
      "id": 0,
      "pitch": 60,
      "pitch_name": "C4",
      "onset": 0.0,
      "offset": 0.5,
      "duration": 0.5,
      "velocity": 80
    }
  ],
  "edges": [
    { "source": 0, "target": 1, "edge_type": "onset" },
    { "source": 1, "target": 2, "edge_type": "voice" },
    { "source": 2, "target": 3, "edge_type": "rest"  }
  ]
}
```

## Structural analysis stub

`run_structural_analysis(graph)` in `pipeline.py` is a placeholder for
AutoSchA GNN inference. It currently returns `0.5` for every node. The
comment in that function describes what the full implementation would do:
convert the graph to a PyTorch Geometric `Data` object, run it through the
trained AutoSchA model, and return per-note Schenkerian depth scores
(0.0 = ornamental, 1.0 = deep structural).

## Dependencies

| Package | Purpose |
|---------|---------|
| `basic-pitch` | Polyphonic audio-to-MIDI transcription (Spotify, ICASSP 2022) |
| `music21` | Symbolic music parsing — pitch names, rhythm, tempo |
| `networkx` | Directed note graph construction and manipulation |
| `matplotlib` | Piano-roll visualisation |
| `numpy` | Array handling (transitively required by all of the above) |
