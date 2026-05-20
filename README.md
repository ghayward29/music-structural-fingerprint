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

## Classifiers: surface vs structural

Two complementary approaches classify a MIDI file as AI-generated or
human-composed.  They answer different questions and are deliberately kept
independent so each can be evaluated and improved separately.

### Surface classifier (piano-roll CNN)

**What it measures:** the visual appearance of the piano roll — which pixels
are lit in a 512×128 grayscale image.  Captures note density patterns, rhythmic
grids, and broad register usage that are visible at a glance.

**Why it works:** AI-generated MIDIs from models like Suno often have
characteristic textures — very even rhythmic grids, limited pitch variety in
specific registers — that the CNN can learn from the image alone.

**Why it's evasable:** re-quantizing, adding humanisation, or transposing a
piece changes the piano roll substantially without changing its musical
structure.  An adversary who knows the classifier uses images can modify the
roll to shift predictions.

### Structural classifier (logistic regression + random forest)

**What it measures:** 14 hand-crafted features derived from music21 note data:
interval distributions, harmonic rhythm, rhythmic CoV, stepwise motion ratio,
pitch range, and phrase-ending cadences.  These target the compositional logic
underneath the surface.

**Why it's harder to evade:** these features describe *how* notes relate to
each other structurally — voice-leading intervals, harmonic pacing, phrase
articulation.  Changing timing or register shifts the piano roll but leaves
most structural features intact.  An adversary would need to alter the
underlying compositional decisions, not just the surface appearance.

**Two models, two purposes:**
- **Logistic regression** is interpretable: each feature has a signed
  coefficient directly showing its direction and magnitude.  Use it to
  understand *which* structural properties differ between AI and human music.
- **Random forest** captures non-linear interactions (e.g. "high stepwise ratio
  AND low rhythmic CV together") that logistic regression misses.  Use it when
  accuracy matters more than interpretability.

### Known limitations

- **Training data is small.** ~100 AI files and ~200 human files is far too few
  for robust generalisation.  Both classifiers will likely overfit.  Treat
  predictions as exploratory signals, not reliable judgements.
- **Tonic detection is approximate.** The cadence feature estimates the tonic as
  the most common pitch class, which is wrong for modal, chromatic, or
  modulatory music.
- **AI samples are Suno-specific.** The classifier is trained on one AI source.
  It may fail to detect AI music from Udio, MusicLM, or other generators.
- **Human samples are not performance-diverse.** The Lakh MIDI Dataset is drawn
  from commercial MIDI transcriptions, which are themselves often quantized and
  may not reflect the full range of human compositional styles.

## AI vs Human Classifier

A CNN classifier that predicts whether a piece of music is AI-generated or
human-composed, based solely on its piano-roll image.

### How it works

Each MIDI file is rendered as a 512×128 grayscale piano-roll image (time on
the x-axis, pitch on the y-axis, time-normalised to fill the full width).  A
small CNN is trained to distinguish the visual patterns of human vs AI
composition.

Architecture: `Conv(1→16) → ReLU → MaxPool` → `Conv(16→32) → ReLU → MaxPool`
→ `AdaptiveAvgPool(4×4)` → `Linear(512→1)` → Sigmoid.

### Step 1 — Collect training data

Drop MIDI files into the appropriate folders:
```
data/human/   ← MIDI files of human-composed music
data/ai/      ← MIDI files of AI-generated music
```

### Step 2 — Generate piano rolls

```bash
python make_rolls.py --input data/human/ --output data/rolls/human/
python make_rolls.py --input data/ai/    --output data/rolls/ai/
```

Each MIDI becomes a 512×128 grayscale PNG saved in the rolls folder.

### Step 3 — Train the classifier

```bash
python train_classifier.py
# optional flags: --epochs 30  --lr 0.0005  --rolls data/rolls/
```

Trains for 20 epochs (80/20 split), prints accuracy and loss per epoch, saves:
- `classifier/model.pth` — model weights
- `classifier/training_curves.png` — accuracy plot

### Step 4 — Predict on a new file

```bash
python predict.py path/to/file.mid
# AI-generated probability: 0.87
# → Likely AI-generated
```

### Via the web app

The `/classify` route accepts a MIDI upload and returns:
```json
{ "ai_probability": 0.87 }
```
The probability bar is shown below the piano-roll visualisation in the UI.

## Web app

A Flask UI wraps the pipeline so you can upload audio from the browser.

```bash
source venv/bin/activate
python app.py
```

Open [http://localhost:5000](http://localhost:5000), upload an MP3 or WAV, and
click **Run Analysis**. The page returns the piano-roll visualisation and the
raw JSON graph. The server exposes three routes:

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Upload UI |
| `/analyze` | POST | Runs the pipeline; returns `{graph, image_b64, stats}` |
| `/health` | GET | Returns `{"status": "ok"}` |

## Dependencies

| Package | Purpose |
|---------|---------|
| `basic-pitch` | Polyphonic audio-to-MIDI transcription (Spotify, ICASSP 2022) |
| `music21` | Symbolic music parsing — pitch names, rhythm, tempo |
| `networkx` | Directed note graph construction and manipulation |
| `matplotlib` | Piano-roll visualisation |
| `numpy` | Array handling (transitively required by all of the above) |
