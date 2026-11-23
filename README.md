# StructuralPrint

StructuralPrint detects AI-generated music by analysing **compositional structure** rather
than acoustic surface. Instead of listening for AI "sound", it transcribes audio to MIDI
and measures features of the underlying note-graph: melodic interval distributions,
harmonic rhythm, rhythmic variability, phrase structure, and voice-leading patterns.
The main finding from the SONICS v5 training run is that modern AI music generators
(Suno, Udio) are **structurally nearly indistinguishable** from human commercial music —
Cohen's d < 0.35 on all 16 purely compositional features extracted from 964 matched songs,
with the two strongest signals (`all_interval_leap` d=0.325, `mel_interval_std` d=0.229)
both small enough that no reliable classifier can be built on them alone.

## Pipeline

```
Audio (MP3/WAV)
  └─ Demucs htdemucs ──────────── stem separation (vocals, bass, other; drums excluded)
       └─ Basic Pitch ONNX ──────── per-stem polyphonic transcription → MIDI
            └─ music21 ──────────── merge stems → multi-track MIDI, parse note events
                 └─ extract_features_28()
                      ├─ 14 melody features (mel_*)   ← highest-pitch part as melody proxy
                      └─ 14 all-notes features (all_*) ← full merged note stream
                           └─ Logistic Regression classifiers
                                ├─ structural_logreg_sonics.pkl      (L2, primary)
                                └─ structural_logreg_sonics_lasso.pkl (L1, interpretable)
```

A piano-roll CNN (`model.pth`) runs in parallel as a second opinion based on the
visual texture of the MIDI rather than hand-crafted structural features.

Artist fingerprinting (`match_artist.py`) compares the structural feature vector of an
uploaded file against a library of artist fingerprints built from known MIDI files,
returning similarity scores for the nearest matches.

## Key Finding

On 964 songs from the SONICS dataset — 500 AI-generated (Suno + Udio) and 464 real
(YouTube downloads), all processed through the same Demucs + Basic Pitch pipeline to
eliminate transcription confounds — no compositional feature shows more than a small
effect size:

| Feature | Cohen's d | Direction |
|---------|-----------|-----------|
| `all_interval_leap` | 0.325 | AI slightly more large-leap intervals |
| `mel_interval_std` | 0.229 | AI melody slightly more variable in interval size |
| All other 14 features | < 0.2 | Negligible |

The logistic regression classifiers achieve ~72–74% accuracy on a held-out test set,
barely above a majority-class baseline of ~52% given the near-balanced dataset.
**This is not a failure of the classifier — it is the correct result.** Modern AI music
generators have learned to produce music that is structurally indistinguishable from
human commercial music at the note level.

## Setup

Requires Python 3.10+ and [FFmpeg](https://ffmpeg.org/) (for Demucs audio loading).

```bash
git clone <repo>
cd music-structural-fingerprint

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Install torchcodec (required by torchaudio 2.11+ for audio loading)
pip install torchcodec --index-url https://download.pytorch.org/whl/cpu

# Install FFmpeg (macOS)
brew install ffmpeg
# Ubuntu/Debian: sudo apt install ffmpeg
```

Run the Flask web app:

```bash
python app.py
# Open http://localhost:5001 and upload an MP3 or WAV
```

Upload an audio file via the web UI or run the pipeline directly:

```bash
python pipeline.py path/to/audio.mp3 --output-dir out/
```

## Demo (fast, pre-baked)

The shippable demo runs on precomputed data only — no Demucs / Basic Pitch / torch
needed at runtime, so it loads instantly and always works. It shows the core finding
(note-level vs. macro-structural vs. merged), honest held-out metrics, and an
interactive gallery of 16 out-of-sample songs with a per-feature "why" breakdown
and a guess-first *Detective mode*.

```bash
source venv/bin/activate
python train_merged_model.py     # trains + saves classifier/structural_merged38.pkl
python build_demo_data.py        # writes demo/gallery.json
python serve_demo.py             # open http://localhost:5002
```

The improved **v6 merged 38-feature model** (28 note-level + 10 macro-structural,
HistGradientBoosting) reaches **0.83 held-out accuracy / 0.92 AUC** — a real jump
over the note-level-only model (~0.73 / 0.80). Macro features are audio-derived
(chroma self-similarity, chorus edit-distance, chord compression), so they require
audio, not just MIDI — which is why the demo ships pre-computed.

## Data

### AI training data
SONICS dataset — 500 fake songs (Suno + Udio), CC BY-NC 4.0.
Download from Kaggle: [awsaf49/sonics-dataset](https://www.kaggle.com/datasets/awsaf49/sonics-dataset).
Not included in this repo (35 GB).

### Real song training data
Not included in this repo. Obtained via YouTube IDs from SONICS `real_songs.csv`.
Not redistributed due to copyright. See `data/README.md` for reproduction instructions.

### Feature vectors
Feature CSVs used for the SONICS v5 training run are available on request but not
committed to the repo due to size. The trained models in `classifier/` are the
primary shareable artifact.

See `data/README.md` for the full data directory layout and what is and is not included.

## Model Versions

See [`classifier/README.md`](classifier/README.md) for the full version history.

Current models (v5, SONICS):
- `classifier/structural_logreg_sonics.pkl` — L2 logistic regression, 28 features, accuracy=0.72, F1=0.74
- `classifier/structural_logreg_sonics_lasso.pkl` — L1 lasso, 28 features, accuracy=0.74, F1=0.76

Archived models (v1–v4) are in `classifier/archive/`. Full training metadata for all
versions is in `classifier/model_registry.json`.

## Project Structure

```
.
├── app.py                      # Flask web app (upload UI + /analyze route)
├── pipeline.py                 # Core pipeline: transcribe → parse → graph
├── extract_features.py         # 28-feature structural extraction (mel_* + all_*)
├── separate_stems.py           # Demucs htdemucs wrapper
├── transcribe_stems.py         # Per-stem Basic Pitch + GM instrument assignment
├── predict.py                  # Piano-roll CNN inference
├── predict_structural.py       # Structural classifier inference
├── match_artist.py             # Artist fingerprint similarity
├── evaluate_batch.py           # Batch evaluation across a MIDI folder
├── retrain_on_stems.py         # Retrain structural classifiers on stems data
├── make_rolls.py               # Generate piano-roll PNGs for CNN training
├── train_classifier.py         # Train the piano-roll CNN
├── classifier/
│   ├── structural_logreg_sonics.pkl       # Current primary (L2)
│   ├── structural_logreg_sonics_lasso.pkl # Current primary (L1)
│   ├── model_registry.json               # Full metadata for all model versions
│   ├── README.md                         # Model version documentation
│   └── archive/                          # v1–v4 prototype models
├── data/
│   ├── ai_stems/               # 42 Suno stem MIDIs (early prototype)
│   └── README.md               # Data directory documentation
└── templates/
    └── index.html              # Single-page upload + results UI
```

## Citations

**SONICS dataset:**
> Awsaf, Abdullah Al, et al. "SONICS: Synthetic Or Not — Identifying Contrasted Songs."
> *arXiv preprint* arXiv:2408.14858 (2024).
> Dataset: [kaggle.com/datasets/awsaf49/sonics-dataset](https://www.kaggle.com/datasets/awsaf49/sonics-dataset)

**AutoSchA (Schenkerian graph representation this project's note graph is based on):**
> Ji, Shiqi, et al. "AutoSchA: Automatic Schenkerian Analysis via Graph Neural Networks."
> *ISMIR 2023.*
