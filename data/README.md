# Data Directory

## What is NOT included in this repo

These directories and files are excluded for legal and size reasons.
The `.gitignore` at the repo root enforces these exclusions.

| Path | Reason | How to obtain |
|------|--------|---------------|
| `data/sonics_fake/` | 35 GB of AI-generated MP3s | Download from Kaggle: [awsaf49/sonics-dataset](https://www.kaggle.com/datasets/awsaf49/sonics-dataset) (CC BY-NC 4.0) |
| `data/sonics_real/` | YouTube downloads of real songs — not redistributed due to copyright | Download via `yt-dlp` using YouTube IDs from SONICS `real_songs.csv` (see Colab notebook) |
| `data/ai/` | Original Suno MP3s used in early prototype (v1–v3 models) | N/A — superseded by SONICS dataset |
| `data/human/midi/` | 200 Lakh MIDI files used in early prototype | Download from [hog.ee.columbia.edu/craffel/lmd](http://hog.ee.columbia.edu/craffel/lmd) |
| `data/human_audio/` | FMA Small audio used for v2 models | Download from [os.unil.cloud.switch.ch/fma/fma_small.zip](http://os.unil.cloud.switch.ch/fma/fma_small.zip) |
| `fma_small/` | FMA Small dataset root | Same as above |
| `data/artists/` | Per-artist MIDI feature archives | Rebuild with `build_fingerprints.py` after obtaining source MIDIs |
| `clean_midi/` | Clean MIDI corpus | Various sources — not redistributed |
| `stems/` | Pre-separated Suno stems (WAV) | N/A — superseded |

## What IS included

| Path | Contents |
|------|----------|
| `data/ai_stems/` | 42 MIDI files transcribed from Suno stems using Demucs + Basic Pitch (early prototype, used for v3–v4 model training) |
| `data/ai/.gitkeep` | Placeholder maintaining folder structure |
| `data/human/.gitkeep` | Placeholder maintaining folder structure |
| `data/rolls/ai/.gitkeep` | Placeholder for piano-roll PNG output |
| `data/rolls/human/.gitkeep` | Placeholder for piano-roll PNG output |

## Reproducing the SONICS v5 training data

To reproduce the full training run from scratch:

1. Download `awsaf49/sonics-dataset` from Kaggle → extract to `data/sonics_fake/`
2. Download real songs via YouTube IDs in SONICS `real_songs.csv` → `data/sonics_real/`
   (use `yt-dlp`; see `StructuralPrint_Training.ipynb` for the full Colab pipeline)
3. Run Demucs + Basic Pitch on both sets → MIDI files in `data/midi/fake/` and `data/midi/real/`
4. Run `extract_features_28()` on all MIDIs → feature matrix
5. Retrain classifiers → place PKL files in `classifier/`

The `StructuralPrint_Training.ipynb` notebook automates steps 2–5 on Google Colab.

## Feature vectors

`.npz` feature matrices (`data/features.npz`, `data/features_stems.npz`, etc.) are not
committed — they are excluded by `*.npz` in `.gitignore`. The trained classifiers in
`classifier/` are the primary shareable artifact from each training run.
