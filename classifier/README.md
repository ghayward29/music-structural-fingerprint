# Classifier Models

## Current Models — SONICS Dataset (v5)

| File | Type | Accuracy | F1 |
|------|------|----------|----|
| `structural_logreg_sonics.pkl` | L2 logistic regression, 28 features | 0.72 | 0.74 |
| `structural_logreg_sonics_lasso.pkl` | L1 lasso, 28 features | 0.74 | 0.76 |
| `model.pth` | Piano-roll CNN (early prototype, not retrained on SONICS) | — | — |

**Training data:** 500 AI-generated songs (Suno + Udio, SONICS dataset) vs 464 real songs
(YouTube downloads via SONICS `real_songs.csv`, not redistributed).

**Pipeline:** Demucs htdemucs stem separation → Basic Pitch ONNX transcription per stem →
merged multi-track MIDI → 28 structural features (14 melody-only `mel_*` + 14 all-notes `all_*`).

**Key finding:** Cohen's d < 0.35 on all 16 purely compositional features extracted from
964 matched songs. Only two features show even small effect sizes:
- `all_interval_leap` d = 0.325 (AI slightly more large-leap intervals across all notes)
- `mel_interval_std` d = 0.229 (AI melody slightly more variable in interval size)

Modern AI music generators (Suno, Udio) are **structurally nearly indistinguishable** from
human commercial music at the note level. The classifiers learn real but weak signals;
they are not reliable detectors.

**L1 survivors** (features with non-zero coefficients after lasso regularisation):
`mel_interval_std` (+1.78, → AI), `mel_interval_large_leap` (−0.87, → human),
`mel_harm_rhythm_std` (−0.63, → human), `mel_pitch_range` (−0.61, → human),
`mel_rhythmic_cv` (−0.58, → human). All are melody-level features, suggesting
what small signal exists lives in melodic rather than harmonic structure.

## Archived Models

See `archive/` subfolder. Models v1–v4 used progressively cleaner training data but
all suffered from training/inference mismatches or genre confounds:

| Version | Files | Training data | Key flaw |
|---------|-------|---------------|----------|
| v1 | `v1_lakh_logreg.pkl`, `v1_lakh_rf.pkl` | Lakh MIDI (human) vs Suno Basic Pitch (AI) | Human train data is native MIDI; inference goes through Basic Pitch. Classifier partially detects transcription artifacts, not composition. |
| v2 | `v2_fma_logreg.pkl`, `v2_fma_rf.pkl` | FMA Small audio → Basic Pitch (human) vs Suno Basic Pitch (AI) | Both classes through Basic Pitch (coherent), but FMA genre distribution (electronic/experimental) doesn't match typical uploaded content. |
| v3 | `v3_stems_logreg.pkl`, `v3_stems_rf.pkl` | Suno stems → Demucs+BasicPitch (AI) vs Lakh native MIDI (human) | Stem-separated AI vs native MIDI human — pipeline mismatch persists for human class. |
| v4 | `v4_stems_v2_logreg.pkl`, `v4_stems_lasso.pkl` | Suno stems → Demucs+BasicPitch (AI) vs Lakh native MIDI (human) | Same mismatch as v3; added melody-aware 28-feature extraction but human data still native MIDI. |
| v5 (current) | `structural_logreg_sonics.pkl`, `_lasso.pkl` | SONICS fake → Demucs+BasicPitch (AI) vs SONICS real → Demucs+BasicPitch (human) | No pipeline mismatch. First fully coherent training run. Larger dataset (964 vs ~240 samples). |

## Feature Set

The 28-feature vector is computed by `extract_features_28()` in `extract_features.py`:

- **14 melody features** (`mel_*`): extracted from the highest-pitch part (melody proxy)
- **14 all-notes features** (`all_*`): extracted from the merged full-score note stream

Base features (applied to both melody and all-notes):
`note_density`, `rhythmic_cv`, `interval_mean`, `interval_std`, `interval_step`,
`interval_skip`, `interval_leap`, `interval_large_leap`, `interval_unison`,
`stepwise_ratio`, `pitch_range`, `harm_rhythm_mean`, `harm_rhythm_std`, `cadence_count`.

## Model Registry

`model_registry.json` contains full training metadata for all versions including
sample counts, preprocessing details, accuracy/F1, and known limitations.
