# Pipeline Coherence Audit

*Generated 2026-05-21. Update this document whenever training data or
preprocessing changes.*

---

## 1. Model inventory

### 1.1 `classifier/model.pth` — Piano-Roll CNN

| Property | Value |
|----------|-------|
| Trained by | `train_classifier.py` |
| AI training data | `data/rolls/ai/` — 101 piano-roll PNGs |
| Human training data | `data/rolls/human/` — 200 piano-roll PNGs |
| AI source MIDIs | `data/ai/` — Suno audio → Basic Pitch full-mix → MIDI |
| Human source MIDIs | `data/human/midi/` — 200 native Lakh MIDI files (randomly sampled by `sample_human.py`) |
| Train/val split | 80/20 stratified |
| Preprocessing at training | `midi_to_roll()` → 512×128 grayscale PNG → `ROLL_TRANSFORM` (normalize tensor) |
| Preprocessing at inference | Same `midi_to_roll()` + `ROLL_TRANSFORM` via shared `roll_utils.py` |

**Mismatch:** The AI training MIDIs are Basic Pitch transcriptions of Suno
audio. The human training MIDIs are native Lakh MIDI files with no
transcription step. At inference, every uploaded file goes through Basic Pitch.
The CNN may therefore be partially learning to distinguish "Basic Pitch output"
from "native MIDI" rather than "AI composition" from "human composition". How
large an effect this has depends on how visually distinctive Basic Pitch's
output artifacts are compared to the compositional signal.

---

### 1.2 `classifier/structural_logreg.pkl` + `structural_rf.pkl` — Lakh-baseline structural models

| Property | Value |
|----------|-------|
| Trained by | `build_feature_dataset.py` → `train_structural_classifier.py` |
| AI training data | `data/ai/` — 101 Suno MIDIs (Basic Pitch full-mix transcriptions) |
| Human training data | `data/human/midi/` — 200 native Lakh MIDI files |
| Feature extraction | `extract_features.py` — 14 hand-crafted structural features |
| Preprocessing at training | `extract_features()` applied directly to MIDI files |
| Preprocessing at inference | Same `extract_features()` on Basic Pitch output |

**Mismatch:** Same transcription-confound as the CNN but more analytically
consequential. `extract_features` measures interval distributions, harmonic
rhythm, note density, pitch range, and cadence counts. Basic Pitch introduces
systematic artifacts — pitch errors, spurious short notes, missing low-velocity
notes, timing quantization — that shift every one of these features. The
classifier was trained on human MIDIs where none of those artifacts exist. At
inference it sees human MIDIs processed through a transcription layer that adds
noise. The resulting feature shift could cause real human music to score as
AI-generated (false positives) because its transcribed features look "wrong"
compared to the native-MIDI human training distribution.

These models should be considered **unreliable for inference** until the human
training data is re-transcribed through the same pipeline used at inference.
They are retained for comparison purposes.

---

### 1.3 `classifier/structural_logreg_audio.pkl` + `structural_rf_audio.pkl` — FMA audio-trained structural models

| Property | Value |
|----------|-------|
| Trained by | `prepare_fma_training.py` → `retrain_on_audio.py` |
| AI training data | `data/ai/` — 101 Suno MIDIs (Basic Pitch full-mix transcriptions) |
| Human training data | `data/human_audio/` — 158 FMA Small tracks (Basic Pitch full-mix transcriptions) |
| Genres covered | 8 FMA genres, 20 tracks each (Electronic, Experimental, Folk, Hip-Hop, Instrumental, International, Pop, Rock) |
| Feature extraction | Same 14 features via `extract_features.py` |
| Preprocessing at training | Both AI and human data were transcribed through Basic Pitch full-mix |
| Preprocessing at inference | Basic Pitch full-mix (or Demucs+per-stem if separation succeeds) |
| Reported test accuracy | LR 90.4%, RF 94.2% (20% hold-out) |

**Mismatch (minor, conditional):** Training used full-mix Basic Pitch for both
classes. At inference, if Demucs separation succeeds, the MIDI is built from
per-stem Basic Pitch outputs merged into a multi-track score. Stem-separated
transcription produces systematically different feature values than full-mix:
higher note density (stems are cleaner, more notes detected), different interval
distributions (per-instrument intervals rather than polyphonic cross-instrument
intervals), different harmonic rhythm (separation removes bleed). The classifier
was not trained on stem-separated MIDIs, so a stem-separated inference MIDI will
fall slightly outside the training distribution.

**This is the most coherent model currently available.** It eliminates the
fundamental transcription-confound by transcribing both classes the same way.
It is the default model used by `app.py`.

---

### 1.4 `classifier/structural_logreg_stems.pkl` + `structural_rf_stems.pkl` — Stems-trained models (NOT YET TRAINED)

| Property | Planned value |
|----------|---------------|
| Trained by | `retrain_on_stems.py` (script exists; not yet run) |
| AI training data | `data/ai_stems/` — Suno stems → Demucs htdemucs → per-stem Basic Pitch → merged MIDI |
| Human training data | `data/human/midi/` — 200 native Lakh MIDI files |
| Status | Blocked on Suno stem availability |

**Planned mismatch:** The AI training data will be stem-separated + per-stem
Basic Pitch. The human training data is still native Lakh MIDI — the same
fundamental transcription-confound as the Lakh baseline. For full coherence,
human training data would need to come from a source where separated stems
(or at minimum audio) are available, so the same Demucs+Basic Pitch pipeline
can be applied to both classes.

---

## 2. Preprocessing summary table

| Model | AI preprocessing | Human preprocessing | Inference preprocessing | Coherent? |
|-------|-----------------|---------------------|------------------------|-----------|
| CNN | BP full-mix → roll | Native MIDI → roll | BP (±Demucs) → roll | No — human training ≠ inference |
| LR/RF Lakh | BP full-mix | Native MIDI | BP (±Demucs) | No — same confound |
| LR/RF FMA audio | BP full-mix | BP full-mix | BP full-mix (±Demucs shift) | Mostly yes |
| LR/RF stems | BP per-stem (pending) | Native MIDI | BP per-stem | No — same confound |

---

## 3. Known bugs

### 3.1 `prepare_fma_training.py` broken by pipeline.py return-type change

`transcribe_audio()` was updated to return `(midi_path, stem_separated)` but
`prepare_fma_training.py` still assigns the result to a bare `midi_path`
variable. This breaks the FMA transcription script at runtime. **Fixed** in the
same commit as this audit (line 126: `midi_path, _ = transcribe_audio(...)`).

---

## 4. What needs to happen for full pipeline coherence

### Immediate (before publishing any results)

1. **Re-transcribe human training data for CNN and Lakh-baseline structural
   models.** Use the same Basic Pitch pipeline applied to AI data. The simplest
   approach: download audio versions of the Lakh MIDI pieces and run
   `prepare_fma_training.py`-style transcription on them. Until this is done, the
   Lakh-baseline models should not be presented as reliable detectors.

2. **Decide the canonical inference path.** Currently `app.py` uses stem
   separation when Demucs is available, and falls back to full-mix. The FMA
   audio model was trained on full-mix. Either:
   - Train a stems model on human audio run through Demucs+BP (requires human
     audio with separable stems), or
   - Keep full-mix as the canonical path (disable stem separation at inference,
     or train a separate model for stem-separated inputs).

### Once Suno stems are available

3. **Run `transcribe_stems.py`** on the Suno stem folders to populate
   `data/ai_stems/`.

4. **Run `retrain_on_stems.py`** to produce `structural_logreg_stems.pkl` +
   `structural_rf_stems.pkl`. This will still have the human-data mismatch
   (native MIDI vs stem-separated AI). Note the mismatch explicitly in any
   results.

5. **Evaluate with `evaluate_batch.py`** on a held-out set before drawing
   conclusions from the stems model.

### Longer term

6. **Acquire human audio with stems or re-transcribable audio** for the FMA
   tracks (audio is available, just re-run `prepare_fma_training.py` through
   Demucs). This would make the stems model fully coherent.

7. **Increase training set size.** 101 AI + 158–200 human is marginal.
   Even the 90–94% test accuracy should be interpreted cautiously: the test set
   has only ~52 samples (20% of 259), so confidence intervals are wide (roughly
   ±5–7% at 95% confidence). A model that achieves 94% on 52 samples may achieve
   anything from 84% to 99% on a different 52 samples from the same distribution.

8. **Diversify the AI training data** beyond Suno. Models trained only on
   Suno outputs are likely to miss AI-generated music from Udio, MusicLM, or
   future generators with different statistical properties.

---

## 5. Feature-level transcription confound analysis

Basic Pitch artifacts affect each of the 14 structural features differently.
The following table describes the expected direction and magnitude of the shift
when native MIDI is replaced with a Basic Pitch transcription:

| Feature | Expected shift under BP transcription | Severity |
|---------|---------------------------------------|----------|
| `note_density` | Increases (BP adds spurious short notes) | High |
| `pitch_range` | May increase (spurious high-frequency detections) | Medium |
| `interval_std` | Increases (noisy pitches widen interval distribution) | Medium |
| `interval_unison` | Decreases (fewer exact pitch repetitions) | Low |
| `interval_large_leap` | Increases (pitch errors create large jumps) | Medium |
| `rhythmic_cv` | Increases (timing of spurious notes is irregular) | High |
| `harm_rhythm_mean` | Increases (more spurious harmonic changes) | Medium |
| `harm_rhythm_std` | Increases (more irregular harmonic change pattern) | Medium |
| `stepwise_ratio` | Decreases (leaps from pitch errors dilute stepwise motion) | Medium |
| `cadence_count` | Unpredictable (phrase boundary detection is onset-gap based) | Low |
| `interval_mean` | Unpredictable (errors can go either direction) | Low |
| `interval_step` | Decreases slightly | Low |
| `interval_skip` | Unpredictable | Low |
| `interval_leap` | Increases slightly | Low |

Features with high shift severity (`note_density`, `rhythmic_cv`) are also
among the top LR coefficients in the FMA audio model (+0.74 and +0.56
respectively). This means the classifier has partially learned the transcription
pipeline itself, not only the underlying composition.
