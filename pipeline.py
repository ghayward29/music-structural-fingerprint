#!/usr/bin/env python3
"""
pipeline.py — Music Structural Analysis Prototype

End-to-end flow:
  Audio (MP3/WAV)
    → Basic Pitch  → MIDI
    → music21      → note list
    → networkx     → directed note graph (onset / voice / rest edges)
    → JSON export  + matplotlib visualization
    → run_structural_analysis (stub for AutoSchA depth assignment)

Usage:
    python pipeline.py path/to/audio.mp3 [--output-dir out/]
"""

import copy
import sys
import os
import json
import argparse
import tempfile

import numpy as np
import music21
import music21.stream
import music21.tempo
import networkx as nx
import matplotlib
matplotlib.use("Agg")          # headless — safe on servers and CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from basic_pitch.inference import predict, Model
from basic_pitch import build_icassp_2022_model_path, FilenameSuffix

# Explicitly target the ONNX backend — avoids the broken TF SavedModel path
# on Apple Silicon / Python 3.12 where tensorflow-macos raises add_slot errors.
ONNX_MODEL_PATH = build_icassp_2022_model_path(FilenameSuffix.onnx)

# ---------------------------------------------------------------------------
# Tuneable constants
# ---------------------------------------------------------------------------

# Two non-overlapping notes are considered part of the same melodic voice if
# their pitches differ by at most this many semitones.  12 = one octave.
VOICE_PITCH_RANGE = 12  # semitones

# Minimum silence between a note's offset and the next note's onset for the
# gap to be labelled a "rest" rather than a smooth melodic succession.
REST_THRESHOLD = 0.1   # seconds

# Edge colours used in the piano-roll visualisation.
EDGE_COLORS = {
    "onset": "#e74c3c",   # red   — harmonic overlap / simultaneity
    "voice": "#2ecc71",   # green — smooth melodic succession
    "rest":  "#3498db",   # blue  — phrase boundary / return after silence
}


# ---------------------------------------------------------------------------
# Step 1 — Audio → MIDI  (Basic Pitch)
# ---------------------------------------------------------------------------

def _stem_to_part(midi_path: str, part_name: str) -> tuple[music21.stream.Part, int]:
    """Parse a Basic Pitch MIDI and return a named music21 Part plus note count."""
    parsed = music21.converter.parse(midi_path)
    flat   = parsed.flatten()
    part   = music21.stream.Part()
    part.id       = part_name
    part.partName = part_name
    n_notes = 0
    for element in flat.notes:
        part.insert(element.offset, copy.deepcopy(element))
        n_notes += len(element.pitches) if hasattr(element, "pitches") else 1
    return part, n_notes


def transcribe_audio(audio_path: str, output_dir: str) -> tuple[str, bool]:
    """
    Transcribe audio to MIDI, preferring Demucs stem separation for accuracy.

    Attempts htdemucs 4-stem separation first: vocals, bass, and other stems
    are each transcribed with Basic Pitch and merged into a multi-track MIDI.
    Falls back to full-mix Basic Pitch if separation fails.

    Returns (midi_path, stem_separated).
    """
    from separate_stems import separate_audio, transcribe_with_basic_pitch

    print(f"[1/5] Transcribing audio: {audio_path}")

    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    with tempfile.TemporaryDirectory() as sep_tmp:
        stem_paths = separate_audio(audio_path, sep_tmp)
        if stem_paths:
            print(f"    Stem separation succeeded ({len(stem_paths)} stems)")
            parts: list[tuple[music21.stream.Part, int]] = []
            for stem_path in stem_paths:
                stem_name = os.path.splitext(os.path.basename(stem_path))[0]
                try:
                    mid = transcribe_with_basic_pitch(stem_path, sep_tmp)
                    part, n = _stem_to_part(mid, stem_name)
                    parts.append((part, n))
                    print(f"      {stem_name}: {n} notes")
                except Exception as exc:
                    print(f"      {stem_name}: transcription failed ({exc})")
            if parts:
                midi_path = os.path.join(output_dir, f"{base_name}.mid")
                score = music21.stream.Score()
                for part, _ in parts:
                    score.append(part)
                score.write("midi", fp=midi_path)
                print(f"    Merged MIDI written → {midi_path}")
                return midi_path, True

    print("    Falling back to full-mix transcription")
    from separate_stems import transcribe_with_basic_pitch as _tbp
    midi_path = _tbp(audio_path, output_dir)
    print(f"    MIDI written → {midi_path}")
    return midi_path, False


# ---------------------------------------------------------------------------
# Step 2 — MIDI → note list  (music21)
# ---------------------------------------------------------------------------

def parse_midi(midi_path: str) -> list[dict]:
    """
    Parse the MIDI file with music21 and return a list of note dictionaries,
    sorted by onset time.  Each dict contains:
        id         — unique integer index (assigned after sorting)
        pitch      — MIDI note number 0–127
        pitch_name — human-readable name, e.g. "C4"
        onset      — start time in seconds
        offset     — end time in seconds
        duration   — duration in seconds
        velocity   — MIDI velocity 0–127  (loudness proxy)

    music21 gives us symbolic pitch/rhythm information that Basic Pitch alone
    does not expose in a convenient form.
    """
    print(f"[2/5] Parsing MIDI with music21: {midi_path}")

    score = music21.converter.parse(midi_path)

    # Extract tempo — Basic Pitch embeds one MetronomeMark; fall back to 120.
    bpm = _get_bpm(score)
    print(f"    Detected tempo: {bpm:.1f} BPM")

    notes: list[dict] = []
    note_id = 0

    for part in score.parts:
        flat = part.flatten()
        for element in flat.notesAndRests:
            if isinstance(element, music21.note.Rest):
                continue

            onset_ql = float(element.offset)

            # music21 may yield Note or Chord elements; unpack chords so
            # every sounding pitch becomes its own node in the graph.
            if isinstance(element, music21.note.Note):
                pitch_rows = [
                    (element.pitch.midi,
                     element.pitch.nameWithOctave,
                     element.volume.velocity or 64,
                     float(element.duration.quarterLength))
                ]
            elif isinstance(element, music21.chord.Chord):
                pitch_rows = [
                    (n.pitch.midi,
                     n.pitch.nameWithOctave,
                     element.volume.velocity or 64,
                     float(element.duration.quarterLength))
                    for n in element.notes
                ]
            else:
                continue

            for pitch_midi, pitch_name, velocity, dur_ql in pitch_rows:
                onset_sec  = onset_ql * (60.0 / bpm)
                dur_sec    = dur_ql   * (60.0 / bpm)
                notes.append({
                    "id":         note_id,
                    "pitch":      pitch_midi,
                    "pitch_name": pitch_name,
                    "onset":      round(onset_sec, 4),
                    "offset":     round(onset_sec + dur_sec, 4),
                    "duration":   round(dur_sec, 4),
                    "velocity":   int(velocity),
                })
                note_id += 1

    # Canonical order: earliest onset first; ties broken by pitch (low→high).
    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    for i, n in enumerate(notes):
        n["id"] = i

    print(f"    Extracted {len(notes)} notes")
    return notes


def _get_bpm(score: music21.stream.Score) -> float:
    """Return the first MetronomeMark BPM found in the score, else 120."""
    for el in score.flatten().getElementsByClass(music21.tempo.MetronomeMark):
        return float(el.number)
    return 120.0


# ---------------------------------------------------------------------------
# Step 3 — Note list → directed graph  (networkx)
# ---------------------------------------------------------------------------

def build_note_graph(notes: list[dict]) -> nx.DiGraph:
    """
    Build a directed graph whose nodes are notes and whose edges capture
    three structural relationships, mirroring the AutoSchA paper:

    'onset'  — Notes a and b overlap in time (b starts before a ends).
               These edges encode harmonic/contrapuntal simultaneity.

    'voice'  — b starts right after a ends (gap < REST_THRESHOLD) and both
               pitches lie within VOICE_PITCH_RANGE semitones.  These edges
               trace melodic continuity — the same voice line uninterrupted.

    'rest'   — Like voice edges but with a perceivable silence between a's
               offset and b's onset.  Rest edges mark phrase boundaries and
               resumptions after silence.

    Edge direction is always earlier-onset → later-onset, reflecting the
    temporal flow of the music.  For voice/rest, only the nearest
    pitch-compatible successor is connected to avoid runaway fan-out.
    """
    print("[3/5] Building note graph with networkx")

    G = nx.DiGraph()

    # Add every note as a node; store the full feature dict as attributes so
    # downstream GNN code can read them directly from the graph object.
    for note in notes:
        G.add_node(note["id"], **note)

    n = len(notes)

    for i in range(n):
        a = notes[i]
        # Flag: have we already assigned a voice/rest successor to note a?
        # We limit each note to exactly one forward voice/rest edge (the
        # nearest pitch-compatible non-overlapping successor) to keep the
        # graph sparse and musically interpretable.
        found_successor = False

        for j in range(i + 1, n):
            b = notes[j]

            if b["onset"] < a["offset"]:
                # --- Onset edge: a and b sound simultaneously ---
                # b starts before a finishes → harmonic overlap.
                G.add_edge(a["id"], b["id"], edge_type="onset")

            else:
                # b starts after a ends — candidate for voice or rest edge.

                # Stop searching forward once we've placed a successor edge.
                if found_successor:
                    break

                # Skip b if it is more than 2 s away and we have not yet
                # found a successor — avoids connecting notes across large
                # silences that span multiple phrases.
                if b["onset"] > a["offset"] + 2.0:
                    break

                pitch_diff = abs(b["pitch"] - a["pitch"])
                if pitch_diff > VOICE_PITCH_RANGE:
                    # Different register — likely a different voice; keep
                    # looking forward for a pitch-compatible successor.
                    continue

                gap = b["onset"] - a["offset"]
                edge_type = "voice" if gap < REST_THRESHOLD else "rest"
                G.add_edge(a["id"], b["id"], edge_type=edge_type)
                found_successor = True

    onset_count = sum(1 for *_, d in G.edges(data=True) if d["edge_type"] == "onset")
    voice_count = sum(1 for *_, d in G.edges(data=True) if d["edge_type"] == "voice")
    rest_count  = sum(1 for *_, d in G.edges(data=True) if d["edge_type"] == "rest")

    print(f"    {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(onset={onset_count}, voice={voice_count}, rest={rest_count})")
    return G


# ---------------------------------------------------------------------------
# Step 4 — Export graph to JSON
# ---------------------------------------------------------------------------

def export_graph_json(G: nx.DiGraph, output_path: str) -> None:
    """
    Write the note graph to a self-contained JSON file.  The format is
    intentionally flat so it can be fed directly into a GNN data loader
    (e.g. PyTorch Geometric's from_dict) or a front-end visualisation tool.
    """
    print(f"[4/5] Exporting graph JSON → {output_path}")

    data = {
        "nodes": [
            {"id": node_id, **attrs}
            for node_id, attrs in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "edge_type": d["edge_type"]}
            for u, v, d in G.edges(data=True)
        ],
    }

    with open(output_path, "w") as fh:
        json.dump(data, fh, indent=2)

    print(f"    Written {len(data['nodes'])} nodes, {len(data['edges'])} edges")


# ---------------------------------------------------------------------------
# Step 5 — Visualise the graph  (matplotlib)
# ---------------------------------------------------------------------------

def visualize_graph(G: nx.DiGraph, output_path: str) -> None:
    """
    Render the note graph in a piano-roll coordinate system:
        x-axis = onset time (seconds)
        y-axis = MIDI pitch number

    This layout makes the three edge types immediately readable:
    - red onset edges run roughly vertically (simultaneous notes)
    - green voice edges run diagonally forward (melodic lines)
    - blue rest edges are similar but skip over silence

    Node size scales with velocity; node colour encodes pitch class
    (repeating palette across octaves via the plasma colormap).
    """
    print(f"[5/5] Visualising graph → {output_path}")

    fig, ax = plt.subplots(figsize=(16, 6))

    # Piano-roll positions: place each node at its (time, pitch) coordinates.
    pos = {
        node_id: (attrs["onset"], attrs["pitch"])
        for node_id, attrs in G.nodes(data=True)
    }

    # Draw edge sets one type at a time so each gets its own colour.
    for edge_type, color in EDGE_COLORS.items():
        edges = [(u, v) for u, v, d in G.edges(data=True)
                 if d["edge_type"] == edge_type]
        if edges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                edge_color=color,
                alpha=0.55,
                arrows=True,
                arrowsize=8,
                width=1.2,
                ax=ax,
            )

    # Draw nodes — size ∝ velocity, colour = MIDI pitch (mod 12 via plasma).
    node_ids  = list(G.nodes())
    sizes     = [max(15, G.nodes[n]["velocity"] * 0.6) for n in node_ids]
    colors    = [G.nodes[n]["pitch"] % 12 for n in node_ids]   # pitch class

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_ids,
        node_size=sizes,
        node_color=colors,
        cmap=plt.cm.tab10,     # 12 distinct colours, one per pitch class
        vmin=0, vmax=11,
        ax=ax,
    )

    # Legend for edge types.
    legend_handles = [
        mpatches.Patch(color=c, label=f"{t} edge")
        for t, c in EDGE_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9)

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI Pitch")
    ax.set_title("Music Note Graph — structural relationships (piano-roll view)")
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"    Saved to {output_path}")


# ---------------------------------------------------------------------------
# Step 6 (stub) — Structural depth assignment
# ---------------------------------------------------------------------------

def run_structural_analysis(graph: nx.DiGraph) -> dict[int, float]:
    """
    STUB — This is where AutoSchA inference would go.

    In the full pipeline this function would:
      1. Convert the networkx graph to a PyTorch Geometric Data object,
         encoding node features (pitch, duration, velocity) and categorical
         edge-type embeddings (onset=0, voice=1, rest=2).
      2. Pass the graph through the trained AutoSchA GNN, which applies
         hierarchical structural pooling inspired by Schenkerian analysis
         to assign each note a depth score.
      3. Return a mapping  node_id → depth_score  where:
             0.0  = foreground / ornamental note
             1.0  = deep structural pitch (Urlinie / Bassbrechung level)

    For now every note receives a placeholder score of 0.5.
    """
    return {node_id: 0.5 for node_id in graph.nodes()}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Music structural analysis pipeline "
                    "(Basic Pitch → music21 → networkx graph)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("audio",
                        help="Path to input MP3 or WAV file")
    parser.add_argument("--output-dir", default=".",
                        help="Directory for generated MIDI, JSON, and PNG files")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.audio))[0]

    # Run the pipeline.
    midi_path, _stem_sep = transcribe_audio(args.audio, args.output_dir)
    notes                = parse_midi(midi_path)

    if not notes:
        print("No notes extracted — check the audio and Basic Pitch output.",
              file=sys.stderr)
        sys.exit(1)

    G = build_note_graph(notes)

    json_path = os.path.join(args.output_dir, f"{base}_graph.json")
    export_graph_json(G, json_path)

    img_path = os.path.join(args.output_dir, f"{base}_graph.png")
    visualize_graph(G, img_path)

    depth_scores = run_structural_analysis(G)
    sample = list(depth_scores.items())[:5]
    print(f"\n[stub] Structural depth scores (first 5 of {len(depth_scores)}): {sample}")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
