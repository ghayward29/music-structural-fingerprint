"""
predict.py — run the trained piano-roll classifier on a MIDI file.

Usage:
    python predict.py path/to/file.mid

Output:
    AI-generated probability: 0.87
    → Likely AI-generated

The classifier was trained by train_classifier.py.  It expects the model
weights at classifier/model.pth.  Run train_classifier.py first.
"""

import os
import sys

import torch

from roll_utils import midi_to_roll, ROLL_TRANSFORM
# Import the model class from train_classifier so the architecture definition
# lives in exactly one place.  All module-level training code in that file
# is gated behind `if __name__ == "__main__"`, so importing it is safe.
from train_classifier import PianoRollCNN

MODEL_PATH = os.path.join(os.path.dirname(__file__), "classifier", "model.pth")


def classify_midi(midi_path: str) -> float:
    """
    Render a MIDI file as a piano roll, load the trained CNN, and return
    the AI-generated probability in [0.0, 1.0].

    Raises FileNotFoundError if the model weights have not been trained yet.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. "
            "Run train_classifier.py first."
        )

    # Load the model once per call.  For the web server, app.py could cache
    # this, but keeping predict.py stateless makes it easy to use standalone.
    model = PianoRollCNN()
    # map_location="cpu" ensures it loads correctly even if training used GPU.
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    # Render the piano roll using the exact same function as make_rolls.py so
    # the image format matches what the model was trained on.
    img = midi_to_roll(midi_path)

    # Apply the shared transform (grayscale → normalised tensor).
    tensor = ROLL_TRANSFORM(img).unsqueeze(0)  # add batch dimension → (1, 1, H, W)

    with torch.no_grad():
        prob = model(tensor).item()  # scalar in [0, 1]

    return round(prob, 4)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/file.mid", file=sys.stderr)
        sys.exit(1)

    midi_path = sys.argv[1]
    if not os.path.isfile(midi_path):
        print(f"Error: file not found: {midi_path}", file=sys.stderr)
        sys.exit(1)

    prob  = classify_midi(midi_path)
    label = "Likely AI-generated" if prob >= 0.5 else "Likely human-composed"

    print(f"AI-generated probability: {prob:.2f}")
    print(f"→ {label}")


if __name__ == "__main__":
    main()
