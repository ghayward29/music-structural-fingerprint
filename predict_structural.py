"""
predict_structural.py — run both structural classifiers on a MIDI file.

Loads the logistic regression and random forest trained by
train_structural_classifier.py and returns AI-generated probabilities
from each model.

Usage:
    python predict_structural.py path/to/file.mid

Output:
    Logistic Regression: AI-generated probability: 0.84
    Random Forest:       AI-generated probability: 0.79
"""

import os
import pickle
import sys

from extract_features import extract_features

LOGREG_PATH = os.path.join(os.path.dirname(__file__), "classifier", "structural_logreg.pkl")
RF_PATH     = os.path.join(os.path.dirname(__file__), "classifier", "structural_rf.pkl")


def _load_model(path: str):
    """Load a pickled sklearn Pipeline, raising FileNotFoundError with a clear message."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Run build_feature_dataset.py then train_structural_classifier.py first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def classify_structural(midi_path: str) -> tuple[float, float]:
    """
    Extract structural features from a MIDI file and return AI-generated
    probabilities from both classifiers.

    Returns (logreg_probability, rf_probability), each in [0.0, 1.0].
    1.0 = AI-generated, 0.0 = human-composed.

    Raises FileNotFoundError if either model has not been trained yet.
    """
    lr_pipe = _load_model(LOGREG_PATH)
    rf_pipe = _load_model(RF_PATH)

    # Extract the same 14 features used during training.
    vec = extract_features(midi_path).reshape(1, -1)   # shape (1, 14)

    # predict_proba returns [[p_human, p_ai]]; we want p_ai (column 1).
    lr_prob = float(lr_pipe.predict_proba(vec)[0, 1])
    rf_prob = float(rf_pipe.predict_proba(vec)[0, 1])

    return round(lr_prob, 4), round(rf_prob, 4)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python predict_structural.py path/to/file.mid", file=sys.stderr)
        sys.exit(1)

    midi_path = sys.argv[1]
    if not os.path.isfile(midi_path):
        print(f"Error: file not found: {midi_path}", file=sys.stderr)
        sys.exit(1)

    lr_prob, rf_prob = classify_structural(midi_path)

    def verdict(p: float) -> str:
        return "Likely AI-generated" if p >= 0.5 else "Likely human-composed"

    print(f"Logistic Regression: AI-generated probability: {lr_prob:.2f}  ({verdict(lr_prob)})")
    print(f"Random Forest:       AI-generated probability: {rf_prob:.2f}  ({verdict(rf_prob)})")


if __name__ == "__main__":
    main()
