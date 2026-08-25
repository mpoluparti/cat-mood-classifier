"""Predict the context label for a single audio clip.

Usage: python -m src.predict path/to/clip.wav
"""

import sys
from pathlib import Path

import joblib

from .features import extract_features
from .preprocess import load_and_normalize

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "classifier.joblib"

_bundle = None


def _get_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def predict_with_probs(path: str):
    """Return (predicted_label, {label: probability}) for one audio clip."""
    bundle = _get_bundle()
    clf, encoder = bundle["model"], bundle["label_encoder"]

    clip = load_and_normalize(path)
    features = extract_features(clip).reshape(1, -1)

    probs = clf.predict_proba(features)[0]
    label = encoder.inverse_transform([probs.argmax()])[0]
    probabilities = dict(zip(encoder.classes_, probs.tolist()))
    return label, probabilities


def predict(path: str) -> str:
    label, _ = predict_with_probs(path)
    return label


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict path/to/clip.wav")
        sys.exit(1)

    label, probabilities = predict_with_probs(sys.argv[1])
    print(f"Predicted: {label}")
    for cls, prob in sorted(probabilities.items(), key=lambda kv: -kv[1]):
        print(f"  {cls}: {prob:.3f}")
