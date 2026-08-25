"""Predict the context label for a single audio clip.

Usage: python -m src.predict path/to/clip.wav
"""

import sys
from pathlib import Path

import joblib

from .features import extract_features
from .preprocess import load_and_normalize

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "classifier.joblib"


def predict(path: str) -> str:
    bundle = joblib.load(MODEL_PATH)
    clf, encoder = bundle["model"], bundle["label_encoder"]

    clip = load_and_normalize(path)
    features = extract_features(clip).reshape(1, -1)

    pred = clf.predict(features)[0]
    return encoder.inverse_transform([pred])[0]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.predict path/to/clip.wav")
        sys.exit(1)

    label = predict(sys.argv[1])
    print(f"Predicted: {label}")
