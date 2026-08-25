"""Load and normalize a raw audio clip for feature extraction."""

import librosa
import numpy as np

SAMPLE_RATE = 22050
CLIP_DURATION_SEC = 4.0
CLIP_LENGTH = int(SAMPLE_RATE * CLIP_DURATION_SEC)


def load_and_normalize(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)

    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak

    if len(y) < CLIP_LENGTH:
        y = np.pad(y, (0, CLIP_LENGTH - len(y)))
    else:
        y = y[:CLIP_LENGTH]

    return y
