"""Extract YAMNet embeddings as an alternative to hand-crafted features.

YAMNet expects 16kHz mono audio in [-1, 1] and produces one 1024-dim
embedding per ~0.48s frame; we mean-pool across frames to get a single
fixed-length vector per clip.
"""

import librosa
import numpy as np
import tensorflow_hub as hub

YAMNET_SAMPLE_RATE = 16000

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = hub.load("https://tfhub.dev/google/yamnet/1")
    return _model


def extract_yamnet_embedding(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=YAMNET_SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)

    peak = np.max(np.abs(y)) if y.size else 0.0
    if peak > 0:
        y = y / peak

    if y.size == 0:
        return np.zeros(1024, dtype=np.float32)

    model = _get_model()
    _, embeddings, _ = model(y.astype(np.float32))
    return embeddings.numpy().mean(axis=0)
