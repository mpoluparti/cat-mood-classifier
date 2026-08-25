"""Extract a fixed-length feature vector from a preprocessed audio clip."""

import librosa
import numpy as np

from .preprocess import SAMPLE_RATE


def extract_features(y: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    f0, voiced_flag, _ = librosa.pyin(
        y, sr=SAMPLE_RATE, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])
    pitch_mean = np.mean(voiced_f0) if voiced_f0.size else 0.0
    pitch_std = np.std(voiced_f0) if voiced_f0.size else 0.0

    features = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        centroid.mean(axis=1), centroid.std(axis=1),
        zcr.mean(axis=1), zcr.std(axis=1),
        rms.mean(axis=1), rms.std(axis=1),
        [pitch_mean, pitch_std],
    ])
    return features
