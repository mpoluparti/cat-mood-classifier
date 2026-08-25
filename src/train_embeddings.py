"""Train a classifier on YAMNet embeddings instead of hand-crafted features.

Usage: python -m src.train_embeddings
"""

from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .data_utils import build_index
from .embeddings import extract_yamnet_embedding

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "classifier_yamnet.joblib"
FEATURE_CACHE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "yamnet_cache.npz"
)


def build_dataset(index):
    X, y = [], []
    for _, row in index.iterrows():
        X.append(extract_yamnet_embedding(row["path"]))
        y.append(row["label"])
    return np.array(X), np.array(y)


def load_or_build_dataset(index):
    if FEATURE_CACHE_PATH.exists():
        cached = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
        if len(cached["y"]) == len(index):
            print(f"Loaded cached embeddings from {FEATURE_CACHE_PATH}")
            return cached["X"], cached["y"]

    X, y = build_dataset(index)
    FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FEATURE_CACHE_PATH, X=X, y=y)
    return X, y


def evaluate(name, clf, X_train, y_train, X_test, y_test, encoder):
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"\n[{name}] CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return clf


def main():
    index = build_index(RAW_DIR)
    print(f"Found {len(index)} labeled clips: {index['label'].value_counts().to_dict()}")

    X, y = load_or_build_dataset(index)

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )
    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    rf = evaluate(
        "RandomForest",
        RandomForestClassifier(n_estimators=200, random_state=42),
        X_train, y_train, X_test, y_test, encoder,
    )
    svm = evaluate(
        "SVM (RBF)",
        SVC(kernel="rbf", probability=True, random_state=42),
        X_train, y_train, X_test, y_test, encoder,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": svm, "label_encoder": encoder}, MODEL_PATH)
    print(f"\nSaved SVM model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
