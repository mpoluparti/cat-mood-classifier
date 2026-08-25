"""Build the feature dataset from data/raw, train a classifier, and save it.

Usage: python -m src.train
"""

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

from .data_utils import build_index
from .features import extract_features
from .preprocess import load_and_normalize

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "classifier.joblib"


def build_dataset(index):
    X, y = [], []
    for _, row in index.iterrows():
        clip = load_and_normalize(row["path"])
        X.append(extract_features(clip))
        y.append(row["label"])
    return np.array(X), np.array(y)


def main():
    index = build_index(RAW_DIR)
    print(f"Found {len(index)} labeled clips: {index['label'].value_counts().to_dict()}")

    X, y = build_dataset(index)

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "label_encoder": encoder}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
