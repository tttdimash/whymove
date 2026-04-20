"""Training utilities for the intent classifier."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from whymove.classifier.labels import ALL_LABELS
from whymove.classifier.model import IntentClassifier
from whymove.features.vectorizer import FEATURE_NAMES


def load_training_data(
    features_path: Path,
    labels_path: Path,
) -> tuple[np.ndarray, list[list[str]]]:
    """Load features + labels and join them by move_id.

    Args:
        features_path: Path to features.parquet (columns: move_id, + FEATURE_NAMES)
        labels_path: Path to labels JSONL (each line: {"move_id": ..., "labels": [...]})

    Returns:
        X: float32 array of shape (n_samples, N_FEATURES)
        y: list of label-value lists
    """
    features_df = pd.read_parquet(features_path)

    labels: list[dict] = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))

    labels_df = pd.DataFrame(labels)
    merged = features_df.merge(labels_df[["move_id", "labels"]], on="move_id", how="inner")

    X = merged[FEATURE_NAMES].values.astype(np.float32)
    y = [row if isinstance(row, list) else [] for row in merged["labels"].tolist()]
    return X, y


def train_model(
    X: np.ndarray,
    y: list[list[str]],
    threshold: float = 0.3,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[IntentClassifier, dict]:
    """Train and evaluate the intent classifier.

    Returns:
        (trained classifier, metrics dict)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    clf = IntentClassifier(threshold=threshold)
    clf.fit(X_train, y_train)

    metrics = evaluate_model(clf, X_test, y_test)
    return clf, metrics


def evaluate_model(
    clf: IntentClassifier,
    X_test: np.ndarray,
    y_test: list[list[str]],
) -> dict:
    """Compute per-label and aggregate F1 scores."""
    mlb = MultiLabelBinarizer(classes=[l.value for l in ALL_LABELS])
    mlb.fit([[]])

    Y_true = mlb.transform(y_test)

    # Get raw probas and threshold
    probas = clf._get_probas(X_test)
    Y_pred = (probas >= clf.threshold).astype(int)

    macro_f1 = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))

    per_label = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    per_label_f1 = {
        label.value: float(score)
        for label, score in zip(ALL_LABELS, per_label)
    }

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "per_label_f1": per_label_f1,
        "n_train": len(X_test),
    }
