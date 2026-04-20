"""Unit tests for the intent classifier."""

from __future__ import annotations

import numpy as np
import pytest

from whymove.classifier.labels import ALL_LABELS, IntentLabel, N_LABELS
from whymove.classifier.model import IntentClassifier
from whymove.features.vectorizer import N_FEATURES


def _make_synthetic_data(n: int = 50):
    """Create small synthetic training data."""
    rng = np.random.default_rng(42)
    X = rng.random((n, N_FEATURES)).astype(np.float32)
    # Assign random subsets of labels
    y = []
    for i in range(n):
        labels = [l.value for l in ALL_LABELS if rng.random() > 0.85]
        if not labels:
            labels = [ALL_LABELS[i % N_LABELS].value]
        y.append(labels)
    return X, y


def test_classifier_fit_and_predict():
    X, y = _make_synthetic_data(60)
    clf = IntentClassifier(threshold=0.1)
    clf.fit(X, y)
    assert clf.is_trained

    preds = clf.predict(X[:5])
    assert len(preds) == 5
    for pred in preds:
        assert isinstance(pred, list)
        for item in pred:
            assert 0.0 <= item.confidence <= 1.0


def test_classifier_threshold_filtering():
    X, y = _make_synthetic_data(60)
    clf_low = IntentClassifier(threshold=0.0)
    clf_high = IntentClassifier(threshold=0.99)
    clf_low.fit(X, y)
    clf_high.fit(X, y)

    pred_low = clf_low.predict(X[:1])[0]
    pred_high = clf_high.predict(X[:1])[0]
    assert len(pred_low) >= len(pred_high)


def test_classifier_save_load(tmp_path):
    X, y = _make_synthetic_data(60)
    clf = IntentClassifier(threshold=0.3)
    clf.fit(X, y)

    model_path = tmp_path / "test_model.joblib"
    clf.save(model_path)
    assert model_path.exists()

    loaded = IntentClassifier.load(model_path)
    assert loaded.threshold == 0.3
    assert loaded.is_trained

    # Predictions should match
    orig_preds = clf.predict(X[:3])
    loaded_preds = loaded.predict(X[:3])
    for o, l in zip(orig_preds, loaded_preds):
        orig_labels = {i.label for i in o}
        loaded_labels = {i.label for i in l}
        assert orig_labels == loaded_labels


def test_classifier_load_missing_file():
    with pytest.raises(FileNotFoundError):
        IntentClassifier.load("nonexistent_model.joblib")


def test_predict_one_uses_features(mock_classifier, sample_features):
    """predict_one calls through to predict correctly."""
    result = mock_classifier.predict_one(sample_features)
    assert len(result) > 0
    mock_classifier.predict_one.assert_called_once_with(sample_features)


def test_predict_sorted_by_confidence():
    X, y = _make_synthetic_data(60)
    clf = IntentClassifier(threshold=0.0)
    clf.fit(X, y)
    preds = clf.predict(X[:1])[0]
    confidences = [p.confidence for p in preds]
    assert confidences == sorted(confidences, reverse=True)
