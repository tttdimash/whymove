"""Multi-label intent classifier wrapping scikit-learn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from whymove.classifier.labels import ALL_LABELS, IntentLabel
from whymove.features.vectorizer import FEATURE_NAMES, N_FEATURES, features_to_vector
from whymove.models import LabeledIntent, PositionFeatures


class IntentClassifier:
    """Multi-label classifier that predicts chess move intent with confidence scores.

    Uses MultiOutputClassifier(GradientBoostingClassifier) — one binary classifier
    per intent label. predict_proba gives calibrated per-label confidence.
    """

    def __init__(self, threshold: float = 0.3) -> None:
        self.threshold = threshold
        self._model: MultiOutputClassifier | None = None
        self._mlb: MultiLabelBinarizer = MultiLabelBinarizer(classes=[l.value for l in ALL_LABELS])
        self._mlb.fit([[]])  # Initialize with empty data to set classes
        self.feature_names: list[str] = FEATURE_NAMES

    def fit(self, X: np.ndarray, y_multilabel: list[list[str]]) -> None:
        """Train the classifier.

        Args:
            X: Feature matrix of shape (n_samples, N_FEATURES)
            y_multilabel: List of label-value lists per sample
        """
        Y = self._mlb.fit_transform(y_multilabel)
        base = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self._model = MultiOutputClassifier(base, n_jobs=-1)
        self._model.fit(X, Y)

    def predict(self, X: np.ndarray) -> list[list[LabeledIntent]]:
        """Predict intent labels for a batch of feature vectors.

        Returns:
            List of sorted LabeledIntent lists (descending by confidence),
            filtered by threshold.
        """
        probas = self._get_probas(X)
        results = []
        for row in probas:
            labeled = [
                LabeledIntent(label=ALL_LABELS[i], confidence=float(p))
                for i, p in enumerate(row)
                if p >= self.threshold
            ]
            labeled.sort(key=lambda x: x.confidence, reverse=True)
            results.append(labeled)
        return results

    def predict_one(self, features: PositionFeatures) -> list[LabeledIntent]:
        """Predict intent labels for a single position."""
        if self._model is None:
            raise RuntimeError("Classifier has not been trained or loaded.")
        vec = features_to_vector(features).reshape(1, -1)
        return self.predict(vec)[0]

    def _get_probas(self, X: np.ndarray) -> np.ndarray:
        """Return probability matrix of shape (n_samples, n_labels)."""
        if self._model is None:
            raise RuntimeError("Classifier has not been trained or loaded.")
        proba_cols = [est.predict_proba(X)[:, 1] for est in self._model.estimators_]
        return np.column_stack(proba_cols)

    def save(self, path: Path) -> None:
        """Serialize the trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self._model,
                "mlb": self._mlb,
                "threshold": self.threshold,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> IntentClassifier:
        """Load a previously saved classifier."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run `whymove train` to train and save a model first."
            )
        data: dict[str, Any] = joblib.load(path)
        obj = cls(threshold=data["threshold"])
        obj._model = data["model"]
        obj._mlb = data["mlb"]
        obj.feature_names = data.get("feature_names", FEATURE_NAMES)
        return obj

    @property
    def is_trained(self) -> bool:
        return self._model is not None
