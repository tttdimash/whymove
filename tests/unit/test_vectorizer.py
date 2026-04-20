"""Unit tests for the feature vectorizer."""

from __future__ import annotations

import chess
import numpy as np
import pytest

from whymove.features.vectorizer import (
    FEATURE_NAMES,
    N_FEATURES,
    features_to_vector,
    vector_to_feature_dict,
)


def test_vector_length(sample_features):
    vec = features_to_vector(sample_features)
    assert len(vec) == N_FEATURES


def test_feature_names_count():
    assert len(FEATURE_NAMES) == N_FEATURES


def test_vector_dtype(sample_features):
    vec = features_to_vector(sample_features)
    assert vec.dtype == np.float32


def test_eval_delta_at_correct_index(sample_features):
    vec = features_to_vector(sample_features)
    idx = FEATURE_NAMES.index("eval_delta_cp")
    assert vec[idx] == float(sample_features.eval_delta_cp)


def test_piece_type_one_hot_knight(sample_features):
    """Knight should have piece_type_knight=1, others=0."""
    vec = features_to_vector(sample_features)
    assert vec[FEATURE_NAMES.index("piece_type_knight")] == 1.0
    assert vec[FEATURE_NAMES.index("piece_type_pawn")] == 0.0
    assert vec[FEATURE_NAMES.index("piece_type_queen")] == 0.0


def test_tactical_fork_flag(sample_features):
    vec = features_to_vector(sample_features)
    idx = FEATURE_NAMES.index("is_fork")
    assert vec[idx] == 1.0  # sample_features has is_fork=True


def test_vector_to_feature_dict_round_trip(sample_features):
    vec = features_to_vector(sample_features)
    d = vector_to_feature_dict(vec)
    assert set(d.keys()) == set(FEATURE_NAMES)
    assert d["eval_delta_cp"] == float(sample_features.eval_delta_cp)


def test_is_capture_false_in_sample(sample_features):
    vec = features_to_vector(sample_features)
    idx = FEATURE_NAMES.index("is_capture")
    assert vec[idx] == 0.0  # sample_features has is_capture=False
