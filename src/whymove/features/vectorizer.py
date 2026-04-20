"""Convert PositionFeatures to a flat numpy array for the classifier."""

from __future__ import annotations

import numpy as np

import chess

from whymove.models import PositionFeatures

# The feature order is the contract between extractor and classifier.
# Never reorder without retraining the model.
FEATURE_NAMES: list[str] = [
    # Evaluation
    "eval_delta_cp",
    # Material
    "material_delta",
    # Mobility
    "moved_piece_mobility_before",
    "moved_piece_mobility_after",
    "total_white_mobility_delta",
    "total_black_mobility_delta",
    # Piece type (one-hot: pawn=0, knight=1, bishop=2, rook=3, queen=4, king=5)
    "piece_type_pawn",
    "piece_type_knight",
    "piece_type_bishop",
    "piece_type_rook",
    "piece_type_queen",
    "piece_type_king",
    # Piece color (1 = white, 0 = black)
    "piece_color",
    # Geometry
    "distance_moved",
    "destination_rank",
    "destination_file",
    # King safety (4 features)
    "white_king_zone_attacks_delta",
    "black_king_zone_attacks_delta",
    "white_open_files_near_king_delta",
    "black_open_files_near_king_delta",
    # Pawn structure (4 features)
    "doubled_pawns_delta",
    "isolated_pawns_delta",
    "passed_pawns_delta",
    "pawn_islands_delta",
    # Square control (2 features)
    "center_control_delta",
    "key_square_control_delta",
    # Tactical flags (16 bool → float features)
    "is_fork",
    "is_pin",
    "is_skewer",
    "is_discovered_attack",
    "is_double_attack",
    "is_double_check",
    "is_mating_threat",
    "is_capture",
    "is_promotion",
    "is_check",
    "is_zwischenzug",
    "is_x_ray",
    "is_overloading",
    "is_sacrifice",
    "is_deflection",
    "is_interposition",
]

N_FEATURES: int = len(FEATURE_NAMES)

# Map chess piece type constants to one-hot index
_PIECE_TYPE_TO_IDX: dict[int, int] = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def features_to_vector(features: PositionFeatures) -> np.ndarray:
    """Convert a PositionFeatures object to a float32 numpy array of shape (N_FEATURES,)."""
    t = features.tactical
    ks = features.king_safety
    ps = features.pawn_structure
    piece_one_hot = [0.0] * 6
    piece_one_hot[_PIECE_TYPE_TO_IDX.get(features.piece.piece_type, 0)] = 1.0

    vec = [
        float(features.eval_delta_cp),
        float(features.material_delta),
        float(features.moved_piece_mobility_before),
        float(features.moved_piece_mobility_after),
        float(features.total_white_mobility_delta),
        float(features.total_black_mobility_delta),
        *piece_one_hot,
        float(features.piece.color),
        float(features.distance_moved),
        float(features.destination_rank),
        float(features.destination_file),
        float(ks.white_king_zone_attacks_delta),
        float(ks.black_king_zone_attacks_delta),
        float(ks.white_open_files_near_king_delta),
        float(ks.black_open_files_near_king_delta),
        float(ps.doubled_pawns_delta),
        float(ps.isolated_pawns_delta),
        float(ps.passed_pawns_delta),
        float(ps.pawn_islands_delta),
        float(features.center_control_delta),
        float(features.key_square_control_delta),
        float(t.is_fork),
        float(t.is_pin),
        float(t.is_skewer),
        float(t.is_discovered_attack),
        float(t.is_double_attack),
        float(t.is_double_check),
        float(t.is_mating_threat),
        float(t.is_capture),
        float(t.is_promotion),
        float(t.is_check),
        float(t.is_zwischenzug),
        float(t.is_x_ray),
        float(t.is_overloading),
        float(t.is_sacrifice),
        float(t.is_deflection),
        float(t.is_interposition),
    ]

    assert len(vec) == N_FEATURES, f"Feature count mismatch: {len(vec)} != {N_FEATURES}"
    return np.array(vec, dtype=np.float32)


def vector_to_feature_dict(vec: np.ndarray) -> dict[str, float]:
    """Convert a feature vector back to a named dict (for human-readable output)."""
    return dict(zip(FEATURE_NAMES, vec.tolist()))
