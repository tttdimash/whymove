"""Shared Pydantic schemas — the single source of truth for all data flowing between components."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from whymove.classifier.labels import IntentLabel


class MoveInput(BaseModel):
    fen: str | None = None
    pgn: str | None = None
    move_uci: str = Field(..., description="Move in UCI notation, e.g. 'e2e4'")
    engine_depth: int = Field(default=20, ge=1, le=30)
    top_k_labels: int = Field(default=5, ge=1, le=10)


class PieceInfo(BaseModel):
    piece_type: int   # chess.PAWN=1 .. chess.KING=6
    color: bool       # chess.WHITE=True, chess.BLACK=False
    from_square: int  # 0-63
    to_square: int    # 0-63


class PawnStructureFeatures(BaseModel):
    doubled_pawns_delta: int   # after - before (positive = more doubled = worse)
    isolated_pawns_delta: int
    passed_pawns_delta: int    # positive = more passed pawns = better for mover
    pawn_islands_delta: int


class KingSafetyFeatures(BaseModel):
    white_king_zone_attacks_delta: int   # opponent attackers near white king: after - before
    black_king_zone_attacks_delta: int   # opponent attackers near black king: after - before
    white_open_files_near_king_delta: int
    black_open_files_near_king_delta: int


class TacticalFlags(BaseModel):
    is_fork: bool = False
    is_pin: bool = False
    is_skewer: bool = False
    is_discovered_attack: bool = False
    is_double_attack: bool = False
    is_double_check: bool = False
    is_mating_threat: bool = False
    is_capture: bool = False
    is_promotion: bool = False
    is_check: bool = False
    is_zwischenzug: bool = False
    is_x_ray: bool = False
    is_overloading: bool = False
    is_sacrifice: bool = False
    is_deflection: bool = False
    is_interposition: bool = False


class PositionFeatures(BaseModel):
    # Evaluation (white-relative centipawns)
    eval_before_cp: int
    eval_after_cp: int
    eval_delta_cp: int

    # Material (white - black, in centipawns using standard values)
    material_before: int
    material_after: int
    material_delta: int

    # Mobility
    moved_piece_mobility_before: int
    moved_piece_mobility_after: int
    total_white_mobility_delta: int
    total_black_mobility_delta: int

    # Piece info
    piece: PieceInfo
    distance_moved: int    # Chebyshev distance
    destination_rank: int  # 0-7
    destination_file: int  # 0-7

    # Structural
    king_safety: KingSafetyFeatures
    pawn_structure: PawnStructureFeatures
    center_control_delta: int   # white center squares controlled: after - before
    key_square_control_delta: int

    # Tactical
    tactical: TacticalFlags


class LabeledIntent(BaseModel):
    label: IntentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)


class ExplanationResult(BaseModel):
    move_san: str
    fen_before: str
    intents: list[LabeledIntent]
    explanation: str
    feature_summary: dict[str, Any]
    model_version: str
