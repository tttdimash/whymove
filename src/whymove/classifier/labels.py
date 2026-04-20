"""Intent label taxonomy — 40 labels covering all chess strategic and tactical concepts."""

from __future__ import annotations

from enum import Enum


class IntentLabel(str, Enum):
    # ── Tactical (18) ─────────────────────────────────────────────────────────
    FORK = "fork"
    PIN = "pin"
    SKEWER = "skewer"
    DISCOVERED_ATTACK = "discovered_attack"
    DOUBLE_ATTACK = "double_attack"
    DOUBLE_CHECK = "double_check"
    MATING_THREAT = "mating_threat"
    MATING_COMBINATION = "mating_combination"
    CAPTURE = "capture"
    EXCHANGE = "exchange"
    SACRIFICE = "sacrifice"
    DEFLECTION = "deflection"
    DECOY = "decoy"
    ZWISCHENZUG = "zwischenzug"
    X_RAY = "x_ray"
    OVERLOADING = "overloading"
    INTERPOSITION = "interposition"
    DEFENSIVE = "defensive"

    # ── Positional / Strategic (10) ───────────────────────────────────────────
    IMPROVE_PIECE = "improve_piece"
    OUTPOST_CREATION = "outpost_creation"
    PAWN_STRUCTURE = "pawn_structure"
    CONTROL_KEY_SQUARE = "control_key_square"
    PIECE_COORDINATION = "piece_coordination"
    RESTRICT_OPPONENT = "restrict_opponent"
    OPEN_FILE = "open_file"
    SPACE_ADVANTAGE = "space_advantage"
    BISHOP_PAIR = "bishop_pair"
    WEAK_SQUARE_EXPLOITATION = "weak_square_exploitation"

    # ── Plan Preparation (6) ──────────────────────────────────────────────────
    PROPHYLAXIS = "prophylaxis"
    PAWN_BREAK_PREP = "pawn_break_prep"
    PIECE_REROUTING = "piece_rerouting"
    CASTLING_PREP = "castling_prep"
    OPEN_FILE_PREP = "open_file_prep"
    ATTACK_PREPARATION = "attack_preparation"

    # ── Endgame (6) ───────────────────────────────────────────────────────────
    KING_ACTIVATION = "king_activation"
    PAWN_PROMOTION_PREP = "pawn_promotion_prep"
    SIMPLIFICATION = "simplification"
    ZUGZWANG = "zugzwang"
    OPPOSITION = "opposition"
    PASSED_PAWN = "passed_pawn"


ALL_LABELS: list[IntentLabel] = list(IntentLabel)
LABEL_INDEX: dict[IntentLabel, int] = {label: i for i, label in enumerate(ALL_LABELS)}
N_LABELS: int = len(ALL_LABELS)
