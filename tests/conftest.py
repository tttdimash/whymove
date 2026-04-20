"""Shared pytest fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import chess
import pytest

from whymove.classifier.labels import IntentLabel
from whymove.classifier.model import IntentClassifier
from whymove.engine.base import ChessEngine, EngineEvaluation
from whymove.explainer.claude_client import ClaudeExplainer
from whymove.models import (
    ExplanationResult,
    KingSafetyFeatures,
    LabeledIntent,
    PawnStructureFeatures,
    PieceInfo,
    PositionFeatures,
    TacticalFlags,
)


# ── Known positions ───────────────────────────────────────────────────────────

@pytest.fixture
def starting_fen() -> str:
    return chess.STARTING_FEN


@pytest.fixture
def italian_fen() -> str:
    """Italian game position after 1.e4 e5 2.Nf3 Nc6 3.Bc4"""
    return "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"


@pytest.fixture
def fork_fen() -> str:
    """Position where Nf3-d4 forks queen and rook."""
    # White knight can fork black queen and rook
    return "r5k1/8/8/8/8/5N2/8/4K3 w - - 0 1"


# ── Sample PositionFeatures ───────────────────────────────────────────────────

@pytest.fixture
def sample_features() -> PositionFeatures:
    return PositionFeatures(
        eval_before_cp=30,
        eval_after_cp=80,
        eval_delta_cp=50,
        material_before=0,
        material_after=0,
        material_delta=0,
        moved_piece_mobility_before=4,
        moved_piece_mobility_after=8,
        total_white_mobility_delta=3,
        total_black_mobility_delta=-2,
        piece=PieceInfo(piece_type=chess.KNIGHT, color=chess.WHITE, from_square=21, to_square=36),
        distance_moved=2,
        destination_rank=4,
        destination_file=4,
        king_safety=KingSafetyFeatures(
            white_king_zone_attacks_delta=0,
            black_king_zone_attacks_delta=1,
            white_open_files_near_king_delta=0,
            black_open_files_near_king_delta=0,
        ),
        pawn_structure=PawnStructureFeatures(
            doubled_pawns_delta=0,
            isolated_pawns_delta=0,
            passed_pawns_delta=0,
            pawn_islands_delta=0,
        ),
        center_control_delta=1,
        key_square_control_delta=2,
        tactical=TacticalFlags(is_fork=True, is_check=False, is_capture=False),
    )


# ── Mock engine ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_engine() -> MagicMock:
    engine = MagicMock(spec=ChessEngine)
    engine.evaluate.return_value = EngineEvaluation(
        score_cp=30, score_mate=None, depth=15, best_move_uci="d1f3"
    )
    engine.get_best_move.return_value = "d1f3"
    return engine


# ── Mock classifier ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_classifier() -> MagicMock:
    clf = MagicMock(spec=IntentClassifier)
    clf.predict_one.return_value = [
        LabeledIntent(label=IntentLabel.FORK, confidence=0.87),
        LabeledIntent(label=IntentLabel.MATING_THREAT, confidence=0.42),
        LabeledIntent(label=IntentLabel.IMPROVE_PIECE, confidence=0.31),
    ]
    clf.is_trained = True
    return clf


# ── Mock explainer ────────────────────────────────────────────────────────────

@pytest.fixture
def mock_explainer() -> MagicMock:
    explainer = MagicMock(spec=ClaudeExplainer)
    explainer.explain.return_value = (
        "This knight move creates a powerful fork, simultaneously attacking "
        "the queen and rook and forcing material loss."
    )
    return explainer


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture
def test_app(mock_engine, mock_classifier, mock_explainer):
    """FastAPI app with all external dependencies mocked."""
    from whymove.api.app import create_app
    from whymove.container import AppConfig, Container
    from pathlib import Path

    config = AppConfig(model_path=Path("nonexistent.joblib"))
    app = create_app(config)

    # Override the container with mocked services
    container = Container(config)
    container._engine = mock_engine
    container._classifier = mock_classifier
    container._explainer = mock_explainer
    app.state.container = container

    return app
