"""Integration tests for the analysis pipeline (all external deps mocked)."""

from __future__ import annotations

import pytest

from whymove.models import MoveInput
from whymove.pipeline import AnalysisPipeline


@pytest.fixture
def pipeline(mock_engine, mock_classifier, mock_explainer):
    return AnalysisPipeline(
        engine=mock_engine,
        classifier=mock_classifier,
        explainer=mock_explainer,
        engine_depth=15,
    )


def test_pipeline_analyze_with_fen(pipeline):
    # Italian game: bishop on c4 plays Nc3 (b1c3)
    result = pipeline.analyze(MoveInput(
        fen="rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        move_uci="b1c3",
    ))
    assert result.move_san == "Nc3"


def test_pipeline_analyze_starting_position(pipeline):
    result = pipeline.analyze(MoveInput(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="e2e4",
    ))
    assert result.move_san == "e4"
    assert result.fen_before == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    assert len(result.intents) > 0
    assert result.explanation != ""
    assert result.model_version == "0.1.0"


def test_pipeline_analyze_with_pgn(pipeline):
    # PGN ends after 2...Nc6, so it's White to move; Ng5 (f3g5) is legal
    pgn = """[Event "Test"]
1. e4 e5 2. Nf3 Nc6"""
    result = pipeline.analyze(MoveInput(pgn=pgn, move_uci="f3g5"))
    assert result.move_san == "Ng5"


def test_pipeline_illegal_move_raises(pipeline):
    with pytest.raises(ValueError, match="[Ii]llegal"):
        pipeline.analyze(MoveInput(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move_uci="e2e5",  # illegal — pawn can't jump 3 squares
        ))


def test_pipeline_invalid_uci_raises(pipeline):
    with pytest.raises(ValueError):
        pipeline.analyze(MoveInput(
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            move_uci="zzz",  # invalid UCI
        ))


def test_pipeline_no_fen_or_pgn_raises(pipeline):
    with pytest.raises(ValueError, match="fen.*pgn|pgn.*fen"):
        pipeline.analyze(MoveInput(move_uci="e2e4"))


def test_pipeline_feature_summary_keys(pipeline):
    result = pipeline.analyze(MoveInput(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="e2e4",
    ))
    fs = result.feature_summary
    assert "eval_delta_cp" in fs
    assert "is_check" in fs
    assert "is_capture" in fs


def test_pipeline_top_k_respected(pipeline):
    result = pipeline.analyze(MoveInput(
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_uci="e2e4",
        top_k_labels=2,
    ))
    assert len(result.intents) <= 2
