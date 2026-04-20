"""Main analysis pipeline — orchestrates engine, features, classifier, and explainer."""

from __future__ import annotations

import io
from typing import Any

import chess
import chess.pgn

from whymove.classifier.model import IntentClassifier
from whymove.engine.base import ChessEngine
from whymove.explainer.claude_client import ClaudeExplainer
from whymove.features.extractor import FeatureExtractor
from whymove.models import ExplanationResult, LabeledIntent, MoveInput, PositionFeatures


class AnalysisPipeline:
    """End-to-end pipeline: MoveInput → ExplanationResult."""

    def __init__(
        self,
        engine: ChessEngine,
        classifier: IntentClassifier,
        explainer: ClaudeExplainer,
        engine_depth: int = 20,
        top_k_labels: int = 5,
    ) -> None:
        self._extractor = FeatureExtractor(engine)
        self._classifier = classifier
        self._explainer = explainer
        self._engine_depth = engine_depth
        self._top_k = top_k_labels

    def analyze(self, request: MoveInput) -> ExplanationResult:
        """Analyze a chess move and return labeled intents + human-readable explanation.

        Steps:
            1. Resolve FEN from PGN if needed
            2. Validate move legality
            3. Extract position features (engine + python-chess)
            4. Classify intent (multi-label with confidence scores)
            5. Generate explanation (Claude)
            6. Return ExplanationResult
        """
        # Step 1: Resolve FEN
        fen = self._resolve_fen(request)

        # Step 2: Validate move
        board = chess.Board(fen)
        try:
            move = chess.Move.from_uci(request.move_uci)
        except ValueError as e:
            raise ValueError(f"Invalid UCI move '{request.move_uci}': {e}") from e

        if move not in board.legal_moves:
            raise ValueError(
                f"Illegal move '{request.move_uci}' in position '{fen}'. "
                f"Legal moves: {', '.join(m.uci() for m in board.legal_moves)}"
            )
        move_san = board.san(move)

        # Step 3: Extract features
        features: PositionFeatures = self._extractor.extract(
            fen, request.move_uci, self._engine_depth
        )

        # Step 4: Classify intent
        top_k = request.top_k_labels if request.top_k_labels > 0 else self._top_k
        all_intents: list[LabeledIntent] = self._classifier.predict_one(features)
        intents = all_intents[:top_k]

        # Step 5: Generate explanation
        explanation = self._explainer.explain(fen, move_san, intents, features)

        # Step 6: Assemble result
        return ExplanationResult(
            move_san=move_san,
            fen_before=fen,
            intents=intents,
            explanation=explanation,
            feature_summary=_build_feature_summary(features),
            model_version="0.1.0",
        )

    def _resolve_fen(self, request: MoveInput) -> str:
        if request.fen:
            # Validate the FEN
            try:
                chess.Board(request.fen)
            except ValueError as e:
                raise ValueError(f"Invalid FEN: {e}") from e
            return request.fen
        if request.pgn:
            return _pgn_to_fen(request.pgn)
        raise ValueError("Either 'fen' or 'pgn' must be provided.")


def _pgn_to_fen(pgn_text: str) -> str:
    """Parse PGN text and return the FEN of the final position."""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Could not parse PGN text.")
    board = game.end().board()
    return board.fen()


def _build_feature_summary(features: PositionFeatures) -> dict[str, Any]:
    """Build a human-readable subset of features for API/CLI consumers."""
    t = features.tactical
    return {
        "eval_delta_cp": features.eval_delta_cp,
        "material_delta": features.material_delta,
        "piece_type": features.piece.piece_type,
        "is_check": t.is_check,
        "is_capture": t.is_capture,
        "is_promotion": t.is_promotion,
        "tactical_flags": {
            k: v
            for k, v in t.model_dump().items()
            if v is True
        },
        "mobility_delta": features.moved_piece_mobility_after - features.moved_piece_mobility_before,
        "center_control_delta": features.center_control_delta,
    }
