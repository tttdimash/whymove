"""Lc0 (Leela Chess Zero) engine adapter — planned, not yet implemented."""

from __future__ import annotations

from whymove.engine.base import ChessEngine, EngineEvaluation

_NOT_IMPLEMENTED = (
    "Lc0 support is planned but not yet implemented. "
    "Use engine_type='stockfish' for now."
)


class Lc0Engine(ChessEngine):
    """Stub for Lc0 integration. Will use chess.engine.SimpleEngine.popen_uci('lc0')."""

    def __init__(self, path: str | None = None, weights_path: str | None = None) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def evaluate(self, fen: str, depth: int = 20) -> EngineEvaluation:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def get_best_move(self, fen: str, depth: int = 20) -> str:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def close(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)
