"""Stockfish engine adapter using the stockfish Python package."""

from __future__ import annotations

import shutil

from stockfish import Stockfish

from whymove.engine.base import ChessEngine, EngineEvaluation


class StockfishEngine(ChessEngine):
    """Wraps the stockfish Python package."""

    def __init__(self, path: str | None = None, depth: int = 20) -> None:
        resolved = path or shutil.which("stockfish")
        if resolved is None:
            raise RuntimeError(
                "Stockfish binary not found. Install it (e.g. `brew install stockfish`) "
                "or set STOCKFISH_PATH."
            )
        self._sf = Stockfish(path=resolved, depth=depth)
        self._default_depth = depth

    def evaluate(self, fen: str, depth: int = 20) -> EngineEvaluation:
        self._sf.set_depth(depth)
        self._sf.set_fen_position(fen)
        raw = self._sf.get_evaluation()
        best = self._sf.get_best_move()

        if raw["type"] == "mate":
            return EngineEvaluation(
                score_cp=0,
                score_mate=int(raw["value"]),
                depth=depth,
                best_move_uci=best,
            )
        return EngineEvaluation(
            score_cp=int(raw["value"]),
            score_mate=None,
            depth=depth,
            best_move_uci=best,
        )

    def get_best_move(self, fen: str, depth: int = 20) -> str:
        self._sf.set_depth(depth)
        self._sf.set_fen_position(fen)
        move = self._sf.get_best_move()
        if move is None:
            raise ValueError(f"No legal moves in position: {fen}")
        return move

    def close(self) -> None:
        # stockfish Python package doesn't expose an explicit close method;
        # the subprocess is cleaned up on GC. This is a no-op.
        pass
