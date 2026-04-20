"""Engine factory — create engine instances by type name."""

from __future__ import annotations

from typing import Literal

from whymove.engine.base import ChessEngine

EngineType = Literal["stockfish", "lc0"]


def create_engine(engine_type: EngineType = "stockfish", **kwargs: object) -> ChessEngine:
    """Instantiate and return a chess engine by type.

    Args:
        engine_type: "stockfish" or "lc0"
        **kwargs: Passed to the engine constructor (e.g. path=, depth=)
    """
    if engine_type == "stockfish":
        from whymove.engine.stockfish_engine import StockfishEngine
        return StockfishEngine(**kwargs)  # type: ignore[arg-type]
    if engine_type == "lc0":
        from whymove.engine.lc0_engine import Lc0Engine
        return Lc0Engine(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown engine type: {engine_type!r}. Choose 'stockfish' or 'lc0'.")
