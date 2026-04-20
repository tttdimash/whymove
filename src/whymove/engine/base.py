"""Abstract chess engine interface — all engine interaction is mediated through this ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class EngineEvaluation(BaseModel):
    score_cp: int           # centipawns from side-to-move perspective
    score_mate: int | None  # None if not a forced mate
    depth: int
    best_move_uci: str | None = None


class ChessEngine(ABC):
    """Abstract interface for chess engines. Implement this to add a new engine backend."""

    @abstractmethod
    def evaluate(self, fen: str, depth: int = 20) -> EngineEvaluation:
        """Evaluate a position and return the score."""
        ...

    @abstractmethod
    def get_best_move(self, fen: str, depth: int = 20) -> str:
        """Return the best move in UCI notation."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release engine resources."""
        ...

    def __enter__(self) -> ChessEngine:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
