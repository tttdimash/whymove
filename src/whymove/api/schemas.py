"""FastAPI-specific request/response schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fen": "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
                "move_uci": "f1c4",
                "engine_depth": 15,
                "top_k_labels": 5,
            }
        }
    )

    fen: str | None = Field(default=None, description="Position in FEN notation")
    pgn: str | None = Field(default=None, description="Game in PGN notation (uses final position)")
    move_uci: str = Field(..., description="Move in UCI notation, e.g. 'e2e4'")
    engine_depth: int = Field(default=15, ge=1, le=30, description="Stockfish search depth")
    top_k_labels: int = Field(default=5, ge=1, le=10, description="Number of top intent labels to return")


class IntentItem(BaseModel):
    label: str
    confidence: float


class AnalyzeResponse(BaseModel):
    move_san: str
    fen_before: str
    intents: list[IntentItem]
    explanation: str
    feature_summary: dict[str, Any]
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    engine: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
