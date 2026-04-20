"""FastAPI route handlers."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from whymove import __version__
from whymove.api.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, IntentItem
from whymove.classifier.labels import ALL_LABELS
from whymove.container import Container
from whymove.models import MoveInput

router = APIRouter()


def get_container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_move(
    request: AnalyzeRequest,
    container: Container = Depends(get_container),
) -> AnalyzeResponse:
    """Analyze a chess move and explain its strategic/tactical intent."""
    if request.fen is None and request.pgn is None:
        raise HTTPException(status_code=422, detail="Either 'fen' or 'pgn' must be provided.")

    move_input = MoveInput(
        fen=request.fen,
        pgn=request.pgn,
        move_uci=request.move_uci,
        engine_depth=request.engine_depth,
        top_k_labels=request.top_k_labels,
    )

    try:
        result = container.pipeline().analyze(move_input)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Analysis failed: {e}")

    return AnalyzeResponse(
        move_san=result.move_san,
        fen_before=result.fen_before,
        intents=[IntentItem(label=i.label.value, confidence=i.confidence) for i in result.intents],
        explanation=result.explanation,
        feature_summary=result.feature_summary,
        model_version=result.model_version,
    )


@router.get("/health", response_model=HealthResponse)
async def health(container: Container = Depends(get_container)) -> HealthResponse:
    """Health check endpoint."""
    try:
        engine = container.engine()
        engine_name = type(engine).__name__
    except Exception:
        engine_name = "unavailable"

    model_loaded = False
    try:
        clf = container.classifier()
        model_loaded = clf.is_trained
    except Exception:
        pass

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        engine=engine_name,
        model_loaded=model_loaded,
        version=__version__,
    )


@router.get("/labels", response_model=list[str])
async def list_labels() -> list[str]:
    """Return all available intent label values."""
    return [label.value for label in ALL_LABELS]
