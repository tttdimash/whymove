"""AppConfig and Container — dependency injection for engine, classifier, and explainer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from whymove.engine.base import ChessEngine
from whymove.engine.factory import EngineType, create_engine


@dataclass
class AppConfig:
    stockfish_path: str | None = None
    engine_type: EngineType = "stockfish"
    engine_depth: int = 20
    model_path: Path = field(default_factory=lambda: Path("models/intent_classifier.joblib"))
    claude_model: str = "claude-opus-4-5"
    claude_max_tokens: int = 300
    classifier_threshold: float = 0.3

    @classmethod
    def from_env(cls) -> AppConfig:
        """Build config from environment variables, falling back to defaults."""
        return cls(
            stockfish_path=os.environ.get("STOCKFISH_PATH"),
            engine_type=os.environ.get("ENGINE_TYPE", "stockfish"),  # type: ignore[arg-type]
            engine_depth=int(os.environ.get("ENGINE_DEPTH", "20")),
            model_path=Path(os.environ.get("MODEL_PATH", "models/intent_classifier.joblib")),
            claude_model=os.environ.get("CLAUDE_MODEL", "claude-opus-4-5"),
            claude_max_tokens=int(os.environ.get("CLAUDE_MAX_TOKENS", "300")),
            classifier_threshold=float(os.environ.get("CLASSIFIER_THRESHOLD", "0.3")),
        )


class Container:
    """Builds and caches singleton instances of all services.

    Manages the lifecycle of the chess engine subprocess.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._engine: ChessEngine | None = None
        self._classifier = None
        self._explainer = None
        self._pipeline = None

    def engine(self) -> ChessEngine:
        if self._engine is None:
            self._engine = create_engine(
                self._config.engine_type,
                path=self._config.stockfish_path,
                depth=self._config.engine_depth,
            )
        return self._engine

    def classifier(self):
        if self._classifier is None:
            from whymove.classifier.model import IntentClassifier
            self._classifier = IntentClassifier.load(self._config.model_path)
        return self._classifier

    def explainer(self):
        if self._explainer is None:
            from whymove.explainer.claude_client import ClaudeExplainer
            self._explainer = ClaudeExplainer(
                model=self._config.claude_model,
                max_tokens=self._config.claude_max_tokens,
            )
        return self._explainer

    def pipeline(self):
        if self._pipeline is None:
            from whymove.pipeline import AnalysisPipeline
            self._pipeline = AnalysisPipeline(
                engine=self.engine(),
                classifier=self.classifier(),
                explainer=self.explainer(),
                engine_depth=self._config.engine_depth,
            )
        return self._pipeline

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None
        self._pipeline = None
