"""Claude API integration for generating human-readable move explanations."""

from __future__ import annotations

import json
import time

import anthropic

from whymove.classifier.labels import ALL_LABELS
from whymove.explainer.prompts import (
    LABELING_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    format_labeling_prompt,
    format_user_prompt,
)
from whymove.models import LabeledIntent, PositionFeatures


class ClaudeExplainer:
    """Generates human-readable chess move explanations using Claude."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-opus-4-5",
        max_tokens: int = 300,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def explain(
        self,
        fen: str,
        move_san: str,
        intents: list[LabeledIntent],
        features: PositionFeatures,
    ) -> str:
        """Generate a human-readable explanation for a chess move."""
        user_prompt = format_user_prompt(fen, move_san, intents, features)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text  # type: ignore[union-attr]

    def generate_labels(
        self,
        fen: str,
        move_san: str,
        features: PositionFeatures,
    ) -> list[str]:
        """Generate structured intent labels for a position (for training data).

        Returns a list of label strings from the taxonomy.
        """
        all_label_values = [l.value for l in ALL_LABELS]
        user_prompt = format_labeling_prompt(fen, move_san, features, all_label_values)
        message = self._client.messages.create(
            model=self._model,
            max_tokens=100,
            system=LABELING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = message.content[0].text.strip()  # type: ignore[union-attr]
        try:
            labels = json.loads(text)
            # Filter to valid labels only
            valid = {l.value for l in ALL_LABELS}
            return [l for l in labels if l in valid]
        except json.JSONDecodeError:
            return []

    def explain_batch(
        self,
        requests: list[tuple[str, str, list[LabeledIntent], PositionFeatures]],
        delay_seconds: float = 0.5,
    ) -> list[str]:
        """Generate explanations for a batch of moves with rate-limit handling."""
        results = []
        for fen, move_san, intents, features in requests:
            result = self._explain_with_retry(fen, move_san, intents, features)
            results.append(result)
            if delay_seconds > 0:
                time.sleep(delay_seconds)
        return results

    def _explain_with_retry(
        self,
        fen: str,
        move_san: str,
        intents: list[LabeledIntent],
        features: PositionFeatures,
        max_retries: int = 3,
    ) -> str:
        for attempt in range(max_retries):
            try:
                return self.explain(fen, move_san, intents, features)
            except anthropic.RateLimitError:
                wait = 2 ** attempt * 5
                time.sleep(wait)
            except anthropic.APIStatusError as e:
                if e.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
        return self.explain(fen, move_san, intents, features)
