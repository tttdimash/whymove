"""Unit tests for the prompt templates."""

from __future__ import annotations

import pytest

from whymove.classifier.labels import IntentLabel
from whymove.explainer.prompts import (
    SYSTEM_PROMPT,
    format_user_prompt,
)
from whymove.models import LabeledIntent


def test_system_prompt_nonempty():
    assert len(SYSTEM_PROMPT) > 100


def test_system_prompt_contains_key_terms():
    assert "chess" in SYSTEM_PROMPT.lower()
    assert "ELO" in SYSTEM_PROMPT or "elo" in SYSTEM_PROMPT.lower()


def test_format_user_prompt_contains_fen(sample_features):
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    intents = [LabeledIntent(label=IntentLabel.FORK, confidence=0.87)]
    prompt = format_user_prompt(fen, "e4", intents, sample_features)
    assert fen in prompt


def test_format_user_prompt_contains_move(sample_features):
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    intents = [LabeledIntent(label=IntentLabel.IMPROVE_PIECE, confidence=0.65)]
    prompt = format_user_prompt(fen, "Nf6", intents, sample_features)
    assert "Nf6" in prompt


def test_format_user_prompt_contains_intent(sample_features):
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    intents = [LabeledIntent(label=IntentLabel.FORK, confidence=0.87)]
    prompt = format_user_prompt(fen, "e4", intents, sample_features)
    assert "fork" in prompt


def test_format_user_prompt_no_intents(sample_features):
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    prompt = format_user_prompt(fen, "e4", [], sample_features)
    assert "no strong signals" in prompt


def test_format_user_prompt_eval_delta(sample_features):
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    prompt = format_user_prompt(fen, "e4", [], sample_features)
    # eval_delta_cp=50 should appear
    assert "+50" in prompt or "50" in prompt
