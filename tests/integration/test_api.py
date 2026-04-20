"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(test_app):
    return TestClient(test_app, raise_server_exceptions=False)


def test_health_endpoint(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert "version" in data
    assert "model_loaded" in data


def test_labels_endpoint(client):
    resp = client.get("/v1/labels")
    assert resp.status_code == 200
    labels = resp.json()
    assert isinstance(labels, list)
    assert len(labels) == 40
    assert "fork" in labels
    assert "improve_piece" in labels
    assert "king_activation" in labels


def test_analyze_valid_fen(client):
    resp = client.post("/v1/analyze", json={
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move_uci": "e2e4",
        "engine_depth": 10,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["move_san"] == "e4"
    assert "intents" in data
    assert "explanation" in data
    assert "feature_summary" in data


def test_analyze_illegal_move_returns_422(client):
    resp = client.post("/v1/analyze", json={
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move_uci": "e2e5",  # illegal
    })
    assert resp.status_code in (422, 503)


def test_analyze_missing_fen_and_pgn_returns_422(client):
    resp = client.post("/v1/analyze", json={"move_uci": "e2e4"})
    assert resp.status_code == 422


def test_analyze_response_schema(client):
    resp = client.post("/v1/analyze", json={
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move_uci": "e2e4",
    })
    assert resp.status_code == 200
    data = resp.json()
    # Validate response schema fields
    required_fields = {"move_san", "fen_before", "intents", "explanation", "feature_summary", "model_version"}
    assert required_fields.issubset(data.keys())
    # Validate intent items
    for intent in data["intents"]:
        assert "label" in intent
        assert "confidence" in intent
        assert 0.0 <= intent["confidence"] <= 1.0


def test_analyze_top_k_labels(client):
    resp = client.post("/v1/analyze", json={
        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "move_uci": "e2e4",
        "top_k_labels": 2,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["intents"]) <= 2
