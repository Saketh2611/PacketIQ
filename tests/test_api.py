"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from document_intelligence.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_missing_input(client):
    resp = client.post("/analyze")
    assert resp.status_code == 400


def test_retrieve_no_index(client):
    resp = client.post("/retrieve", json={"query": "test", "top_k": 5})
    assert resp.status_code in (200, 404)
