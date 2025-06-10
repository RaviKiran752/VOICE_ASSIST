import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "tts_requests_total" in response.text 