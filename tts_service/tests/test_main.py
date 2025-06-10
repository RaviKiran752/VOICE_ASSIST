import pytest
import os
import sys
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tts_service.src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "tts_requests_total" in response.text 