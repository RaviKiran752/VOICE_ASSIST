import pytest
from fastapi.testclient import TestClient
import io
import soundfile as sf
import numpy as np
from src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "cache_stats" in data

def test_tts_endpoint():
    # Create a test audio file
    sample_rate = 22050
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Save to a buffer
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format='WAV')
    buffer.seek(0)
    
    # Create form data
    files = {
        'audio': ('test.wav', buffer, 'audio/wav')
    }
    data = {
        'text': 'Hello, this is a test.'
    }
    
    response = client.post("/tts", files=files, data=data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav" 