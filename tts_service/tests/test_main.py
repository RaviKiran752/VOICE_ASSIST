import pytest
from fastapi.testclient import TestClient
import io
import soundfile as sf
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock proto modules before importing app
with patch('voice_assist.proto.voice_assist_pb2') as mock_pb2, \
     patch('voice_assist.proto.voice_assist_pb2_grpc') as mock_pb2_grpc, \
     patch('redis.ConnectionPool') as mock_pool, \
     patch('redis.Redis') as mock_redis, \
     patch('TTS.api.TTS') as mock_tts:
    
    # Setup Redis mock
    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    mock_redis_client.get.return_value = None
    mock_redis_client.setex.return_value = True
    mock_redis_client.hgetall.return_value = {
        "total_cached": "0",
        "total_size": "0"
    }
    mock_redis.return_value = mock_redis_client
    
    # Setup TTS mock
    mock_tts_instance = MagicMock()
    mock_tts_instance.tts_to_file.return_value = None
    mock_tts.return_value = mock_tts_instance
    
    # Setup proto mocks
    mock_pb2.SynthesizeResponse = MagicMock
    mock_pb2_grpc.TTSServiceServicer = MagicMock
    
    from tts_service.src.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "cache_stats" in data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "tts_requests_total" in response.text

def test_synthesize_endpoint():
    response = client.post("/synthesize", json={"text": "Hello, this is a test."})
    assert response.status_code == 200
    data = response.json()
    assert "audio_data" in data
    assert "audio_format" in data
    assert data["audio_format"] == "wav"

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