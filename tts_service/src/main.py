import grpc
from concurrent import futures
from TTS.api import TTS
import tempfile
import os
import hashlib
import json
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import uvicorn
import redis
from redis.exceptions import RedisError
from prometheus_client import Counter, Histogram, generate_latest, Gauge, CONTENT_TYPE_LATEST
import voice_assist.proto.voice_assist_pb2 as voice_assist_pb2
import voice_assist.proto.voice_assist_pb2_grpc as voice_assist_pb2_grpc
from contextlib import contextmanager
import time
import io
import soundfile as sf
import numpy as np
import asyncio

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis connection pool
redis_pool = redis.ConnectionPool(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    max_connections=10,
    decode_responses=False,  # We need binary data for audio
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True
)

# Initialize Prometheus metrics
# TTS Service Metrics
TTS_REQUESTS = Counter('tts_requests_total', 'Total number of TTS requests')
TTS_CACHE_HITS = Counter('tts_cache_hits_total', 'Number of cache hits')
TTS_CACHE_MISSES = Counter('tts_cache_misses_total', 'Number of cache misses')
TTS_GENERATION_TIME = Histogram('tts_generation_seconds', 'Time spent generating speech')
TTS_ERRORS = Counter('tts_errors_total', 'Number of TTS errors', ['error_type'])

# Redis Metrics
REDIS_OPERATION_TIME = Histogram('redis_operation_seconds', 'Time spent on Redis operations', ['operation'])
REDIS_CONNECTION_STATUS = Gauge('redis_connection_status', 'Redis connection status (1 = connected, 0 = disconnected)')
REDIS_CACHE_SIZE = Gauge('redis_cache_size_bytes', 'Total size of cached audio data in bytes')
REDIS_CACHE_ITEMS = Gauge('redis_cache_items_total', 'Total number of cached items')
REDIS_CACHE_HIT_RATIO = Gauge('redis_cache_hit_ratio', 'Cache hit ratio (hits / (hits + misses))')
REDIS_CACHE_EVICTIONS = Counter('redis_cache_evictions_total', 'Number of cache evictions')
REDIS_CACHE_EXPIRATIONS = Counter('redis_cache_expirations_total', 'Number of cache expirations')

# Cache Performance Metrics
CACHE_OPERATION_TIME = Histogram('cache_operation_seconds', 'Time spent on cache operations', ['operation'])
CACHE_METADATA_SIZE = Gauge('cache_metadata_size_bytes', 'Size of cache metadata in bytes')
CACHE_AVERAGE_ITEM_SIZE = Gauge('cache_average_item_size_bytes', 'Average size of cached items')
CACHE_UTILIZATION = Gauge('cache_utilization_ratio', 'Cache utilization ratio (current items / max items)')

# Cache configuration
CACHE_VERSION = "v1"
CACHE_PREFIX = "tts"
CACHE_EXPIRY = 86400  # 24 hours
CACHE_MAX_SIZE = 1000  # Maximum number of cached items

@contextmanager
def get_redis_client():
    """Context manager for Redis client with connection pooling."""
    client = redis.Redis(connection_pool=redis_pool)
    try:
        yield client
    except RedisError as e:
        TTS_ERRORS.labels(error_type='redis_error').inc()
        REDIS_CONNECTION_STATUS.set(0)
        raise HTTPException(status_code=503, detail=f"Redis error: {str(e)}")
    finally:
        client.close()

def get_cache_key(text: str, voice_id: str = "default") -> str:
    """Generate a cache key for the given text and voice."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return f"{CACHE_PREFIX}:{CACHE_VERSION}:{voice_id}:{text_hash}"

def get_metadata_key(cache_key: str) -> str:
    """Generate a metadata key for the given cache key."""
    return f"{cache_key}:metadata"

def get_cache_stats_key() -> str:
    """Generate a key for cache statistics."""
    return f"{CACHE_PREFIX}:{CACHE_VERSION}:stats"

def get_cached_audio(text: str, voice_id: str = "default") -> tuple[bytes, dict]:
    """Try to get cached audio and its metadata."""
    cache_key = get_cache_key(text, voice_id)
    metadata_key = get_metadata_key(cache_key)
    start_time = time.time()
    
    try:
        with get_redis_client() as redis_client:
            # Get both audio and metadata in a pipeline
            pipe = redis_client.pipeline()
            pipe.get(cache_key)
            pipe.get(metadata_key)
            audio_data, metadata_json = pipe.execute()
            
            REDIS_OPERATION_TIME.labels(operation='get_with_metadata').observe(time.time() - start_time)
            
            if audio_data:
                TTS_CACHE_HITS.inc()
                metadata = json.loads(metadata_json) if metadata_json else {}
                return audio_data, metadata
            
            TTS_CACHE_MISSES.inc()
            return None, None
            
    except Exception as e:
        TTS_ERRORS.labels(error_type='cache_get_error').inc()
        return None, None

def cache_audio(text: str, audio_data: bytes, voice_id: str = "default", expire_seconds: int = CACHE_EXPIRY):
    """Cache the audio data with metadata."""
    cache_key = get_cache_key(text, voice_id)
    metadata_key = get_metadata_key(cache_key)
    stats_key = get_cache_stats_key()
    
    start_time = time.time()
    
    try:
        with get_redis_client() as redis_client:
            # Start a pipeline for atomic operations
            pipe = redis_client.pipeline()
            
            # Store audio data
            pipe.setex(cache_key, expire_seconds, audio_data)
            
            # Store metadata
            metadata = {
                "text": text,
                "voice_id": voice_id,
                "created_at": datetime.utcnow().isoformat(),
                "size_bytes": len(audio_data),
                "expires_at": (datetime.utcnow().timestamp() + expire_seconds)
            }
            pipe.setex(metadata_key, expire_seconds, json.dumps(metadata))
            
            # Update cache statistics
            pipe.hincrby(stats_key, "total_cached", 1)
            pipe.hincrby(stats_key, "total_size", len(audio_data))
            
            # Check if we need to evict old entries
            total_cached = int(redis_client.hget(stats_key, "total_cached") or 0)
            if total_cached > CACHE_MAX_SIZE:
                # Get oldest entries to evict
                keys_to_evict = redis_client.zrange(f"{CACHE_PREFIX}:{CACHE_VERSION}:access_time", 0, total_cached - CACHE_MAX_SIZE)
                if keys_to_evict:
                    for key in keys_to_evict:
                        pipe.delete(key)
                        pipe.delete(get_metadata_key(key))
                    pipe.hincrby(stats_key, "total_cached", -len(keys_to_evict))
            
            # Execute all operations
            pipe.execute()
            
            REDIS_OPERATION_TIME.labels(operation='set_with_metadata').observe(time.time() - start_time)
            
    except Exception as e:
        TTS_ERRORS.labels(error_type='cache_set_error').inc()
        print(f"Failed to cache audio: {str(e)}")

def update_cache_metrics():
    """Update cache-related metrics."""
    try:
        with get_redis_client() as redis_client:
            stats_key = get_cache_stats_key()
            stats = redis_client.hgetall(stats_key)
            
            total_cached = int(stats.get("total_cached", 0))
            total_size = int(stats.get("total_size", 0))
            
            REDIS_CACHE_SIZE.set(total_size)
            REDIS_CACHE_ITEMS.set(total_cached)
            REDIS_CACHE_HIT_RATIO.set(
                TTS_CACHE_HITS._value.get() / 
                (TTS_CACHE_HITS._value.get() + TTS_CACHE_MISSES._value.get() + 1e-9)
            )
            CACHE_UTILIZATION.set(total_cached / CACHE_MAX_SIZE)
            CACHE_AVERAGE_ITEM_SIZE.set(total_size / (total_cached + 1e-9))
            
    except Exception as e:
        print(f"Failed to update cache metrics: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check Redis connection and service health."""
    try:
        with get_redis_client() as redis_client:
            redis_client.ping()
            REDIS_CONNECTION_STATUS.set(1)
            update_cache_metrics()  # Update cache metrics
            return {
                "status": "healthy",
                "redis": "connected",
                "cache_stats": {
                    "total_items": REDIS_CACHE_ITEMS._value.get(),
                    "total_size": REDIS_CACHE_SIZE._value.get(),
                    "hit_ratio": REDIS_CACHE_HIT_RATIO._value.get()
                }
            }
    except Exception as e:
        REDIS_CONNECTION_STATUS.set(0)
        return {
            "status": "unhealthy",
            "redis": "disconnected",
            "error": str(e)
        }

# Initialize TTS model
# Using a fast and high-quality model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

class TTSServicer(voice_assist_pb2_grpc.TTSServiceServicer):
    def SynthesizeSpeech(self, request, context):
        try:
            TTS_REQUESTS.inc()
            
            # Try to get from cache first
            cached_audio, _ = get_cached_audio(request.text)
            if cached_audio:
                return voice_assist_pb2.SynthesizeResponse(
                    audio_data=cached_audio,
                    audio_format="wav"
                )

            # If not in cache, generate new audio
            with TTS_GENERATION_TIME.time():
                # Create a temporary file for the output
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    output_path = temp_file.name

                # Generate speech
                tts.tts_to_file(
                    text=request.text,
                    file_path=output_path,
                    speaker_wav=None  # Using default voice
                )

                # Read the generated audio file
                with open(output_path, "rb") as f:
                    audio_data = f.read()

                # Cache the generated audio
                cache_audio(request.text, audio_data)

                # Clean up the temporary file
                os.unlink(output_path)

                return voice_assist_pb2.SynthesizeResponse(
                    audio_data=audio_data,
                    audio_format="wav"
                )
        except Exception as e:
            TTS_ERRORS.inc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return voice_assist_pb2.SynthesizeResponse()

@app.post("/synthesize")
async def synthesize_speech(text: str):
    try:
        TTS_REQUESTS.inc()
        
        # Try to get from cache first
        cached_audio, _ = get_cached_audio(text)
        if cached_audio:
            return {
                "audio_data": cached_audio,
                "audio_format": "wav"
            }

        # If not in cache, generate new audio
        with TTS_GENERATION_TIME.time():
            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                output_path = temp_file.name

            # Generate speech
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=None  # Using default voice
            )

            # Read the generated audio file
            with open(output_path, "rb") as f:
                audio_data = f.read()

            # Cache the generated audio
            cache_audio(text, audio_data)

            # Clean up the temporary file
            os.unlink(output_path)

            return {
                "audio_data": audio_data,
                "audio_format": "wav"
            }
    except Exception as e:
        TTS_ERRORS.inc()
        return {"error": str(e)}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics."""
    try:
        with get_redis_client() as redis_client:
            stats_key = get_cache_stats_key()
            stats = redis_client.hgetall(stats_key)
            return {
                "total_cached": int(stats.get("total_cached", 0)),
                "total_size_bytes": int(stats.get("total_size", 0)),
                "max_size": CACHE_MAX_SIZE,
                "version": CACHE_VERSION
            }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to get cache stats: {str(e)}")

@app.get("/cache/metadata/{text_hash}")
async def get_cache_metadata(text_hash: str):
    """Get metadata for a specific cached item."""
    try:
        with get_redis_client() as redis_client:
            # Search for the key pattern
            pattern = f"{CACHE_PREFIX}:{CACHE_VERSION}:*:{text_hash}"
            keys = redis_client.keys(pattern)
            
            if not keys:
                raise HTTPException(status_code=404, detail="Cache entry not found")
            
            metadata_key = get_metadata_key(keys[0])
            metadata = redis_client.get(metadata_key)
            
            if not metadata:
                raise HTTPException(status_code=404, detail="Metadata not found")
            
            return json.loads(metadata)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to get metadata: {str(e)}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    voice_assist_pb2_grpc.add_TTSServiceServicer_to_server(
        TTSServicer(), server
    )
    server.add_insecure_port("[::]:50053")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    # Start gRPC server in a separate thread
    import threading
    grpc_thread = threading.Thread(target=serve)
    grpc_thread.start()
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000) 