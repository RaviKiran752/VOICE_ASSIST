import grpc
from concurrent import futures
from TTS.api import TTS
import tempfile
import os
import hashlib
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import redis
from prometheus_client import Counter, Histogram, generate_latest
import proto.voice_assist_pb2 as voice_assist_pb2
import proto.voice_assist_pb2_grpc as voice_assist_pb2_grpc

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    db=0,
    decode_responses=False  # We need binary data for audio
)

# Initialize Prometheus metrics
TTS_REQUESTS = Counter('tts_requests_total', 'Total number of TTS requests')
TTS_CACHE_HITS = Counter('tts_cache_hits_total', 'Number of cache hits')
TTS_GENERATION_TIME = Histogram('tts_generation_seconds', 'Time spent generating speech')
TTS_ERRORS = Counter('tts_errors_total', 'Number of TTS errors')

# Initialize TTS model
# Using a fast and high-quality model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def get_cache_key(text: str) -> str:
    """Generate a cache key for the given text."""
    return f"tts:{hashlib.md5(text.encode()).hexdigest()}"

def get_cached_audio(text: str) -> bytes:
    """Try to get cached audio for the given text."""
    cache_key = get_cache_key(text)
    cached_data = redis_client.get(cache_key)
    if cached_data:
        TTS_CACHE_HITS.inc()
        return cached_data
    return None

def cache_audio(text: str, audio_data: bytes, expire_seconds: int = 86400):
    """Cache the audio data for the given text."""
    cache_key = get_cache_key(text)
    redis_client.setex(cache_key, expire_seconds, audio_data)

class TTSServicer(voice_assist_pb2_grpc.TTSServiceServicer):
    def SynthesizeSpeech(self, request, context):
        try:
            TTS_REQUESTS.inc()
            
            # Try to get from cache first
            cached_audio = get_cached_audio(request.text)
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
        cached_audio = get_cached_audio(text)
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