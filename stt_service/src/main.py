import grpc
from concurrent import futures
import whisper
import tempfile
import os
import redis
from prometheus_client import Counter, Histogram, generate_latest
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
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
    decode_responses=True  # We need text for transcriptions
)

# Initialize Prometheus metrics
STT_REQUESTS = Counter('stt_requests_total', 'Total number of STT requests')
STT_CACHE_HITS = Counter('stt_cache_hits_total', 'Number of cache hits')
STT_PROCESSING_TIME = Histogram('stt_processing_seconds', 'Time spent processing speech')
STT_ERRORS = Counter('stt_errors_total', 'Number of STT errors')

# Initialize Whisper model
model = whisper.load_model("base")

def get_cache_key(audio_data: bytes) -> str:
    """Generate a cache key for the given audio data."""
    import hashlib
    return f"stt:{hashlib.md5(audio_data).hexdigest()}"

def get_cached_transcription(audio_data: bytes) -> str:
    """Try to get cached transcription for the given audio."""
    cache_key = get_cache_key(audio_data)
    cached_text = redis_client.get(cache_key)
    if cached_text:
        STT_CACHE_HITS.inc()
        return cached_text
    return None

def cache_transcription(audio_data: bytes, text: str, expire_seconds: int = 86400):
    """Cache the transcription for the given audio."""
    cache_key = get_cache_key(audio_data)
    redis_client.setex(cache_key, expire_seconds, text)

class STTServicer(voice_assist_pb2_grpc.STTServiceServicer):
    def TranscribeSpeech(self, request, context):
        try:
            STT_REQUESTS.inc()
            
            # Try to get from cache first
            cached_text = get_cached_transcription(request.audio_data)
            if cached_text:
                return voice_assist_pb2.TranscribeResponse(
                    text=cached_text
                )

            # If not in cache, process the audio
            with STT_PROCESSING_TIME.time():
                # Create a temporary file for the audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(request.audio_data)
                    temp_file.flush()
                    
                    # Transcribe using Whisper
                    result = model.transcribe(temp_file.name)
                    text = result["text"].strip()

                    # Cache the transcription
                    cache_transcription(request.audio_data, text)

                    # Clean up
                    os.unlink(temp_file.name)

                    return voice_assist_pb2.TranscribeResponse(
                        text=text
                    )
        except Exception as e:
            STT_ERRORS.inc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return voice_assist_pb2.TranscribeResponse()

@app.post("/transcribe")
async def transcribe_speech(audio_data: bytes):
    try:
        STT_REQUESTS.inc()
        
        # Try to get from cache first
        cached_text = get_cached_transcription(audio_data)
        if cached_text:
            return {"text": cached_text}

        # If not in cache, process the audio
        with STT_PROCESSING_TIME.time():
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                
                # Transcribe using Whisper
                result = model.transcribe(temp_file.name)
                text = result["text"].strip()

                # Cache the transcription
                cache_transcription(audio_data, text)

                # Clean up
                os.unlink(temp_file.name)

                return {"text": text}
    except Exception as e:
        STT_ERRORS.inc()
        return {"error": str(e)}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    voice_assist_pb2_grpc.add_STTServiceServicer_to_server(
        STTServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    # Start gRPC server in a separate thread
    import threading
    grpc_thread = threading.Thread(target=serve)
    grpc_thread.start()
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000) 