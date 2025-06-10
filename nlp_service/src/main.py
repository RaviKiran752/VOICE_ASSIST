import grpc
from concurrent import futures
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
from fastapi import FastAPI, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import redis
from redis.exceptions import RedisError
from prometheus_client import Counter, Histogram, generate_latest, Gauge
import proto.voice_assist_pb2 as voice_assist_pb2
import proto.voice_assist_pb2_grpc as voice_assist_pb2_grpc
import json
import hashlib
from datetime import datetime
from contextlib import contextmanager
import time
import os

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
    db=1,  # Using different DB than TTS/STT
    max_connections=10,
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True
)

# Initialize Prometheus metrics
NLP_REQUESTS = Counter('nlp_requests_total', 'Total number of NLP requests')
NLP_CACHE_HITS = Counter('nlp_cache_hits_total', 'Number of cache hits')
NLP_CACHE_MISSES = Counter('nlp_cache_misses_total', 'Number of cache misses')
NLP_PROCESSING_TIME = Histogram('nlp_processing_seconds', 'Time spent processing text')
NLP_ERRORS = Counter('nlp_errors_total', 'Number of NLP errors', ['error_type'])
NLP_INTENT_DISTRIBUTION = Counter('nlp_intent_distribution_total', 'Distribution of detected intents', ['intent'])
NLP_ENTITY_DISTRIBUTION = Counter('nlp_entity_distribution_total', 'Distribution of detected entities', ['entity_type'])
NLP_CONFIDENCE = Histogram('nlp_confidence', 'Confidence scores for intent detection')
REDIS_OPERATION_TIME = Histogram('nlp_redis_operation_seconds', 'Time spent on Redis operations', ['operation'])
REDIS_CONNECTION_STATUS = Gauge('nlp_redis_connection_status', 'Redis connection status (1 = connected, 0 = disconnected)')

# Cache configuration
CACHE_VERSION = "v1"
CACHE_PREFIX = "nlp"
CACHE_EXPIRY = 86400  # 24 hours
CACHE_MAX_SIZE = 1000  # Maximum number of cached items

@contextmanager
def get_redis_client():
    """Context manager for Redis client with connection pooling."""
    client = redis.Redis(connection_pool=redis_pool)
    try:
        yield client
    except RedisError as e:
        NLP_ERRORS.labels(error_type='redis_error').inc()
        REDIS_CONNECTION_STATUS.set(0)
        raise HTTPException(status_code=503, detail=f"Redis error: {str(e)}")
    finally:
        client.close()

def get_cache_key(text: str) -> str:
    """Generate a cache key for the given text."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return f"{CACHE_PREFIX}:{CACHE_VERSION}:{text_hash}"

def get_metadata_key(cache_key: str) -> str:
    """Generate a metadata key for the given cache key."""
    return f"{cache_key}:metadata"

def get_cache_stats_key() -> str:
    """Generate a key for cache statistics."""
    return f"{CACHE_PREFIX}:{CACHE_VERSION}:stats"

def cache_result(text: str, result: dict, expire_seconds: int = CACHE_EXPIRY):
    """Cache the NLP result with metadata."""
    cache_key = get_cache_key(text)
    metadata_key = get_metadata_key(cache_key)
    stats_key = get_cache_stats_key()
    
    start_time = time.time()
    
    try:
        with get_redis_client() as redis_client:
            # Start a pipeline for atomic operations
            pipe = redis_client.pipeline()
            
            # Store result
            pipe.setex(cache_key, expire_seconds, json.dumps(result))
            
            # Store metadata
            metadata = {
                "text": text,
                "created_at": datetime.utcnow().isoformat(),
                "intent": result.get("intent"),
                "confidence": result.get("confidence"),
                "entities": result.get("entities"),
                "expires_at": (datetime.utcnow().timestamp() + expire_seconds)
            }
            pipe.setex(metadata_key, expire_seconds, json.dumps(metadata))
            
            # Update cache statistics
            pipe.hincrby(stats_key, "total_cached", 1)
            
            # Execute all operations
            pipe.execute()
            
            REDIS_OPERATION_TIME.labels(operation='set').observe(time.time() - start_time)
            
    except Exception as e:
        NLP_ERRORS.labels(error_type='cache_set_error').inc()
        print(f"Failed to cache NLP result: {str(e)}")

def get_cached_result(text: str) -> tuple[dict, dict]:
    """Try to get cached NLP result and its metadata."""
    cache_key = get_cache_key(text)
    metadata_key = get_metadata_key(cache_key)
    start_time = time.time()
    
    try:
        with get_redis_client() as redis_client:
            # Get both result and metadata in a pipeline
            pipe = redis_client.pipeline()
            pipe.get(cache_key)
            pipe.get(metadata_key)
            result_json, metadata_json = pipe.execute()
            
            REDIS_OPERATION_TIME.labels(operation='get').observe(time.time() - start_time)
            
            if result_json:
                NLP_CACHE_HITS.inc()
                result = json.loads(result_json)
                metadata = json.loads(metadata_json) if metadata_json else {}
                return result, metadata
            
            NLP_CACHE_MISSES.inc()
            return None, None
            
    except Exception as e:
        NLP_ERRORS.labels(error_type='cache_get_error').inc()
        return None, None

# Load spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

# Load intent classification model
# Using a more sophisticated model for better intent understanding
model_name = "facebook/bart-large-mnli"  # Using BART for zero-shot classification
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define intent categories with descriptions for zero-shot classification
INTENT_TEMPLATES = {
    "greeting": "This is a greeting or salutation",
    "farewell": "This is a goodbye or farewell message",
    "weather": "This is a question about weather or temperature",
    "time": "This is a question about current time or schedule",
    "date": "This is a question about date or calendar",
    "general_question": "This is a general question seeking information",
    "command": "This is a command or instruction to perform an action",
    "unknown": "This is an unknown or unclear intent"
}

class NLPServicer(voice_assist_pb2_grpc.NLPServiceServicer):
    def ProcessText(self, request, context):
        try:
            NLP_REQUESTS.inc()
            text = request.text.lower()
            
            # Try to get from cache first
            cached_result, _ = get_cached_result(text)
            if cached_result:
                return voice_assist_pb2.ProcessTextResponse(
                    intent=cached_result["intent"],
                    entities=cached_result["entities"],
                    confidence=cached_result["confidence"]
                )

            # If not in cache, process the text
            with NLP_PROCESSING_TIME.time():
                # Extract entities using spaCy
                doc = nlp(text)
                entities = {}
                for ent in doc.ents:
                    entities[ent.label_] = ent.text
                    NLP_ENTITY_DISTRIBUTION.labels(entity_type=ent.label_).inc()

                # Zero-shot classification for intent
                intent_scores = {}
                for intent, template in INTENT_TEMPLATES.items():
                    inputs = tokenizer(
                        text,
                        template,
                        return_tensors="pt",
                        truncation=True,
                        padding=True
                    )
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        scores = torch.softmax(outputs.logits, dim=1)
                        intent_scores[intent] = float(scores[0][1])

                # Get the intent with highest confidence
                intent = max(intent_scores.items(), key=lambda x: x[1])
                confidence = intent[1]
                intent = intent[0]

                # If confidence is too low, mark as unknown
                if confidence < 0.5:
                    intent = "unknown"
                    confidence = 0.0

                NLP_INTENT_DISTRIBUTION.labels(intent=intent).inc()
                NLP_CONFIDENCE.observe(confidence)

                result = {
                    "intent": intent,
                    "entities": entities,
                    "confidence": confidence
                }

                # Cache the result
                cache_result(text, result)

                return voice_assist_pb2.ProcessTextResponse(
                    intent=intent,
                    entities=entities,
                    confidence=confidence
                )
        except Exception as e:
            NLP_ERRORS.labels(error_type='processing_error').inc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return voice_assist_pb2.ProcessTextResponse()

@app.post("/process")
async def process_text(text: str):
    try:
        NLP_REQUESTS.inc()
        
        # Try to get from cache first
        cached_result, _ = get_cached_result(text)
        if cached_result:
            return cached_result

        # If not in cache, process the text
        with NLP_PROCESSING_TIME.time():
            # Create a gRPC request
            request = voice_assist_pb2.ProcessTextRequest(text=text)
            
            # Process using the same logic as gRPC service
            doc = nlp(text.lower())
            entities = {}
            for ent in doc.ents:
                entities[ent.label_] = ent.text
                NLP_ENTITY_DISTRIBUTION.labels(entity_type=ent.label_).inc()

            # Zero-shot classification for intent
            intent_scores = {}
            for intent, template in INTENT_TEMPLATES.items():
                inputs = tokenizer(
                    text,
                    template,
                    return_tensors="pt",
                    truncation=True,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    scores = torch.softmax(outputs.logits, dim=1)
                    intent_scores[intent] = float(scores[0][1])

            # Get the intent with highest confidence
            intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = intent[1]
            intent = intent[0]

            # If confidence is too low, mark as unknown
            if confidence < 0.5:
                intent = "unknown"
                confidence = 0.0

            NLP_INTENT_DISTRIBUTION.labels(intent=intent).inc()
            NLP_CONFIDENCE.observe(confidence)

            result = {
                "intent": intent,
                "entities": entities,
                "confidence": confidence
            }

            # Cache the result
            cache_result(text, result)

            return result
    except Exception as e:
        NLP_ERRORS.labels(error_type='processing_error').inc()
        return {"error": str(e)}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/health")
async def health_check():
    """Check Redis connection and service health."""
    try:
        with get_redis_client() as redis_client:
            redis_client.ping()
            REDIS_CONNECTION_STATUS.set(1)
            return {
                "status": "healthy",
                "redis": "connected",
                "model": "loaded",
                "cache_stats": {
                    "total_cached": int(redis_client.hget(get_cache_stats_key(), "total_cached") or 0)
                }
            }
    except Exception as e:
        REDIS_CONNECTION_STATUS.set(0)
        return {
            "status": "unhealthy",
            "redis": "disconnected",
            "error": str(e)
        }

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    voice_assist_pb2_grpc.add_NLPServiceServicer_to_server(
        NLPServicer(), server
    )
    server.add_insecure_port("[::]:50052")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    # Start gRPC server in a separate thread
    import threading
    grpc_thread = threading.Thread(target=serve)
    grpc_thread.start()
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000) 