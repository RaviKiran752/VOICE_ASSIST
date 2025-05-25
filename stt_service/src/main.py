import grpc
from concurrent import futures
import whisper
import tempfile
import os
from fastapi import FastAPI, UploadFile, File
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

# Load Whisper model
model = whisper.load_model("base")

class STTServicer(voice_assist_pb2_grpc.STTServiceServicer):
    def TranscribeAudio(self, request, context):
        try:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=f".{request.audio_format}", delete=False) as temp_file:
                temp_file.write(request.audio_data)
                temp_file_path = temp_file.name

            # Transcribe the audio
            result = model.transcribe(temp_file_path)
            
            # Clean up the temporary file
            os.unlink(temp_file_path)

            return voice_assist_pb2.TranscribeResponse(
                text=result["text"],
                confidence=float(result.get("confidence", 0.0))
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return voice_assist_pb2.TranscribeResponse()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        audio_data = await file.read()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{file.filename.split('.')[-1]}", delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Transcribe the audio
        result = model.transcribe(temp_file_path)
        
        # Clean up the temporary file
        os.unlink(temp_file_path)

        return {
            "text": result["text"],
            "confidence": float(result.get("confidence", 0.0))
        }
    except Exception as e:
        return {"error": str(e)}

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