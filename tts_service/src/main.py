import grpc
from concurrent import futures
from TTS.api import TTS
import tempfile
import os
from fastapi import FastAPI
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

# Initialize TTS model
# Using a fast and high-quality model
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

class TTSServicer(voice_assist_pb2_grpc.TTSServiceServicer):
    def SynthesizeSpeech(self, request, context):
        try:
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

            # Clean up the temporary file
            os.unlink(output_path)

            return voice_assist_pb2.SynthesizeResponse(
                audio_data=audio_data,
                audio_format="wav"
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return voice_assist_pb2.SynthesizeResponse()

@app.post("/synthesize")
async def synthesize_speech(text: str):
    try:
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

        # Clean up the temporary file
        os.unlink(output_path)

        return {
            "audio_data": audio_data,
            "audio_format": "wav"
        }
    except Exception as e:
        return {"error": str(e)}

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