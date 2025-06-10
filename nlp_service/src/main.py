import grpc
from concurrent import futures
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
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
            text = request.text.lower()
            
            # Extract entities using spaCy
            doc = nlp(text)
            entities = {}
            for ent in doc.ents:
                entities[ent.label_] = ent.text

            # Zero-shot classification for intent
            intent_scores = {}
            for intent, template in INTENT_TEMPLATES.items():
                # Prepare the input for zero-shot classification
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
                    intent_scores[intent] = float(scores[0][1])  # Using entailment score

            # Get the intent with highest confidence
            intent = max(intent_scores.items(), key=lambda x: x[1])
            confidence = intent[1]
            intent = intent[0]

            # If confidence is too low, mark as unknown
            if confidence < 0.5:
                intent = "unknown"
                confidence = 0.0

            return voice_assist_pb2.ProcessTextResponse(
                intent=intent,
                entities=entities,
                confidence=confidence
            )
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return voice_assist_pb2.ProcessTextResponse()

@app.post("/process")
async def process_text(text: str):
    try:
        # Create a gRPC request
        request = voice_assist_pb2.ProcessTextRequest(text=text)
        
        # Process using the same logic as gRPC service
        doc = nlp(text.lower())
        entities = {}
        for ent in doc.ents:
            entities[ent.label_] = ent.text

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

        return {
            "intent": intent,
            "entities": entities,
            "confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

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