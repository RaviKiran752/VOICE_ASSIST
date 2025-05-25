import grpc
from concurrent import futures
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
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
# Using DistilBERT for faster inference
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define intent categories
INTENTS = {
    "greeting": ["hello", "hi", "hey", "greetings"],
    "farewell": ["bye", "goodbye", "see you", "farewell"],
    "weather": ["weather", "temperature", "forecast", "rain", "sunny"],
    "time": ["time", "clock", "hour", "schedule"],
    "date": ["date", "day", "calendar", "month"],
    "general_question": ["what", "how", "why", "when", "where", "who"],
    "command": ["play", "stop", "pause", "resume", "next", "previous"],
    "unknown": []
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

            # Classify intent
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                scores = torch.softmax(outputs.logits, dim=1)
                confidence = float(scores[0][1])  # Using positive class confidence

            # Determine intent based on keywords and model confidence
            intent = "unknown"
            max_matches = 0
            
            for intent_name, keywords in INTENTS.items():
                matches = sum(1 for keyword in keywords if keyword in text)
                if matches > max_matches:
                    max_matches = matches
                    intent = intent_name

            # If no keyword matches but model confidence is high, use model prediction
            if max_matches == 0 and confidence > 0.7:
                intent = "general_question"

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

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            confidence = float(scores[0][1])

        intent = "unknown"
        max_matches = 0
        
        for intent_name, keywords in INTENTS.items():
            matches = sum(1 for keyword in keywords if keyword in text.lower())
            if matches > max_matches:
                max_matches = matches
                intent = intent_name

        if max_matches == 0 and confidence > 0.7:
            intent = "general_question"

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