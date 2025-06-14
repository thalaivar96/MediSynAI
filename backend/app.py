import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Load the Google API key from environment variable
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Gemini model
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-04-17",
    system_instruction="""
        You are MediSynAI, a highly knowledgeable and empathetic AI healthcare assistant created by Thalaivar.
        You must respond strictly only to medical-related queries.
        Politely refuse to anything outside the medical domain.
        Your tone is respectful, formal, friendly, and clear.
        Do not diagnose. Always encourage users to consult a medical professional.
    """
)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend (e.g., GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["thalaivar96.github.io/MediSynAI"],  # For production, set your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming request
class Query(BaseModel):
    query: str

@app.get("/")
async def root():
    return {"message": "MediSynAI backend is live"}

@app.post("/ask")
async def ask_medical_ai(query: Query):
    try:
        convo = model.start_chat()

        response = convo.send_message(query.query)
        return {"response": response.text}

    except Exception as e:
        return {"response": f"⚠️ An error occurred: {str(e)}"}