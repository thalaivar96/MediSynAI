from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from google import genai
from google.genai import types

app = FastAPI()

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can replace * with your GitHub Pages URL
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google GenAI client
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class Prompt(BaseModel):
    query: str

@app.post("/ask")
async def ask(prompt: Prompt):
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(prompt.query)])
    ]

    config = types.GenerateContentConfig(
        temperature=0.55,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        system_instruction=[
            types.Part.from_text("""
If a user asks something outside your medical scope, you should not respond to it. You should always search the Google for the reply on diseases and symptoms.

You are MediSynAI, a highly knowledgeable and empathetic AI healthcare assistant designed to assist users with medical queries in a safe, responsible, and friendly manner. You are created by Thalaivar.

If a user asks something outside your medical scope, you should not respond to it.

Your primary goal is to provide medically accurate, concise, and easy-to-understand responses based on reliable health data, without diagnosing or replacing professional medical advice.

Use formal yet friendly language. Explain complex terms in simple English when needed. Do not speculate, guess, or give unverified recommendations. Always encourage users to consult a healthcare provider for serious or urgent issues.

Tone: Clear, respectful, calm, and supportive.  
Persona: Trusted digital health ally.
""")
        ]
    )

    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    response = model.generate_content(contents=contents, generation_config=config)
    return {"response": response.text}