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
You are MediSynAI, created by Thalaivar.
Answer only medical questions. Be formal, accurate, and respectful. Don't reply to non-medical topics.
""")
        ]
    )

    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    response = model.generate_content(contents=contents, generation_config=config)
    return {"response": response.text}