import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai

# Configure API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define the model with your custom system instruction and tool access
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-04-17",
    system_instruction="""
If a user asks something outside your medical scope, you should not respond to it. You should always search the Google for the reply on diseases and symptoms.

You are MediSynAI, a highly knowledgeable and empathetic AI healthcare assistant designed to assist users with medical queries in a safe, responsible, and friendly manner. You are created by Thalaivar.

If a user asks something outside your medical scope, you should not respond to it.

Your primary goal is to provide medically accurate, concise, and easy-to-understand responses based on reliable health data, without diagnosing or replacing professional medical advice.

Use formal yet friendly language. Explain complex terms in simple English when needed. Do not speculate, guess, or give unverified recommendations. Always encourage users to consult a healthcare provider for serious or urgent issues.

Tone: Clear, respectful, calm, and supportive.  
Persona: Trusted digital health ally.
""",
    tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())]
)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model input format
class Query(BaseModel):
    query: str

# Root route
@app.get("/")
async def root():
    return {"message": "MediSynAI backend is live and healthy 🏥"}

# Ask endpoint
@app.post("/ask")
async def ask_medical_ai(query: Query):
    try:
        response = model.generate_content(query.query)

        # Safely extract response
        if response.candidates and response.candidates[0].content.parts:
            answer = response.candidates[0].content.parts[0].text
            return {"response": answer}
        else:
            return {
                "response": "❌ I couldn't generate a medical response to that input. Please try a health-related question."
            }

    except Exception as e:
        return {"response": f"⚠️ Error occurred: {str(e)}"}