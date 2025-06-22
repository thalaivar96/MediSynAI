# app.py 

import os
from flask import Flask, request, jsonify # Import Flask components
import google.generativeai as genai # Keep this import style, it works
from google.generativeai import types
import json
import firebase_admin
from firebase_admin import credentials, firestore
from flask_cors import CORS # Import Flask-CORS

app = Flask(__name__)
CORS(app) # Enable CORS for your Flask app to allow frontend requests

# --- Firebase Initialization ---
firebase_credentials_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
if not firebase_credentials_json:
    raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not set.")

try:
    # Parse the JSON string into a dictionary
    cred_dict = json.loads(firebase_credentials_json)
    cred = credentials.Certificate(cred_dict)
    
    # Check if a Firebase app is already initialized to prevent re-initialization errors
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase initialized successfully!")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    # You might want to raise an exception or handle this more gracefully
    # if Firebase is critical for your app.

# --- Gemini Client Configuration (UPDATED) ---
gemini_api_key = os.getenv("GOOGLE_API_KEY") # Use GOOGLE_API_KEY for Render
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# *** KEY CHANGE: Configure genai directly, no 'Client' object ***
genai.configure(api_key=gemini_api_key)


# --- Helper functions for Gemini interactions (UPDATED) ---

def predict_medical_data(symptom_text, history_contents):
    model_name = "gemini-2.5-flash-preview-04-17" # Renamed 'model' to 'model_name' for clarity
    
    # *** KEY CHANGE: Get the GenerativeModel instance directly ***
    model_obj = genai.GenerativeModel(model_name)

    contents = list(history_contents) # Start with provided history
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=symptom_text)]))

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type=genai.types.Type.OBJECT,
            required=["conditions", "treatments", "red_flags"],
            properties={
                "conditions": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(
                        type=genai.types.Type.OBJECT,
                        required=["name", "confidence_in_percent"],
                        properties={
                            "name": genai.types.Schema(type=genai.types.Type.STRING),
                            "confidence_in_percent": genai.types.Schema(type=genai.types.Type.NUMBER),
                        },
                    ),
                ),
                "treatments": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(type=genai.types.Type.STRING),
                ),
                "red_flags": genai.types.Schema(
                    type=genai.types.Type.ARRAY,
                    items=genai.types.Schema(type=genai.types.Type.STRING),
                ),
            },
        ),
        system_instruction=[
            types.Part.from_text(text="""
You are AltheaAI, a medically-trained assistant. Based on the user's symptom input and the conversation history, return structured JSON with:
- conditions (name + confidence)
- treatments (basic suggestions)
- red_flags (warnings to seek medical help)

Output only the JSON format without explanation. Confidence should be a number (e.g., 75 for 75%).
""")
        ]
    )

    response_json = ""
    try:
        # *** KEY CHANGE: Call generate_content_stream on the model_obj ***
        # *** Also: 'config' parameter is now 'generation_config' ***
        for chunk in model_obj.generate_content_stream(
            contents=contents,
            generation_config=generate_content_config,
        ):
            response_json += chunk.text
    except Exception as e:
        print(f"Error during Gemini prediction: {e}")
        return json.dumps({"error": str(e), "message": "Failed to get medical prediction."})

    return response_json.strip()

def explain_medical_output(user_input, prediction_json_str, history_contents):
    model_name = "gemini-2.5-flash-preview-04-17" # Renamed 'model' to 'model_name' for clarity
    
    # *** KEY CHANGE: Get the GenerativeModel instance directly ***
    model_obj = genai.GenerativeModel(model_name)

    contents = list(history_contents)
    
    combined_prompt = f"""
User Prompt:
{user_input}

Health-Model's Prediction:
{prediction_json_str}
    """
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=combined_prompt)]))

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""
You are AltheaAI, designed to explain medical diagnoses in a friendly, easy-to-understand way.
First check the Prompt given by the User, then relate it with the prediction made by Health-Model.
If the prediction made by Health-Model is empty or irrelevant to the user's current query, reply to the User's prompt on your Own, offering general helpful information.
Mostly use Lists to respond.
If the Symptoms are severe, always recommend the user to meet a healthcare professional like Doctors.
You are created by Thalaivar.

Instructions:
0. You will be provided the JSON of the symptoms, conditions, treatments, and red flags from an Health-Model.
1. Greet the user and acknowledge their symptoms.
2. Describe likely conditions with confidence levels.
3. Suggest simple treatments.
4. Calmly mention red flags, if any.
5. Avoid JSON or technical terms.
6. Use kind, conversational, reassuring language.
7. Maintain continuity with the previous conversation (if any) based on the provided chat history.
""")
        ]
    )

    full_explanation = ""
    try:
        # *** KEY CHANGE: Call generate_content_stream on the model_obj ***
        # *** Also: 'config' parameter is now 'generation_config' ***
        for chunk in model_obj.generate_content_stream(
            contents=contents,
            generation_config=generate_content_config,
        ):
            full_explanation += chunk.text
    except Exception as e:
        print(f"Error during Gemini explanation: {e}")
        return f"I'm sorry, I encountered an error while explaining. Please try again. ({e})"
    return full_explanation


# --- Flask Routes ---

@app.route('/chat', methods=['POST'])
def chat():
    # Expect JSON payload: {"userId": "...", "message": "..."}
    data = request.get_json()
    user_id = data.get('userId')
    user_message = data.get('message')

    if not user_id or not user_message:
        return jsonify({"error": "Missing userId or message"}), 400

    # 1. Retrieve conversation history from Firebase
    # Conversation history will be stored as a list of dicts:
    # [{"role": "user", "parts": [{"text": "..."}]}, {"role": "model", "parts": [{"text": "..."}]}]
    conversation_ref = db.collection('conversations').document(user_id)
    doc = conversation_ref.get()
    
    # Initialize chat_history for Gemini
    gemini_chat_history_contents = [] 
    firebase_history_data = [] # To store/update in Firebase

    if doc.exists:
        firebase_history_data = doc.to_dict().get('history', [])
        # Convert Firebase history (dicts) back to Google AI types.Content objects
        for item in firebase_history_data:
            if 'role' in item and 'parts' in item:
                # Ensure 'parts' is a list of dicts, convert text parts
                parts_list = []
                for part in item['parts']:
                    if 'text' in part:
                        parts_list.append(types.Part.from_text(text=part['text']))
                
                if parts_list: # Only add if there are valid parts
                    gemini_chat_history_contents.append(types.Content(role=item['role'], parts=parts_list))
    
    # IMPORTANT: Apply context window management here if history gets too long
    # For simplicity, not doing it in this example, but crucial for production
    # e.g., if len(gemini_chat_history_contents) > MAX_TURNS: gemini_chat_history_contents = gemini_chat_history_contents[-MAX_TURNS:]

    # 2. Predict Medical Data (Gemini #1)
    prediction_json_str = predict_medical_data(user_message, gemini_chat_history_contents)
    
    try:
        parsed_prediction = json.loads(prediction_json_str)
    except json.JSONDecodeError:
        print(f"Warning: Could not parse prediction JSON: {prediction_json_str}")
        parsed_prediction = {"conditions": [], "treatments": [], "red_flags": []}

    # 3. Explain Medical Output (Gemini #2)
    # The explanation model still needs the full original user message for its prompt.
    althea_response_text = explain_medical_output(user_message, json.dumps(parsed_prediction, indent=2), gemini_chat_history_contents)

    # 4. Update conversation history in Firebase
    firebase_history_data.append({"role": "user", "parts": [{"text": user_message}]})
    firebase_history_data.append({"role": "model", "parts": [{"text": althea_response_text}]})
    
    # Save the updated history back to Firebase
    conversation_ref.set({'history': firebase_history_data})

    return jsonify({
        "response": althea_response_text,
        "prediction_data": parsed_prediction # Optionally send this for debugging/frontend display
    })

# Render will look for the PORT environment variable
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True) # debug=True only for local testing
