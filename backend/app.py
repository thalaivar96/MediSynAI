import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore

# --- App Setup ---
app = Flask(__name__)
CORS(app)

# --- Firebase Setup ---
firebase_credentials_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_KEY")
if not firebase_credentials_json:
    raise ValueError("FIREBASE_SERVICE_ACCOUNT_KEY not set")

cred_dict = json.loads(firebase_credentials_json)
cred = credentials.Certificate(cred_dict)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# --- Gemini Setup ---
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY not set")
genai.configure(api_key=gemini_api_key)

# --- Medical Prediction ---
def predict_medical_data(symptom_text, history_contents):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    contents = list(history_contents)
    contents.append(genai.Content(role="user", parts=[genai.Part.from_text(symptom_text)]))

    config = genai.GenerateContentConfig(
        thinking_config=genai.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=genai.Schema(
            type=genai.Type.OBJECT,
            required=["conditions", "treatments", "red_flags"],
            properties={
                "conditions": genai.Schema(
                    type=genai.Type.ARRAY,
                    items=genai.Schema(
                        type=genai.Type.OBJECT,
                        required=["name", "confidence_in_percent"],
                        properties={
                            "name": genai.Schema(type=genai.Type.STRING),
                            "confidence_in_percent": genai.Schema(type=genai.Type.NUMBER),
                        },
                    ),
                ),
                "treatments": genai.Schema(type=genai.Type.ARRAY, items=genai.Schema(type=genai.Type.STRING)),
                "red_flags": genai.Schema(type=genai.Type.ARRAY, items=genai.Schema(type=genai.Type.STRING)),
            },
        ),
        system_instruction=[
            genai.Part.from_text("""
You are AltheaAI, a medically-trained assistant. Based on the user's symptom input and conversation history, return structured JSON with:
- conditions (name + confidence)
- treatments (basic suggestions)
- red_flags (warnings to seek medical help)
Only output JSON. Confidence should be a number (e.g., 75 for 75%).
""")
        ]
    )

    response_json = ""
    try:
        for chunk in model.generate_content_stream(contents=contents, generation_config=config):
            response_json += chunk.text
    except Exception as e:
        print("Gemini prediction error:", e)
        return json.dumps({"error": str(e), "message": "Prediction failed"})

    return response_json.strip()

# --- Friendly Explanation ---
def explain_medical_output(user_input, prediction_json_str, history_contents):
    model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
    contents = list(history_contents)
    prompt = f"""
User Prompt:
{user_input}

Health-Model's Prediction:
{prediction_json_str}
"""
    contents.append(genai.Content(role="user", parts=[genai.Part.from_text(prompt)]))

    config = genai.GenerateContentConfig(
        thinking_config=genai.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
        system_instruction=[
            genai.Part.from_text("""
You are AltheaAI, designed to explain medical diagnoses in a friendly, easy-to-understand way.
If symptoms are severe, recommend meeting a doctor. Use kind language and avoid JSON.
Respond in list form and relate new message with previous history if available.
""")
        ]
    )

    full_response = ""
    try:
        for chunk in model.generate_content_stream(contents=contents, generation_config=config):
            full_response += chunk.text
    except Exception as e:
        print("Gemini explanation error:", e)
        return f"Sorry, I encountered an error: {e}"

    return full_response

# --- Flask Route ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = data.get('userId')
    user_msg = data.get('message')

    if not user_id or not user_msg:
        return jsonify({"error": "Missing userId or message"}), 400

    # --- History fetch ---
    ref = db.collection('conversations').document(user_id)
    doc = ref.get()
    history = []
    content_list = []

    if doc.exists:
        history = doc.to_dict().get('history', [])
        for entry in history:
            parts = [genai.Part.from_text(part['text']) for part in entry.get('parts', []) if 'text' in part]
            if parts:
                content_list.append(genai.Content(role=entry['role'], parts=parts))

    # --- Predict ---
    prediction_str = predict_medical_data(user_msg, content_list)
    try:
        prediction_data = json.loads(prediction_str)
    except:
        prediction_data = {"conditions": [], "treatments": [], "red_flags": []}

    # --- Explain ---
    explanation = explain_medical_output(user_msg, json.dumps(prediction_data, indent=2), content_list)

    # --- Save history ---
    history.append({"role": "user", "parts": [{"text": user_msg}]})
    history.append({"role": "model", "parts": [{"text": explanation}]})
    ref.set({"history": history})

    return jsonify({"response": explanation, "prediction_data": prediction_data})

# --- Run Locally (for development) ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
