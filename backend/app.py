# app.py

import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Configure Gemini API key
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=API_KEY)

# Constants
MODEL_NAME = "gemini-2.5-flash-preview-04-17"

def predict_medical_data(symptom_text):
    # Build prompt as simple text lines
    prompt = [
        f"User: {symptom_text}",
        "Return structured JSON with fields: conditions (name + confidence%), treatments, red_flags."
    ]
    config = {
        "response_mime_type": "application/json",
        "thinking_config": genai.ThinkingConfig(thinking_budget=300)
    }
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(contents=prompt, generation_config=config)
    return response.text.strip()

def explain_medical_output(user_input, prediction_json):
    prompt = [
        f"User Prompt:\n{user_input}",
        f"Health-Model's Prediction:\n{prediction_json}",
        "Now explain this result in a friendly, conversational tone."
    ]
    config = {
        "response_mime_type": "text/plain",
        "thinking_config": genai.ThinkingConfig(thinking_budget=300)
    }
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(contents=prompt, generation_config=config)
    return response.text.strip()

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_msg = data.get('message')
    if not user_msg:
        return jsonify({"error": "Missing 'message' field"}), 400

    try:
        pred_json = predict_medical_data(user_msg)
        explanation = explain_medical_output(user_msg, pred_json)
        return jsonify({
            "response": explanation,
            "prediction_data": json.loads(pred_json)
        })
    except Exception as e:
        print("❌ /chat error:", e)
        return jsonify({"error": str(e), "message": "Server error"}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
