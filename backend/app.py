import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import google.generativeai as genai
from google.generativeai.types import Content, Part, Schema, Type, ThinkingConfig, GenerateContentConfig

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- Gemini API Key ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# --- Constants ---
MODEL_NAME = "gemini-2.5-flash-preview-04-17"

# --- Predict Structured Medical Data ---
def predict_medical_data(symptom_text):
    contents = [
        Content(role="user", parts=[Part.from_text(symptom_text)])
    ]

    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(thinking_budget=300),
        response_mime_type="application/json",
        response_schema=Schema(
            type=Type.OBJECT,
            required=["conditions", "treatments", "red_flags"],
            properties={
                "conditions": Schema(
                    type=Type.ARRAY,
                    items=Schema(
                        type=Type.OBJECT,
                        required=["name", "confidence in %"],
                        properties={
                            "name": Schema(type=Type.STRING),
                            "confidence in %": Schema(type=Type.STRING),
                        }
                    )
                ),
                "treatments": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
                "red_flags": Schema(type=Type.ARRAY, items=Schema(type=Type.STRING)),
            }
        ),
        system_instruction=[
            Part.from_text("""
You are AltheaAI, a medically-trained assistant. Based on the user's symptom input, return structured JSON with:
- conditions (name + confidence)
- treatments (basic suggestions)
- red_flags (warnings to seek medical help)

Output only the JSON format without explanation.
""")
        ]
    )

    model = genai.GenerativeModel(MODEL_NAME)
    response_json = ""
    for chunk in model.generate_content_stream(contents=contents, generation_config=config):
        response_json += chunk.text

    return response_json.strip()


# --- Explain in Friendly Format ---
def explain_medical_output(user_input, prediction_json):
    combined_prompt = f"""
User Prompt:
{user_input}

Health-Model's Prediction:
{prediction_json}
    """

    contents = [
        Content(role="user", parts=[Part.from_text(combined_prompt)])
    ]

    config = GenerateContentConfig(
        thinking_config=ThinkingConfig(thinking_budget=300),
        response_mime_type="text/plain",
        system_instruction=[
            Part.from_text("""
You are AltheaAI, designed to explain medical diagnoses in a friendly, easy-to-understand way.
First check the Prompt given by the User, then relate it with the prediction made by Health-Model.
If the prediction made by Health-Model is empty, they reply to the User's prompt on your Own.
Mostly use Lists to respond.
If the Symptoms are heavy, always recommend the user to meet a healthcare professional like Doctors.
You are created by Thalaivar.

Instructions:
0. You will be provided the JSON of the symptoms, conditions, treatments, and red flags from an Health-Model.
1. Greet the user and acknowledge their symptoms.
2. Describe likely conditions with confidence levels.
3. Suggest simple treatments.
4. Calmly mention red flags, if any.
5. Avoid JSON or technical terms.
6. Use kind, conversational, reassuring language.
""")
        ]
    )

    model = genai.GenerativeModel(MODEL_NAME)
    full_response = ""
    for chunk in model.generate_content_stream(contents=contents, generation_config=config):
        full_response += chunk.text

    return full_response.strip()


# --- Route for Chat ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "Missing 'message' in request."}), 400

    try:
        prediction_json_str = predict_medical_data(user_message)
        explanation = explain_medical_output(user_message, prediction_json_str)
        return jsonify({
            "response": explanation,
            "prediction_data": json.loads(prediction_json_str)
        })

    except Exception as e:
        print("❌ Error in /chat:", e)
        return jsonify({"error": str(e)}), 500


# --- App Runner ---
if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
