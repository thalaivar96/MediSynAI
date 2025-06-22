from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from flask_cors import CORS

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Set your Gemini API key
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Initialize the model
model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        user_message = data.get("message", "")

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        response = model.generate_content(user_message)

        return jsonify({
            "response": response.text.strip()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Optional: health check route
@app.route("/health")
def health():
    return jsonify({"status": "AltheaAI is healthy"}), 200

# Run if executed directly
if __name__ == "__main__":
    app.run(debug=True)

