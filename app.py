from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "Chatbot is running!"}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    # Simple response (echo bot for now)
    if user_message:
        return jsonify({"reply": f"You said: {user_message}"})
    else:
        return jsonify({"reply": "Please send me a message."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
