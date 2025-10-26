from flask import Flask, request, jsonify
import sys
import os

# Add mobilebot path so Flask can find modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mobilebot')))

# Import the get_response function from your existing autogen_module
from agents.autogen_module import get_response

app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    """Endpoint for Streamlit to send user queries."""
    data = request.get_json()
    user_query = data.get("query", "")
    
    if not user_query.strip():
        return jsonify({"response": "Please provide a question."}), 400

    try:
        # Call your existing backend logic
        answer = get_response(user_query)
        return jsonify({"response": answer})
    except Exception as e:
        print(f"❌ Flask error: {e}")
        return jsonify({"response": "⚠️ Error processing your request."}), 500


if __name__ == "__main__":
    # Run Flask app
    app.run(debug=True)
