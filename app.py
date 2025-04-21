import os
import logging
from flask import Flask, render_template, request, jsonify
from chatbot import InfoBot

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")


infobot = InfoBot()

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Process chat requests from the user."""
    user_message = request.json.get('message', '')
    
    if not user_message:
        return jsonify({'response': 'Please provide a message.'})
    
    try:
        
        response = infobot.get_response(user_message)
        return jsonify({'response': response})
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'response': f"I'm sorry, I encountered an error: {str(e)}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
