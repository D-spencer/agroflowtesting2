import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

# Import from your existing chatbot modules
from chatbot.database import get_supabase
from chatbot.embeddings import load_embedding_model
from chatbot.llm import get_llm
from chatbot.chat_engine import chat
from chatbot.logger import logger

app = Flask(__name__)
CORS(app)  # Allow frontend to call this API

# ── GLOBAL INITIALIZATION ──────────────────────────────────
# These only load once when the server starts
logger.info("Initializing AgroFlow AI API...")

try:
    supabase = get_supabase()
    logger.info("✅ Supabase connected.")

    embedding_model = load_embedding_model()
    logger.info("✅ Embedding model loaded.")

    llm = get_llm()
    logger.info("✅ LLM initialized.")

    logger.info("✅ AgroFlow AI API is ready.")
except Exception as e:
    logger.error(f"❌ Failed to initialize: {e}")
    supabase = None
    embedding_model = None
    llm = None

# ── CHAT ENDPOINT ──────────────────────────────────────────
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    try:
        # Get the user's message from the request
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        # Check if everything is initialized
        if supabase is None or embedding_model is None or llm is None:
            return jsonify({'error': 'AI service not properly initialized'}), 500

        logger.info(f"📨 User Question: {user_message}")

        # Call your existing chat engine
        answer = chat(
            question=user_message,
            supabase=supabase,
            embedding_model=embedding_model,
            llm=llm
        )

        logger.info("✅ Answer generated successfully.")

        return jsonify({
            'response': answer,
            'success': True
        })

    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

# ── HEALTH CHECK ───────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    status = 'healthy'
    details = {}
    
    # Check if all components are ready
    if supabase is None:
        status = 'degraded'
        details['supabase'] = 'not connected'
    else:
        details['supabase'] = 'connected'
    
    if embedding_model is None:
        status = 'degraded'
        details['embedding_model'] = 'not loaded'
    else:
        details['embedding_model'] = 'loaded'
    
    if llm is None:
        status = 'degraded'
        details['llm'] = 'not initialized'
    else:
        details['llm'] = 'initialized'
    
    return jsonify({
        'status': status,
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'details': details
    })

# ── ROOT ENDPOINT ──────────────────────────────────────────
@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'AgroFlow AI',
        'version': '1.0.0',
        'endpoints': {
            '/chat': 'POST - Send messages to AI',
            '/health': 'GET - Check service health'
        }
    })

# ── START SERVER ──────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🚀 Starting AgroFlow AI API on port {port}")
    app.run(host='0.0.0.0', port=port)
