from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tiktoken

from llama_index import set_global_tokenizer, set_global_service_context
from custom_logging import log_interaction
from llm_setup import get_query_engine, get_service_context
from llama_index.callbacks import TokenCountingHandler

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from auth import firebase_auth_required
from dotenv import load_dotenv

app = Flask(__name__)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,  # Use the remote address of the client for rate limiting
    default_limits=["100 per day", "50 per hour"]  # Default rate limits
)

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo").encode
token_counter = TokenCountingHandler(tokenizer=tokenizer)

service_context = get_service_context(token_counter)
query_engine = get_query_engine(service_context)

set_global_service_context(service_context)
set_global_tokenizer(tokenizer)

@app.route('/query/m-and-a/', methods=['POST'])
@limiter.limit("5 per minute")  # Custom rate limit for this endpoint
@firebase_auth_required
def query_m_and_a():
    try:
        # Extract user query from request JSON body
        user_input = request.json.get('query')
        if not user_input:
            return jsonify({"error": "No query provided"}), 400

        # Get response from the chat engine
        response = query_engine.query(user_input)
        response_text = response.response
        log_interaction(user_input, response_text, token_counter)

        final_obj = {"response": response_text,
                    'total_embedding_token_count': token_counter.total_embedding_token_count,
                    'prompt_llm_token_count': token_counter.prompt_llm_token_count,
                    'completion_llm_token_count': token_counter.completion_llm_token_count,
                    'total_llm_token_count': token_counter.total_llm_token_count,
                    'prompt': token_counter.llm_token_counts[0].prompt,
                    'completion': token_counter.llm_token_counts[0].completion }
        
        token_counter.reset_counts()
        return jsonify(final_obj), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.errorhandler(429)
def ratelimit_handler(e):
    return "You have exceeded your rate limit.", 429

if __name__ == '__main__':
    app.run(debug=True)
