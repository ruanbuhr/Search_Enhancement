from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
from google_search import *
from query_summarizer import *
from deep_learning import *

load_dotenv()

app = Flask(__name__)

# Accept cross origin requests
CORS(app)

port = int(os.getenv('PORT', 8080))

query_summarizer = QuerySummarize()

@app.route('/search', methods=['POST'])
def search():
    data = request.json

    if data is None:
        return jsonify({ "error": "No JSON received."}), 400

    query = data.get('query')

    if query is None:
        return jsonify({"error": "Query not found."}), 400
    
    try:
        keywords = query_summarizer.summarize(query)
        keyword_string = ' '.join(keywords)

        results = google_search(keyword_string)

        enhanced_results = rank_snippets(query, results)

        return jsonify(enhanced_results), 200
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run()