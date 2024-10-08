from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from flask_cors import CORS
from google_search import *

load_dotenv()

app = Flask(__name__)

# Accept cross origin requests
CORS(app)

port = int(os.getenv('PORT', 8080))

@app.route('/search', methods=['POST'])
def search():
    data = request.json

    if data is None:
        return jsonify({ "error": "No JSON received."}), 400

    query = data.get('query')

    if query is None:
        return jsonify({"error": "Query not found."}), 400

    results = google_search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run()