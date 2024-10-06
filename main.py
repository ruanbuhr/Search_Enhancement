from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from google_search import *

load_dotenv()

app = Flask(__name__)

port = int(os.getenv('API_PORT', 5000))

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    results = google_search(query)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)