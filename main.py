from flask import Flask
import os

app = Flask(__name__)

port = int(os.getenv('API_PORT', 5000))

app.route('/search', methods=['POST'])
def search():
    pass

if __name__ == '__main__':
    app.run(debug=True)