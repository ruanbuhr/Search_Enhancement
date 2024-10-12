import os
import requests

def google_search(query):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={SEARCH_ENGINE_ID}'

    response = requests.get(url)
    results = response.json()

    # Returns a dictionary of links and snippets
    res_dict = {item['link']: item['snippet'] for item in results.get('items', [])}

    return res_dict