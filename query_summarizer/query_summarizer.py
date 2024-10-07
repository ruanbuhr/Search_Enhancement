from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re
import pandas as pd
from datasets import load_dataset

class QuerySummarize:
    def __init__(self):
        # Load the SpaCy NER model
        self.nlp = spacy.load("en_core_web_sm")

        # Load the dataset and create a corpus (replace with a larger corpus as needed)
        ds = load_dataset("microsoft/ms_marco", "v1.1")
        self.corpus = list(pd.DataFrame(ds['train'])['query'])

    def normalize(self, text):
        text = text.lower().strip()

        char_re = re.compile(r'[^\w\s]')
        text = char_re.sub('', text)
        # Lowercase and strip any extra spaces
        return text

    def remove_substrings(self, terms):
        # Normalize all terms first
        terms = [self.normalize(term) for term in terms]

        # Sort terms by length (longest to shortest) to handle substrings
        terms.sort(key=len, reverse=True)
        filtered_terms = []

        for i, term in enumerate(terms):
            # Check if the term is a substring of any longer term in filtered_terms
            if not any(term in longer_term for longer_term in filtered_terms):
                filtered_terms.append(term)  # Add term if it's not a substring
        
        return filtered_terms

    def get_keywords_in_original_order(self, text, keywords):
        """
        Sorts the keywords to match the order they appear as substrings in the original text.
        """
        normalized_text = self.normalize(text)  # Normalize the text for substring matching
        keyword_positions = []

        for keyword in keywords:
            # Find the position of the keyword in the normalized text
            position = normalized_text.find(keyword)
            if position != -1:
                keyword_positions.append((keyword, position))
        
        # Sort keywords by their position in the text
        keyword_positions.sort(key=lambda x: x[1])
        
        # Return the keywords in order of their appearance
        ordered_keywords = [keyword for keyword, pos in keyword_positions]
        
        return ordered_keywords

    def summerize(self, text):
        # Named Entity Recognition using SpaCy
        doc = self.nlp(text)
        entities = [self.normalize(ent.text) for ent in doc.ents]  # Normalize entities

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text] + self.corpus)  # Fit TF-IDF using the corpus

        # Extract top terms with highest TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix[0].toarray()[0]

        # Extract top 10 terms, and normalize
        top_10_terms = [(self.normalize(feature_names[i]), tfidf_scores[i]) for i in tfidf_scores.argsort()[-8:]]  # Top 10

        # Remove substrings (if one word is a substring of another, remove the shorter one)
        keywords = self.remove_substrings(entities + [term for term, score in top_10_terms])

        # Get the keywords in the original order they appeared in the text
        ordered_keywords = self.get_keywords_in_original_order(text, keywords)

        return ordered_keywords