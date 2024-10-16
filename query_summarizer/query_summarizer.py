from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import re

class QuerySummarize:
    def __init__(self):
        # Load the SpaCy NER model
        self.nlp = spacy.load("en_core_web_sm")
        # Define common question words
        self.question_words = {"what", "why", "how", "when", "where", "which", "who", "whom", "whose"}

    def normalize(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        # Lowercase and strip any extra spaces
        return text

    def remove_substrings(self, terms):
        # Normalize all terms first
        terms = [self.normalize(term) for term in terms]

        # Sort terms by length (longest to shortest) to handle substrings
        terms.sort(key=len, reverse=True)  
        filtered_terms = []

        for term in terms:
            if not any(term in longer_term for longer_term in filtered_terms):
                filtered_terms.append(term)  # Add term if it's not a substring

        return filtered_terms

    def get_keywords_in_original_order(self, text, keywords):
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

    def extract_question_related_keywords(self, text):
        tokens = text.split()
        question_related_keywords = []

        for i, token in enumerate(tokens):
            normalized_token = self.normalize(token)
            if normalized_token in self.question_words:
                question_related_keywords.append(normalized_token)
                # Add the next token if it exists (to capture adjacent keywords)
                if i + 1 < len(tokens):
                    next_token = self.normalize(tokens[i + 1])
                    question_related_keywords.append(next_token)

        return question_related_keywords

    def summarize(self, text):
        # Named Entity Recognition using SpaCy
        doc = self.nlp(text)
        entities = [self.normalize(ent.text) for ent in doc.ents]  # Normalize entities

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=10)  # Limit to top 10 features
        tfidf_matrix = vectorizer.fit_transform([text]) 

        # Extract top terms with highest TF-IDF scores
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]

        # Extract top 10 terms, and normalize
        top_10_terms = [(self.normalize(feature_names[i]), tfidf_scores[i]) for i in tfidf_scores.argsort()[-10:]]

        # Remove substrings (if one word is a substring of another, remove the shorter one) 
        keywords = self.remove_substrings(entities + [term for term, score in top_10_terms])

        # Extract question-related keywords
        question_related_keywords = self.extract_question_related_keywords(text)

        # Combine all keywords and remove duplicates
        all_keywords = list(set(keywords + question_related_keywords))

        # Get the keywords in the original order they appeared in the text
        ordered_keywords = self.get_keywords_in_original_order(text, all_keywords)

        return ordered_keywords
