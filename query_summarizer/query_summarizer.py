import spacy
import re
from collections import Counter, defaultdict

class QuerySummarize:
    def __init__(self):
        # Load SpaCy NER model and stop words
        self.nlp = spacy.load("en_core_web_sm")
        self.stopwords = set(spacy.lang.en.stop_words.STOP_WORDS)
        self.question_words = {"what", "why", "how", "when", "where", "which", "who", "whom", "whose"}

    def normalize(self, text):
        """Normalize text by lowering case and removing punctuation."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def tokenize(self, text):
        """Tokenize text while filtering out stopwords."""
        return [word for word in self.normalize(text).split() if word and word not in self.stopwords]

    def remove_substrings(self, terms):
        """Remove terms that are substrings of longer terms."""
        terms = [self.normalize(term) for term in terms]
        terms.sort(key=len, reverse=True)
        filtered_terms = []
        for term in terms:
            if not any(term in longer_term for longer_term in filtered_terms):
                filtered_terms.append(term)
        return filtered_terms

    def extract_phrases(self, text):
        """Extract meaningful phrases from text using SpaCy's POS tags."""
        doc = self.nlp(text)
        phrases = []
        for chunk in doc.noun_chunks:
            phrase = self.normalize(chunk.text)
            if phrase not in self.stopwords:
                phrases.append(phrase)
        return phrases

    def calculate_word_scores(self, phrases):
        """Calculate word scores based on frequency and co-occurrence in phrases."""
        word_freq = Counter()
        word_degree = defaultdict(int)

        for phrase in phrases:
            words = phrase.split()
            for word in words:
                word_freq[word] += 1
                word_degree[word] += len(words) - 1

        word_scores = {word: degree / freq for word, (degree, freq) in 
                       zip(word_degree.keys(), zip(word_degree.values(), word_freq.values()))}
        return word_scores

    def extract_keywords(self, text):
        """Extract and rank keywords from text."""
        phrases = self.extract_phrases(text)
        word_scores = self.calculate_word_scores(phrases)
        phrase_scores = {phrase: sum(word_scores.get(word, 0) for word in phrase.split()) for phrase in phrases}
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [phrase for phrase, _ in sorted_phrases]
        return keywords

    def find_positions(self, text, keywords):
        """Find the starting index of each keyword/phrase in the original text."""
        keyword_positions = []
        for keyword in keywords:
            # Use regex to find the first occurrence of the keyword/phrase
            match = re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)
            if match:
                keyword_positions.append((keyword, match.start()))
        # Sort by the starting index to preserve original order
        keyword_positions.sort(key=lambda x: x[1])
        return [kw for kw, _ in keyword_positions]

    def summarize(self, text):
        """Summarize the text to extract main keywords."""
        doc = self.nlp(text)
        entities = [self.normalize(ent.text) for ent in doc.ents]

        # Extract keywords and remove redundant ones
        keywords = self.extract_keywords(text)
        keywords = self.remove_substrings(entities + keywords)

        # Remove redundant keywords (substrings)
        keywords = [kw for kw in keywords if kw not in self.stopwords]

        # Ensure keywords are ordered by appearance in the original text
        keywords = self.find_positions(text, keywords)

        return keywords