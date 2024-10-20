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
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def tokenize(self, text):
        return [word for word in self.normalize(text).split() if word and word not in self.stopwords]

    def remove_substrings(self, terms):
        terms = [self.normalize(term) for term in terms]
        terms.sort(key=len, reverse=True)
        filtered_terms = []
        for term in terms:
            if not any(term in longer_term for longer_term in filtered_terms):
                filtered_terms.append(term)
        return filtered_terms

    def extract_phrases(self, text):
        tokens = self.tokenize(text)
        phrases = []
        phrase = []
        for token in tokens:
            if token in self.stopwords:
                if phrase:
                    phrases.append(" ".join(phrase))
                    phrase = []
            else:
                phrase.append(token)
        if phrase:
            phrases.append(" ".join(phrase))
        return phrases

    def calculate_word_scores(self, phrases):
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
        phrases = self.extract_phrases(text)
        word_scores = self.calculate_word_scores(phrases)
        phrase_scores = {phrase: sum(word_scores.get(word, 0) for word in phrase.split()) for phrase in phrases}
        sorted_phrases = sorted(phrase_scores.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, _ in sorted_phrases]

    def summarize(self, text):
        doc = self.nlp(text)
        entities = [self.normalize(ent.text) for ent in doc.ents]

        keywords = self.extract_keywords(text)

        keywords = self.remove_substrings(entities + keywords)

        keywords = [kw for kw in keywords if kw not in self.stopwords]

        return keywords