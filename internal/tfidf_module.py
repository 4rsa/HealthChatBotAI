"""
tfidf_module.py

This file encapsulates all TF-IDF related logic, including:
  - Preprocessing (lowercasing, punctuation removal, lemmatization)
  - Building/fitting the TfidfVectorizer
  - Computing cosine similarities
"""

import string
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Auto-download necessary NLTK resources (safe to run multiple times)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

try:
    nltk.data.find("corpora/omw-1.4")
except LookupError:
    nltk.download("omw-1.4")

class TfidfManager:
    def __init__(self, questions, answers):
        """
        :param questions: list of question strings
        :param answers:   list of answer strings
        """
        self.lemmatizer = WordNetLemmatizer()
        self.questions = [self.preprocess_input(q) for q in questions]
        self.answers = answers

        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)

    def preprocess_input(self, text: str) -> str:
        """
        Preprocess the text:
        - lowercase
        - remove punctuation
        - tokenize
        - lemmatize
        - rejoin
        """
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized).strip()

    def get_most_similar_answer(self, user_input: str, threshold=0.1) -> str:
        cleaned_input = self.preprocess_input(user_input)
        user_tfidf = self.vectorizer.transform([cleaned_input])
        similarities = cosine_similarity(user_tfidf, self.tfidf_matrix).flatten()

        best_match_idx = similarities.argmax()
        best_match_score = similarities[best_match_idx]

        if best_match_score > threshold:
            return self.answers[best_match_idx]
        else:
            return None
