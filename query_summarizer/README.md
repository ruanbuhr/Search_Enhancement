# **NLP Query Summarizer**

## **Project Overview**

The **NLP Query Summarizer** project is designed to extract relevant keywords from a given text input and summarize them in the same order they appear within the text. It combines Named Entity Recognition (NER) and TF-IDF vectorization to produce high-quality, succinct keywords. This is useful for tasks like search query generation, where the goal is to extract the most important terms from a larger body of text.

The summarizer leverages:
- **SpaCy** for Named Entity Recognition (NER).
- **TF-IDF (Term Frequency-Inverse Document Frequency)** to score the relevance of terms from a pre-loaded corpus.
- A **custom algorithm** that removes redundant terms and substrings to provide a cleaner set of keywords.

The goal of this project is to summarize a paragraph or a sentence into key terms or phrases that can be used as search queries. The summarization process involves two major steps:

1. **Named Entity Recognition (NER)**: This step uses SpaCy to identify key entities such as people, locations, organizations, and dates from the text input. These entities are important search terms and are given higher relevance in the summarization process.
   
2. **TF-IDF Vectorization**: This technique calculates the importance of terms in the text relative to a larger corpus (in this case, the Microsoft MS Marco dataset). Terms that are frequent in the text but not common across the corpus are considered more relevant and are added to the keyword list.

After combining the NER entities and the top TF-IDF terms, a **filtering step** ensures that no duplicate or substring terms are included. The final keywords are sorted based on their appearance order in the original text, ensuring that the context is preserved.
