# Snippet Relevance Ranking using Sentence Transformers

This project ranks text snippets based on their relevance to a user query using Sentence Transformers. Given a long-form query and a list of snippets, the program calculates cosine similarity scores to determine how closely each snippet aligns with the query. It then ranks the snippets in descending order of relevance.

# Model Used

The model all-MiniLM-L6-v2 is a pre-trained sentence transformer from the Sentence Transformers library, based on Microsoft’s MiniLM architecture. 

## Model Architecture:

MiniLM (Minimal Language Model) is a lightweight transformer model that uses a small architecture, allowing for faster inference and reduced memory usage.It has 6 layers which makes the model fast while retaining good accuracy.

## Embedding

The model produces 384-dimensional embeddings. THis makes the model fast and require less memory that that of a higher dimension model.
