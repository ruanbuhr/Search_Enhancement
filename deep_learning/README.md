# Snippet Relevance Ranking using Sentence Transformers

This project ranks text snippets based on their relevance to a user query using **Sentence Transformers**. Given a long-form query and a list of snippets, the program calculates similarity scores to determine how closely each snippet aligns with the query. It then ranks the snippets in descending order of relevance.

## Model Used

The model `all-MiniLM-L6-v2` is a pre-trained sentence transformer from the Sentence Transformers library, based on Microsoft’s **MiniLM** architecture. It provides an efficient tradeoff between speed and accuracy for sentence embedding generation.

### Model Architecture: MiniLM

MiniLM (**Minimal Language Model**) is a lightweight transformer model optimized for fast inference and reduced memory usage. The architecture consists of 6 layers, allowing the model to process text quickly while maintaining a good level of accuracy.

## Embedding

The model produces **384-dimensional embeddings** for input sentences or phrases. These embeddings capture the semantic meaning of the sentences, and the relatively small size of the embedding (384 dimensions) helps make this model fast and memory-efficient compared to larger models like BERT or RoBERTa.

## Siamese Neural Network

In addition to the sentence transformer, this project uses a **Siamese Neural Network** to compute similarity scores between pairs of sentences. A Siamese Network is a special type of neural network architecture that takes two input embeddings and passes them through a shared set of layers to produce a similarity score between them.

### Siamese Network Architecture

The Siamese network used in this project is composed of fully connected layers applied to the concatenation of the two input sentence embeddings (one for the query and one for each snippet). The structure of the network is as follows:

- **Input Layer**: Two sentence embeddings are concatenated into a single vector of size `embedding_dim * 2` (i.e., 384 * 2 = 768).
- **Fully Connected Layers**:
  - **First Layer**: A linear transformation from 768 to 128 dimensions, followed by a **ReLU activation** and a **Dropout layer** (to prevent overfitting).
  - **Second Layer**: A linear transformation from 128 to 64 dimensions, followed by another **ReLU activation** and **Dropout layer**.
  - **Output Layer**: A linear transformation from 64 to 1 dimension, followed by a **Sigmoid activation function**, which outputs a value between 0 and 1, representing the similarity score.

The model architecture allows it to compare pairs of embeddings and output a score indicating the relevance of the snippet to the query. The higher the score, the more similar the two sentences are.

### How the Siamese Network Works

1. **Input**: For each query-snippet pair, the model first obtains the embeddings of both the query and the snippet using the pre-trained Sentence Transformer.
2. **Concatenation**: The query and snippet embeddings are concatenated into a single vector.
3. **Forward Pass**: The concatenated embeddings are passed through the fully connected layers of the Siamese network.
4. **Output**: The final output is a similarity score between 0 and 1. This score represents how closely the snippet matches the query in terms of semantic meaning.
5. **Ranking**: All snippets are scored, and they are ranked in descending order based on their similarity scores.

