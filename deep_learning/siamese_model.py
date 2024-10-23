import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# Load a pre-trained Sentence Transformer model
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

# Get the embedding dimension
embedding_dim = sentence_transformer.get_sentence_embedding_dimension()

# Define the Siamese Network architecture used for comparing sentence embeddings
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
          # Fully connected layers for processing the combined embeddings
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),  # Input size is twice the embedding dimension
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout for regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(64, 1),
            nn.Sigmoid() # Output is a similarity score between 0 and 1
        )
 # Forward method to process input embeddings
    def forward(self, query_embedding, snippet_embedding):
         # Concatenate the query and snippet embeddings
        combined_embedding = torch.cat((query_embedding, snippet_embedding), dim=-1)
        # Pass the concatenated embeddings through the fully connected layers
        return self.fc(combined_embedding)

# Initiate the Siamese model
loaded_model = SiameseNetwork(embedding_dim)

# Load the saved trained model checkpoint containing the model's weights
checkpoint = torch.load("deep_learning/siamese_model.pth", map_location=torch.device('cpu'))

# Load the model's state (weights and biases) from the checkpoint
loaded_model.load_state_dict(checkpoint['model_state_dict'])

# Set the model to evaluation mode
loaded_model.eval()

# Function to rank snippets based on their similarity to a query
def rank_snippets(query, google_results):
     # Encode the query sentence into an embedding using the sentence transformer
    query_embedding = sentence_transformer.encode(query, convert_to_tensor=True)
    
    link_scores = []
     # Loop through each link and snippet in the Google search results
    for link, snippet in google_results.items():
        # Encode the snippet into an embedding
        snippet_embedding = sentence_transformer.encode(snippet, convert_to_tensor=True)
         # Pass the query and snippet embeddings through the Siamese network to get the similarity score
        score = loaded_model(query_embedding, snippet_embedding).item()
         # Append the link and score to the link_scores list
        link_scores.append((link, score))

    ranked_link = sorted(link_scores, key=lambda x: x[1], reverse=True)

     # Sort the links based on their similarity scores in descending order
    ranked_links_only = [link for link, score in ranked_link]
    return ranked_links_only

loaded_model.eval()
