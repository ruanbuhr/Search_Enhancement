import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

# Load STS Benchmark dataset from Hugging Face's 'sentence-transformers' repository
ds = load_dataset("sentence-transformers/stsb")

# Convert training dataset to pandas DataFrame
train_dataset = ds['train'].to_pandas()

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pretrained Sentence Transformer model
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2').to(device)

# Get the embedding dimension size
embedding_dim = sentence_transformer.get_sentence_embedding_dimension()


# Dataset class for sentence pairs
class SentencePairsDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    # Return the sentences and score
    def __getitem__(self, idx):
        sentence1 = self.dataframe.iloc[idx]['sentence1']
        sentence2 = self.dataframe.iloc[idx]['sentence2']
        label = self.dataframe.iloc[idx]['score']
        return sentence1, sentence2, label


# Create an instance of SentencePairsDataset and DataLoader for batching
train_data = SentencePairsDataset(train_dataset)
batch_size = 256  # Define the batch size for DataLoader
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Define the Siamese Network architecture for comparing sentence embeddings
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        # A fully connected neural network with dropout layers for regularization
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output is a score between 0 and 1, representing similarity
        )

    # Forward pass combining embeddings of sentence pairs
    def forward(self, query_embedding, snippet_embedding):
        combined_embedding = torch.cat((query_embedding, snippet_embedding), dim=-1)
        return self.fc(combined_embedding)

# Instantiate the Siamese model and move it to the device
siamese_model = SiameseNetwork(embedding_dim).to(device)

# Define binary cross-entropy loss and Adam optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(siamese_model.parameters(), lr=1e-3)

# Set the number of epochs
epochs = 20

# Training loop
for epoch in range(epochs):
    total_loss = 0
    siamese_model.train()  # Set the model to training mode

    for batch in train_loader:
        sentence1_list, sentence2_list, labels = batch

        # Generate sentence embeddings in batches to utilize GPU effectively
        query_embeddings = sentence_transformer.encode(sentence1_list, convert_to_tensor=True, device=device)
        snippet_embeddings = sentence_transformer.encode(sentence2_list, convert_to_tensor=True, device=device)

        # Forward pass through the Siamese network with the embeddings
        output = siamese_model(query_embeddings, snippet_embeddings)

        # Compute loss
        labels = labels.to(device).float()
        loss = criterion(output.squeeze(), labels)

        # Backpropagation and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss for monitoring
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


# Function to rank snippets based on similarity with the query
def rank_snippets(query, snippets):

    # Encode the query sentence into an embedding using the sentence transformer
    query_embedding = sentence_transformer.encode([query], convert_to_tensor=True, device=device)

    # Encode all snippets into embeddings
    snippet_embeddings = sentence_transformer.encode(snippets, convert_to_tensor=True, device=device)

    snippet_scores = []
    # Calculate similarity score for each snippet with the query embedding
    for i, snippet_embedding in enumerate(snippet_embeddings):
        # Pass embeddings through the Siamese network and compute the score
        score = siamese_model(query_embedding, snippet_embedding.unsqueeze(0)).item()
        snippet_scores.append((snippets[i], score))

    # Sort snippets by similarity score in descending order
    ranked_snippets = sorted(snippet_scores, key=lambda x: x[1], reverse=True)
    return ranked_snippets


# Example query and snippets to rank
query = "How is AI impacting industries?"
snippets = [
    "AI is transforming industries across various sectors.",
    "Machine learning is a subset of artificial intelligence.",
    "Climate change affects the environment significantly.",
    "Deep learning is a branch of machine learning.",
    "Reinforcement learning is about learning from interaction."
]

# Rank the snippets and print them with their similarity scores
ranked_results = rank_snippets(query, snippets)
for snippet, score in ranked_results:
    print(f"Snippet: {snippet}, Score: {score:.4f}")
