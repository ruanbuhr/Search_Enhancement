import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

ds = load_dataset("sentence-transformers/stsb")

train_dataset = ds['train'].to_pandas()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2').to(device)
embedding_dim = sentence_transformer.get_sentence_embedding_dimension()


class SentencePairsDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sentence1 = self.dataframe.iloc[idx]['sentence1']
        sentence2 = self.dataframe.iloc[idx]['sentence2']
        label = self.dataframe.iloc[idx]['score']
        return sentence1, sentence2, label


train_data = SentencePairsDataset(train_dataset)
batch_size = 256  
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Siamese Network Architecture
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, query_embedding, snippet_embedding):
        combined_embedding = torch.cat((query_embedding, snippet_embedding), dim=-1)
        return self.fc(combined_embedding)

# Instantiate the Siamese model and move it to the same device
siamese_model = SiameseNetwork(embedding_dim).to(device)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(siamese_model.parameters(), lr=1e-3)

epochs = 0

for epoch in range(epochs):
    total_loss = 0
    siamese_model.train()

    for batch in train_loader:
        sentence1_list, sentence2_list, labels = batch

        # Get embeddings in batch (efficient GPU utilization)
        query_embeddings = sentence_transformer.encode(sentence1_list, convert_to_tensor=True, device=device)
        snippet_embeddings = sentence_transformer.encode(sentence2_list, convert_to_tensor=True, device=device)

        # Forward pass through the Siamese network
        output = siamese_model(query_embeddings, snippet_embeddings)

        # Compute the loss (move labels to GPU)
        labels = labels.to(device).float()
        loss = criterion(output.squeeze(), labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")


def rank_snippets(query, snippets):

    query_embedding = sentence_transformer.encode([query], convert_to_tensor=True, device=device)


    snippet_embeddings = sentence_transformer.encode(snippets, convert_to_tensor=True, device=device)

    snippet_scores = []
    for i, snippet_embedding in enumerate(snippet_embeddings):
        score = siamese_model(query_embedding, snippet_embedding.unsqueeze(0)).item()
        snippet_scores.append((snippets[i], score))

    ranked_snippets = sorted(snippet_scores, key=lambda x: x[1], reverse=True)
    return ranked_snippets


query = "How is AI impacting industries?"
snippets = [
    "AI is transforming industries across various sectors.",
    "Machine learning is a subset of artificial intelligence.",
    "Climate change affects the environment significantly.",
    "Deep learning is a branch of machine learning.",
    "Reinforcement learning is about learning from interaction."
]

ranked_results = rank_snippets(query, snippets)
for snippet, score in ranked_results:
    print(f"Snippet: {snippet}, Score: {score:.4f}")

