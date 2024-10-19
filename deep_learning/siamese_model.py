import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
embedding_dim = sentence_transformer.get_sentence_embedding_dimension()
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


loaded_model = SiameseNetwork(embedding_dim)

# Load the entire checkpoint
checkpoint = torch.load("deep_learning/siamese_model.pth", map_location=torch.device('cpu'))

loaded_model.load_state_dict(checkpoint['model_state_dict'])

loaded_model.eval()

def rank_snippets(query, google_results):
    query_embedding = sentence_transformer.encode(query, convert_to_tensor=True)
    link_scores = []

    for link, snippet in google_results.items():
        snippet_embedding = sentence_transformer.encode(snippet, convert_to_tensor=True)
        score = loaded_model(query_embedding, snippet_embedding).item()
        link_scores.append((link, score))

    ranked_link = sorted(link_scores, key=lambda x: x[1], reverse=True)

    # Return only the links
    ranked_links_only = [link for link, score in ranked_link]
    return ranked_links_only

loaded_model.eval()
