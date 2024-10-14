from sentence_transformers import SentenceTransformer, util

# Init model once
model = SentenceTransformer('all-MiniLM-L6-v2')

def rank_search_results(user_query, link_snippet_dict):
    links = list(link_snippet_dict.keys())
    snippets = list(link_snippet_dict.keys())

    # Compute embeddings
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    snippet_embeddings = model.encode(snippets, convert_to_tensor=True)

    # Compute cosine similarity scores
    cosine_scores = util.cos_sim(query_embedding, snippet_embeddings)

    link_score_pairs = list(zip(links, cosine_scores[0].cpu().numpy()))

    # Sort links based on similarity scores
    ranked_links = sorted(link_score_pairs, key=lambda x: x[1], reverse=True)

    sorted_links = [link for link, score in ranked_links]

    return sorted_links

