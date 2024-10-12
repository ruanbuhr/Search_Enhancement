from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
# Example user query
user_query = "I'm interested in becoming a more proficient data scientist and want to improve my coding skills specifically for data science applications. I already have a basic understanding of Python, but I need guidance on the most effective ways to advance my skills in data manipulation, machine learning, and statistical analysis. Additionally, I'm looking for resources that can help me learn more about data visualization, working with large datasets, and using popular libraries like Pandas, NumPy, and Scikit-Learn. How can I structure my learning to effectively build these skills and become proficient in data science coding?"

# Example snippets
snippets = [
    "Coding skills are essential for data science. Start with learning Python and R.",
    "Data science coding involves learning various algorithms and data processing techniques.",
    "To improve coding skills, practice regularly on platforms like Kaggle and GitHub.",
    "Building projects and collaborating with others can also enhance your coding abilities.",
    "Focusing on statistics and machine learning can help you become a better data scientist.",
    "Engage in coding challenges and competitions to hone your problem-solving skills.",
    "Reading code written by experienced data scientists can provide insights into best practices.",
    "Learning data visualization techniques in tools like Matplotlib, Seaborn, and Tableau can be helpful.",
    "Master data cleaning and preprocessing techniques, as they are crucial in data science.",
    "Understanding the basics of SQL is important for handling data in relational databases.",
    "Try working with different datasets to get familiar with various data structures and formats.",
    "Explore popular data science libraries such as Pandas, NumPy, and Scikit-Learn.",
    "Attend data science meetups and conferences to network and learn from industry experts.",
    "Stay updated with the latest trends in data science by reading blogs, articles, and research papers.",
    "Work on end-to-end data science projects to understand the full lifecycle of data analysis.",
    "Use platforms like Coursera and edX to take courses on specialized topics like deep learning.",
    "Regularly review your code and seek feedback from peers to identify areas for improvement.",
    "Develop a solid understanding of foundational concepts in linear algebra and statistics.",
    "Use version control systems like Git to manage your code and collaborate with others.",
    "water sucks ass",
    "i dont understand coding at all",
    "How can I improve my coding skills for data science?",
    "dino people suck",
    "where is my waffles",
    "how many beavers does it take to paint a wall",
    "is running a sport",
    "where is my pizza?",
    "how many elephants does a dorito contain"
]
# Compute embedding for the user query
query_embedding = model.encode(user_query, convert_to_tensor=True)

# Compute embeddings for the snippets
snippet_embeddings = model.encode(snippets, convert_to_tensor=True)
# Compute cosine similarity scores

cosine_scores = util.cos_sim(query_embedding, snippet_embeddings)

# Combine snippets with their scores
snippet_score_pairs = list(zip(snippets, cosine_scores[0].cpu().numpy()))

# Sort snippets based on similarity scores
ranked_snippets = sorted(snippet_score_pairs, key=lambda x: x[1], reverse=True)
for snippet, score in ranked_snippets:
    print(f"Score: {score:.4f} - Snippet: {snippet}")

snippets = [snippet for snippet, score in ranked_snippets]

for snippet in snippets:
    print(snippet)

