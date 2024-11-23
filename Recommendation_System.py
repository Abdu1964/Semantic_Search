from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine
import numpy as np

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example dataset: News article titles
articles = [
    "Machine learning is fascinating.",
    "AI and machine learning are closely related fields.",
    "Climate change impacts global weather patterns.",
    "Stock markets see significant growth.",
    "AI is transforming the healthcare industry."
]

# User query
query = "How is AI used in healthcare?"

# Generate embeddings for articles
article_embeddings = model.encode(articles)

# Generate embedding for the query
query_embedding = model.encode(query)

# Calculate cosine similarity for the query against each article
similarities = []
for i, article_embedding in enumerate(article_embeddings):
    # Using Sentence Transformers util.cos_sim
    similarity_transformers = util.cos_sim(query_embedding, article_embedding).item()
    similarities.append((articles[i], similarity_transformers))

# Rank articles by similarity (highest first)
ranked_articles = sorted(similarities, key=lambda x: x[1], reverse=True)

# Display the ranked results
print(f"Query: '{query}'\n")
print("Top Recommendations:")
for i, (article, score) in enumerate(ranked_articles[:5]):  # Top 5 recommendations
    print(f"{i+1}. {article} (Score: {score:.4f})")
