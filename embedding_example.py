from sentence_transformers import SentenceTransformer, util
from scipy.spatial.distance import cosine

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example sentences
sentence1 = "Machine learning is fascinating."
sentence2 = "AI and machine learning are closely related fields."

# Generate embeddings
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

# Calculate cosine similarity between the embeddings using both Sentence Transformers and Scipy
cosine_sim_transformers = util.cos_sim(embedding1, embedding2)
cosine_sim_scipy = 1 - cosine(embedding1, embedding2)

print("Cosine Similarity (Text Embedding - Sentence Transformers):", cosine_sim_transformers.item())
print("Cosine Similarity (Text Embedding - Scipy):", cosine_sim_scipy)
