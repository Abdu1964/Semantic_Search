Similarity Search Methods and Optimization Techniques
This repository contains code and implementations for exploring various similarity search methods and optimization techniques aimed at improving both search efficiency and quality. It includes practical examples using different similarity metrics, Approximate Nearest Neighbors (ANN) for faster search, and methods to improve the quality of text-based similarity searches.

Key Concepts
Similarity Search: A technique to find items in a dataset that are similar to a given query.
Similarity Metrics: Methods to quantify the similarity between two items (e.g., Cosine Similarity, Euclidean Distance, Jaccard Similarity).
Optimization Techniques: Methods like Approximate Nearest Neighbors (ANN) and Dimensionality Reduction to improve search efficiency.
Features
Cosine Similarity: Measures the angle between vectors, commonly used for text similarity.
Euclidean Distance: Measures the straight-line distance between two vectors.
Jaccard Similarity: Measures the similarity between two sets.
Hamming Distance: Measures the difference between two strings.
Approximate Nearest Neighbors (ANN): Faster methods for finding nearest neighbors in high-dimensional spaces.
Dimensionality Reduction (PCA): Reduces the number of features while maintaining data patterns.
Installation
Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/Abdu1964/Semantic_Search.git
cd Semantic_Search
Install the required Python libraries:

bash
Copy code
pip install -r requirements.txt
Usage
Similarity Metrics Example:

Cosine Similarity:
python
Copy code
from sklearn.metrics.pairwise import cosine_similarity
vector1 = [1, 2, 3]
vector2 = [4, 5, 6]
print(cosine_similarity([vector1], [vector2]))
Euclidean Distance:
python
Copy code
from numpy import linalg as la
import numpy as np
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])
print(la.norm(vector1 - vector2))
Optimizing with Approximate Nearest Neighbors:

Install Annoy library:
bash
Copy code
pip install annoy
Example code for ANN:
python
Copy code
from annoy import AnnoyIndex
index = AnnoyIndex(3, 'angular')
index.add_item(0, [1, 2, 3])
index.add_item(1, [4, 5, 6])
index.build(10)
print(index.get_nns_by_item(0, 2))  # Find nearest neighbors
Dimensionality Reduction Example:

python
Copy code
from sklearn.decomposition import PCA
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
print(reduced_data)
Text-Based Similarity with Embeddings:

Install sentence-transformers:
bash
Copy code
pip install sentence-transformers
Example code for Sentence Embedding:
python
Copy code
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
sentence1 = "Machine learning is fascinating."
sentence2 = "AI and ML are related."
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)
from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity([embedding1], [embedding2]))
