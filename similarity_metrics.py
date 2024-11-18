import numpy as np
from scipy.spatial.distance import hamming

# Cosine Similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Euclidean Distance
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# Jaccard Similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Hamming Distance
def hamming_distance(str1, str2):
    # Normalize length of strings
    max_len = max(len(str1), len(str2))
    str1 = str1.ljust(max_len)
    str2 = str2.ljust(max_len)
    return hamming(list(str1), list(str2)) * max_len

# Example Vectors and Sets
vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
set1 = {1, 2, 3}
set2 = {2, 3, 4}
str1 = "101010"
str2 = "111000"

# Results
print("Cosine Similarity:", cosine_similarity(vec1, vec2))
print("Euclidean Distance:", euclidean_distance(vec1, vec2))
print("Jaccard Similarity:", jaccard_similarity(set1, set2))
print("Hamming Distance:", hamming_distance(str1, str2))
