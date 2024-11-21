import numpy as np  #library for numerical operations
from scipy.spatial.distance import hamming  #Computes the normalized Hamming distance
import matplotlib.pyplot as plt # for plotting and visualization

# Cosine Similarity ,How two vectors are similar
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Euclidean Distance ,used to measure the distance between 2 sets 
#Computes the straight-line distance between two points in space
#for clusturing and classfication(Nearest neighbor)
def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

# Jaccard Similarity  
# Measures similarity between two sets, ignoring element frequency
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

# Hamming Distance ,count the number of differences where the corsponding binary or letter are diffrenet at the same position
#the number of positions where the corresponding elements differ, the sum of the differences of a and b
def hamming_distance(str1, str2):
    max_len = max(len(str1), len(str2))
    str1 = str1.ljust(max_len)
    str2 = str2.ljust(max_len)
    return hamming(list(str1), list(str2)) * max_len

# Dot Product
def dot_product(vec1, vec2):
    return np.dot(vec1, vec2)

# Visualize Vector-based Metrics
# Visualizes vectors in 2D space
def visualize_vectors(vec1, vec2):
    plt.figure(figsize=(8, 6))
    
    # Plot vectors
    plt.quiver(0, 0, vec1[0], vec1[1], angles='xy', scale_units='xy', scale=1, color='r', label='vec1') #to draw arrows representing vector
    plt.quiver(0, 0, vec2[0], vec2[1], angles='xy', scale_units='xy', scale=1, color='b', label='vec2')
    
    plt.xlim(-1, max(vec1[0], vec2[0]) + 1) #adjust axis limits for x
    plt.ylim(-1, max(vec1[1], vec2[1]) + 1)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    
    plt.legend()
    plt.title('Vector Visualization')
    plt.grid()
    plt.show()

# Visualize Binary Similarity
# Compares two binary sequences visually
def visualize_binary_similarity(str1, str2):
    max_len = max(len(str1), len(str2))
    x = np.arange(max_len)
    
    # Convert strings to binary arrays
    bin_str1 = np.array([int(bit) for bit in str1.ljust(max_len)])
    bin_str2 = np.array([int(bit) for bit in str2.ljust(max_len)])
    
    plt.figure(figsize=(10, 2))
    plt.plot(x, bin_str1, label='str1', marker='o')
    plt.plot(x, bin_str2, label='str2', marker='o')
    plt.yticks([0, 1])
    plt.xlabel('Bit Position')
    plt.title('Binary Sequence Comparison')
    plt.legend()
    plt.grid()
    plt.show()

# Example Vectors and Sets
vec1 = np.array([1, 2])
vec2 = np.array([2, 3])
set1 = {1, 2, 3}
set2 = {2, 3, 4}
str1 = "101010"
str2 = "111000"

# Calculate and display results
print("Cosine Similarity:", cosine_similarity(vec1, vec2))
print("Euclidean Distance:", euclidean_distance(vec1, vec2))
print("Jaccard Similarity:", jaccard_similarity(set1, set2))
print("Hamming Distance:", hamming_distance(str1, str2))
print("Dot Product:", dot_product(vec1, vec2))

# Visualize metrics
visualize_vectors(vec1, vec2)
visualize_binary_similarity(str1, str2)
