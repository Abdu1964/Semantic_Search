#1. Approximate Nearest Neighbor Search with Annoy (LSH)
from annoy import AnnoyIndex #Approximate Nearest Neighbors Oh Yeah

# Initialize with dimension of vectors (3D vectors)
f = 3 # the vectors are 3-dimensional.
index = AnnoyIndex(f, 'angular') #Creates an Annoy index with f dimensions,angular is angular distance for cosine similarity

# Add vectors to the annoy index
index.add_item(0, [1, 2, 3]) #0,1,2 are indexes and the rest are vectors
index.add_item(1, [4, 5, 6])
index.add_item(2, [7, 8, 9])

# Build the index with 10 trees
index.build(50)

# Query for the nearest neighbor
nearest_neighbors = index.get_nns_by_item(0, 2)  # Find 2 nearest neighbors for item 0 , it will do the similarity  between the vectors and 
# put by their id

print("1. Approximate Nearest Neighbor Search with Annoy (LSH)")
print ("------------------------------------------------------")
print("Nearest Neighbors:", nearest_neighbors)
print(                                                           )

from sklearn.cluster import KMeans  # K-means for clustering data into groups
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset: 6 points in a 3D space
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 0], [2, 3, 4], [9, 8, 7]])

# Apply KMeans clustering (here, choosing 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)  # Applies K-Means clustering to divide the dataset into 2 clusters

# Assign clusters
labels = kmeans.labels_  # Retrieves the cluster label of the first point

# Find the cluster of the first point (index 0)
cluster = labels[0]
cluster_points = X[labels == cluster]  # Get all points in the same cluster as the first point

# Exclude the first point from the cluster (so it doesn't compare with itself)
cluster_points_without_first = np.delete(cluster_points, 0, axis=0)

# Calculate the cosine similarity of the first point with others in the same cluster
similarities = cosine_similarity([X[0]], cluster_points_without_first)  # Computes the cosine similarity of [1, 2, 3] with the remaining points in the same cluster

# Find the nearest neighbor in the cluster
nearest_neighbor_index = np.argmax(similarities)  # Finds the index of the most similar point (excluding itself)
nearest_neighbor = cluster_points_without_first[nearest_neighbor_index]  # Gets the actual point

print("2.Clustering with K-Means for Efficient Search")
print("---------------------------------------------")
print(f"Clustered Neighbors for point {X[0]}: {nearest_neighbor}")
print("                                                       ")


#3. Vector Quantization (VQ)
from sklearn.cluster import KMeans
import numpy as np

# Sample data: 6 points in a 3D space
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 0], [2, 3, 4], [9, 8, 7]])

# Apply KMeans to quantize data into 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Centroids are the representative "quantized" vectors 
# Retrieves the centroids (mean positions of the clusters)
centroids = kmeans.cluster_centers_

# Assign each point to the closest centroid
labels = kmeans.labels_

print("3.Vector Quantization (VQ)")
print ("---------------------------------------------")
print("Centroids of Quantized Vectors:", centroids)
print("Assigned Labels for Each Point:", labels)
print("                                                       ")
#4. Using KD-Tree for Efficient Search
from sklearn.neighbors import KDTree  #a binary tree structure for nearest neighbor search
import numpy as np

# Sample data: 6 points in a 3D space
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 0], [2, 3, 4], [9, 8, 7]])

# Build a KD-tree from the data
tree = KDTree(X)

# Query the nearest neighbor to the first point (index 0) dist is distance  and ind is indice which indcts k NN  
dist, ind = tree.query([X[0]], k=2)  # Find the 2 nearest neighbors
print("4.Using KD-Tree for Efficient Search")
print ("---------------------------------------------")
print("Nearest Neighbors Indices:", ind)
print("Nearest Neighbors Distances:", dist)
print ("                                      ")

#5. Ball Tree for High-Dimensional Data
from sklearn.neighbors import BallTree # sklearn provides tools and algorithms for finding for NN
import numpy as np

# Sample data: 6 points in a 3D space
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 0], [2, 3, 4], [9, 8, 7]])

# Build a Ball tree from the data
tree = BallTree(X)

# Query the nearest neighbor to the first point (index 0)
dist, ind = tree.query([X[0]], k=2)  # Find the 2 nearest neighbors
print ("5. Ball Tree for High-Dimensional Data")
print ("---------------------------------------------")
print("Nearest Neighbors Indices:", ind)
print("Nearest Neighbors Distances:", dist)