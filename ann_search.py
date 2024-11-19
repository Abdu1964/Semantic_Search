#1. Approximate Nearest Neighbor Search with Annoy (LSH)
from annoy import AnnoyIndex

# Initialize with dimension of vectors (3D vectors)
f = 3
index = AnnoyIndex(f, 'angular')

# Add vectors to the index
index.add_item(0, [1, 2, 3])
index.add_item(1, [4, 5, 6])
index.add_item(2, [7, 8, 9])

# Build the index with 10 trees
index.build(10)

# Query for the nearest neighbor
nearest_neighbors = index.get_nns_by_item(0, 2)  # Find 2 nearest neighbors for item 0

print("1. Approximate Nearest Neighbor Search with Annoy (LSH)")
print ("------------------------------------------------------")
print("Nearest Neighbors:", nearest_neighbors)
print(                                                           )

#2. Clustering with K-Means for Efficient Search
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset: 6 points in a 3D space
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 0], [2, 3, 4], [9, 8, 7]])

# Apply KMeans clustering (here, choosing 2 clusters)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# Assign clusters
labels = kmeans.labels_

# Find the nearest point to the first point (index 0) within the same cluster
cluster = labels[0]  # Cluster of the first point
cluster_points = X[labels == cluster]  # Get all points in the same cluster

# Calculate the cosine similarity of the first point with others in the cluster
similarities = cosine_similarity([X[0]], cluster_points)

# Find the nearest neighbor in the cluster
nearest_neighbor_index = np.argmax(similarities)
nearest_neighbor = cluster_points[nearest_neighbor_index]

print("2.Clustering with K-Means for Efficient Search")
print ("---------------------------------------------")
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
centroids = kmeans.cluster_centers_

# Assign each point to the closest centroid
labels = kmeans.labels_

print("3.Vector Quantization (VQ)")
print ("---------------------------------------------")
print("Centroids of Quantized Vectors:", centroids)
print("Assigned Labels for Each Point:", labels)
print("                                                       ")
#4. Using KD-Tree for Efficient Search
from sklearn.neighbors import KDTree
import numpy as np

# Sample data: 6 points in a 3D space
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 0, 0], [2, 3, 4], [9, 8, 7]])

# Build a KD-tree from the data
tree = KDTree(X)

# Query the nearest neighbor to the first point (index 0)
dist, ind = tree.query([X[0]], k=2)  # Find the 2 nearest neighbors
print("4.Using KD-Tree for Efficient Search")
print ("---------------------------------------------")
print("Nearest Neighbors Indices:", ind)
print("Nearest Neighbors Distances:", dist)
print ("                                      ")

#5. Ball Tree for High-Dimensional Data
from sklearn.neighbors import BallTree
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




