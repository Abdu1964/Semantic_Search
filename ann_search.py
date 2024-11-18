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
print("Nearest Neighbors:", nearest_neighbors)
