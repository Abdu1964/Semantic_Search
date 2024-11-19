import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Example dataset with higher dimensions
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Visualize original data
plt.figure(figsize=(6, 4))
plt.scatter(data[:, 0], data[:, 1], c='blue', label='Original Data')
plt.title("Original Data (Higher Dimensions)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend(loc='best')
plt.show()

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# Visualize reduced data
plt.figure(figsize=(6, 4))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='red', label='Reduced Data (2D)')
plt.title("Reduced Data after PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(loc='best')
plt.show()

# Print the explained variance ratio
print(f"Explained Variance Ratio by Components: {pca.explained_variance_ratio_}")
print("Reduced Data:\n", reduced_data)
