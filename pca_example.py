import numpy as np
from sklearn.decomposition import PCA

# Example dataset with higher dimensions
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
print("Reduced Data:\n", reduced_data)
