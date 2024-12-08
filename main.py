import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load the image
image_path = "lena.png"
points_per_cluster = 50
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Reshape image into a 2D array of pixels (RGB)
pixels = image_rgb.reshape(-1, 3)


# K-means from scratch
def kmeans_clustering(data, k=3, max_iters=100, tol=1e-4):
    np.random.seed(0)
    # Randomly initialize centroids
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]

    for iteration in range(max_iters):
        # Assign points to nearest centroid
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return labels, centroids


# Fuzzy C-means from scratch
def fuzzy_cmeans_clustering(data, c=3, m=2, max_iters=100, error=1e-4):
    np.random.seed(0)
    # Initialize membership matrix randomly
    u = np.random.dirichlet(np.ones(c), size=data.shape[0])

    for iteration in range(max_iters):
        # Calculate cluster centers
        centroids = (u.T ** m) @ data / np.sum(u.T ** m, axis=1)[:, np.newaxis]

        # Update membership matrix
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2) + 1e-6  # Avoid division by zero
        new_u = 1 / distances ** (2 / (m - 1))
        new_u /= np.sum(new_u, axis=1, keepdims=True)

        # Check for convergence
        if np.linalg.norm(new_u - u) < error:
            break
        u = new_u

    labels = np.argmax(u, axis=1)  # Assign each point to the cluster with highest membership
    return labels, centroids


# Apply K-means
kmeans_labels, kmeans_centroids = kmeans_clustering(pixels, k=3)

# Apply Fuzzy C-means
fuzzy_labels, fuzzy_centroids = fuzzy_cmeans_clustering(pixels, c=3)

# Stratified Sampling to Reduce Points

sampled_pixels = []
sampled_kmeans_labels = []
sampled_fuzzy_labels = []

for cluster in range(3):
    cluster_indices = np.where(kmeans_labels == cluster)[0]
    sampled_indices = np.random.choice(cluster_indices, min(len(cluster_indices), points_per_cluster), replace=False)
    sampled_pixels.append(pixels[sampled_indices])
    sampled_kmeans_labels.append(kmeans_labels[sampled_indices])
    sampled_fuzzy_labels.append(fuzzy_labels[sampled_indices])

sampled_pixels = np.vstack(sampled_pixels)
sampled_kmeans_labels = np.hstack(sampled_kmeans_labels)
sampled_fuzzy_labels = np.hstack(sampled_fuzzy_labels)


# Visualization
fig = plt.figure(figsize=(12, 6))

# K-means Plot
ax1 = fig.add_subplot(121, projection='3d')
for i, color in enumerate(['r', 'g', 'b']):
    ax1.scatter(
        sampled_pixels[sampled_kmeans_labels == i, 0],
        sampled_pixels[sampled_kmeans_labels == i, 1],
        sampled_pixels[sampled_kmeans_labels == i, 2],
        edgecolors=color, facecolors='none', label=f"Cluster {i+1}", alpha=0.6, s=10
    )
ax1.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], kmeans_centroids[:, 2], c='black', marker='o', s=100, label='Centroids')
ax1.set_title("K-means Clustering")
ax1.set_xlabel('Red')
ax1.set_ylabel('Green')
ax1.set_zlabel('Blue')
ax1.set_xlim(ax1.get_xlim()[::-1]) 
ax1.legend()

# Fuzzy C-means Plot
ax2 = fig.add_subplot(122, projection='3d')
for i, color in enumerate(['r', 'g', 'b']):
    ax2.scatter(
        sampled_pixels[sampled_fuzzy_labels == i, 0],
        sampled_pixels[sampled_fuzzy_labels == i, 1],
        sampled_pixels[sampled_fuzzy_labels == i, 2],
        edgecolors=color, facecolors='none', label=f"Cluster {i+1}", alpha=0.6, s=10
    )
ax2.scatter(fuzzy_centroids[:, 0], fuzzy_centroids[:, 1], fuzzy_centroids[:, 2], c='black', marker='o', s=100, label='Centroids')
ax2.set_title("Fuzzy C-means Clustering")
ax2.set_xlabel('Red')
ax2.set_ylabel('Green')
ax2.set_zlabel('Blue')
ax2.legend()
ax2.set_xlim(ax2.get_xlim()[::-1]) 
plt.tight_layout()
plt.show()
