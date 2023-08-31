import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

class KMeans:
    def __init__(self, n_clusters=4, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        
    def initialize_clusters(self, X):
        n_samples, n_features = X.shape
        clusters = np.zeros((self.n_clusters, n_features))
        for k in range(self.n_clusters):
            cluster = X[np.random.choice(range(n_samples))]
            clusters[k] = cluster
        return clusters
    
    def compute_cluster_labels(self, X, clusters):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.apply_along_axis(euclidean_distance, 1, X, clusters[k])
        return np.argmin(distances, axis=1)
    
    def compute_new_clusters(self, X, cluster_labels):
        clusters = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            clusters[k] = np.mean(X[cluster_labels == k], axis=0)
        return clusters
    
    def predict(self, X):
        clusters = self.initialize_clusters(X)
        for _ in range(self.max_iterations):
            cluster_labels = self.compute_cluster_labels(X, clusters)
            old_clusters = clusters
            clusters = self.compute_new_clusters(X, cluster_labels)
            if np.sum(clusters - old_clusters) == 0:
                break
        return cluster_labels

X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.6)
kmeans = KMeans(n_clusters=4, max_iterations=100)
y_pred = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
plt.show()
