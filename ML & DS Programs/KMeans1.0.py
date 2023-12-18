from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
x = iris.data  # Features
y = iris.target  # Target labels (not used in clustering)

# Initialize the KMeans model with 3 clusters and a random seed for reproducibility
km = KMeans(n_clusters=3, random_state=42)

# Fit the KMeans model to the feature data
km.fit(x)

# Obtain cluster labels and centroid coordinates
cluster_labels = km.labels_
print("Cluster Labels:", cluster_labels)

centroid = km.cluster_centers_
print("Centroid Coordinates:", centroid)

# Visualize the clusters and centroids using matplotlib
plt.scatter(x[:, 0], x[:, 1], c=cluster_labels, cmap="viridis", marker="o", edgecolors="black")
plt.scatter(centroid[:, 0], centroid[:, 1], c="red", marker="*", s=500, label="Centroid")

plt.title('K-Means Clustering on Iris Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.show()
