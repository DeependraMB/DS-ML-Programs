import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load Iris dataset
iris = load_iris()
x = iris.data
y = iris.target

# Create KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)

# Get cluster labels and centroids
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add new data point
new_data_point = np.array([[5.0, 3.5, 1.5, 0.2]])  # You can modify this with your own values
new_data_label = kmeans.predict(new_data_point)

# Print cluster labels and centroids
print("CLUSTER LABELS:\n", cluster_labels)
print("CENTROIDS: \n", centroids)

# Plot existing data, clusters, centroids, and new data point
plt.scatter(x[:, 0], x[:, 1], c=cluster_labels, cmap="viridis", marker="o", edgecolors="black")
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="*", s=300, edgecolors="yellow")
plt.scatter(new_data_point[:, 0], new_data_point[:, 1], c="blue", marker="*", s=100, )

plt.show()
