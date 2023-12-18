from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

bc=load_breast_cancer()

x=bc.data
y=bc.target

kmeans = KMeans(n_clusters=3,random_state=42)
kmeans.fit(x)

cluster_labels = kmeans.labels_
print("Cluster Labels::\n",cluster_labels)

centroids = kmeans.cluster_centers_
print("Centroids::",centroids)

plt.scatter(x[:,0],x[:,1],c=cluster_labels,cmap="viridis",marker="o",edgecolors="black")
plt.scatter(centroids[:,0],centroids[:,1],c="red",s=300,marker="*")
plt.title("KMEANS ALGORITHM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()