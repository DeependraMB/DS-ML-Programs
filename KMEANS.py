from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris=load_iris()
x=iris.data
y=iris.target

km=KMeans(n_clusters=3,random_state=42)
km.fit(x)

cluster_labels=km.labels_
print(cluster_labels)
centroids=km.cluster_centers_
print(centroids)

plt.scatter(x[:,0],x[:,1],c=cluster_labels,cmap="viridis",marker="o",edgecolors="black")
plt.scatter(centroids[:,0],centroids[:,1],marker="o",s=200,c="red",label="centroid");
plt.xlabel=(iris.feature_names[0])
plt.ylabel=(iris.feature_names[1])
plt.title("KMEANS CLUSTERING")
plt.legend()
plt.show()