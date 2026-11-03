import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 1. Load data and select numerical features
data = pd.read_csv(r"C:\Users\omgow\OneDrive\Desktop\MSIS\AML_LAB\Datasets\income_data.csv")
X = data[["income", "happiness"]]

# 2. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Calculate distances and similarities for first five rows
print("Euclidean distance:\n",
      np.round(euclidean_distances(X_scaled[:5]), 3))
print("Cosine similarity:\n",
      np.round(cosine_similarity(X_scaled[:5]), 3))

# 4. K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
data["KMeans_Cluster"] = kmeans_labels

print("\nK-Means silhouette score:",
      silhouette_score(X_scaled, kmeans_labels))

plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=kmeans_labels, cmap="viridis")
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            color="red", marker="X", s=150, label="Centroids")
plt.xlabel("Standardized Income")
plt.ylabel("Standardized Happiness")
plt.title("K-Means Clustering")
plt.legend()
plt.show()

# 5. Hierarchical clustering with different linkage methods
methods = ["single", "complete", "average", "ward"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for method, ax in zip(methods, axes.ravel()):
    Z = linkage(X_scaled, method=method)
    dendrogram(Z, truncate_mode="lastp", p=20, ax=ax)
    ax.set_title(method.capitalize() + " Linkage")
    ax.set_xlabel("Clusters / Samples")
    ax.set_ylabel("Distance")

plt.tight_layout()
plt.show()

# 6. Hierarchical clustering using Ward linkage
Z = linkage(X_scaled, method="ward")
hier_labels = fcluster(Z, t=3, criterion="maxclust")
data["Hierarchical_Cluster"] = hier_labels

print("Hierarchical silhouette score:",
      silhouette_score(X_scaled, hier_labels))

plt.figure(figsize=(7, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1],
            c=hier_labels, cmap="plasma")
plt.xlabel("Standardized Income")
plt.ylabel("Standardized Happiness")
plt.title("Hierarchical Clustering - Ward Linkage")
plt.show()