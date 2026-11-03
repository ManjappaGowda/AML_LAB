import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Load dataset
data = pd.read_csv(r"C:\Users\omgow\OneDrive\Desktop\MSIS\AML_LAB\Datasets\diabetes.csv")

# 2. Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

print("Original data shape:", X.shape)

# 3. Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply PCA and reduce to two components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Display transformed data and explained variance
pca_data = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

print("\nFirst five PCA rows:")
print(pca_data.head())

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("Total variance retained:",
      pca.explained_variance_ratio_.sum())

# 6. Visualize PCA components
plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Visualization of Diabetes Dataset")
plt.legend(
    *scatter.legend_elements(),
    title="Outcome"
)
plt.grid()
plt.show()

# 7. Plot explained variance
plt.figure(figsize=(6, 4))
plt.bar(
    ["PC1", "PC2"],
    pca.explained_variance_ratio_
)
plt.ylabel("Explained Variance Ratio")
plt.title("PCA Explained Variance")
plt.show()