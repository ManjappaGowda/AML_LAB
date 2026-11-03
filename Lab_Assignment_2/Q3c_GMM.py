import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# 1. Load dataset
data = pd.read_csv(r"C:\Users\omgow\OneDrive\Desktop\MSIS\AML_LAB\Datasets\income_data.csv")
X = data[["income", "happiness"]]

# 2. Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=3,
    covariance_type="full",
    random_state=42
)

gmm.fit(X_scaled)

# 4. Assign cluster labels
labels = gmm.predict(X_scaled)
data["GMM_Cluster"] = labels

# 5. Display results
print("First five cluster assignments:")
print(data[["income", "happiness", "GMM_Cluster"]].head())

print("\nConverged:", gmm.converged_)
print("Number of iterations:", gmm.n_iter_)
print("AIC:", gmm.aic(X_scaled))
print("BIC:", gmm.bic(X_scaled))
print("Silhouette Score:", silhouette_score(X_scaled, labels))

# 6. Plot clusters
plt.figure(figsize=(7, 5))
plt.scatter(
    X_scaled[:, 0], X_scaled[:, 1],
    c=labels, cmap="viridis"
)
plt.xlabel("Standardized Income")
plt.ylabel("Standardized Happiness")
plt.title("Gaussian Mixture Model Clustering")
plt.show()

# 7. Show membership probabilities for first five rows
probabilities = gmm.predict_proba(X_scaled)
print("\nCluster membership probabilities (first five):")
print(probabilities[:5])

