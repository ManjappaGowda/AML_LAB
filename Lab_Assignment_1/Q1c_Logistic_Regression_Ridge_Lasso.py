import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report

# 1. Load dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# 3. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("\nSigmoid Function")
print("Sigmoid(0):", sigmoid(0))
print("Sigmoid(2):", sigmoid(2))
print("Sigmoid(-2):", sigmoid(-2))

# 5. Logistic Regression without regularization penalty
# Scikit-learn uses L2 by default, so penalty=None is specified.
logistic = LogisticRegression(
    penalty=None,
    max_iter=5000
)

logistic.fit(X_train, y_train)

y_pred = logistic.predict(X_test)

print("\nLogistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n",
      classification_report(y_test, y_pred,
                            target_names=data.target_names))

# 6. Ridge Regularization (L2)
ridge = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=5000
)

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

print("\nRidge Regularization (L2)")
print("Accuracy:", accuracy_score(y_test, ridge_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, ridge_pred))

# 7. Lasso Regularization (L1)
lasso = LogisticRegression(
    penalty="l1",
    solver="liblinear",
    C=1.0,
    max_iter=5000
)

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

print("\nLasso Regularization (L1)")
print("Accuracy:", accuracy_score(y_test, lasso_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lasso_pred))

# 8. Display coefficients
print("\nModel Coefficients")

print("Logistic Regression:", logistic.coef_)
print("Ridge Coefficients:", ridge.coef_)
print("Lasso Coefficients:", lasso.coef_)

# 9. Calculate predicted probabilities
probabilities = ridge.predict_proba(X_test)

print("\nPredicted Probabilities (First 5):")
print(probabilities[:5])

# 10. Plot Sigmoid Function
z = np.linspace(-10, 10, 100)

plt.figure(figsize=(7, 5))
plt.plot(z, sigmoid(z))
plt.axhline(y=0.5, linestyle="--")
plt.axvline(x=0, linestyle="--")
plt.xlabel("Input (z)")
plt.ylabel("Probability")
plt.title("Sigmoid Function")
plt.grid()
plt.show()

# 11. Decision Boundary using two features
# Use mean radius and mean texture

X_2 = data.data[:, [0, 1]]

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_2, y, test_size=0.2, random_state=42, stratify=y
)

scaler2 = StandardScaler()

X_train2 = scaler2.fit_transform(X_train2)
X_test2 = scaler2.transform(X_test2)

boundary_model = LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=5000
)

boundary_model.fit(X_train2, y_train2)

# Create mesh grid
x_min, x_max = X_train2[:, 0].min() - 1, X_train2[:, 0].max() + 1
y_min, y_max = X_train2[:, 1].min() - 1, X_train2[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict over grid
Z = boundary_model.predict(
    np.c_[xx.ravel(), yy.ravel()]
)

Z = Z.reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(8, 6))

plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")

plt.scatter(
    X_train2[:, 0],
    X_train2[:, 1],
    c=y_train2,
    cmap="coolwarm",
    edgecolors="black"
)

plt.xlabel("Mean Radius (Scaled)")
plt.ylabel("Mean Texture (Scaled)")
plt.title("Logistic Regression Decision Boundary")
plt.show()