import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Breast Cancer dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Entropy calculation
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    return -np.sum(probabilities * np.log2(probabilities))

# Gini impurity calculation
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)

    return 1 - np.sum(probabilities ** 2)

# Information Gain calculation
def information_gain(parent, left, right):
    parent_entropy = entropy(parent)

    weighted_entropy = (
        len(left) / len(parent) * entropy(left)
        + len(right) / len(parent) * entropy(right)
    )

    return parent_entropy - weighted_entropy

# Example calculation
parent = np.array([0, 0, 1, 1, 1, 0])
left = np.array([0, 0, 0])
right = np.array([1, 1, 1])

print("\nEntropy:", entropy(parent))
print("Gini Impurity:", gini(parent))
print("Information Gain:", information_gain(parent, left, right))

# Decision Tree using Gini
model_gini = DecisionTreeClassifier(
    criterion="gini",
    random_state=42
)

model_gini.fit(X_train, y_train)

y_pred_gini = model_gini.predict(X_test)

print("\nDecision Tree - Gini Impurity")
print("Accuracy:", accuracy_score(y_test, y_pred_gini))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_gini))
print("Classification Report:\n",
      classification_report(y_test, y_pred_gini,
                            target_names=data.target_names))

# Decision Tree using Entropy
model_entropy = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model_entropy.fit(X_train, y_train)

y_pred_entropy = model_entropy.predict(X_test)

print("\nDecision Tree - Entropy")
print("Accuracy:", accuracy_score(y_test, y_pred_entropy))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_entropy))
print("Classification Report:\n",
      classification_report(y_test, y_pred_entropy,
                            target_names=data.target_names))

# Visualize Decision Tree
plt.figure(figsize=(20, 10))

plot_tree(
    model_gini,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree - Breast Cancer Dataset")
plt.show()