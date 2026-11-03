# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Load dataset
data = pd.read_csv(r"C:\Users\omgow\OneDrive\Desktop\MSIS\AML_LAB\Datasets\Social_Network_Ads.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:", data.shape)

# 2. Select input features and target
X = data[['Gender', 'Age', 'EstimatedSalary']].copy()
y = data['Purchased']

# 3. Convert Gender into numerical values
X['Gender'] = X['Gender'].map({
    'Female': 0,
    'Male': 1
})

# 4. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 5. Feature scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Create KNN model
knn = KNeighborsClassifier(n_neighbors=5)

# 7. Train the model
knn.fit(X_train, y_train)

# 8. Predict test data
y_pred = knn.predict(X_test)

# 9. Evaluate the model
print("\nKNN Classification Results")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=['Not Purchased', 'Purchased']
))

# 10. Test different K values
k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    acc = accuracy_score(y_test, prediction)
    accuracies.append(acc)

    print("K =", k, "Accuracy =", acc)

# 11. Find best K
best_k = list(k_values)[accuracies.index(max(accuracies))]

print("\nBest K:", best_k)
print("Best Accuracy:", max(accuracies))

# 12. Plot K vs Accuracy
plt.figure(figsize=(8, 5))

plt.plot(k_values, accuracies, marker='o')

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN - K Value vs Accuracy")

plt.xticks(list(k_values))
plt.grid()
plt.show()

# 13. Confusion Matrix Visualization
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=['Not Purchased', 'Purchased'],
    cmap='Blues'
)

plt.title("KNN Confusion Matrix")
plt.show()

# 14. Predict a new customer
new_customer = pd.DataFrame({
    'Gender': [1],
    'Age': [35],
    'EstimatedSalary': [60000]
})

new_customer_scaled = scaler.transform(new_customer)

prediction = knn.predict(new_customer_scaled)

print("\nNew Customer Prediction:")

if prediction[0] == 1:
    print("Customer will purchase the product")
else:
    print("Customer will not purchase the product")