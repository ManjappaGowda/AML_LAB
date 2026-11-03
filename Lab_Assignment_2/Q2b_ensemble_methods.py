import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

# 1. Load dataset
data = pd.read_csv(r"C:\Users\omgow\OneDrive\Desktop\MSIS\AML_LAB\Datasets\Social_Network_Ads.csv")

# 2. Prepare features and target
X = data[["Gender", "Age", "EstimatedSalary"]].copy()
X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
y = data["Purchased"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Define ensemble models
models = {
    "Random Forest (Bagging)": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=42
    )
}

# 5. Train and evaluate each model
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\n", name)
    print("Accuracy:", acc)
    print(classification_report(
        y_test, y_pred,
        target_names=["Not Purchased", "Purchased"]
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Not Purchased", "Purchased"],
        cmap="Blues"
    )
    plt.title(name)
    plt.show()

# 6. Compare model accuracy
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values())
plt.ylabel("Accuracy")
plt.title("Comparison of Ensemble Models")
plt.xticks(rotation=15)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

print("\nAccuracy comparison:")
for name, score in results.items():
    print(name, ":", score)