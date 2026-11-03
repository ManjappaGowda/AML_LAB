# Import libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# 1. Load dataset
data = pd.read_csv(r"C:\Users\omgow\OneDrive\Desktop\MSIS\AML_LAB\Datasets\diabetes.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset Shape:", data.shape)

# 2. Separate features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

print("\nFeatures:")
print(X.columns)

print("\nTarget Distribution:")
print(y.value_counts())

# 3. Handle invalid zero values
# Zero is not a valid measurement for these attributes
zero_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[zero_columns] = X[zero_columns].replace(0, np.nan)

print("\nMissing Values:")
print(X.isnull().sum())

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# 5. Create Naive Bayes model
# Median imputation handles missing values
model = make_pipeline(
    SimpleImputer(strategy="median"),
    GaussianNB()
)

# 6. Train model
model.fit(X_train, y_train)

# 7. Predict test data
y_pred = model.predict(X_test)

# 8. Evaluate model
print("\nNaive Bayes Classification Results")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["No Diabetes", "Diabetes"]
))

# 9. Predict probability
y_probability = model.predict_proba(X_test)

print("\nPredicted Probabilities (First 5):")
print(y_probability[:5])

# 10. Predict a new patient
new_patient = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 100,
    "BMI": 28.5,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30
}])

prediction = model.predict(new_patient)

print("\nNew Patient Prediction:")

if prediction[0] == 1:
    print("Predicted Class: Diabetes")
else:
    print("Predicted Class: No Diabetes")

print("Prediction Probabilities:", model.predict_proba(new_patient))