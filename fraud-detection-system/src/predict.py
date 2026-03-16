import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# Load saved model
model = joblib.load("models/best_fraud_model.pkl")


# Load test dataset
test = pd.read_csv("data/processed/test_processed.csv")


# Separate features and target
X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]


# Probability predictions
y_prob = model.predict_proba(X_test)[:, 1]


# Threshold tuning
y_pred = (y_prob > 0.8).astype(int)


# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)


# Print results
print("Model Evaluation on Test Data")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)


# Save predictions
test["fraud_prediction"] = y_pred
test.to_csv("predictions.csv", index=False)

print("Predictions saved to predictions.csv")