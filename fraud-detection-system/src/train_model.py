from unicodedata import name

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE


# Load dataset
train = pd.read_csv("data/processed/train_processed.csv")
test = pd.read_csv("data/processed/test_processed.csv")

X = train.drop("is_fraud", axis=1)
y = train["is_fraud"]

X_test = test.drop("is_fraud", axis=1)
y_test = test["is_fraud"]

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

##MLFlow experiment setup
mlflow.set_experiment("Fraud Detection Experiment")

## Variable to track the best model
best_model = None
best_f1 = 0
best_model_name = ""

# RANDOM FOREST
with mlflow.start_run(run_name="Random Forest"):

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

    rf_model.fit(X_resampled, y_resampled)

    ##probability predictions
    y_prob = rf_model.predict_proba(X_test)[:,1]

    # threshold tuning
    y_pred = (y_prob > 0.8).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    mlflow.log_param("model", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.log_model(rf_model, name="random_forest_model")

    if f1 > best_f1:
        best_f1 = f1
        best_model = rf_model
        best_model_name = "Random Forest"

# LOGISTIC REGRESSION
with mlflow.start_run(run_name="Logistic Regression"):

    lr_model = LogisticRegression(max_iter=1000, random_state=42)

    lr_model.fit(X_resampled, y_resampled)
    
    ##probability predictions
    y_prob = lr_model.predict_proba(X_test)[:,1]
    # threshold tuning
    y_pred = (y_prob > 0.8).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.log_model(lr_model, name="logistic_regression_model")

    if f1 > best_f1:
        best_f1 = f1
        best_model = lr_model
        best_model_name = "LogisticRegression"

print(f"Best Model: {best_model_name}")
print(f"Best F1 Score: {best_f1}")

import os
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_fraud_model.pkl")

print("Best model saved successfully in models/best_fraud_model.pkl")