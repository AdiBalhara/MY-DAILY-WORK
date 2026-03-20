import os
import sys
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import load_data, engineer_features
from config import (
    DATA_PATH, TARGET_COL, DROP_COLS, CAT_FEATURES,
    NUM_FEATURES, TEST_SIZE, RANDOM_STATE, MODEL_DIR
)


# ── Load dataset ───────────────────────────────────────────────────────────────

df = load_data(DATA_PATH)

# Drop non-predictive columns
df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

# Feature engineering
df = engineer_features(df)

# Encode categorical columns
extra_cats = ["AgeBucket"]
for col in CAT_FEATURES + extra_cats:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Split features and target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Scale numerical features
num_cols = [c for c in NUM_FEATURES + [
    "Balance_Salary_Ratio", "Products_Per_Tenure", "EngagementScore"
] if c in X_train.columns]

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

# Save scaler
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("Scaler saved.")

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=RANDOM_STATE)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
print(f"After SMOTE — class counts: {dict(pd.Series(y_resampled).value_counts())}")


# ── MLflow experiment setup ────────────────────────────────────────────────────

# Use SQLite backend — works on all platforms including Windows
# (file:// URIs with backslashes are not supported on Windows)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path  = os.path.join(BASE_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{db_path}")

mlflow.set_experiment("Churn Prediction Experiment")

# Variables to track the best model
best_model      = None
best_f1         = 0
best_model_name = ""


# ── LOGISTIC REGRESSION ────────────────────────────────────────────────────────

with mlflow.start_run(run_name="Logistic Regression"):

    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs"
    )
    lr_model.fit(X_resampled, y_resampled)

    y_prob = lr_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model",        "LogisticRegression")
    mlflow.log_param("C",            1.0)
    mlflow.log_param("max_iter",     1000)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("solver",       "lbfgs")

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("roc_auc",   roc_auc)

    mlflow.sklearn.log_model(lr_model, name="logistic_regression_model")

    print(f"\nLogistic Regression — F1: {f1:.4f}  ROC-AUC: {roc_auc:.4f}")

    if f1 > best_f1:
        best_f1         = f1
        best_model      = lr_model
        best_model_name = "Logistic Regression"


# ── RANDOM FOREST ──────────────────────────────────────────────────────────────

with mlflow.start_run(run_name="Random Forest"):

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf_model.fit(X_resampled, y_resampled)

    y_prob = rf_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model",        "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth",    10)
    mlflow.log_param("class_weight", "balanced")

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("roc_auc",   roc_auc)

    mlflow.sklearn.log_model(rf_model, name="random_forest_model")

    print(f"Random Forest      — F1: {f1:.4f}  ROC-AUC: {roc_auc:.4f}")

    if f1 > best_f1:
        best_f1         = f1
        best_model      = rf_model
        best_model_name = "Random Forest"


# ── GRADIENT BOOSTING ──────────────────────────────────────────────────────────

with mlflow.start_run(run_name="Gradient Boosting"):

    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    gb_model.fit(X_resampled, y_resampled)

    y_prob = gb_model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    mlflow.log_param("model",         "GradientBoosting")
    mlflow.log_param("n_estimators",  100)
    mlflow.log_param("learning_rate", 0.1)
    mlflow.log_param("max_depth",     3)
    mlflow.log_param("subsample",     0.8)

    mlflow.log_metric("accuracy",  accuracy)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("roc_auc",   roc_auc)

    mlflow.sklearn.log_model(gb_model, name="gradient_boosting_model")

    print(f"Gradient Boosting  — F1: {f1:.4f}  ROC-AUC: {roc_auc:.4f}")

    if f1 > best_f1:
        best_f1         = f1
        best_model      = gb_model
        best_model_name = "Gradient Boosting"


# ── Save best model ────────────────────────────────────────────────────────────

print(f"\nBest Model   : {best_model_name}")
print(f"Best F1 Score: {best_f1:.4f}")

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_churn_model.pkl"))
print(f"Best model saved successfully in models/best_churn_model.pkl")
