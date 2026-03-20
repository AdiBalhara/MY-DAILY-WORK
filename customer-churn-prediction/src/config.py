"""
config.py — Centralized project configuration
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Churn_Modelling.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
LOG_DIR    = os.path.join(BASE_DIR, "logs")

# ── Data ───────────────────────────────────────────────────────────────────────
TARGET_COL    = "Exited"
DROP_COLS     = ["RowNumber", "CustomerId", "Surname"]   # non-predictive IDs
CAT_FEATURES  = ["Geography", "Gender"]
NUM_FEATURES  = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
]

# ── Modelling ──────────────────────────────────────────────────────────────────
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
CV_FOLDS       = 5
SCORING_METRIC = "roc_auc"       # primary metric for hyper-param search

# ── Model hyper-parameters (default / grid search candidates) ─────────────────
LR_PARAMS = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["lbfgs"],
    "max_iter": [1000],
    "class_weight": ["balanced"],
}

RF_PARAMS = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "class_weight": ["balanced"],
}

GB_PARAMS = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
    "subsample": [0.8, 1.0],
}
