"""
data_preprocessing.py — Data loading, cleaning, and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (DATA_PATH, TARGET_COL, DROP_COLS,
                    CAT_FEATURES, NUM_FEATURES, TEST_SIZE, RANDOM_STATE)


# ── Load ───────────────────────────────────────────────────────────────────────

def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"[DATA] Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── EDA helpers ────────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> None:
    """Print a quick overview of the dataset."""
    print("\n── Shape ──────────────────────────────")
    print(df.shape)
    print("\n── Dtypes ─────────────────────────────")
    print(df.dtypes)
    print("\n── Missing values ─────────────────────")
    print(df.isnull().sum())
    print("\n── Target distribution ────────────────")
    counts = df[TARGET_COL].value_counts()
    print(counts)
    churn_rate = counts[1] / counts.sum() * 100
    print(f"Churn rate: {churn_rate:.1f}%")


# ── Feature engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features to improve model signal."""
    df = df.copy()

    # Balance-to-salary ratio
    df["Balance_Salary_Ratio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    # Products-per-tenure
    df["Products_Per_Tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # Age buckets (young / middle / senior)
    df["AgeBucket"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Young", "Middle", "Senior", "Elder"]
    ).astype(str)

    # Zero-balance flag
    df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

    # Engagement score: active × has card × num products
    df["EngagementScore"] = (
        df["IsActiveMember"] * df["HasCrCard"] * df["NumOfProducts"]
    )

    return df


# ── Preprocessing pipeline ─────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    apply_smote: bool = True,
    scaler: StandardScaler = None,
    fit_scaler: bool = True,
):
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler, feature_names
    """
    df = df.copy()

    # 1. Drop non-predictive columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # 2. Feature engineering
    df = engineer_features(df)

    # 3. Encode categoricals
    extra_cats = ["AgeBucket"]
    all_cats   = CAT_FEATURES + extra_cats
    for col in all_cats:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # 4. Separate features / target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = X.columns.tolist()

    # 5. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 6. Scale numerical features
    num_cols = [c for c in NUM_FEATURES + [
        "Balance_Salary_Ratio", "Products_Per_Tenure", "EngagementScore"
    ] if c in X_train.columns]

    if fit_scaler:
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    else:
        assert scaler is not None, "Provide a fitted scaler when fit_scaler=False"

    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 7. Handle class imbalance with SMOTE (training set only)
    if apply_smote:
        sm = SMOTE(random_state=RANDOM_STATE)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(f"[SMOTE] After resampling: {dict(pd.Series(y_train).value_counts())}")

    print(f"[SPLIT] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test, scaler, feature_names


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data()
    summarise(df)
    X_tr, X_te, y_tr, y_te, sc, feats = preprocess(df)
    print(f"\nFeatures ({len(feats)}): {feats}")
