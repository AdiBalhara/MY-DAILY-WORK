"""
train.py
~~~~~~~~
Trains three classifiers in separate MLflow runs, compares them by
CV accuracy, and saves the best model to disk.

Pattern
-------
  setup_mlflow()               <- set URI + experiment ONCE here
  for each model:
      mlflow.start_run(run_name=...)
          log params
          stratified CV  ->  log cv metrics
          full fit
          log sklearn artefact
  pick best by CV accuracy
  save best_model.pkl

Classifiers
-----------
  1. Logistic Regression
  2. LinearSVC
  3. Random Forest
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mlflow
import mlflow.sklearn
from scipy.sparse import spmatrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.config import (
    BEST_MODEL_PATH, CV_FOLDS,
    LR_C, LR_MAX_ITER, LR_MULTI_CLASS, LR_SOLVER,
    MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI,
    MODEL_DIR, RANDOM_STATE,
    RF_MAX_DEPTH, RF_N_ESTIMATORS,
    SVM_C, SVM_MAX_ITER,
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_NGRAM_RANGE,
    TEST_SIZE,
)
from src.utils import ensure_dirs, get_logger, save_object

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# MLflow setup  (called once at the top of train())
# ---------------------------------------------------------------------------

def _setup_mlflow() -> None:
    """
    Point MLflow at the SQLite backend and select the experiment.
    Calling this inside train.py (not just main.py) ensures tracking
    works whether you run main.py OR call train() directly in a notebook.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info("MLflow URI        : %s", MLFLOW_TRACKING_URI)
    logger.info("MLflow experiment : %s", MLFLOW_EXPERIMENT)


# ---------------------------------------------------------------------------
# Label encoding
# ---------------------------------------------------------------------------

def encode_labels(series) -> tuple[np.ndarray, LabelEncoder]:
    """
    Encode a Series of genre strings to a contiguous int32 array.

    Returns
    -------
    y  : int32 ndarray of shape (n_samples,)
    le : fitted LabelEncoder — keep it to decode predictions later
    """
    le = LabelEncoder()
    y  = le.fit_transform(series.str.strip()).astype("int32")
    logger.info("Labels: %d classes encoded.", len(le.classes_))
    return y, le


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cv_score(model, X: spmatrix, y: np.ndarray) -> tuple[float, float]:
    """Run stratified CV and return (mean_accuracy, std_accuracy)."""
    cv     = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def _run_model(
    model,
    run_name: str,
    params: dict,
    X: spmatrix,
    y: np.ndarray,
    artifact_name: str,
) -> tuple[object, float]:
    """
    Train one model inside its own top-level MLflow run.

    Steps
    -----
    1. Log hyper-parameters + shared TF-IDF settings.
    2. Stratified CV -> log cv_accuracy_mean / std.
    3. Fit on full (X, y).
    4. Log sklearn artefact.

    Returns
    -------
    (fitted_model, cv_accuracy_mean)
    """
    # nested=False  ->  each model gets its own TOP-LEVEL run, fully visible
    # in the MLflow UI as a separate row (not buried under a parent run).
    mlflow.end_run()   # close any stale run left by a previous crash
    with mlflow.start_run(run_name=run_name, nested=False):

        # 1. Params
        mlflow.log_params({
            **params,
            "cv_folds":           CV_FOLDS,
            "test_size":          TEST_SIZE,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_ngram_range":  str(TFIDF_NGRAM_RANGE),
            "tfidf_min_df":       TFIDF_MIN_DF,
        })

        # 2. Cross-validation
        logger.info("[%s]  Running %d-fold CV ...", run_name, CV_FOLDS)
        cv_mean, cv_std = _cv_score(model, X, y)
        logger.info("[%s]  CV accuracy: %.4f +/- %.4f", run_name, cv_mean, cv_std)
        mlflow.log_metrics({
            "cv_accuracy_mean": cv_mean,
            "cv_accuracy_std":  cv_std,
        })

        # 3. Full fit
        logger.info("[%s]  Fitting on %d samples ...", run_name, X.shape[0])
        model.fit(X, y)

        # 4. Log model artefact
        mlflow.sklearn.log_model(model, artifact_path=artifact_name)
        logger.info("[%s]  Logged to MLflow.", run_name)

    return model, cv_mean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train(X: spmatrix, y: np.ndarray) -> object:
    """
    Train all three classifiers, each in its own MLflow run.
    Compare by CV accuracy and persist the winner as best_model.pkl.

    Parameters
    ----------
    X : Sparse TF-IDF feature matrix  (n_samples x n_features).
    y : Encoded integer label array   (n_samples,).

    Returns
    -------
    The best fitted estimator.
    """
    ensure_dirs(MODEL_DIR)

    # Point MLflow at the local SQLite store — safe to call multiple times
    _setup_mlflow()

    best_model      = None
    best_score      = 0.0
    best_model_name = ""

    # ------------------------------------------------------------------ #
    # 1. Logistic Regression                                               #
    # ------------------------------------------------------------------ #
    lr_model, lr_score = _run_model(
        model = LogisticRegression(
            C            = LR_C,
            max_iter     = LR_MAX_ITER,
            solver       = LR_SOLVER,
            multi_class  = LR_MULTI_CLASS,
            random_state = RANDOM_STATE,
            n_jobs       = -1,
        ),
        run_name      = "Logistic Regression",
        params        = {
            "model":       "LogisticRegression",
            "lr_C":        LR_C,
            "lr_solver":   LR_SOLVER,
            "lr_max_iter": LR_MAX_ITER,
        },
        X             = X,
        y             = y,
        artifact_name = "logistic_regression_model",
    )
    print(f"\nLogistic Regression  — CV accuracy: {lr_score:.4f}")
    if lr_score > best_score:
        best_score, best_model, best_model_name = lr_score, lr_model, "Logistic Regression"

    # ------------------------------------------------------------------ #
    # 2. LinearSVC                                                         #
    # ------------------------------------------------------------------ #
    svm_model, svm_score = _run_model(
        model = LinearSVC(
            C            = SVM_C,
            max_iter     = SVM_MAX_ITER,
            random_state = RANDOM_STATE,
        ),
        run_name      = "LinearSVC",
        params        = {
            "model":        "LinearSVC",
            "svm_C":        SVM_C,
            "svm_max_iter": SVM_MAX_ITER,
        },
        X             = X,
        y             = y,
        artifact_name = "linearsvc_model",
    )
    print(f"LinearSVC            — CV accuracy: {svm_score:.4f}")
    if svm_score > best_score:
        best_score, best_model, best_model_name = svm_score, svm_model, "LinearSVC"

    # ------------------------------------------------------------------ #
    # 3. Random Forest                                                     #
    # ------------------------------------------------------------------ #
    rf_model, rf_score = _run_model(
        model = RandomForestClassifier(
            n_estimators = RF_N_ESTIMATORS,
            max_depth    = RF_MAX_DEPTH,
            random_state = RANDOM_STATE,
            n_jobs       = -1,
        ),
        run_name      = "Random Forest",
        params        = {
            "model":           "RandomForest",
            "rf_n_estimators": RF_N_ESTIMATORS,
            "rf_max_depth":    str(RF_MAX_DEPTH),
        },
        X             = X,
        y             = y,
        artifact_name = "random_forest_model",
    )
    print(f"Random Forest        — CV accuracy: {rf_score:.4f}")
    if rf_score > best_score:
        best_score, best_model, best_model_name = rf_score, rf_model, "Random Forest"

    # ------------------------------------------------------------------ #
    # Save best                                                            #
    # ------------------------------------------------------------------ #
    print(f"\nBest model   : {best_model_name}")
    print(f"Best CV acc  : {best_score:.4f}")

    save_object(best_model, BEST_MODEL_PATH)
    print(f"Best model saved -> {BEST_MODEL_PATH}")

    return best_model
