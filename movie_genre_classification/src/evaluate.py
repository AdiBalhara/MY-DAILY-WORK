"""
evaluate.py
~~~~~~~~~~~
Compute classification metrics, log them to MLflow, generate figures, and
write a plain-text report.

Public surface
--------------
evaluate()               — core metrics + MLflow logging
plot_confusion_matrix()  — heatmap saved to reports/figures/ + MLflow
plot_genre_distribution()— bar chart saved to reports/figures/ + MLflow
"""

from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from src.config import FIG_DIR, GENRE_COL, RESULTS_PATH
from src.utils import ensure_dirs, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder | None = None,
    *,
    save: bool = True,
) -> dict:
    """
    Compute accuracy, F1 (macro + weighted), precision, and recall.
    Log all scalars to the active MLflow run and optionally save a text report.

    Parameters
    ----------
    y_true        : Ground-truth encoded labels.
    y_pred        : Predicted encoded labels.
    label_encoder : Used to decode integer indices back to genre strings.
    save          : When True, write results.txt and log it to MLflow.

    Returns
    -------
    Dict with keys: accuracy, macro_f1, weighted_f1, report (str).
    """
    ensure_dirs(FIG_DIR)

    class_names = label_encoder.classes_ if label_encoder is not None else None

    acc          = float(accuracy_score(y_true, y_pred))
    macro_f1     = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    weighted_f1  = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    precision    = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall       = float(recall_score(y_true, y_pred,    average="weighted", zero_division=0))
    report_str   = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    logger.info("─" * 55)
    logger.info("Accuracy         : %.4f", acc)
    logger.info("Macro F1         : %.4f", macro_f1)
    logger.info("Weighted F1      : %.4f", weighted_f1)
    logger.info("Weighted Precision: %.4f", precision)
    logger.info("Weighted Recall  : %.4f", recall)
    logger.info("─" * 55)
    logger.info("\n%s", report_str)

    # Log scalars to MLflow
    mlflow.log_metrics({
        "test_accuracy":           acc,
        "test_macro_f1":           macro_f1,
        "test_weighted_f1":        weighted_f1,
        "test_weighted_precision": precision,
        "test_weighted_recall":    recall,
    })
    logger.info("MLflow: test metrics logged.")

    if save:
        ensure_dirs(os.path.dirname(RESULTS_PATH))
        with open(RESULTS_PATH, "w", encoding="utf-8") as fh:
            fh.write(f"Accuracy          : {acc:.4f}\n")
            fh.write(f"Macro F1          : {macro_f1:.4f}\n")
            fh.write(f"Weighted F1       : {weighted_f1:.4f}\n")
            fh.write(f"Weighted Precision: {precision:.4f}\n")
            fh.write(f"Weighted Recall   : {recall:.4f}\n\n")
            fh.write(report_str)
        mlflow.log_artifact(RESULTS_PATH, artifact_path="reports")
        logger.info("Results saved → %s (also logged to MLflow)", RESULTS_PATH)

    return {
        "accuracy":    acc,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "report":      report_str,
    }


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder | None = None,
    top_n: int = 20,
) -> None:
    """
    Plot a (truncated) confusion matrix and log the figure to MLflow.

    When the number of classes exceeds *top_n*, only the *top_n* classes with
    the highest support are shown to keep the figure readable.

    Parameters
    ----------
    y_true        : Ground-truth labels.
    y_pred        : Predicted labels.
    label_encoder : To decode integer indices.
    top_n         : Maximum number of classes to display.
    """
    ensure_dirs(FIG_DIR)
    class_names = label_encoder.classes_.copy() if label_encoder is not None else None

    # Restrict to top-N classes by support
    if class_names is not None and len(class_names) > top_n:
        counts      = np.bincount(y_true, minlength=len(class_names))
        top_indices = np.argsort(counts)[::-1][:top_n]
        mask        = np.isin(y_true, top_indices)
        y_true_plot = y_true[mask]
        y_pred_plot = y_pred[mask]
        class_names = class_names[top_indices]
    else:
        y_true_plot, y_pred_plot = y_true, y_pred

    cm   = confusion_matrix(y_true_plot, y_pred_plot)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot       = True,
        fmt         = "d",
        cmap        = "Blues",
        xticklabels = class_names if class_names is not None else "auto",
        yticklabels = class_names if class_names is not None else "auto",
        linewidths  = 0.4,
        ax          = ax,
    )
    ax.set_title("Confusion Matrix", fontsize=15, pad=12)
    ax.set_xlabel("Predicted genre", fontsize=12)
    ax.set_ylabel("True genre",      fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,             fontsize=9)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="figures")
    logger.info("Confusion matrix saved → %s", path)


# ---------------------------------------------------------------------------
# Genre distribution
# ---------------------------------------------------------------------------

def plot_genre_distribution(df: pd.DataFrame) -> None:
    """
    Bar chart of genre frequencies in *df* (typically the training split).
    Saved locally and logged to the active MLflow run.
    """
    ensure_dirs(FIG_DIR)
    counts = df[GENRE_COL].str.strip().value_counts()

    fig, ax = plt.subplots(figsize=(14, 5))
    counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white", width=0.75)
    ax.set_title("Genre Distribution — Training Set", fontsize=14, pad=10)
    ax.set_xlabel("Genre",  fontsize=11)
    ax.set_ylabel("Samples", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()

    path = os.path.join(FIG_DIR, "genre_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="figures")
    logger.info("Genre distribution saved → %s", path)
