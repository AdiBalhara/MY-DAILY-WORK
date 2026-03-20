"""
predict.py
~~~~~~~~~~
Inference helpers for both batch and single-sample prediction.

``load_artefacts``     — load model + vectoriser from disk
``predict_dataframe``  — run inference on a raw DataFrame
``predict_single``     — convenience wrapper for one title + description
"""

from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

from src.config import BEST_MODEL_PATH, VECTORIZER_PATH, TITLE_COL, DESC_COL
from src.feature_engineering import transform
from src.preprocessing import preprocess_dataframe
from src.utils import get_logger, load_object

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

def load_artefacts() -> tuple[object, TfidfVectorizer]:
    """
    Load and return ``(model, vectorizer)`` from their default disk paths.

    Raises FileNotFoundError if either artefact is missing (i.e., training
    has not been run yet).
    """
    model      = load_object(BEST_MODEL_PATH)
    vectorizer = load_object(VECTORIZER_PATH)
    return model, vectorizer


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_dataframe(
    df: pd.DataFrame,
    *,
    model: object = None,
    vectorizer: TfidfVectorizer | None = None,
    label_encoder: LabelEncoder | None = None,
) -> pd.DataFrame:
    """
    Run inference on a raw (un-preprocessed) DataFrame.

    Parameters
    ----------
    df            : DataFrame with at least TITLE_COL and DESC_COL columns.
    model         : Fitted classifier.  Loaded from disk when None.
    vectorizer    : Fitted TF-IDF vectoriser.  Loaded from disk when None.
    label_encoder : sklearn LabelEncoder to decode integer predictions back to
                    genre strings.  When None, raw integer indices are stored.

    Returns
    -------
    A copy of *df* with an extra ``predicted_genre`` column.
    """
    if model is None or vectorizer is None:
        model, vectorizer = load_artefacts()

    df_out   = df.copy()
    df_clean = preprocess_dataframe(df_out)
    X        = transform(df_clean["clean_text"], vectorizer)
    y_pred   = model.predict(X)

    if label_encoder is not None:
        df_out["predicted_genre"] = label_encoder.inverse_transform(y_pred)
    else:
        df_out["predicted_genre"] = y_pred

    logger.info("Predicted %d samples.", len(df_out))
    return df_out


# ---------------------------------------------------------------------------
# Single-sample prediction
# ---------------------------------------------------------------------------

def predict_single(
    title: str,
    description: str,
    *,
    model: object = None,
    vectorizer: TfidfVectorizer | None = None,
    label_encoder: LabelEncoder | None = None,
) -> str:
    """
    Predict the genre for one movie given its title and plot synopsis.

    Parameters
    ----------
    title       : Movie title string.
    description : Plot synopsis string.
    model / vectorizer / label_encoder : Same as ``predict_dataframe``.

    Returns
    -------
    Predicted genre string (or integer index when no encoder is provided).
    """
    row    = pd.DataFrame([{TITLE_COL: title, DESC_COL: description}])
    result = predict_dataframe(
        row,
        model         = model,
        vectorizer    = vectorizer,
        label_encoder = label_encoder,
    )
    return result["predicted_genre"].iloc[0]
