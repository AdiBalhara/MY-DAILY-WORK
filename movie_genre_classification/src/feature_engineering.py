"""
feature_engineering.py
~~~~~~~~~~~~~~~~~~~~~~
Wraps scikit-learn's TfidfVectorizer with project defaults and provides
a clean fit/transform/load interface.

The fitted vectoriser is saved to disk after ``fit_transform`` so that:
  - ``train``    calls ``fit_transform`` once
  - ``evaluate`` and ``predict`` call ``transform`` with the saved vectoriser
"""

from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF, TFIDF_SUBLINEAR_TF,
    VECTORIZER_PATH,
)
from src.utils import get_logger, save_object, load_object

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Vectoriser factory
# ---------------------------------------------------------------------------

def _build_vectorizer() -> TfidfVectorizer:
    """Return a fresh TF-IDF vectoriser configured with project defaults."""
    return TfidfVectorizer(
        max_features  = TFIDF_MAX_FEATURES,
        ngram_range   = TFIDF_NGRAM_RANGE,
        min_df        = TFIDF_MIN_DF,
        sublinear_tf  = TFIDF_SUBLINEAR_TF,
        strip_accents = "unicode",
        analyzer      = "word",
        dtype         = "float32",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_transform(texts: pd.Series) -> tuple[spmatrix, TfidfVectorizer]:
    """
    Fit a new vectoriser on *texts*, transform them, persist the vectoriser,
    and return ``(feature_matrix, vectoriser)``.

    Parameters
    ----------
    texts : Series of pre-processed (cleaned) strings.

    Returns
    -------
    X   : sparse float32 matrix of shape (n_samples, n_features)
    vec : the fitted TfidfVectorizer
    """
    logger.info("Fitting TF-IDF on %d documents …", len(texts))
    vec = _build_vectorizer()
    X   = vec.fit_transform(texts)
    logger.info("Feature matrix : shape=%s  nnz=%d", X.shape, X.nnz)
    save_object(vec, VECTORIZER_PATH)
    return X, vec


def transform(texts: pd.Series,
              vectorizer: TfidfVectorizer | None = None) -> spmatrix:
    """
    Transform *texts* with an already-fitted vectoriser.

    If *vectorizer* is None, the persisted vectoriser at VECTORIZER_PATH is
    loaded automatically.

    Parameters
    ----------
    texts      : Series of pre-processed strings.
    vectorizer : Fitted TfidfVectorizer, or None to load from disk.

    Returns
    -------
    Sparse feature matrix of shape (n_samples, n_features).
    """
    if vectorizer is None:
        vectorizer = load_object(VECTORIZER_PATH)
    return vectorizer.transform(texts)
