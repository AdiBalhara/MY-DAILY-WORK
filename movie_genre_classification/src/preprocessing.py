"""
preprocessing.py
~~~~~~~~~~~~~~~~
Text cleaning pipeline:

    raw text
        → lower-case
        → strip HTML tags
        → remove punctuation & digits
        → tokenise (whitespace split)
        → remove stop-words and very short tokens
        → (optional) Porter stemming
        → rejoin as a single clean string

The ``preprocess_dataframe`` function combines title (repeated for emphasis)
and description, then applies the cleaning pipeline to produce a single
``clean_text`` column consumed by the vectoriser.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Path fix — makes `from src.X import Y` work whether you run this file
# directly (`python src/preprocessing.py`) or via the project root
# (`python main.py`).  Must come BEFORE any `from src.*` imports.
# ---------------------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------------------------------------------------------------------------

import re
import string

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from src.config import (
    STOPWORDS_LANG, MIN_TOKEN_LENGTH, TITLE_REPEAT,
    TITLE_COL, DESC_COL,
)
from src.utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# NLTK one-time resource downloads
# ---------------------------------------------------------------------------
for _resource, _kind in [("stopwords", "corpora"), ("punkt", "tokenizers")]:
    try:
        nltk.data.find(f"{_kind}/{_resource}")
    except LookupError:
        logger.info("Downloading NLTK resource: %s", _resource)
        nltk.download(_resource, quiet=True)

# ---------------------------------------------------------------------------
# Module-level singletons (built once, reused for every row)
# ---------------------------------------------------------------------------
_STOP_WORDS  = set(stopwords.words(STOPWORDS_LANG))
_STEMMER     = PorterStemmer()
_RE_HTML     = re.compile(r"<[^>]+>")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_RE_DIGITS   = re.compile(r"\d+")
_RE_SPACES   = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Core cleaning function
# ---------------------------------------------------------------------------

def clean_text(text: str, *, stem: bool = False) -> str:
    """
    Apply the full cleaning pipeline to a single string.

    Parameters
    ----------
    text : Raw input string (title, description, or a combination).
    stem : When True, apply Porter stemming after stop-word removal.

    Returns
    -------
    A cleaned, space-joined string.  Returns "" for non-string input.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Lower-case
    text = text.lower()

    # 2. Strip HTML tags
    text = _RE_HTML.sub(" ", text)

    # 3. Remove punctuation
    text = text.translate(_PUNCT_TABLE)

    # 4. Remove digit sequences
    text = _RE_DIGITS.sub(" ", text)

    # 5. Normalise whitespace
    text = _RE_SPACES.sub(" ", text).strip()

    # 6. Tokenise + filter
    tokens = [
        t for t in text.split()
        if len(t) >= MIN_TOKEN_LENGTH and t not in _STOP_WORDS
    ]

    # 7. Optional stemming
    if stem:
        tokens = [_STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# DataFrame-level preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame, *, stem: bool = False) -> pd.DataFrame:
    """
    Add a ``clean_text`` column to *df* and return the modified copy.

    The column is built by concatenating:
        (title × TITLE_REPEAT) + description
    then passing the combined text through ``clean_text``.

    Parameters
    ----------
    df   : DataFrame containing at least TITLE_COL and DESC_COL columns.
    stem : Forwarded to ``clean_text``.

    Returns
    -------
    A new DataFrame with the original columns plus ``clean_text``.
    """
    logger.info("Pre-processing %d rows  (stem=%s) …", len(df), stem)

    df = df.copy()

    title = df[TITLE_COL].fillna("").astype(str)
    desc  = df[DESC_COL].fillna("").astype(str)

    # Repeat title to increase its term frequency relative to the description
    combined = (title + " ") * TITLE_REPEAT + desc

    df["clean_text"] = combined.apply(lambda t: clean_text(t, stem=stem))

    empty_mask = df["clean_text"].str.len() == 0
    if empty_mask.any():
        logger.warning("  %d rows produced empty clean_text", empty_mask.sum())

    logger.info("Pre-processing complete.")
    return df
