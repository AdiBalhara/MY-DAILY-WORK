"""
data_loader.py
~~~~~~~~~~~~~~
Responsible for reading the raw pipe-delimited data files into DataFrames and
performing light structural validation.

Expected file format  (no header row, delimiter " ::: "):
    ID ::: TITLE ::: GENRE ::: DESCRIPTION
The test file omits the GENRE column:
    ID ::: TITLE ::: DESCRIPTION
"""

from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from src.config import (
    TRAIN_PATH, TEST_PATH, SOLUTION_PATH,
    ID_COL, TITLE_COL, GENRE_COL, DESC_COL,
)
from src.utils import get_logger

logger = get_logger(__name__)

_DELIMITER   = " ::: "
_TRAIN_COLS  = [ID_COL, TITLE_COL, GENRE_COL, DESC_COL]
_TEST_COLS   = [ID_COL, TITLE_COL, DESC_COL]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read(path: str, columns: list[str]) -> pd.DataFrame:
    """Read one raw file into a DataFrame and log basic stats."""
    logger.info("Reading  %s", path)
    df = pd.read_csv(
        path,
        sep         = _DELIMITER,
        header      = None,
        names       = columns,
        engine      = "python",
        on_bad_lines= "skip",
    )
    logger.info("  rows=%d  cols=%s", len(df), list(df.columns))
    return df


def _validate(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Drop rows whose critical text columns are missing and warn the caller.
    Also strips leading/trailing whitespace from all string columns.
    """
    text_cols = [c for c in [TITLE_COL, DESC_COL, GENRE_COL] if c in df.columns]
    before    = len(df)

    # Strip whitespace
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Drop rows with empty title or description
    df = df.dropna(subset=[TITLE_COL, DESC_COL])
    df = df[df[TITLE_COL].str.len() > 0]
    df = df[df[DESC_COL].str.len() > 0]

    dropped = before - len(df)
    if dropped:
        logger.warning("%s: dropped %d malformed rows", name, dropped)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_train() -> pd.DataFrame:
    """Return the labelled training DataFrame."""
    df = _read(TRAIN_PATH, _TRAIN_COLS)
    return _validate(df, "train")


def load_test() -> pd.DataFrame:
    """Return the unlabelled test DataFrame (no GENRE column)."""
    df = _read(TEST_PATH, _TEST_COLS)
    return _validate(df, "test")


def load_test_solution() -> pd.DataFrame:
    """Return the ground-truth labels for the test split."""
    df = _read(SOLUTION_PATH, _TRAIN_COLS)
    return _validate(df, "solution")


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning (train, test, solution)."""
    return load_train(), load_test(), load_test_solution()
