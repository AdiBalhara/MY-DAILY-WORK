"""
utils.py
~~~~~~~~
Shared helpers: logging, directory creation, object serialisation.
"""

from __future__ import annotations
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
import logging
import joblib

from src.config import VERBOSE


def get_logger(name: str) -> logging.Logger:
    """Return a consistently-formatted logger for the given module name."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level   = logging.INFO if VERBOSE else logging.WARNING
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt     = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                datefmt = "%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def ensure_dirs(*paths: str) -> None:
    """Create every supplied directory (including parents) if it does not exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_object(obj: object, path: str) -> None:
    """
    Persist any Python object to *path* using joblib compression.

    The parent directory is created automatically.
    """
    ensure_dirs(os.path.dirname(path))
    joblib.dump(obj, path, compress=3)
    get_logger(__name__).info("Saved  → %s", path)


def load_object(path: str) -> object:
    """
    Deserialise and return the object stored at *path*.

    Raises
    ------
    FileNotFoundError
        When the artefact does not exist on disk.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artefact not found: {path}\n"
            "Run  python main.py --mode train  first."
        )
    obj = joblib.load(path)
    get_logger(__name__).info("Loaded ← %s", path)
    return obj
