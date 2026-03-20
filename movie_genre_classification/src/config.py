"""
config.py
~~~~~~~~~
Single source of truth for every path, constant, and hyper-parameter used
across the pipeline.  Nothing else should hard-code strings or magic numbers.
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data", "raw-data")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR    = os.path.join(REPORT_DIR, "figures")

TRAIN_PATH    = os.path.join(DATA_DIR, "train_data.txt")
TEST_PATH     = os.path.join(DATA_DIR, "test_data.txt")
SOLUTION_PATH = os.path.join(DATA_DIR, "test_data_solution.txt")

MODEL_PATH      = os.path.join(MODEL_DIR, "model.pkl")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
RESULTS_PATH    = os.path.join(REPORT_DIR, "results.txt")

# ---------------------------------------------------------------------------
# Data column names
# ---------------------------------------------------------------------------
ID_COL    = "id"
TITLE_COL = "title"
GENRE_COL = "genre"
DESC_COL  = "description"

GENRES: list[str] = [
    "action", "adult", "adventure", "animation", "biography", "comedy",
    "crime", "documentary", "drama", "family", "fantasy", "game-show",
    "history", "horror", "music", "musical", "mystery", "news",
    "reality-tv", "romance", "sci-fi", "short", "sport", "talk-show",
    "thriller", "war", "western",
]

# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------
STOPWORDS_LANG   = "english"
MIN_TOKEN_LENGTH = 2
TITLE_REPEAT     = 2

# ---------------------------------------------------------------------------
# TF-IDF vectorisation
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 30_000
TFIDF_NGRAM_RANGE  = (1, 2)
TFIDF_MIN_DF       = 2
TFIDF_SUBLINEAR_TF = True

# ---------------------------------------------------------------------------
# Model hyper-parameters
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

LR_C           = 5.0
LR_MAX_ITER    = 1_000
LR_SOLVER      = "lbfgs"
LR_MULTI_CLASS = "auto"

SVM_C        = 1.0
SVM_MAX_ITER = 1_000

RF_N_ESTIMATORS = 300
RF_MAX_DEPTH    = None

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
# Local SQLite backend — NO server needed.
# Runs are stored in mlflow.db at the project root.
# View UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
MLFLOW_TRACKING_URI = f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
MLFLOW_EXPERIMENT   = "movie-genre-classification"

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
VERBOSE = True
