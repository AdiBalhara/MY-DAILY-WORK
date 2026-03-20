
from __future__ import annotations
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

# ---------------------------------------------------------------------------
# Path fix — makes `from src.*` work when uvicorn is started from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config import (
    BEST_MODEL_PATH, GENRES, RESULTS_PATH,
    VECTORIZER_PATH,
)
from src.data_loader import load_train
from src.predict import load_artefacts, predict_dataframe, predict_single
from src.train import encode_labels
from src.utils import get_logger

logger = get_logger("api")


# ---------------------------------------------------------------------------
# App-level state  (model loaded once at startup, reused for every request)
# ---------------------------------------------------------------------------

class _ModelStore:
    """Simple container that holds the loaded artefacts."""
    model         = None
    vectorizer    = None
    label_encoder = None
    loaded        = False
    load_error    = ""


store = _ModelStore()


def _load_model() -> None:
    """Load model, vectoriser, and label encoder into the global store."""
    try:
        store.model, store.vectorizer = load_artefacts()
        df_train              = load_train()
        _, store.label_encoder = encode_labels(df_train["genre"])
        store.loaded          = True
        store.load_error      = ""
        logger.info("Model artefacts loaded successfully.")
    except FileNotFoundError as exc:
        store.loaded     = False
        store.load_error = str(exc)
        logger.error("Failed to load model: %s", exc)


# ---------------------------------------------------------------------------
# Lifespan  (replaces deprecated @app.on_event("startup"))
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts when the server starts."""
    logger.info("Starting up — loading model artefacts …")
    _load_model()
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title       = "Movie Genre Classification API",
    description = (
        "Predict the genre of a movie from its title and plot synopsis.\n\n"
        "Run `python main.py --mode train` first to generate the model artefacts."
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*", "null"],   # "null" covers file:// origin
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class MovieRequest(BaseModel):
    title: str = Field(
        ...,
        min_length = 1,
        max_length = 500,
        examples   = ["Inception"],
        description = "Movie title.",
    )
    description: str = Field(
        ...,
        min_length = 10,
        max_length = 5000,
        examples   = ["A thief who steals corporate secrets through dream-sharing technology."],
        description = "Plot synopsis.",
    )

    @field_validator("title", "description", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class MovieBatchRequest(BaseModel):
    movies: list[MovieRequest] = Field(
        ...,
        min_length  = 1,
        max_length  = 100,
        description = "List of movies to classify (max 100).",
    )


class PredictionResponse(BaseModel):
    title           : str
    description     : str
    predicted_genre : str
    confidence      : str = "n/a"


class BatchPredictionResponse(BaseModel):
    results      : list[PredictionResponse]
    total        : int
    elapsed_ms   : float


class HealthResponse(BaseModel):
    status       : str
    model_loaded : bool
    model_path   : str
    vectorizer_path: str
    error        : Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_loaded    : bool
    model_path      : str
    vectorizer_path : str
    supported_genres: list[str]
    total_genres    : int
    load_error      : Optional[str] = None


# ---------------------------------------------------------------------------
# Dependency — raise 503 if artefacts are not loaded
# ---------------------------------------------------------------------------

def _require_model() -> None:
    if not store.loaded:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = (
                f"Model not loaded. {store.load_error} "
                "Run `python main.py --mode train` first."
            ),
        )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

# Mount the static folder so index.html + assets are served
_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get(
    "/",
    summary  = "UI",
    tags     = ["Health"],
    response_class = FileResponse,
)
def root():
    """Serve the main HTML UI."""
    return FileResponse(os.path.join(_STATIC, "index.html"))


@app.get(
    "/health",
    response_model = HealthResponse,
    summary        = "Health check",
    tags           = ["Health"],
)
def health():
    """Detailed health check including model load status."""
    return HealthResponse(
        status          = "ok" if store.loaded else "degraded",
        model_loaded    = store.loaded,
        model_path      = BEST_MODEL_PATH,
        vectorizer_path = VECTORIZER_PATH,
        error           = store.load_error or None,
    )


@app.post(
    "/predict",
    response_model = PredictionResponse,
    summary        = "Predict genre for a single movie",
    tags           = ["Prediction"],
    status_code    = status.HTTP_200_OK,
)
def predict(request: MovieRequest):
    """
    Predict the genre for **one** movie given its title and plot synopsis.

    Example request body:
    ```json
    {
        "title": "The Dark Knight",
        "description": "Batman faces the Joker, a criminal mastermind who plunges Gotham into chaos."
    }
    ```
    """
    _require_model()

    try:
        genre = predict_single(
            request.title,
            request.description,
            model         = store.model,
            vectorizer    = store.vectorizer,
            label_encoder = store.label_encoder,
        )
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Prediction failed: {exc}",
        )

    return PredictionResponse(
        title           = request.title,
        description     = request.description,
        predicted_genre = str(genre),
    )


@app.post(
    "/predict/batch",
    response_model = BatchPredictionResponse,
    summary        = "Predict genres for multiple movies",
    tags           = ["Prediction"],
    status_code    = status.HTTP_200_OK,
)
def predict_batch(request: MovieBatchRequest):
    """
    Predict genres for a **list** of movies in one call (max 100).

    Example request body:
    ```json
    {
        "movies": [
            {"title": "Titanic",    "description": "A love story aboard a doomed ship."},
            {"title": "The Matrix", "description": "A hacker discovers reality is a simulation."}
        ]
    }
    ```
    """
    _require_model()

    t0 = time.perf_counter()

    try:
        df = pd.DataFrame([
            {"title": m.title, "description": m.description}
            for m in request.movies
        ])
        df_result = predict_dataframe(
            df,
            model         = store.model,
            vectorizer    = store.vectorizer,
            label_encoder = store.label_encoder,
        )
    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Batch prediction failed: {exc}",
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = [
        PredictionResponse(
            title           = row["title"],
            description     = row["description"],
            predicted_genre = str(row["predicted_genre"]),
        )
        for _, row in df_result.iterrows()
    ]

    return BatchPredictionResponse(
        results    = results,
        total      = len(results),
        elapsed_ms = round(elapsed_ms, 2),
    )


@app.get(
    "/genres",
    summary = "List all supported genres",
    tags    = ["Info"],
)
def list_genres():
    """Return all 27 genre classes the model can predict."""
    return {
        "genres": sorted(GENRES),
        "total" : len(GENRES),
    }


@app.get(
    "/model/info",
    response_model = ModelInfoResponse,
    summary        = "Model artefact info",
    tags           = ["Info"],
)
def model_info():
    """Return model load status, artefact paths, and supported genres."""
    return ModelInfoResponse(
        model_loaded     = store.loaded,
        model_path       = BEST_MODEL_PATH,
        vectorizer_path  = VECTORIZER_PATH,
        supported_genres = sorted(GENRES),
        total_genres     = len(GENRES),
        load_error       = store.load_error or None,
    )


@app.post(
    "/model/reload",
    summary = "Reload model artefacts from disk",
    tags    = ["Info"],
)
def reload_model():
    """
    Hot-reload the model and vectoriser from disk without restarting the server.
    Useful after re-training.
    """
    _load_model()
    if store.loaded:
        return {"status": "ok", "message": "Model reloaded successfully."}
    raise HTTPException(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail      = f"Reload failed: {store.load_error}",
    )
