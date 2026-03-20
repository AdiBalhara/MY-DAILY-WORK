"""
main.py
~~~~~~~
CLI entry point for the Movie Genre Classification pipeline.

MLflow backend: local SQLite (no server needed to RECORD runs).
To VIEW runs in the browser UI run this in a separate terminal:

    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Then open http://localhost:5000

Usage
-----
    python main.py --mode train
    python main.py --mode evaluate
    python main.py --mode all
    python main.py --mode predict --title "Inception" --description "A thief..."
"""

from __future__ import annotations
import argparse
import sys

import mlflow

from src.config import (
    BEST_MODEL_PATH,
    FIG_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI,
    MODEL_DIR, REPORT_DIR, VECTORIZER_PATH,
)
from src.data_loader import load_test, load_test_solution, load_train
from src.evaluate import evaluate, plot_confusion_matrix, plot_genre_distribution
from src.feature_engineering import fit_transform, transform
from src.predict import load_artefacts, predict_single
from src.preprocessing import preprocess_dataframe
from src.train import encode_labels, train
from src.utils import ensure_dirs, get_logger

logger = get_logger("main")


# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------

def _init_mlflow() -> None:
    """Configure MLflow to use the local SQLite backend."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    logger.info("MLflow URI        : %s", MLFLOW_TRACKING_URI)
    logger.info("MLflow experiment : %s", MLFLOW_EXPERIMENT)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_train() -> tuple:
    """
    Full training pipeline.

    Steps
    -----
    1. Load + validate raw training data.
    2. Plot genre distribution.
    3. Pre-process text.
    4. Encode labels + fit TF-IDF.
    5. Train 3 models (each in its own MLflow run) → save best.

    Returns
    -------
    (best_model, vectorizer, label_encoder)
    """
    ensure_dirs(MODEL_DIR, REPORT_DIR, FIG_DIR)
    _init_mlflow()

    logger.info("=" * 55)
    logger.info("TRAINING")
    logger.info("=" * 55)

    # 1. Load
    df_train = load_train()
    plot_genre_distribution(df_train)

    # 2. Pre-process
    df_clean = preprocess_dataframe(df_train)

    # 3. Labels + features
    y, le  = encode_labels(df_clean["genre"])
    X, vec = fit_transform(df_clean["clean_text"])

    # 4. Train — opens 3 separate MLflow runs internally
    model = train(X, y)

    logger.info("Training complete.")
    logger.info("View runs:  mlflow ui --backend-store-uri %s --port 5000", MLFLOW_TRACKING_URI)
    return model, vec, le


def run_evaluate() -> dict:
    """
    Evaluate the best saved model against the held-out test solution.
    Logs metrics + plots to a dedicated MLflow run.

    Returns
    -------
    Dict with keys: accuracy, macro_f1, weighted_f1, report.
    """
    ensure_dirs(REPORT_DIR, FIG_DIR)
    _init_mlflow()

    logger.info("=" * 55)
    logger.info("EVALUATION")
    logger.info("=" * 55)

    with mlflow.start_run(run_name="Evaluation"):

        model, vec = load_artefacts()

        df_test  = load_test()
        df_sol   = load_test_solution()

        df_clean = preprocess_dataframe(df_test)
        X_test   = transform(df_clean["clean_text"], vec)
        y_pred   = model.predict(X_test)

        # Rebuild encoder (must match training)
        df_train = load_train()
        _, le    = encode_labels(df_train["genre"])

        y_true = le.transform(df_sol["genre"].str.strip())
        y_pred = y_pred[: len(y_true)]

        results = evaluate(y_true, y_pred, label_encoder=le, save=True)
        plot_confusion_matrix(y_true, y_pred, label_encoder=le)

    return results


def run_predict(title: str, description: str) -> str:
    """Predict genre for a single movie — no MLflow run needed."""
    logger.info("=" * 55)
    logger.info("PREDICT")
    logger.info("=" * 55)

    model, vec = load_artefacts()

    df_train = load_train()
    _, le    = encode_labels(df_train["genre"])

    genre = predict_single(
        title, description,
        model         = model,
        vectorizer    = vec,
        label_encoder = le,
    )

    print(f"\n  Title      : {title}")
    print(f"  Description: {description[:120]}{'...' if len(description) > 120 else ''}")
    print(f"  Predicted  : {genre}\n")
    return genre


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "main.py",
        description = "Movie Genre Classification pipeline",
        formatter_class = argparse.RawDescriptionHelpFormatter,
        epilog = (
            "Examples:\n"
            "  python main.py --mode train\n"
            "  python main.py --mode evaluate\n"
            "  python main.py --mode all\n"
            '  python main.py --mode predict --title "Jaws" '
            '--description "A shark terrorises a beach town."\n\n'
            "View MLflow UI:\n"
            "  mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000\n"
        ),
    )
    parser.add_argument(
        "--mode", required=True,
        choices=["train", "evaluate", "predict", "all"],
    )
    parser.add_argument("--title",       default="")
    parser.add_argument("--description", default="")
    return parser


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    if args.mode == "train":
        run_train()

    elif args.mode == "evaluate":
        run_evaluate()

    elif args.mode == "predict":
        if not args.title.strip() or not args.description.strip():
            sys.exit("Error: --title and --description are required for predict mode.")
        run_predict(args.title, args.description)

    elif args.mode == "all":
        run_train()
        run_evaluate()


if __name__ == "__main__":
    main()
