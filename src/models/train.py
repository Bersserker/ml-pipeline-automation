"""Train and register the credit default model with MLflow."""

import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.prepare_dataset import PROCESSED_DATA_PATH
from src.models.evaluation import test_metrics, training_metrics
from src.models.modeling import MODEL_NAMES
from src.models.pipeline import create_pipeline
from src.models.tracking import (
    configure_tracking,
    log_fitted_model,
    log_search_configuration,
    log_search_results,
)
from src.models.training_data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    prepare_training_data,
    save_training_outputs,
)

MLFLOW_DIR = Path(
    os.environ.get(
        "MLFLOW_DIR", Path(__file__).resolve().parents[2] / "artifacts" / "mlflow"
    )
).resolve()
BEST_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best_model.joblib"
EXPERIMENT_NAME = "credit-default"


def train(
    df: pd.DataFrame, *, models=None, param_grid=None, n_jobs=-1
) -> tuple[Pipeline, dict[str, float]]:
    """Validate processed credit data, train, and return the pipeline and metrics."""
    models = list(MODEL_NAMES if models is None else models)
    search = create_pipeline(
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, param_grid, models=models, n_jobs=n_jobs
    )
    reference_data, (X_train, X_test, y_train, y_test) = prepare_training_data(df)
    configure_tracking(MLFLOW_DIR, EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Model selection and hyperparameter tuning"):
        log_search_configuration(search, models)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        selected_model = log_search_results(search, models)
        print(f"Best model: {selected_model}")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV accuracy: {search.best_score_:.6f}")

        mlflow.log_metrics(training_metrics(pipeline, X_train, y_train))
        metrics = test_metrics(pipeline, X_test, y_test)
        mlflow.log_metrics(metrics)
        log_fitted_model(pipeline, selected_model, X_train)

    save_training_outputs(pipeline, BEST_MODEL_PATH, reference_data, X_train, X_test)
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    return pipeline, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROCESSED_DATA_PATH,
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    _, metrics = train(
        pd.read_csv(args.data_path), models=args.models, n_jobs=args.n_jobs
    )
    print(metrics)


if __name__ == "__main__":
    main()
