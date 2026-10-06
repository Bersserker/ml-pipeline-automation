"""Train and register the credit default model with MLflow."""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.clean_dataset import clean_dataset
from src.features.build_features import build_features
from src.models.pipeline import create_pipeline

MLFLOW_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "mlflow"
EXPERIMENT_NAME = "credit-default"
PARAM_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
}

NUMERIC_FEATURES = (
    ["limit_bal", "age"]
    + [f"bill_amt{i}" for i in range(1, 7)]
    + [f"pay_amt{i}" for i in range(1, 7)]
    + [f"avg_exp_{i}" for i in range(1, 6)]
)
CATEGORICAL_FEATURES = [
    "sex",
    "education",
    "marriage",
    "pay_0",
    "pay_2",
    "pay_3",
    "pay_4",
    "pay_5",
    "pay_6",
    "se_ma_2",
    "agebin",
]


def train(df: pd.DataFrame, *, param_grid=None, n_jobs=-1):
    """Prepare raw credit data, train a pipeline, and return it with test metrics."""
    df = build_features(clean_dataset(df))
    X = (
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    y = df["default"]
    if set(y.unique()) != {0, 1}:
        raise ValueError("The default column must contain both binary classes: 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    artifact_dir = MLFLOW_DIR / "mlartifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Log explicitly: MLflow autolog currently calls log_loss with deprecated y_pred.
    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run(run_name="GradientBoosting Hyperparameter Tuning"):
        search = create_pipeline(
            NUMERIC_FEATURES,
            CATEGORICAL_FEATURES,
            PARAM_GRID if param_grid is None else param_grid,
            n_jobs=n_jobs,
        )
        mlflow.log_param("model_type", "GradientBoosting")
        mlflow.log_params({"cv_folds": search.cv.n_splits, "scoring": search.scoring})
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        mlflow.log_params(search.best_params_)
        mlflow.log_metric("best_cv_accuracy", search.best_score_)
        mlflow.log_param("search_candidates", len(search.cv_results_["params"]))
        mlflow.log_table(pd.DataFrame(search.cv_results_), artifact_file="cv_results.json")
        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV accuracy: {search.best_score_:.6f}")

        train_proba = pipeline.predict_proba(X_train)
        train_pred = pipeline.predict(X_train)
        mlflow.log_metrics(
            {
                "training_score": pipeline.score(X_train, y_train),
                "training_accuracy_score": accuracy_score(y_train, train_pred),
                "training_precision_score": precision_score(
                    y_train, train_pred, average="weighted", zero_division=0
                ),
                "training_recall_score": recall_score(
                    y_train, train_pred, average="weighted", zero_division=0
                ),
                "training_f1_score": f1_score(
                    y_train, train_pred, average="weighted", zero_division=0
                ),
                "training_log_loss": log_loss(y_train, train_proba),
                "training_roc_auc": roc_auc_score(y_train, train_proba[:, 1]),
            }
        )

        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)
        metrics = {
            "test_auc": roc_auc_score(y_test, y_pred_proba),
            "test_f1": f1_score(y_test, y_pred, zero_division=0),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name="CreditDefaultModel",
            input_example=X_train.head(5),
            signature=ModelSignature(
                inputs=Schema([ColSpec("double", name) for name in X_train.columns]),
                outputs=Schema([ColSpec("long")]),
            ),
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

    return pipeline, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/raw/UCI_Credit_Card.csv",
    )
    args = parser.parse_args()
    _, metrics = train(pd.read_csv(args.data_path))
    print(metrics)


if __name__ == "__main__":
    main()
