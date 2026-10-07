"""Train and register the credit default model with MLflow."""

import argparse
from pathlib import Path

import mlflow
from mlflow.models import ModelSignature
import mlflow.sklearn
from mlflow.types.schema import ColSpec, Schema
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data.prepare_dataset import PROCESSED_DATA_PATH
from src.data.validation import PROCESSED_SCHEMA
from src.features.build_features import build_features
from src.models.modeling import MODEL_NAMES
from src.models.pipeline import create_pipeline

MLFLOW_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "mlflow"
EXPERIMENT_NAME = "credit-default"

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


def train(df: pd.DataFrame, *, models=None, param_grid=None, n_jobs=-1):
    """Validate processed credit data, train, and return the pipeline and metrics."""
    df = PROCESSED_SCHEMA.validate(df, lazy=True)
    models = list(MODEL_NAMES if models is None else models)
    search = create_pipeline(
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, param_grid, models=models, n_jobs=n_jobs
    )
    df = build_features(df)
    X = (
        df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    y = df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    artifact_dir = MLFLOW_DIR / "mlartifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            EXPERIMENT_NAME, artifact_location=artifact_dir.as_uri()
        )
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Log explicitly: MLflow autolog currently calls log_loss with deprecated y_pred.
    mlflow.sklearn.autolog(disable=True)

    with mlflow.start_run(run_name="Model selection and hyperparameter tuning"):
        mlflow.log_param("candidate_models", ",".join(models))
        mlflow.log_params({"cv_folds": search.cv.n_splits, "scoring": search.scoring})
        search.fit(X_train, y_train)
        # Persist the refitted pipeline, without GridSearchCV's scorer and CV objects.
        pipeline = search.best_estimator_
        selected_model = next(
            name
            for name, grid in zip(models, search.param_grid)
            if isinstance(
                pipeline.named_steps["classifier"], type(grid["classifier"][0])
            )
        )
        mlflow.log_param("model_type", selected_model)
        mlflow.log_params(
            {
                key: value
                for key, value in search.best_params_.items()
                if key != "classifier"
            }
        )
        mlflow.log_metric("best_cv_accuracy", search.best_score_)
        mlflow.log_param("search_candidates", len(search.cv_results_["params"]))
        results = pd.DataFrame(search.cv_results_)
        results["params"] = [
            {
                key: (type(value).__name__ if key == "classifier" else value)
                for key, value in params.items()
            }
            for params in search.cv_results_["params"]
        ]
        results["param_classifier"] = results["param_classifier"].map(
            lambda model: type(model).__name__
        )
        mlflow.log_table(results, artifact_file="cv_results.json")
        print(f"Best model: {selected_model}")
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
            # CatBoost's native extension is not supported by skops.
            serialization_format=(
                mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
                if selected_model == "catboost"
                else mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS
            ),
            skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
        )

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
