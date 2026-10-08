"""Configure MLflow and log model selection results and fitted models."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.models.modeling import build_model


def configure_tracking(directory: Path, experiment_name: str) -> None:
    artifact_dir = directory / "mlartifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{directory / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=artifact_dir.as_uri()
        )
    mlflow.set_experiment(experiment_name)
    # Log explicitly: autolog calls log_loss with deprecated y_pred.
    mlflow_sklearn.autolog(disable=True)


def log_search_configuration(
    search: GridSearchCV[Pipeline], models: Sequence[str]
) -> None:
    parameters = search.get_params(deep=False)
    mlflow.log_param("candidate_models", ",".join(models))
    mlflow.log_params(
        {
            "cv_folds": parameters["cv"].get_n_splits(),
            "scoring": parameters["scoring"],
        }
    )


def log_search_results(search: GridSearchCV[Pipeline], models: Sequence[str]) -> str:
    classifier = search.best_estimator_.named_steps["classifier"]
    selected_model = next(
        name for name in models if isinstance(classifier, type(build_model(name)))
    )
    mlflow.log_param("model_type", selected_model)
    mlflow.log_params(
        {
            key: value
            for key, value in search.best_params_.items()
            if key != "classifier"
        }
    )
    mlflow.log_metric("best_cv_accuracy", float(search.best_score_))
    mlflow.log_param("search_candidates", len(search.cv_results_["params"]))
    results = pd.DataFrame(search.cv_results_)
    serialized_params = [
        {
            key: (type(value).__name__ if key == "classifier" else value)
            for key, value in params.items()
        }
        for params in search.cv_results_["params"]
    ]
    results["params"] = pd.Series(serialized_params, index=results.index, dtype=object)
    results["param_classifier"] = results["param_classifier"].map(
        lambda model: type(model).__name__
    )
    mlflow.log_table(results, artifact_file="cv_results.json")
    return selected_model


def log_fitted_model(
    pipeline: Pipeline, selected_model: str, X_train: pd.DataFrame
) -> None:
    mlflow_sklearn.log_model(
        pipeline,
        name="model",
        registered_model_name="CreditDefaultModel",
        input_example=X_train.head(5),
        signature=ModelSignature(
            inputs=Schema([ColSpec("double", str(name)) for name in X_train.columns]),
            outputs=Schema([ColSpec("long")]),
        ),
        # CatBoost's native extension is not supported by skops.
        serialization_format=(
            mlflow_sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            if selected_model == "catboost"
            else mlflow_sklearn.SERIALIZATION_FORMAT_SKOPS
        ),
        skops_trusted_types=["numpy.dtype", "sklearn.tree._tree.Tree"],
    )
