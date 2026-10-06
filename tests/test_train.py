import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from src.data.clean_dataset import clean_dataset
from src.features.build_features import build_features
from src.models import train as training
from src.models.pipeline import create_pipeline


def test_search_covers_grid_and_refits_pipeline():
    rng = np.random.default_rng(42)
    features = pd.DataFrame({"amount": rng.normal(size=60), "category": [0, 1] * 30})
    features.loc[0, "amount"] = np.nan
    target = np.array([0, 1] * 30)
    grid = {"n_estimators": [2, 3], "max_depth": [1, 2]}
    search = create_pipeline(["amount"], ["category"], grid, n_jobs=1)
    search.fit(features, target)
    assert isinstance(search.estimator, Pipeline)
    assert len(search.cv_results_["params"]) == 4
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    assert search.best_score_ == max(search.cv_results_["mean_test_score"])
    assert isinstance(search.best_estimator_, Pipeline)
    assert (
        search.best_estimator_.named_steps["classifier"].n_estimators
        == (search.best_params_["classifier__n_estimators"])
    )


def test_full_grid_mlflow_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "MLFLOW_DIR", tmp_path / "mlflow")
    raw = pd.read_csv("data/raw/UCI_Credit_Card.csv").sample(n=200, random_state=42)
    pipeline, metrics = training.train(raw, n_jobs=1)
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(training.EXPERIMENT_NAME)
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.status == "FINISHED"
    assert (
        int(run.data.params["search_candidates"]) == len(ParameterGrid(training.PARAM_GRID)) == 36
    )
    assert np.isfinite(list(metrics.values())).all()
    versions = client.search_model_versions("name='CreditDefaultModel'")
    loaded = mlflow.sklearn.load_model(versions[0].source)
    features = (
        build_features(clean_dataset(raw))[
            training.NUMERIC_FEATURES + training.CATEGORICAL_FEATURES
        ]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    np.testing.assert_array_equal(loaded.predict(features), pipeline.predict(features))
    np.testing.assert_allclose(loaded.predict_proba(features), pipeline.predict_proba(features))
