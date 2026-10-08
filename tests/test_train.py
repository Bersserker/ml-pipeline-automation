import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from src.data.clean_dataset import clean_dataset
from src.features.build_features import build_features
from src.models import train as training
from src.models.modeling import build_model
from src.models.pipeline import create_pipeline


def test_search_covers_grid_and_refits_pipeline():
    rng = np.random.default_rng(42)
    features = pd.DataFrame({"amount": rng.normal(size=60), "category": [0, 1] * 30})
    features.loc[0, "amount"] = np.nan
    target = np.array([0, 1] * 30)
    grid = {"n_estimators": [2, 3], "max_depth": [1, 2]}
    search = create_pipeline(
        ["amount"],
        ["category"],
        {
            "random_forest": grid,
            "log_reg": {"C": [0.1, 1.0]},
            "catboost": {"iterations": [2], "depth": [2]},
        },
        n_jobs=1,
    )
    search.fit(features, target)
    assert isinstance(search.estimator, Pipeline)
    assert len(search.cv_results_["params"]) == 7
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()
    assert search.best_score_ == max(search.cv_results_["mean_test_score"])
    assert isinstance(search.best_estimator_, Pipeline)


@pytest.mark.parametrize(
    "model,grid",
    [
        ("log_reg", {"C": [0.1, 1.0]}),
        ("random_forest", {"n_estimators": [2, 3], "max_depth": [2]}),
        ("catboost", {"iterations": [2, 3], "depth": [2]}),
    ],
)
def test_full_grid_mlflow_round_trip(tmp_path, monkeypatch, model, grid):
    monkeypatch.setattr(training, "MLFLOW_DIR", tmp_path / "mlflow")
    model_path = tmp_path / "models" / "best_model.joblib"
    monkeypatch.setattr(training, "BEST_MODEL_PATH", model_path)
    raw = pd.read_csv("data/raw/UCI_Credit_Card.csv").sample(n=200, random_state=42)
    processed = clean_dataset(raw)
    pipeline, metrics = training.train(
        processed, models=[model], param_grid={model: grid}, n_jobs=1
    )
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(training.EXPERIMENT_NAME)
    run = client.search_runs([experiment.experiment_id])[0]
    assert run.info.status == "FINISHED"
    assert int(run.data.params["search_candidates"]) == len(ParameterGrid(grid))
    assert np.isfinite(list(metrics.values())).all()
    versions = client.search_model_versions("name='CreditDefaultModel'")
    loaded = mlflow.sklearn.load_model(versions[0].source)
    assert run.data.params["model_type"] == model
    assert isinstance(loaded, Pipeline)
    assert isinstance(loaded.named_steps["classifier"], type(build_model(model)))
    features = (
        build_features(clean_dataset(raw))[
            training.NUMERIC_FEATURES + training.CATEGORICAL_FEATURES
        ]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    np.testing.assert_array_equal(loaded.predict(features), pipeline.predict(features))
    np.testing.assert_allclose(
        loaded.predict_proba(features), pipeline.predict_proba(features)
    )
    saved = joblib.load(model_path)
    reference = pd.read_csv(model_path.parent / "train_reference.csv")
    held_out = pd.read_csv(model_path.parent / "test_reference.csv")
    assert len(reference) == 160
    assert len(held_out) == 40
    assert set(reference["id"]).isdisjoint(held_out["id"])
    assert set(reference["id"]) | set(held_out["id"]) == set(processed["id"])
    assert isinstance(saved, Pipeline)
    np.testing.assert_array_equal(saved.predict(features), pipeline.predict(features))
    np.testing.assert_allclose(
        saved.predict_proba(features), pipeline.predict_proba(features)
    )


def test_invalid_model_selection():
    with pytest.raises(ValueError, match="Unknown model"):
        create_pipeline(["amount"], [], models=["missing"])
    with pytest.raises(ValueError, match="at least one"):
        create_pipeline(["amount"], [], models=[])
    with pytest.raises(ValueError, match="keyed"):
        create_pipeline(["amount"], [], {"C": [1]}, models=["log_reg"])
