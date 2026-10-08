import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.api import app as api
from src.features.build_features import build_features


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "best_model.joblib")
    api.load_model.cache_clear()
    with TestClient(api.app) as client:
        yield client
    api.load_model.cache_clear()


@pytest.fixture
def payload():
    return {
        "limit_bal": 20000,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 24,
        **{f"pay_{i}": 0 for i in (0, 2, 3, 4, 5, 6)},
        **{f"bill_amt{i}": 100 for i in range(1, 7)},
        **{f"pay_amt{i}": 50 for i in range(1, 7)},
    }


def test_predict_saved_pipeline(client, payload):
    features = build_features(pd.DataFrame([payload, payload])).astype(float)
    model = Pipeline(
        [
            ("imputer", SimpleImputer()),
            ("scaler", StandardScaler()),
            ("classifier", DummyClassifier()),
        ]
    ).fit(features, [0, 1])
    joblib.dump(model, api.MODEL_PATH)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] == int(model.predict(features)[0])
    assert result["default_probability"] == model.predict_proba(features)[0, 1]
    # A zero credit limit produces missing derived ratios, as during training.
    zero_limit = {**payload, "limit_bal": 0}
    features = (
        build_features(pd.DataFrame([zero_limit]))
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    response = client.post("/predict", json=zero_limit)
    assert response.status_code == 200
    assert response.json()["default_probability"] == model.predict_proba(features)[0, 1]


def test_missing_model(client, payload):
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=payload).status_code == 503


@pytest.mark.parametrize(
    "change", [{"age": -1}, {"sex": 3}, {"pay_amt1": -1}, {"unknown": 1}]
)
def test_invalid_input(client, payload, change):
    assert client.post("/predict", json={**payload, **change}).status_code == 422


def test_missing_field(client, payload):
    del payload["age"]
    assert client.post("/predict", json=payload).status_code == 422
