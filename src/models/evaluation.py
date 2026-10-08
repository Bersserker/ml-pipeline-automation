"""Calculate scalar training and held-out metrics for credit models."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def training_metrics(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series
) -> dict[str, float]:
    probabilities = pipeline.predict_proba(X)
    predictions = pipeline.predict(X)
    return {
        "training_score": float(pipeline.score(X, y)),
        "training_accuracy_score": float(accuracy_score(y, predictions)),
        "training_precision_score": float(
            precision_score(y, predictions, average="weighted", zero_division=0)
        ),
        "training_recall_score": float(
            recall_score(y, predictions, average="weighted", zero_division=0)
        ),
        "training_f1_score": float(
            f1_score(y, predictions, average="weighted", zero_division=0)
        ),
        "training_log_loss": float(log_loss(y, probabilities)),
        "training_roc_auc": float(roc_auc_score(y, probabilities[:, 1])),
    }


def test_metrics(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = pipeline.predict(X)
    return {
        "test_auc": float(roc_auc_score(y, probabilities)),
        "test_f1": float(f1_score(y, predictions, zero_division=0)),
    }
