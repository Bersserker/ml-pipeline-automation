"""Prepare training data and persist the model with its reference splits."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data.validation import PROCESSED_SCHEMA
from src.features.build_features import build_features

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


def prepare_training_data(df: pd.DataFrame):
    """Validate data, build features, and make a reproducible stratified split."""
    reference_data = PROCESSED_SCHEMA.validate(df, lazy=True)
    features = build_features(reference_data)
    X = (
        features[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    y = features["default"]
    return reference_data, train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def save_training_outputs(
    pipeline: Pipeline,
    model_path: Path,
    reference_data: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Save the fitted pipeline and the original rows of each split."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    reference_data.loc[X_train.index].to_csv(
        model_path.parent / "train_reference.csv", index=False
    )
    reference_data.loc[X_test.index].to_csv(
        model_path.parent / "test_reference.csv", index=False
    )
