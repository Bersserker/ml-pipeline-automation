# pipeline.py
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.modeling import MODEL_NAMES, PARAM_GRIDS, build_model


def create_pipeline(
    numeric_features, categorical_features, parameters=None, *, models=None, n_jobs=-1
):
    """Compare selected models; parameters maps model names to custom grids."""
    models = list(MODEL_NAMES if models is None else models)
    if not models or len(set(models)) != len(models):
        raise ValueError("Select at least one model, without duplicates.")
    if parameters is not None and set(parameters) - set(models):
        raise ValueError("Parameter grids must be keyed by selected model names.")
    grids = []
    for name in models:
        estimator = build_model(name)
        grid = (parameters or {}).get(name, PARAM_GRIDS[name])
        grids.append(
            {
                "classifier": [estimator],
                **{f"classifier__{key}": values for key, values in grid.items()},
            }
        )
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", grids[0]["classifier"][0]),
        ]
    )
    return GridSearchCV(
        pipeline,
        grids,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=n_jobs,
        error_score="raise",
    )
