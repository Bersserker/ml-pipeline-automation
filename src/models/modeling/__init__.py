"""Model factories and their default hyperparameter grids."""

from importlib import import_module

MODEL_NAMES = ("log_reg", "random_forest", "catboost")
PARAM_GRIDS = {
    "log_reg": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [50, 100], "max_depth": [5, None]},
    "catboost": {"iterations": [500, 1000], "depth": [4, 6, 6]},
}


def build_model(name):
    if name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model {name!r}. Choose from: {', '.join(MODEL_NAMES)}"
        )
    return import_module(f"{__name__}.{name}").build_model({})
