from sklearn.ensemble import RandomForestClassifier


def build_model(params: dict):
    return RandomForestClassifier(**{"random_state": 42, "n_jobs": 1, **params})
