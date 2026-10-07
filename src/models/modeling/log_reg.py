from sklearn.linear_model import LogisticRegression


def build_model(params: dict):
    return LogisticRegression(**{"max_iter": 2000, "random_state": 42, **params})
