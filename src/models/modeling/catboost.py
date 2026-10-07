from catboost import CatBoostClassifier


def build_model(params: dict):
    return CatBoostClassifier(
        **{
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
            **params,
        }
    )
