# ml-pipeline-automation

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Default of Credit Card

## Шпаргалка: окружение и загрузка данных

Запускайте команды из корня проекта. Нужны `uv` и Python 3.13;
для команд `make` также нужен GNU Make.

```bash
# Установить зависимости из pyproject.toml через uv
uv sync

# Скачать, проверить и очистить датасет (uv подготовит окружение)
make data

# То же самое без Makefile
uv run python -m src.data.make_dataset
```

Скрипт `src/data/make_dataset.py` загружает датасет
`uciml/default-of-credit-card-clients-dataset` в `data/raw/UCI_Credit_Card.csv`.
Если файл уже существует, повторная загрузка пропускается.
Затем Pandera проверяет исходные данные, очистка удаляет одинаковые строки,
приводит названия столбцов к нижнему регистру и переименовывает целевой столбец
в `default`. Очищенные данные повторно проверяются и сохраняются в
`data/processed/UCI_Credit_Card.csv`. Исходный файл остаётся неизменным.
Пустой датасет, отсутствующие столбцы, пропуски и недопустимые значения
останавливают подготовку; существующий обработанный файл при этом сохраняется.

```bash
make prepare-data  # Проверить и обработать уже загруженный raw
# Или без Makefile:
uv run python -m src.data.prepare_dataset
# Повторить обработку через DVC при изменении исходных данных или кода:
uv run dvc repro prepare
```

Зависимость `kagglehub` указана в `pyproject.toml`, версии фиксируются в `uv.lock`.
Для установки зависимостей также можно использовать `make requirements`.

## Запуск проекта

```bash
make requirements  # Установить зависимости
make run           # Запустить FastAPI на http://127.0.0.1:8000
# Или make dev для автоматической перезагрузки при изменении кода
```

Документация API: http://127.0.0.1:8000/docs. Сейчас приложение содержит
каркас FastAPI; endpoints для скоринга ещё не реализованы.
Остановка — `Ctrl+C`. Другой порт: `make run APP_PORT=8001`.

## MLflow и обучение

Запустите интерфейс и tracking server из корня проекта:

```bash
make mlflow
# Синонимы: make mlflow-ui или make mlflow-server
```

Откройте http://127.0.0.1:5000. Сервер работает до нажатия `Ctrl+C`.
База SQLite и артефакты сохраняются в `artifacts/mlflow/`.

Обучение сравнивает модели из `src/models/modeling`: `log_reg`, `random_forest`
и `catboost`. Для каждой модели используется своя сетка параметров из
`src/models/modeling/__init__.py`. `GridSearchCV` выбирает модель и параметры
по средней accuracy на пяти стратифицированных фолдах. Препроцессинг обучается
внутри каждого фолда; тестовая выборка используется только для итоговой оценки.

```bash
make train
make train MODELS="log_reg random_forest" N_JOBS=2
make train MODELS=catboost
```

`make train` сначала выполняет загрузку и подготовку данных. Обучение читает
`data/processed/UCI_Credit_Card.csv` и снова проверяет его перед построением
признаков. Для обучения без повторной подготовки:

```bash
uv run python -m src.models.train --models log_reg --n-jobs 2
```

`DATA_PATH` и аргумент `--data-path` позволяют выбрать другой CSV с очищенными
данными, соответствующими схеме `PROCESSED_SCHEMA`.

Лучшая модель переобучается на всей обучающей выборке и регистрируется как
`CreditDefaultModel` в локальной базе `artifacts/mlflow/mlflow.db`.
MLflow сохраняет выбранную модель, параметры, таблицу результатов CV и тестовые
метрики. Для LogisticRegression и RandomForest используется `skops`,
для CatBoost — `cloudpickle`. Загружайте только доверенные артефакты моделей.

Через Python можно передать собственные сетки:

```python
from src.models.train import train
import pandas as pd

df = pd.read_csv("data/processed/UCI_Credit_Card.csv")

pipeline, metrics = train(
    df,
    models=["log_reg", "random_forest"],
    param_grid={
        "log_reg": {"C": [0.1, 1.0]},
        "random_forest": {"n_estimators": [50, 100], "max_depth": [5, 10]},
    },
    n_jobs=2,
)
```

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         ml_pipeline_automation and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── ml_pipeline_automation   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes ml_pipeline_automation a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
