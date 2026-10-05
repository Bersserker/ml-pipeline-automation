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

# Скачать датасет (uv автоматически подготовит окружение)
make data

# То же самое без Makefile
uv run python -m src.data.make_dataset
```

Скрипт `src/data/make_dataset.py` загружает датасет
`uciml/default-of-credit-card-clients-dataset` в `data/raw/UCI_Credit_Card.csv`.
Если файл уже существует, повторная загрузка пропускается.
Это загрузка исходных данных; обработка данных пока не реализована.

Зависимость `kagglehub` указана в `pyproject.toml`, версии фиксируются в `uv.lock`.
Для установки зависимостей также можно использовать `make requirements`.

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
