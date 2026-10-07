# Как создать аналогичный MLOps-проект

Пошаговая инструкция на примере классификации кредитного дефолта. Результат:
структура Cookiecutter Data Science, изолированное Python-окружение, контроль
качества кода и данных, воспроизводимый пайплайн, учёт экспериментов и API модели.
Команды ниже рассчитаны на Bash в Linux/macOS и выполняются из корня нового
проекта, кроме команды генерации шаблона. Для Windows можно использовать WSL.

Шаблон создаёт каркас. Загрузку данных, признаки, обучение и API нужно реализовать
самостоятельно или перенести из этого репозитория. Разделы с конфигурациями —
инструкция для **нового** проекта; они не означают, что все эти настройки уже
включены в текущем репозитории.

## 1. Инструменты и генерация Cookiecutter

Установите Git, Python 3.13, uv и GNU Make. Docker понадобится для контейнерного
запуска. Проверка установки:

```bash
git --version
uv --version
make --version
uv python install 3.13
```

Способ установки uv для своей ОС указан в
[официальной инструкции](https://docs.astral.sh/uv/getting-started/installation/).

В родительской папке запустите генератор Cookiecutter Data Science v2:

```bash
uvx --from cookiecutter-data-science ccds
```

Выберите параметры по названиям, поскольку порядок пунктов может меняться:

| Параметр | Значение для аналогичного проекта |
| --- | --- |
| `project_name` | `Credit Default MLOps` |
| `repo_name` | `credit-default-mlops` |
| `module_name` | `credit_default_mlops` |
| `author_name` | Ваше имя |
| `description` | Классификация кредитного дефолта |
| `python_version_number` | `3.13` |
| `environment_manager` | `uv` |
| `dependency_file` | `pyproject.toml` |
| `dataset_storage` | `none` для локального старта |
| `pydata_packages` | `basic` |
| `testing_framework` | `pytest` |
| `linting_and_formatting` | `flake8+black+isort` |
| `include_code_scaffold` | `Yes` |

Лицензию и генератор документации выберите под свою задачу. Затем:

```bash
cd credit-default-mlops
git init
```

Если нужен именно классический CLI `cookiecutter`, вместо предыдущей генерации
можно использовать старый шаблон v1:

```bash
uvx cookiecutter https://github.com/drivendataorg/cookiecutter-data-science -c v1
```

У v1 другой набор параметров; современный шаблон v2 запускается через `ccds`.
Подробности: [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/).

## 2. Виртуальное окружение и зависимости

В созданном `pyproject.toml` задайте `requires-python = "~=3.13.0"` в секции
`[project]`. Не создавайте вторую секцию `[project]`. Затем:

```bash
uv python pin 3.13
uv venv --python 3.13
source .venv/bin/activate

uv add pandas numpy scikit-learn catboost joblib mlflow skops cloudpickle
uv add fastapi uvicorn kagglehub 'pandera[io,pandas]' python-dotenv
uv add --dev black flake8 isort pytest httpx dvc great-expectations
uv sync
```

По необходимости добавьте инструменты для исследования и подбора параметров:

```bash
uv add --dev jupyterlab matplotlib optuna
```

`uv add` записывает зависимости в `pyproject.toml` и обновляет `uv.lock`.
Храните оба файла в Git. Для повторной установки зафиксированных зависимостей
используйте `uv sync --locked`: команда проверяет согласованность lock-файла
и конфигурации. `uv sync --frozen` устанавливает lock-файл без такой проверки.
Запуск через `uv run` не требует активации окружения.
См. [окружения uv](https://docs.astral.sh/uv/pip/environments/) и
[фиксацию зависимостей](https://docs.astral.sh/uv/concepts/projects/sync/).

В **существующем** `ml-pipeline-automation` достаточно `uv sync --locked`:
зависимости и lock-файл уже есть. `requirements.txt` здесь не фиксирует все версии;
основной способ воспроизведения окружения — `uv.lock`.

## 3. Структура и Git

Для повторения архитектуры этого репозитория подготовьте структуру:

```text
src/
  data/        # загрузка, очистка, схемы и проверки данных
  features/    # вычисление признаков
  models/      # sklearn Pipeline, обучение, предсказания
  api/         # FastAPI
tests/         # проверки данных, обучения и API
data/raw/      # неизменяемые исходные данные
data/processed/# результат подготовки
models/        # сериализованная модель
artifacts/     # локальные эксперименты MLflow
notebooks/     # исследования
reports/       # метрики и отчёты
docs/          # инструкции
```

```bash
mkdir -p src/{data,features,models,api} tests data/{raw,processed}
mkdir -p models artifacts reports
touch src/__init__.py src/{data,features,models,api}/__init__.py
```

Сгенерированный пакет `credit_default_mlops/` можно использовать вместо `src/`.
В таком случае замените пути и имена модулей во всех дальнейших командах,
DVC, тестах и CI. В этом репозитории рабочий пайплайн находится в `src/`,
а `ml_pipeline_automation/` содержит каркас шаблона.

Добавьте в `.gitignore` правила для `.venv/`, `.env`, `__pycache__/`,
`.pytest_cache/`, `artifacts/mlflow/` и генерируемых отчётов. Исходные данные
и модели храните через DVC либо отдельное хранилище артефактов. Например:

```gitignore
.venv/
.env
__pycache__/
.pytest_cache/
/artifacts/mlflow/
/models/*.joblib
/models/*_reference.csv
/reports/drift.json
```

## 4. Black, Flake8 и сортировка импортов

Добавьте в `pyproject.toml` (или дополните существующие секции):

```toml
[tool.black]
line-length = 88
target-version = ['py313']

[tool.isort]
profile = "black"
line_length = 88
```

Создайте `.flake8`; если шаблон уже настроил Flake8 в `setup.cfg`, перенесите
настройки в один файл:

```ini
[flake8]
max-line-length = 88
extend-ignore = E203
exclude = .git,.venv,__pycache__,data,models,artifacts
```

Форматирование и проверка — разные действия:

```bash
uv run isort src tests
uv run black src tests
uv run isort --check-only src tests
uv run black --check src tests
uv run flake8 src tests
```

Flake8 может сообщить о длинной строке, которую Black не разбивает автоматически;
исправьте её вручную. Совместимые настройки описаны в
[документации Black](https://black.readthedocs.io/en/stable/guides/using_black_with_other_tools.html).

В текущем репозитории CI проверяет `src tests` через Black и Flake8.
`make lint` и `make format` используют Ruff, настроенный на пакет
`ml_pipeline_automation/`. Поэтому для рабочего кода запускайте команды
Black/Flake8 выше. Для нового проекта согласуйте команды Makefile и CI.

## 5. Подготовка и проверка данных

Реализуйте отдельные модули; ориентир — файлы этого репозитория:

| Файл | Назначение |
| --- | --- |
| `src/data/make_dataset.py` | Скачивание через KaggleHub и запуск подготовки |
| `src/data/clean_dataset.py` | Очистка, удаление дубликатов, переименование колонок |
| `src/data/validation.py` | Pandera-схемы исходных и обработанных данных |
| `src/data/prepare_dataset.py` | Проверка → очистка → проверка → сохранение |
| `src/data/validation_gx.py` | Отдельная проверка Great Expectations для CI |

Для этого датасета источник — `uciml/default-of-credit-card-clients-dataset`,
исходный файл — `data/raw/UCI_Credit_Card.csv`, обработанный —
`data/processed/UCI_Credit_Card.csv`. Для другой задачи замените источник,
схемы, целевую колонку и допустимые значения.

Проверяйте наличие колонок, типы, пропуски, диапазоны и непустую выборку.
Исходный файл сохраняйте неизменным; ошибочная проверка должна завершать процесс
с ненулевым кодом и сохранять предыдущий корректный обработанный файл.

После реализации модулей:

```bash
uv run python -m src.data.make_dataset
uv run python -m src.data.validation_gx
```

## 6. DVC: версии данных и пайплайн

В новом Git-репозитории и после загрузки данных:

```bash
uv run dvc init
uv run dvc add data/raw
git add .dvc .dvcignore data/raw.dvc
# Если DVC создал data/.gitignore, добавьте и его.
```

Если, как в этом репозитории, `/data/*` уже игнорируется глобально, добавьте
исключения `!/data/*.dvc` и `!/data/.gitignore`: метаданные должны попадать в Git.
Сам CSV туда не добавляйте. В существующем репозитории DVC уже инициализирован.

Создайте `dvc.yaml` с этапом подготовки, как здесь:

```yaml
stages:
  prepare:
    cmd: uv run python -m src.data.prepare_dataset
    deps:
      - data/raw/UCI_Credit_Card.csv
      - src/data/prepare_dataset.py
      - src/data/clean_dataset.py
      - src/data/validation.py
    outs:
      - data/processed
```

```bash
uv run dvc repro
git add dvc.yaml dvc.lock
```

Для нового проекта добавьте `pyproject.toml` и `uv.lock` в `deps`, чтобы изменение
окружения тоже требовало повторной подготовки. При необходимости добавьте этап
обучения с зависимостями от обработанных данных, признаков, кода моделей и
гиперпараметров. В этом репозитории DVC пока управляет только подготовкой;
обучение запускается отдельно. Если параметризуете этап, вынесите значения
в `params.yaml`, объявите `params` в DVC и обеспечьте их чтение Python-кодом.

Для проверки хранения можно использовать локальный remote вне проекта:

```bash
mkdir -p ../credit-default-dvc-storage
uv run dvc remote add --local -d storage ../credit-default-dvc-storage
uv run dvc push
```

`--local` сохраняет машинозависимый путь в `.dvc/config.local`, который не должен
попадать в Git. Коллеге понадобится настроить доступный ему remote отдельно.
Для общего S3-хранилища вместо локального remote:

```bash
uv add --dev 'dvc[s3]'
# Замените адрес на свой доступный bucket и префикс.
uv run dvc remote add -d storage s3://YOUR-BUCKET/credit-default
```

Учётные данные задайте через профиль AWS или окружение. Общий адрес remote
фиксируется в `.dvc/config`; секреты туда не записывайте. Сейчас `.dvc/config`
этого репозитория пуст: перед `dvc push`/`dvc pull` нужно настроить remote.
См. [команды DVC](https://dvc.org/doc/command-reference/) и
[настройку remote](https://dvc.org/doc/command-reference/remote/add).

При изменении исходных данных выполняйте `uv run dvc add data/raw`, затем
`uv run dvc repro` и `uv run dvc push`. Фиксируйте обновлённые `.dvc`-файлы
и `dvc.lock` вместе с кодом. После клонирования: `uv sync --locked`, настройка
remote, `uv run dvc pull`, затем `uv run dvc repro` при необходимости.

## 7. Обучение и MLflow

Реализуйте sklearn `Pipeline`: imputer, масштабирование числовых признаков,
кодирование категорий, классификатор. Обучайте препроцессинг внутри каждого
CV-фолда. Отложенную test-выборку используйте только для итоговой оценки.
Зафиксируйте seed и стратифицированное разделение для классификации.

В этом проекте `src/models/pipeline.py` сравнивает LogisticRegression,
RandomForest и CatBoost через `GridSearchCV`, а `src/models/train.py` записывает
параметры, результаты CV, test-метрики и модель в MLflow. Он явно задаёт SQLite
tracking URI и регистрирует модель `CreditDefaultModel`.

После переноса или реализации модулей запустите обучение:

```bash
uv run python -m src.models.train --models log_reg --n-jobs 2
```

Для просмотра результатов в другом терминале из корня проекта:

```bash
mkdir -p artifacts/mlflow
uv run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///artifacts/mlflow/mlflow.db \
  --artifacts-destination "$PWD/artifacts/mlflow/mlartifacts"
```

Откройте <http://127.0.0.1:5000>. Здесь обучение пишет непосредственно в SQLite;
переменная `MLFLOW_TRACKING_URI` не меняет URI, явно заданный кодом обучения.
Для удалённого tracking server измените код так, чтобы он читал эту настройку.

Сохраняйте полный pipeline с препроцессингом, идентификатор MLflow run, версию
данных и Git-коммит. Здесь результат инференса хранится в
`models/best_model.joblib`. Для повторяемого релиза добавьте версионирование
этого файла через DVC либо загрузку конкретной версии из реестра MLflow.
Не фиксируйте в Git локальную SQLite-базу и бинарные модели.

## 8. Тесты и CI

Напишите pytest-проверки схем и очистки, разделения данных, сериализации pipeline,
а также API: корректный запрос, неверный вход и отсутствующая модель.
Для быстрых тестов используйте небольшие синтетические данные и временные пути;
скачивание полного датасета оставьте отдельной интеграционной проверке.

```bash
uv run pytest tests
uv run python -m compileall -q src tests
```

Для нового проекта создайте `.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]
permissions:
  contents: read
jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv python install 3.13
      - run: uv sync --locked
      - run: uv run isort --check-only src tests
      - run: uv run black --check src tests
      - run: uv run flake8 src tests
      - run: uv run pytest tests
```

Добавляйте проверку данных в отдельный job, обеспечив скачивание или `dvc pull`
и доступ к remote через секреты CI. В текущем репозитории уже есть workflows
для Black/Flake8/Great Expectations и проверки синтаксиса; запуск всех тестов
и автоматический deployment ещё нужно подключить.

## 9. FastAPI, Docker и мониторинг

Реализуйте `src/api/app.py`: загрузку доверенного pipeline, `GET /health`,
`POST /predict`, проверку входа и вычисление тех же признаков, что при обучении.
В этом проекте отсутствие модели даёт 503, некорректный запрос — 422.

```bash
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Документация API: <http://127.0.0.1:8000/docs>. После замены модели перезапустите
процесс, если модель загружается один раз при старте или первом запросе.

Для Docker возьмите за основу `Dockerfile` и `.dockerignore` этого проекта:
Python 3.13, системные библиотеки для CatBoost, установка через
`uv sync --frozen --no-dev`, исходный код, команда Uvicorn и healthcheck.
Исключите из контекста `.git`, `.venv`, секреты и локальные данные.
Предоставьте обученную модель контейнеру через volume:

```bash
docker build -t credit-scoring:local .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v "$PWD/models:/app/models:ro" credit-scoring:local
curl http://127.0.0.1:8000/health
```

Мониторинг должен сохранять train-reference и сравнивать распределения входных
признаков с новыми данными, например через PSI. Скрипт `scripts/simulate_data.py`
в рабочей версии этого проекта отправляет test-клиентов в API и пишет
`reports/drift.json`:

```bash
uv run python scripts/simulate_data.py --samples 100 --interval 0.1
```

Для нового проекта такой скрипт нужно перенести или написать. Пороги и размер
выборки подбирайте под задачу; дрифт признаков сам по себе не доказывает снижение
качества. С поступлением истинных ответов считайте метрики предсказаний,
а также контролируйте ошибки и задержку API.

## 10. Makefile и первый коммит

В новом проекте объедините команды в Makefile. Строки рецептов должны
начинаться с табуляции:

```makefile
.PHONY: requirements format lint test data train run
requirements:
	uv sync --locked
format:
	uv run isort src tests
	uv run black src tests
lint:
	uv run isort --check-only src tests
	uv run black --check src tests
	uv run flake8 src tests
test:
	uv run pytest tests
data:
	uv run python -m src.data.make_dataset
train: data
	uv run python -m src.models.train --models log_reg --n-jobs 2
run:
	uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

После реализации модулей выполните `make format`, `make lint`, `make test`,
`make data`, `uv run dvc repro`, `make train` и проверку API.
Перед коммитом просмотрите файлы:

```bash
git status --short
git add pyproject.toml uv.lock .python-version .gitignore .flake8 Makefile
git add src tests docs .github .dvc .dvcignore data/raw.dvc dvc.yaml dvc.lock
# Дополнительно добавьте data/.gitignore, если он был создан DVC.
git diff --cached --check
git diff --cached --stat
git commit -m "Initialize reproducible MLOps project"
```

Не добавляйте отсутствующие пути из примера; Dockerfile, `.dockerignore` и
сгенерированный Python-пакет добавьте отдельно, если они используются.
Git хранит код, конфигурацию и метаданные версий; DVC — версии данных;
MLflow — эксперименты и модели. Проверяйте каждый слой при воспроизведении проекта.
