"""Download into data/raw and prepare validated data in data/processed."""

from pathlib import Path

import kagglehub

from src.data.prepare_dataset import prepare_dataset

DATASET = "uciml/default-of-credit-card-clients-dataset"
RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET_FILE = "UCI_Credit_Card.csv"


def load_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = RAW_DATA_DIR / DATASET_FILE

    if not dataset_path.is_file():
        kagglehub.dataset_download(
            DATASET, output_dir=str(RAW_DATA_DIR), force_download=True
        )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Файл {dataset_path} не найден")

    print(f"Датасет готов: {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    prepare_dataset(load_dataset())
