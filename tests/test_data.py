import pandas as pd
import pandera.pandas as pa
import pytest
from src.data import make_dataset
from src.data.clean_dataset import clean_dataset
from src.data.prepare_dataset import prepare_dataset
from src.data.validation import PROCESSED_SCHEMA, RAW_SCHEMA
from src.models import train as training


@pytest.fixture
def raw():
    values = {name: [1, 2] for name in RAW_SCHEMA.columns}
    values["default.payment.next.month"] = [0, 1]
    return pd.DataFrame(values)


def test_prepare_validates_cleans_and_saves(tmp_path, raw):
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    raw_path = tmp_path / "raw.csv"
    output = tmp_path / "processed" / "credit.csv"
    raw.to_csv(raw_path, index=False)
    before = raw_path.read_bytes()
    assert prepare_dataset(raw_path, output) == output
    processed = PROCESSED_SCHEMA.validate(pd.read_csv(output), lazy=True)
    assert len(processed) == 2
    assert "default" in processed and "ID" not in processed
    assert raw_path.read_bytes() == before


@pytest.mark.parametrize(
    "defect", ["missing_column", "null", "invalid_target", "empty"]
)
def test_invalid_raw_does_not_replace_processed(tmp_path, raw, defect):
    if defect == "missing_column":
        raw = raw.drop(columns=["AGE"])
    elif defect == "null":
        raw.loc[0, "LIMIT_BAL"] = float("nan")
    elif defect == "invalid_target":
        raw.loc[0, "default.payment.next.month"] = 3
    else:
        raw = raw.iloc[:0]
    raw_path = tmp_path / "raw.csv"
    output = tmp_path / "processed.csv"
    raw.to_csv(raw_path, index=False)
    output.write_text("previous validated dataset")
    with pytest.raises(pa.errors.SchemaErrors):
        prepare_dataset(raw_path, output)
    assert output.read_text() == "previous validated dataset"


def test_processed_validation_before_saving(tmp_path, raw, monkeypatch):
    raw_path = tmp_path / "raw.csv"
    output = tmp_path / "processed.csv"
    raw.to_csv(raw_path, index=False)
    monkeypatch.setattr(
        "src.data.prepare_dataset.clean_dataset",
        lambda df: clean_dataset(df).drop(columns=["age"]),
    )
    with pytest.raises(pa.errors.SchemaErrors):
        prepare_dataset(raw_path, output)
    assert not output.exists()


def test_download_uses_raw_and_reuses_existing_file(tmp_path, raw, monkeypatch):
    monkeypatch.setattr(make_dataset, "RAW_DATA_DIR", tmp_path / "raw")
    calls = []

    def download(dataset, *, output_dir, force_download):
        calls.append((dataset, force_download))
        raw.to_csv(tmp_path / "raw" / make_dataset.DATASET_FILE, index=False)

    monkeypatch.setattr(make_dataset.kagglehub, "dataset_download", download)
    path = make_dataset.load_dataset()
    assert path.parent == tmp_path / "raw"
    assert make_dataset.load_dataset() == path
    assert calls == [(make_dataset.DATASET, True)]


def test_training_rejects_raw_and_invalid_processed(raw):
    with pytest.raises(pa.errors.SchemaErrors):
        training.train(raw)
    processed = clean_dataset(raw)
    processed.loc[0, "default"] = 5
    with pytest.raises(pa.errors.SchemaErrors):
        training.train(processed)
