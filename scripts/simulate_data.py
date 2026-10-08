"""Send held-out clients to the API and compare feature distributions using PSI."""

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


def psi(reference, current, bins=10):
    """Use training quantiles, open outer bins, and smoothing for empty bins."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if not len(reference) or not len(current):
        raise ValueError("PSI requires nonempty samples")
    if not np.isfinite(reference).all() or not np.isfinite(current).all():
        raise ValueError("PSI requires finite values")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) == 1:
        # Distinguish values below, equal to, and above a constant baseline.
        edges = np.array([edges[0], np.nextafter(edges[0], np.inf)])
    else:
        edges = edges[1:-1]
    edges = np.r_[-np.inf, edges, np.inf]
    expected = np.histogram(reference, bins=edges)[0].astype(float) + 0.5
    actual = np.histogram(current, bins=edges)[0].astype(float) + 0.5
    expected /= expected.sum()
    actual /= actual.sum()
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def simulate(reference, test, *, api_url, samples=100, interval=0):
    if samples < 1 or interval < 0:
        raise ValueError("samples must be positive and interval nonnegative")
    incoming = test.sample(n=min(samples, len(test)), random_state=42)
    predictions = []
    for _, row in incoming.drop(columns=["id", "default"]).iterrows():
        payload = row.to_dict()
        # Keep integer API fields as integers when pandas mixes numeric types.
        for column in (
            "sex",
            "education",
            "marriage",
            "age",
            "pay_0",
            "pay_2",
            "pay_3",
            "pay_4",
            "pay_5",
            "pay_6",
        ):
            payload[column] = int(payload[column])
        request = Request(
            api_url.rstrip("/") + "/predict",
            data=json.dumps(payload, allow_nan=False).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30) as response:
            predictions.append(json.load(response))
        if interval:
            time.sleep(interval)
    return {
        "samples_sent": len(predictions),
        "mean_default_probability": float(
            np.mean([p["default_probability"] for p in predictions])
        ),
        "psi": {
            feature: psi(reference[feature], incoming[feature])
            for feature in ("age", "limit_bal", "bill_amt1", "pay_amt1")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--reference", type=Path, default=Path("models/train_reference.csv")
    )
    parser.add_argument("--test", type=Path, default=Path("models/test_reference.csv"))
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("reports/drift.json"))
    args = parser.parse_args()
    result = simulate(
        pd.read_csv(args.reference),
        pd.read_csv(args.test),
        api_url=args.api_url,
        samples=args.samples,
        interval=args.interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
