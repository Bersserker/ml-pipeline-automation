"""Clean credit data without modifying the raw dataframe."""

import pandas as pd


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and normalize column names and the target name."""
    df = df.copy().drop_duplicates()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(".", "_", regex=False)
    )
    df = df.rename(columns={"default_payment_next_month": "default"})
    return df.dropna(subset=["default"]).reset_index(drop=True)
