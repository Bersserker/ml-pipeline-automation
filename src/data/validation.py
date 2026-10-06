"""Pandera schemas for the raw and cleaned credit scoring datasets.

Both schemas require all expected columns and reject missing values.
Extra columns are allowed; values are coerced to the declared types.
"""

import pandera.pandas as pa

# Keep the source codes accepted by the current cleaning pipeline, which
# normalizes column names but does not recode categorical values.
RAW_SCHEMA = pa.DataFrameSchema(
    {
        "ID": pa.Column("int64", pa.Check.ge(0), nullable=False),
        "LIMIT_BAL": pa.Column("float64", pa.Check.ge(0), nullable=False),
        "SEX": pa.Column("int64", pa.Check.isin([1, 2]), nullable=False),
        "EDUCATION": pa.Column(
            "int64", pa.Check.isin([0, 1, 2, 3, 4, 5, 6]), nullable=False
        ),
        "MARRIAGE": pa.Column("int64", pa.Check.isin([0, 1, 2, 3]), nullable=False),
        "AGE": pa.Column("int64", pa.Check.ge(0), nullable=False),
        **{
            column: pa.Column("int64", pa.Check.ge(-3), nullable=False)
            for column in ("PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6")
        },
        **{
            # Negative bill amounts are valid; do not impose a lower bound.
            f"BILL_AMT{i}": pa.Column("float64", nullable=False)
            for i in range(1, 7)
        },
        **{
            f"PAY_AMT{i}": pa.Column("float64", pa.Check.ge(0), nullable=False)
            for i in range(1, 7)
        },
        "default.payment.next.month": pa.Column(
            "int64", pa.Check.isin([0, 1]), nullable=False
        ),
    },
    index=pa.Index("int64", nullable=False),
    coerce=True,
    strict=False,
    name="raw_credit_dataset",
)

# clean_dataset currently changes only column names and removes rows.
# Renaming the schema preserves the source types and validation rules.
PROCESSED_SCHEMA = RAW_SCHEMA.rename_columns(
    {
        column: "default" if column == "default.payment.next.month" else column.lower()
        for column in RAW_SCHEMA.columns
    }
)
PROCESSED_SCHEMA.name = "processed_credit_dataset"
