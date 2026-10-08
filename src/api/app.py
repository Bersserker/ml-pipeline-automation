"""Simple API for the saved credit scoring pipeline."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from src.features.build_features import build_features

MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "best_model.joblib"
Amount = Annotated[float, Field(allow_inf_nan=False)]
Payment = Annotated[float, Field(ge=0, allow_inf_nan=False)]
RepaymentStatus = Annotated[int, Field(ge=-3)]


class CreditRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit_bal: Payment
    sex: Literal[1, 2]
    education: Literal[0, 1, 2, 3, 4, 5, 6]
    marriage: Literal[0, 1, 2, 3]
    age: Annotated[int, Field(ge=0)]
    pay_0: RepaymentStatus
    pay_2: RepaymentStatus
    pay_3: RepaymentStatus
    pay_4: RepaymentStatus
    pay_5: RepaymentStatus
    pay_6: RepaymentStatus
    bill_amt1: Amount
    bill_amt2: Amount
    bill_amt3: Amount
    bill_amt4: Amount
    bill_amt5: Amount
    bill_amt6: Amount
    pay_amt1: Payment
    pay_amt2: Payment
    pay_amt3: Payment
    pay_amt4: Payment
    pay_amt5: Payment
    pay_amt6: Payment


class CreditResponse(BaseModel):
    prediction: Literal[0, 1]
    default_probability: float = Field(ge=0, le=1)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise HTTPException(status_code=503, detail="Model is missing. Run make train.")
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Credit Scoring API")


@app.get("/health")
def health():
    load_model()
    return {"status": "ok"}


@app.post("/predict", response_model=CreditResponse)
def predict(request: CreditRequest):
    model = load_model()
    features = build_features(pd.DataFrame([request.model_dump()]))
    features = (
        features[list(model.feature_names_in_)]
        .replace([np.inf, -np.inf], np.nan)
        .astype(float)
    )
    default_index = list(model.classes_).index(1)
    return CreditResponse(
        prediction=int(model.predict(features)[0]),
        default_probability=float(model.predict_proba(features)[0, default_index]),
    )
