"""
FastAPI inference endpoint for real-time fraud predictions.
Loads the trained model and preprocessor, applies the same scaling
that was used during training, and returns risk scores for incoming
transactions.
"""
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ── Pydantic models ────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    """Schema matching the V1-V28 PCA features plus Amount and Time."""
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., description="Transaction amount")


class BatchTransactionRequest(BaseModel):
    transactions: List[TransactionFeatures]


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    flagged_count: int


# ── App state ──────────────────────────────────────────────────

MODEL_DIR = Path("models")
_state: dict = {"model": None, "preprocessor": None, "threshold": 0.5}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor on startup."""
    model_path = MODEL_DIR / "best_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    meta_path = MODEL_DIR / "model_metadata.json"

    if model_path.exists():
        _state["model"] = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print("Warning: No trained model found. Predictions will return 503.")

    if preprocessor_path.exists():
        _state["preprocessor"] = joblib.load(preprocessor_path)
        print(f"Preprocessor loaded from {preprocessor_path}")

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        _state["threshold"] = meta.get("threshold", 0.5)

    print(f"Decision threshold: {_state['threshold']:.4f}")
    yield


# ── App ────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection powered by XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)


def _risk_level(prob: float) -> str:
    if prob >= 0.9:
        return "CRITICAL"
    elif prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    else:
        return "LOW"


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same scaling that was used during training."""
    preprocessor = _state.get("preprocessor")
    if preprocessor is not None:
        df = preprocessor.transform(df)
    return df


def _predict(tx: TransactionFeatures) -> PredictionResponse:
    """Run a single transaction through preprocessing and the model."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first by running train.py.",
        )

    raw_df = pd.DataFrame([tx.model_dump()])
    processed_df = _preprocess(raw_df)

    prob = float(model.predict_proba(processed_df)[0, 1])
    threshold = _state["threshold"]
    is_fraud = prob >= threshold

    return PredictionResponse(
        fraud_probability=round(prob, 6),
        is_fraud=is_fraud,
        risk_level=_risk_level(prob),
        threshold=threshold,
    )


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": _state.get("model") is not None,
        "preprocessor_loaded": _state.get("preprocessor") is not None,
        "threshold": _state.get("threshold"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionFeatures):
    """Score a single transaction for fraud risk."""
    return _predict(tx)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchTransactionRequest):
    """Score multiple transactions at once."""
    results = [_predict(tx) for tx in req.transactions]
    flagged = sum(1 for r in results if r.is_fraud)
    return BatchPredictionResponse(
        predictions=results,
        total_transactions=len(results),
        flagged_count=flagged,
    )