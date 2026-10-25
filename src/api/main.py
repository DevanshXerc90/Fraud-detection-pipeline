"""
FastAPI inference endpoint for real-time fraud predictions.
Loads the trained model and preprocessor, applies the same scaling
that was used during training, and returns risk scores for incoming
transactions.
"""
import json
import math
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
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
    transactions: List[TransactionFeatures] = Field(
        ..., min_length=1, max_length=1000
    )


class PredictionResponse(BaseModel):
    fraud_score: float = Field(
        ..., description="Uncalibrated positive-class model score"
    )
    is_fraud: bool
    risk_level: str
    threshold: float


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    flagged_count: int


# ── App state ──────────────────────────────────────────────────

MODEL_DIR = Path("models")
_state: dict = {
    "model": None,
    "preprocessor": None,
    "feature_engineer": None,
    "threshold": 0.5,
    "feature_names": None,
    "load_error": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor on startup."""
    _state.update({
        "model": None,
        "preprocessor": None,
        "feature_engineer": None,
        "threshold": 0.5,
        "feature_names": None,
        "load_error": None,
    })
    model_path = MODEL_DIR / "model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    feature_engineer_path = MODEL_DIR / "feature_engineer.pkl"
    meta_path = MODEL_DIR / "metadata.json"

    required = {
        "model": model_path,
        "preprocessor": preprocessor_path,
        "feature_engineer": feature_engineer_path,
        "metadata": meta_path,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        _state["load_error"] = (
            "Missing model artifacts: " + ", ".join(missing)
        )
        print(f"Warning: {_state['load_error']}")
    else:
        try:
            _state["model"] = joblib.load(model_path)
            _state["preprocessor"] = joblib.load(preprocessor_path)
            _state["feature_engineer"] = joblib.load(feature_engineer_path)
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            threshold = float(meta["best_threshold"])
            feature_names = meta.get("feature_names")
            if not 0 <= threshold <= 1 or not feature_names:
                raise ValueError("metadata has an invalid threshold or feature schema")
            _state["threshold"] = threshold
            _state["feature_names"] = feature_names
            print(f"Model artifacts loaded from {MODEL_DIR}")
        except (OSError, KeyError, TypeError, ValueError, AttributeError,
                ImportError, EOFError, pickle.UnpicklingError) as exc:
            _state["load_error"] = f"Unable to load model artifacts: {exc}"
            print(f"Warning: {_state['load_error']}")

    print(f"Decision threshold: {_state['threshold']:.4f}")
    yield


# ── App ────────────────────────────────────────────────────────

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection powered by XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)


# Risk bands: CRITICAL ≥ 0.9 | HIGH ≥ 0.7 | MEDIUM ≥ 0.4 | LOW < 0.4
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
    """Apply the fitted preprocessing and feature engineering sequence."""
    preprocessor = _state["preprocessor"]
    feature_engineer = _state["feature_engineer"]
    transformed = preprocessor.transform(feature_engineer.transform(df))
    expected = _state["feature_names"]
    if list(transformed.columns) != expected:
        raise ValueError("Transformed feature schema does not match the trained model")
    return transformed


def _validate_transaction(tx: TransactionFeatures) -> None:
    values = tx.model_dump()
    if not all(math.isfinite(value) for value in values.values()):
        raise HTTPException(status_code=422, detail="All values must be finite")
    if tx.Time < 0:
        raise HTTPException(status_code=422, detail="Time must be non-negative")
    if tx.Amount < 0:
        raise HTTPException(status_code=422, detail="Amount must be non-negative")


def _predict(tx: TransactionFeatures) -> PredictionResponse:
    """Run a single transaction through preprocessing and the model."""
    model = _state.get("model")
    if model is None or _state.get("preprocessor") is None or _state.get("feature_engineer") is None:
        raise HTTPException(
            status_code=503,
            detail=_state.get("load_error")
            or "Model artifacts are not ready. Train the model first by running train.py.",
        )

    _validate_transaction(tx)
    raw_df = pd.DataFrame([tx.model_dump()])
    try:
        processed_df = _preprocess(raw_df)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"Model artifacts are incompatible: {exc}") from exc

    prob = float(model.predict_proba(processed_df)[0, 1])
    threshold = _state["threshold"]
    is_fraud = prob >= threshold

    return PredictionResponse(
        fraud_score=round(prob, 6),
        is_fraud=is_fraud,
        risk_level=_risk_level(prob),
        threshold=threshold,
    )


# ── Endpoints ──────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Redirect root to interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    ready = (
        _state.get("model") is not None
        and _state.get("preprocessor") is not None
        and _state.get("feature_engineer") is not None
        and _state.get("feature_names") is not None
    )
    return {
        "status": "ready" if ready else "not_ready",
        "application_available": True,
        "model_ready": ready,
        "model_loaded": _state.get("model") is not None,
        "preprocessor_loaded": _state.get("preprocessor") is not None,
        "feature_engineer_loaded": _state.get("feature_engineer") is not None,
        "error": _state.get("load_error"),
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