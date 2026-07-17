# Credit Card Fraud Detection Pipeline

End-to-end machine learning pipeline that detects fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset (284,807 transactions, 0.172% fraud).

## What This Project Does

The pipeline trains three models (Logistic Regression, Random Forest, XGBoost), compares them on validation AUPRC, picks the best one, and serves it through a FastAPI endpoint for real-time scoring.

Because the dataset is heavily imbalanced, the pipeline uses class-weight balancing, scale_pos_weight in XGBoost, and threshold optimization on the precision-recall curve instead of relying on the default 0.5 cutoff.

## Pipeline Architecture

```
Raw CSV
  |
  v
DataLoader (stratified train/val/test split, 72/8/20)
  |
  v
DataPreprocessor (RobustScaler on Amount and Time)
  |
  v
FeatureEngineer (9 derived features)
  |
  v
ModelTrainer (3 models, threshold tuning, MLflow logging)
  |
  v
FastAPI Inference API
```

## Engineered Features

| Feature | Why It Helps |
|---|---|
| LogAmount | Compresses the extreme right skew of transaction amounts |
| HourOfDay | Extracts the hour from the elapsed-time column |
| HourSin / HourCos | Circular encoding so the model knows 11 PM and 1 AM are close |
| IsNightTime | Fraud rates spike between 2 AM and 6 AM |
| V14_x_V17 | Interaction of the two PCA components most correlated with fraud |
| VComponentsMagnitude | Overall magnitude of the anonymous PCA vector, flags anomalies |
| IsMicroTx | Flags charges under $1, a common card-testing pattern |
| AmountPercentile | Rank-based amount score, robust to outliers |
| TimeSinceFirstTxNorm | Normalized position within the 48-hour data window |

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/fraud-detection-pipeline.git
cd fraud-detection-pipeline
pip install -r requirements.txt
```

## Usage

### Training

1. Download the dataset from Kaggle and place it at `data/creditcard.csv`
2. Run the training pipeline:

```bash
python train.py
```

The script will:
- Split the data (72% train / 8% val / 20% test) with stratification
- Scale Amount and Time using RobustScaler
- Generate engineered features
- Train all three models and log metrics to MLflow
- Select the best model by AUPRC
- Save the model, preprocessor, and metadata to `models/`

### Serving the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health` - Check if model and preprocessor are loaded
- `POST /predict` - Score a single transaction
- `POST /predict/batch` - Score multiple transactions
- `GET /docs` - Interactive Swagger UI

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 476, "V1": -1.36, "V2": -0.07, ..., "V28": 0.03, "Amount": 149.62}'
```

Example response:

```json
{
  "fraud_probability": 0.873,
  "is_fraud": true,
  "risk_level": "HIGH",
  "threshold": 0.381
}
```

### Running Tests

```bash
pytest tests/ -v
```

Tests use synthetic data so they run without the Kaggle dataset.

## Docker

```bash
docker-compose up --build
```

## Tech Stack

- Python 3.11+
- scikit-learn, XGBoost
- FastAPI + Uvicorn
- MLflow (experiment tracking)
- ChromaDB (not used here, used in the RAG chatbot project)
- Docker + Docker Compose

## Project Structure

```
fraud-detection-pipeline/
  src/
    data/           # DataLoader, DataPreprocessor
    features/       # FeatureEngineer (builder.py)
    models/         # ModelTrainer, threshold optimization
    api/            # FastAPI inference endpoint
  tests/            # pytest suite with synthetic data
  train.py          # Main training entry point
  models/           # Saved model artifacts (gitignored)
```

## License

MIT