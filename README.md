<div align="center">

# 🛡️ Credit Card Fraud Detection Pipeline

**End-to-end machine learning pipeline for real-time fraud detection**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-EC6F1A?style=for-the-badge)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](./LICENSE)

Trained on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset — **284,807 transactions** with only **0.172% fraud rate**.

</div>

---

## 📖 Overview

This pipeline trains three models — **Logistic Regression**, **Random Forest**, and **XGBoost** — compares them on validation AUPRC, selects the best performer, and serves it through a **FastAPI endpoint** for real-time transaction scoring.

Because the dataset is heavily imbalanced, the pipeline uses:
- ✅ Class-weight balancing
- ✅ `scale_pos_weight` in XGBoost
- ✅ Threshold optimization on the precision-recall curve (instead of the default `0.5` cutoff)

---

## 🏗️ Pipeline Architecture

```
Raw CSV
  │
  ▼
DataLoader          ── stratified train / val / test split (72 / 8 / 20)
  │
  ▼
DataPreprocessor    ── RobustScaler on Amount and Time
  │
  ▼
FeatureEngineer     ── 9 derived features
  │
  ▼
ModelTrainer        ── 3 models · threshold tuning · MLflow logging
  │
  ▼
FastAPI Inference API
```

---

## ⚙️ Engineered Features

| Feature | Description |
|---|---|
| `LogAmount` | Compresses the extreme right skew of transaction amounts |
| `HourOfDay` | Extracts the hour from the elapsed-time column |
| `HourSin` / `HourCos` | Circular encoding so the model knows 11 PM and 1 AM are close |
| `IsNightTime` | Fraud rates spike between 2 AM and 6 AM |
| `V14_x_V17` | Interaction of the two PCA components most correlated with fraud |
| `VComponentsMagnitude` | Overall magnitude of the anonymous PCA vector — flags anomalies |
| `IsMicroTx` | Flags charges under $1, a common card-testing pattern |
| `AmountPercentile` | Rank-based amount score, robust to outliers |
| `TimeSinceFirstTxNorm` | Normalized position within the 48-hour data window |

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/fraud-detection-pipeline.git
cd fraud-detection-pipeline
pip install -r requirements.txt
```

### Training

1. Download the dataset from Kaggle and place it at `data/creditcard.csv`
2. Run the training pipeline:

```bash
python train.py
```

The training script will:
- Split the data **(72% train / 8% val / 20% test)** with stratification
- Scale `Amount` and `Time` using **RobustScaler**
- Generate all engineered features
- Train all three models and log metrics to **MLflow**
- Select the best model by **AUPRC**
- Save the model, preprocessor, and metadata to `models/`

---

## 🌐 API Reference

Start the inference server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check if the model and preprocessor are loaded |
| `POST` | `/predict` | Score a single transaction |
| `POST` | `/predict/batch` | Score multiple transactions |
| `GET` | `/docs` | Interactive Swagger UI |

**Example Request**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 476, "V1": -1.36, "V2": -0.07, ..., "V28": 0.03, "Amount": 149.62}'
```

**Example Response**

```json
{
  "fraud_probability": 0.873,
  "is_fraud": true,
  "risk_level": "HIGH",
  "threshold": 0.381
}
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

> Tests use **synthetic data** and run without the Kaggle dataset — no setup required.

---

## 🐳 Docker

```bash
docker-compose up --build
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML Models | scikit-learn, XGBoost |
| API Server | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Containerization | Docker + Docker Compose |

---

## 📂 Project Structure

```
fraud-detection-pipeline/
├── src/
│   ├── data/           # DataLoader, DataPreprocessor
│   ├── features/       # FeatureEngineer (builder.py)
│   ├── models/         # ModelTrainer, threshold optimization
│   └── api/            # FastAPI inference endpoint
├── tests/              # pytest suite with synthetic data
├── train.py            # Main training entry point
└── models/             # Saved model artifacts (gitignored)
```

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](./LICENSE) for details.