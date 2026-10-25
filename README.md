<div align="center">

# 🛡️ Credit Card Fraud Detection Pipeline

**End-to-end machine learning pipeline for real-time fraud detection**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-EC6F1A?style=for-the-badge)](https://xgboost.readthedocs.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-Enabled-5FB236?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](./LICENSE)

Trained on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset — **284,807 transactions** with only **0.172% fraud rate**.

</div>

---

## 📖 Overview

This pipeline trains four models — **Logistic Regression**, **Random Forest**, **XGBoost**, and **LightGBM** — compares them on validation AUPRC, selects the best performer, and serves it through a **FastAPI endpoint** for real-time transaction scoring.

Because the dataset is heavily imbalanced, the pipeline uses:
- ✅ Class-weight balancing
- ✅ `scale_pos_weight` in XGBoost and LightGBM
- ✅ Threshold optimization on the precision-recall curve (instead of the default `0.5` cutoff)
- ✅ SHAP interpretability on the winning model

---

## 🏗️ Pipeline Architecture

```
Raw CSV
  │
  ▼
DataLoader          ── stratified train / val / test split (72 / 8 / 20)
  │
  ▼
FeatureEngineer     ── fitted, batch-independent derived features
  │
  ▼
DataPreprocessor    ── RobustScaler on Amount and Time
  │
  ▼
ModelTrainer        ── 4 models · threshold tuning · MLflow logging
  │
  ▼
SHAP Explainer      ── TreeExplainer on winning model · summary + waterfall plots
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
| `AmountPercentile` | Percentile against the training amount distribution |
| `TimeSinceFirstTxNorm` | Normalized position using training time bounds |

---

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/DevanshXerc90/Fraud-detection-pipeline.git
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
- Train all **four models** and log metrics to **MLflow**
- Select the best model by **AUPRC**
- Save the model, fitted feature engineer, preprocessor, and schema metadata to `models/`
- Run **SHAP** on the winning model and save plots to `outputs/shap/`

---

## 📊 Model Benchmarks

All models are evaluated on the same stratified validation split (8% of data).
The winner is selected by **AUPRC** (average precision), and its threshold is
selected using validation data only. The held-out test metrics are written to
`models/results.json` after each run; the bundled artifacts are not a fresh benchmark.

| Model | Validation AUPRC |
|---|---:|
| Logistic Regression | 0.6799 |
| Random Forest | 0.7894 |
| **XGBoost** | **0.8040** |
| LightGBM | 0.0430 |

The selected XGBoost model used a validation-only F1 threshold of `0.937918`.
On the separate 56,962-row test set, it achieved AUPRC `0.8812`, ROC AUC
`0.9788`, precision `0.9405`, recall `0.8061`, and F1 `0.8681`. The test set
contained 98 fraud cases (0.1720%). These are results from the bundled
stratified offline run, not a guarantee of future production performance.

---

## 🌐 API Reference

Start the inference server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Check application availability and model readiness |
| `POST` | `/predict` | Score a single transaction |
| `POST` | `/predict/batch` | Score multiple transactions |
| `GET` | `/docs` | Interactive Swagger UI |

**Example Request**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time":476,"V1":0,"V2":0,"V3":0,"V4":0,"V5":0,"V6":0,"V7":0,"V8":0,"V9":0,"V10":0,"V11":0,"V12":0,"V13":0,"V14":0,"V15":0,"V16":0,"V17":0,"V18":0,"V19":0,"V20":0,"V21":0,"V22":0,"V23":0,"V24":0,"V25":0,"V26":0,"V27":0,"V28":0,"Amount":149.62}'
```

**Example Response**

```json
{
  "fraud_probability": 0.00025,
  "is_fraud": false,
  "risk_level": "LOW",
  "threshold": 0.937918
}
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

> Tests use **synthetic data** and run without the Kaggle dataset — no setup required.

The full training run requires the Kaggle CSV. Regenerate compatible artifacts
with `python train.py` before starting the API. The API requires
`model.pkl`, `preprocessor.pkl`, `feature_engineer.pkl`, and `metadata.json`
in `models/`.

---

## 🐳 Docker

```bash
docker compose up --build
```

The Docker image contains the tracked model artifacts but intentionally
excludes the local dataset, MLflow runs, SHAP output, and tests. The Compose
healthcheck uses Python's standard library and does not require `curl`.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Interpretability | SHAP (TreeExplainer / LinearExplainer) |
| API Server | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Containerization | Docker + Docker Compose |

---

## 🔍 Model Interpretability

After the best model is selected, SHAP values are computed on the test set
using `shap.TreeExplainer` (for tree-based models) or `shap.LinearExplainer`
(for Logistic Regression).

Two plots are saved to `outputs/shap/`:

| File | Description |
|---|---|
| `shap_summary.png` | Beeswarm plot — global feature importance across all test samples |
| `shap_waterfall.png` | Waterfall plot — breakdown for the single highest-risk transaction |

The printed rankings are descriptive associations, not proof of causation or
regulatory compliance.

---

## 📂 Project Structure

```
fraud-detection-pipeline/
├── src/
│   ├── data/           # DataLoader, DataPreprocessor
│   ├── features/       # FeatureEngineer (builder.py)
│   ├── models/         # ModelTrainer (trainer.py), SHAP explainer
│   └── api/            # FastAPI inference endpoint
├── tests/              # pytest suite with synthetic data
├── train.py            # Main training entry point
└── models/             # Saved model, preprocessing, and metadata artifacts
```

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](./LICENSE) for details.