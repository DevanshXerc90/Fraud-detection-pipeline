# Credit Card Fraud Detection Pipeline

This project trains and serves an imbalanced-fraud scoring model on the
Kaggle/ULB credit-card dataset. The chronological evaluation leads the report:
XGBoost achieved test AUPRC **0.7983 (95% bootstrap CI 0.7032–0.8788)**,
precision `0.9464`, and recall `0.7067` on later elapsed-time records.

Run the reproducible pipeline after downloading `data/creditcard.csv`:

```bash
python train.py
```

Two held-out examples from the generated test set:

| Example | True label | Score | Predicted class | Threshold |
|---|---:|---:|---:|---:|
| Correctly detected fraud | 1 | 0.999992 | 1 | 0.993095 |
| False negative | 1 | 0.990442 | 0 | 0.993095 |

These are offline dataset examples, not guarantees for new transactions.

## Setup

```bash
git clone https://github.com/DevanshXerc90/Fraud-detection-pipeline.git
cd Fraud-detection-pipeline
pip install -r requirements.txt
```

Download the [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and save it as `data/creditcard.csv`. The dataset is not committed.

## Architecture and data flow

```text
CSV -> split policy -> fitted feature engineering -> fitted RobustScaler
    -> five-fold model comparison -> validation threshold selection
    -> held-out evaluation -> model/preprocessor artifacts -> FastAPI
```

The default split removes exact duplicate rows before a stratified
70%/10%/20% split. Use `DataLoader.split(..., deduplicate=False)` to reproduce
the historical duplicate-containing split. Feature engineering and scaling
are fitted on training data only. Model selection uses five-fold mean AUPRC on
the training partition; validation selects the operating threshold; the test
set is evaluated once.

## Chronological evaluation

`Time` is elapsed seconds, not wall-clock time. Sorting by `Time` gives:

| Split | Rows | Fraud cases |
|---|---:|---:|
| Earlier train | 199,364 | 384 |
| Validation | 28,481 | 33 |
| Later test | 56,962 | 75 |

XGBoost was selected on temporal validation AUPRC `0.8436`. At the threshold
`0.934245` selected from the 33-fraud validation set:

| Metric | Value | Bootstrap 95% CI |
|---|---:|---:|
| AUPRC | 0.7983 | 0.7032–0.8788 |
| Precision | 0.9464 | 0.8863–1.0000 |
| Recall | 0.7067 | 0.6026–0.8052 |

Intervals use 1,000 percentile-bootstrap samples of the temporal test set;
replicates without positives are counted and skipped. They measure test-sample
uncertainty and do not include uncertainty from selecting a model or threshold.
The temporal threshold was tuned using only 33 validation frauds.

## Stratified evaluation

The default deduplicated stratified run contains 198,607/331 train,
28,373/47 validation, and 56,746/95 test rows/frauds. XGBoost was selected by
five-fold mean AUPRC (`0.8582`, std `0.0643`); Random Forest was close
(`0.8543`, std `0.0587`) and is not claimed statistically inferior.

At the validation-selected threshold `0.993095`, the test results are:

| Metric | Value | Bootstrap 95% CI |
|---|---:|---:|
| AUPRC | 0.8111 | 0.7277–0.8878 |
| Precision | 0.9853 | 0.9524–1.0000 |
| Recall | 0.7053 | 0.6144–0.7901 |

The selected threshold confusion matrix is TN=`56,650`, FP=`1`, FN=`28`,
TP=`67`. These random and temporal metrics use different distributions and
are not directly comparable as if they were the same experiment.

## Duplicate sensitivity

Exact duplicates were measured across every feature column plus `Class`.
The dataset has 1,854 duplicate rows in 773 groups, with 1,081 occurrences
beyond each group's first row. In the historical stratified split, 8 of the
98 test frauds have an identical twin in train or validation; 22 of all 492
fraud rows have a twin in another stratified partition.

| Evaluation | Test rows/frauds | AUPRC | Precision | Recall |
|---|---:|---:|---:|---:|
| Current duplicate-containing split | 56,962 / 98 | 0.8812 | 0.9405 | 0.8061 |
| Deduplicate before splitting | 56,746 / 95 | 0.8111 | 0.9853 | 0.7053 |
| Current split, no train/validation twin in test | 56,618 / 90 | 0.8683 | 0.9342 | 0.7889 |

The deduplicated results differ materially from the historical results, so
the default now deduplicates before splitting. All three result objects,
confusion counts, and 1,000-sample bootstrap intervals are in
[`models/results.json`](models/results.json). Duplicates were not silently
dropped from the source data: repeated transactions may be legitimate events,
so both policies remain available for reproducibility.

## Threshold trade-offs

The selected threshold, `0.5`, and the validation threshold targeting
approximately 90% recall are reported below. The target was chosen from
validation predictions; the test results are descriptive.

| Threshold | TN | FP | FN | TP | Test precision | Test recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.993095 selected | 56,650 | 1 | 28 | 67 | 0.9853 | 0.7053 |
| 0.5 | 56,644 | 7 | 20 | 75 | 0.9146 | 0.7895 |
| 0.002949, ~90% validation recall | 56,004 | 647 | 13 | 82 | 0.1125 | 0.8632 |

The precision-recall curve is generated from held-out predictions:

![Held-out precision-recall curve](outputs/evaluation/precision_recall_curve.png)

Source data is [`precision_recall_curve.json`](outputs/evaluation/precision_recall_curve.json).

## Features

| Feature | Formula or assumption |
|---|---|
| `LogAmount` | `log1p(Amount)` |
| `HourOfDay` | `floor(Time / 3600) mod 24`; elapsed-time proxy |
| `HourSin`, `HourCos` | Circular encoding of `HourOfDay` |
| `IsNightTime` | Configured modulo-24-hour bucket, not verified wall-clock time |
| `V14_x_V17` | Product of PCA components `V14` and `V17` |
| `VComponentsMagnitude` | Euclidean magnitude of the PCA vector |
| `IsMicroTx` | `Amount <= 1.0` dataset-unit indicator, not a causal claim |
| `AmountPercentile` | Percentile against the fitted training amount distribution |
| `TimeSinceFirstTxNorm` | Normalized elapsed time using fitted training bounds |

## API and artifacts

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Endpoints are `/health`, `/predict`, `/predict/batch`, and `/docs`.
The response field is `fraud_score`, an uncalibrated positive-class score,
not a calibrated probability. Risk labels are LOW `<0.4`, MEDIUM `0.4–<0.7`,
HIGH `0.7–<0.9`, and CRITICAL `>=0.9`; they are display bands separate from
the binary decision threshold.

Training writes `model.pkl`, `model.xgb.json`, `preprocessor.pkl`,
`feature_engineer.pkl`, `metadata.json`, and `results.json`. The native JSON
XGBoost artifact produced exactly the same predictions as the joblib model in
the verification run. Load pickle/joblib files only from trusted sources.

The local trainer logs one MLflow run per model. Evidence from `mlruns/` is
summarized in [`docs/mlflow_runs.md`](docs/mlflow_runs.md). No remote MLflow
service is claimed.

SHAP output for the selected model:

![SHAP summary](outputs/shap/shap_summary.png)

![SHAP waterfall](outputs/shap/shap_waterfall.png)

## Tests and project structure

```bash
pytest -q
python -m compileall -q src train.py tests
docker compose up --build
```

```text
src/data/       loading, splitting, preprocessing
src/features/  fitted feature engineering
src/models/    trainers, evaluation helpers, SHAP
src/api/       FastAPI application
tests/         unit and API tests
train.py       end-to-end training command
models/        generated compatible artifacts and metrics
outputs/       generated plots and prediction examples
```

## Limitations

This is an offline portfolio experiment on historical 2013 anonymized PCA
data covering approximately two days, with low fraud prevalence and
unspecified currency. Bootstrap intervals do not capture all selection
uncertainty. The scores are not calibrated probabilities. The experiments do
not establish future performance, fairness, drift behavior, production
throughput, or production readiness. Docker and Render deployment require
their respective environments; Render was not tested in this verification.
