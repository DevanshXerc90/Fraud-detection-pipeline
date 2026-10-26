"""
Unit tests for data loading, preprocessing, and feature engineering.
Uses synthetic data so tests run without needing the Kaggle dataset.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models.trainer import find_optimal_threshold, find_threshold_for_recall
from src.models.evaluation import (
    bootstrap_confidence_intervals,
    threshold_metrics,
)
from fastapi.testclient import TestClient
from src.api.main import (
    TransactionFeatures,
    _parse_frontend_origins,
    _predict,
    _risk_level,
    _state,
    app,
)
from sklearn.linear_model import LogisticRegression


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_data():
    """Generate a small synthetic dataset that looks like the real one."""
    np.random.seed(42)
    n = 1000
    data = {"Time": np.random.uniform(0, 172800, n)}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n)
    data["Amount"] = np.random.exponential(50, n)
    # ~5% fraud for testing (much higher than real data, but easier to test)
    data["Class"] = np.random.choice([0, 1], n, p=[0.95, 0.05])
    return pd.DataFrame(data)


# ── DataLoader tests ──────────────────────────────────────────

class TestDataLoader:
    def test_split_shapes(self, sample_data):
        loader = DataLoader()
        train, val, test = loader.split(sample_data)
        total = len(train) + len(val) + len(test)
        assert total == len(sample_data)

    def test_split_preserves_fraud_ratio(self, sample_data):
        loader = DataLoader()
        train, val, test = loader.split(sample_data)
        original_ratio = sample_data["Class"].mean()
        # each split should have roughly the same fraud ratio
        for split in [train, val, test]:
            assert abs(split["Class"].mean() - original_ratio) < 0.02

    def test_missing_file_raises(self):
        loader = DataLoader("nonexistent.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_data()

    def test_get_features_targets(self, sample_data):
        loader = DataLoader()
        X, y = loader.get_feature_targets(sample_data)
        assert "Class" not in X.columns
        assert len(y) == len(sample_data)

    def test_deduplicated_split_has_no_cross_partition_hashes(self, sample_data):
        duplicated = pd.concat(
            [sample_data, sample_data.iloc[[0, 1]]], ignore_index=True
        )
        train, validation, test = DataLoader().split(
            duplicated, deduplicate=True
        )
        hashes = [
            set(pd.util.hash_pandas_object(part, index=False).tolist())
            for part in (train, validation, test)
        ]
        assert not (hashes[0] & hashes[1])
        assert not (hashes[0] & hashes[2])
        assert not (hashes[1] & hashes[2])


# ── Preprocessor tests ────────────────────────────────────────

class TestDataPreprocessor:
    def test_fit_transform(self, sample_data):
        pp = DataPreprocessor()
        result = pp.fit_transform(sample_data)
        assert result.shape == sample_data.shape

    def test_transform_without_fit_raises(self, sample_data):
        pp = DataPreprocessor()
        with pytest.raises(RuntimeError):
            pp.transform(sample_data)

    def test_scaling_changes_amount(self, sample_data):
        pp = DataPreprocessor()
        result = pp.fit_transform(sample_data)
        # After robust scaling, values should be more centered
        assert abs(result["Amount"].mean()) < 1.0


# ── Feature engineering tests ─────────────────────────────────

class TestFeatureEngineer:
    def test_creates_new_columns(self, sample_data):
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_data)
        assert "LogAmount" in result.columns
        assert "HourOfDay" in result.columns
        assert "IsNightTime" in result.columns
        assert "V14_x_V17" in result.columns
        assert "VComponentsMagnitude" in result.columns

    def test_hour_of_day_range(self, sample_data):
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_data)
        assert result["HourOfDay"].min() >= 0
        assert result["HourOfDay"].max() <= 23

    def test_is_micro_tx_flag(self, sample_data):
        fe = FeatureEngineer()
        result = fe.fit_transform(sample_data)
        assert result["IsMicroTx"].isin([0, 1]).all()

    def test_single_row_is_independent_of_batch(self, sample_data):
        fe = FeatureEngineer().fit(sample_data)
        first = fe.transform(sample_data.iloc[[0]])
        larger_batch = fe.transform(sample_data.iloc[[0, 1]])
        pd.testing.assert_series_equal(
            first.iloc[0], larger_batch.iloc[0], check_names=False
        )

    def test_requires_fit(self, sample_data):
        with pytest.raises(RuntimeError):
            FeatureEngineer().transform(sample_data)


# ── Threshold optimization test ───────────────────────────────

class TestThresholdOptimization:
    def test_find_optimal_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = find_optimal_threshold(y_true, y_proba)
        # should land in the middle range given the clear separation in y_proba
        assert 0.3 <= threshold <= 0.7

    def test_bootstrap_and_threshold_metrics(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        intervals = bootstrap_confidence_intervals(
            y_true, scores, threshold=0.7, n_bootstrap=100, seed=42
        )
        assert intervals["requested_replicates"] == 100
        assert intervals["auprc"]["replicates"] > 0
        report = threshold_metrics(y_true, scores, [0.5, 0.7])
        assert report["0.7"]["confusion_matrix"] == [[3, 0], [0, 3]]

    def test_recall_target_uses_validation_scores(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = find_threshold_for_recall(y_true, scores, 0.9)
        assert threshold == 0.7


class TestRiskBands:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.399999, "LOW"),
            (0.4, "MEDIUM"),
            (0.699999, "MEDIUM"),
            (0.7, "HIGH"),
            (0.899999, "HIGH"),
            (0.9, "CRITICAL"),
        ],
    )
    def test_risk_boundaries(self, score, expected):
        assert _risk_level(score) == expected


class TestCors:
    preview_origin = (
        "https://id-preview--0ca650d0-eedc-4a07-8ca5-d96395203209.lovable.app"
    )

    def test_origin_configuration_is_normalized_and_explicit(self):
        origins = _parse_frontend_origins(
            " https://sentra.example/ , http://localhost:3000///, , "
        )
        assert origins == [
            self.preview_origin,
            "https://sentra.example",
            "http://localhost:3000",
        ]
        assert "*" not in _parse_frontend_origins("*")


class TestInferenceConsistency:
    def test_api_uses_training_feature_schema(self, sample_data):
        train = sample_data.drop(columns=["Class"])
        labels = sample_data["Class"]
        feature_engineer = FeatureEngineer().fit(train)
        preprocessor = DataPreprocessor().fit(train)
        training_features = preprocessor.transform(
            feature_engineer.transform(train)
        )
        model = LogisticRegression(max_iter=200).fit(training_features, labels)

        previous = _state.copy()
        _state.update({
            "model": model,
            "preprocessor": preprocessor,
            "feature_engineer": feature_engineer,
            "feature_names": list(training_features.columns),
            "threshold": 0.5,
            "load_error": None,
        })
        try:
            row = train.iloc[0]
            tx = TransactionFeatures(**row.to_dict())
            expected = float(model.predict_proba(training_features.iloc[[0]])[0, 1])
            actual = _predict(tx).fraud_score
            assert actual == round(expected, 6)
        finally:
            _state.clear()
            _state.update(previous)


    def test_batch_and_invalid_requests(self, sample_data):
        train = sample_data.drop(columns=["Class"])
        labels = sample_data["Class"]
        feature_engineer = FeatureEngineer().fit(train)
        preprocessor = DataPreprocessor().fit(train)
        training_features = preprocessor.transform(
            feature_engineer.transform(train)
        )
        model = LogisticRegression(max_iter=200).fit(training_features, labels)
        previous = _state.copy()
        _state.update({
            "model": model,
            "preprocessor": preprocessor,
            "feature_engineer": feature_engineer,
            "feature_names": list(training_features.columns),
            "threshold": 0.5,
            "load_error": None,
        })
        try:
            payload = train.iloc[0].to_dict()
            client = TestClient(app)
            batch = client.post(
                "/predict/batch",
                json={"transactions": [payload, payload]},
            )
            assert batch.status_code == 200
            assert batch.json()["total_transactions"] == 2
            assert len(batch.json()["predictions"]) == 2
            assert client.post(
                "/predict", json={**payload, "Amount": -1}
            ).status_code == 422
            assert client.post(
                "/predict/batch", json={"transactions": []}
            ).status_code == 422

            preflight = client.options(
                "/predict",
                headers={
                    "Origin": TestCors.preview_origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            assert preflight.status_code in (200, 204)
            assert preflight.headers["access-control-allow-origin"] == (
                TestCors.preview_origin
            )
            assert "POST" in preflight.headers["access-control-allow-methods"]
            assert "content-type" in preflight.headers[
                "access-control-allow-headers"
            ].lower()

            response = client.post(
                "/predict",
                json=payload,
                headers={"Origin": TestCors.preview_origin},
            )
            assert response.status_code == 200
            assert set(response.json()) == {
                "fraud_score",
                "is_fraud",
                "risk_level",
                "threshold",
            }
            assert response.headers["access-control-allow-origin"] == (
                TestCors.preview_origin
            )
            unapproved = client.post(
                "/predict",
                json=payload,
                headers={"Origin": "https://unapproved.example"},
            )
            assert "access-control-allow-origin" not in unapproved.headers
        finally:
            _state.clear()
            _state.update(previous)