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
from src.models.trainer import find_optimal_threshold
from src.api.main import TransactionFeatures, _predict, _state
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
            actual = _predict(tx).fraud_probability
            assert actual == round(expected, 6)
        finally:
            _state.clear()
            _state.update(previous)