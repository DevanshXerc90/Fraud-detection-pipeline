"""
Feature engineering for fraud detection.
Creates domain-specific features that help models catch fraudulent patterns
that raw PCA components alone might miss.
"""
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Generates derived features from raw transaction data.

    Each new feature captures a specific fraud signal:
    - LogAmount: compresses the heavy right tail of transaction amounts
    - HourOfDay: extracts the hour from the elapsed-time column
    - IsNightTime: elapsed-time bucket using a documented modulo-24-hour proxy
    - V14_x_V17: interaction of two PCA components most correlated with fraud
    - VComponentsMagnitude: overall magnitude of the anonymous PCA vector
    - IsMicroTx: flag for very small charges often used to test stolen cards
    - AmountPercentile: percentile relative to the training amount distribution
    - TimeSinceFirstTxNorm: normalized transaction timing using training bounds
    """

    V_COLUMNS = [f"V{i}" for i in range(1, 29)]
    NIGHT_START = 2
    NIGHT_END = 6
    MICRO_TX_THRESHOLD = 1.0

    def __init__(self):
        self._amount_reference = None
        self._time_min = None
        self._time_max = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Learn the training-only statistics needed for live transformations."""
        self._validate_input(df)
        amount = self._numeric_series(df, "Amount")
        time = self._numeric_series(df, "Time")
        self._amount_reference = np.sort(amount.to_numpy())
        self._time_min = float(time.min())
        self._time_max = float(time.max())
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and transform it."""
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps and return the enriched frame."""
        if self._amount_reference is None:
            raise RuntimeError("Call fit() before transform()")
        self._validate_input(df)
        # work on a copy so the original dataframe is never modified
        result = df.copy()
        amount = self._numeric_series(result, "Amount")
        time = self._numeric_series(result, "Time")

        # Log-transform the amount to tame extreme skew.
        # Adding 1 avoids log(0) for any zero-amount edge cases.
        result["LogAmount"] = np.log1p(amount.clip(lower=0))

        # This is a modulo-24-hour proxy derived from elapsed seconds, not a
        # wall-clock timestamp.
        result["HourOfDay"] = (
            (time % 86400) / 3600
        ).astype(int).clip(0, 23)

        # Circular encoding so the model understands that hour 23 and hour 0
        # are close together, not far apart.
        result["HourSin"] = np.sin(2 * np.pi * result["HourOfDay"] / 24)
        result["HourCos"] = np.cos(2 * np.pi * result["HourOfDay"] / 24)

        # Binary flag for the selected elapsed-time bucket.
        result["IsNightTime"] = (
            (result["HourOfDay"] >= self.NIGHT_START)
            & (result["HourOfDay"] < self.NIGHT_END)
        ).astype(int)

        # Interaction between V14 and V17.  In the creditcardfraud dataset
        # these two PCA components have the strongest negative correlation
        # with the fraud label.  Their product captures a joint signal.
        result["V14_x_V17"] = result["V14"] * result["V17"]

        # Euclidean magnitude of the full PCA feature vector.
        # A high overall magnitude can indicate anomalous transactions.
        v_values = result[self.V_COLUMNS].values
        result["VComponentsMagnitude"] = np.sqrt(np.sum(v_values ** 2, axis=1))

        # Flag for amounts at or below the documented dataset threshold.
        result["IsMicroTx"] = (
            amount <= self.MICRO_TX_THRESHOLD
        ).astype(int)

        # Compare against the training distribution, not the current batch.
        result["AmountPercentile"] = (
            np.searchsorted(self._amount_reference, amount, side="right")
            / len(self._amount_reference)
        )

        # Normalize using training bounds so a single request is reproducible.
        time_range = self._time_max - self._time_min
        if time_range > 0:
            result["TimeSinceFirstTxNorm"] = (time - self._time_min) / time_range
        else:
            result["TimeSinceFirstTxNorm"] = 0.0

        return result

    @staticmethod
    def _validate_input(df: pd.DataFrame) -> None:
        required = {"Time", "Amount", "V14", "V17", *{f"V{i}" for i in range(1, 29)}}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Missing feature columns: {', '.join(missing)}")

    @staticmethod
    def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
        values = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{column} contains missing or non-finite values")
        return values