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
    - IsNightTime: fraud rates spike between 2 AM and 6 AM
    - V14_x_V17: interaction of two PCA components most correlated with fraud
    - VComponentsMagnitude: overall magnitude of the anonymous PCA vector
    - IsMicroTx: flag for very small charges often used to test stolen cards
    - AmountPercentile: rank-based score relative to all other transactions
    - TimeSinceFirstTxNorm: normalized transaction timing
    """

    V_COLUMNS = [f"V{i}" for i in range(1, 29)]
    NIGHT_START = 2
    NIGHT_END = 6
    MICRO_TX_THRESHOLD = 1.0

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps and return the enriched frame."""
        result = df.copy()

        # Log-transform the amount to tame extreme skew.
        # Adding 1 avoids log(0) for any zero-amount edge cases.
        result["LogAmount"] = np.log1p(result["Amount"].clip(lower=0))

        # Pull the hour-of-day from the raw Time column (seconds since first tx).
        # The dataset spans ~48 hours so we wrap around modulo 86400.
        result["HourOfDay"] = (
            (result["Time"] % 86400) / 3600
        ).astype(int).clip(0, 23)

        # Circular encoding so the model understands that hour 23 and hour 0
        # are close together, not far apart.
        result["HourSin"] = np.sin(2 * np.pi * result["HourOfDay"] / 24)
        result["HourCos"] = np.cos(2 * np.pi * result["HourOfDay"] / 24)

        # Binary flag for late-night transactions.
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

        # Flag for micro-transactions.  Fraudsters often run a tiny charge
        # first to verify the card is alive before larger purchases.
        result["IsMicroTx"] = (
            result["Amount"] <= self.MICRO_TX_THRESHOLD
        ).astype(int)

        # Percentile rank of the amount.  This is robust to outliers and
        # gives the model a stable, bounded signal.
        result["AmountPercentile"] = (
            result["Amount"].rank(pct=True)
        )

        # Normalized time within the dataset window (0 to 1).
        max_time = result["Time"].max()
        if max_time > 0:
            result["TimeSinceFirstTxNorm"] = result["Time"] / max_time
        else:
            result["TimeSinceFirstTxNorm"] = 0.0

        return result