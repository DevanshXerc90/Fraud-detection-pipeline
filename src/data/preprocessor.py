"""
Preprocessing pipeline — scaling, handling skewness in Amount,
and preparing data for model training.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


class DataPreprocessor:
    """
    Wraps all preprocessing steps so the same transformations
    can be applied consistently during training and inference.
    """

    def __init__(self):
        self.amount_scaler = RobustScaler()
        self.time_scaler = RobustScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Learn scaling parameters from training data."""
        # Amount and Time are the only non-anonymized columns
        # Using RobustScaler because both have heavy outliers
        self.amount_scaler.fit(df[["Amount"]].values)
        self.time_scaler.fit(df[["Time"]].values)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")

        result = df.copy()
        result["Amount"] = self.amount_scaler.transform(result[["Amount"]].values)
        result["Time"] = self.time_scaler.transform(result[["Time"]].values)
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience method — fit then transform."""
        return self.fit(df).transform(df)