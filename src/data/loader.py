"""
Data loader for credit card transaction data.
Handles reading from CSV, basic validation, and train/test splitting
with stratification to preserve the fraud ratio.
"""
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Load and split transaction data for fraud detection."""

    def __init__(self, data_path: str = "data/creditcard.csv"):
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        """Read the raw CSV and do a quick sanity check."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Download it from Kaggle: "
                "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )

        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df):,} transactions "
              f"with {df.shape[1]} features")

        # sanity check — we expect a 'Class' column (0 = legit, 1 = fraud)
        if "Class" not in df.columns:
            raise ValueError("Expected 'Class' column not found in dataset")

        fraud_pct = df["Class"].mean() * 100
        print(f"Fraud ratio: {fraud_pct:.3f}% "
              f"({df['Class'].sum():,} fraudulent transactions)")

        return df

    def split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified split into train / val / test.
        Keeps the fraud ratio consistent across all three sets.
        """
        # first pull out the test set
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df["Class"],
            random_state=random_state,
        )

        # then carve validation out of what's left
        # adjust val_size relative to the remaining (non-test) fraction
        adjusted_val = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=adjusted_val,
            stratify=train_val["Class"],
            random_state=random_state,
        )

        print(f"Train: {len(train):,} | "
              f"Val: {len(val):,} | "
              f"Test: {len(test):,}")

        return train.reset_index(drop=True), \
            val.reset_index(drop=True), \
            test.reset_index(drop=True)

    @staticmethod
    def get_feature_targets(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate features (X) and label (y)."""
        X = df.drop(columns=["Class"])
        y = df["Class"]
        return X, y