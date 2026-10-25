"""
Model training, threshold optimisation, and MLflow logging.

Trains four models:
    - Logistic Regression  (sklearn, class_weight="balanced")
    - Random Forest        (sklearn, class_weight="balanced")
    - XGBoost              (scale_pos_weight = neg / pos)
    - LightGBM             (scale_pos_weight = neg / pos)

All four are evaluated on validation AUPRC; the highest-scoring model
is designated the "best" and is the one saved and used for inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
import lightgbm as lgb


# ── Threshold helper ──────────────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> float:
    """
    Sweep the precision-recall curve and return the threshold that
    maximises the F1 score on the provided labels/probabilities.

    Used both at training time (to tune each model's decision boundary)
    and in tests (to verify the function's behaviour).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision and recall have one extra element; align them with thresholds
    precision = precision[:-1]
    recall = recall[:-1]

    denom = precision + recall
    # avoid division by zero for degenerate thresholds
    f1_scores = np.where(
        denom > 0,
        2 * precision * recall / denom,
        0.0,
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx])


# ── ModelTrainer ──────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    End-to-end trainer that fits four models, logs each to MLflow,
    selects the best by validation AUPRC, tunes its decision threshold,
    and exposes the winning model for downstream evaluation and saving.
    """

    def __init__(self, experiment_name: str = "fraud-detection"):
        mlflow.set_experiment(experiment_name)
        self._models: Dict[str, object] = {}   # name → fitted estimator
        self._val_auprc: Dict[str, float] = {}
        self._thresholds: Dict[str, float] = {}
        self.failures: Dict[str, str] = {}
        self.best_model_name: str = ""
        self.best_model = None
        self.best_threshold: float = 0.5

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Dict[str, float]:
        """
        Train all four models; log each to a separate MLflow run.

        Returns
        -------
        dict
            Mapping of model name → validation AUPRC, used by train.py
            to record pipeline results.
        """
        # Class imbalance ratio used by XGBoost and LightGBM
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        model_specs = self._build_model_specs(scale_pos_weight)

        print(f"\n{'-' * 60}")
        print(f"  Training {len(model_specs)} models")
        print(f"  Class ratio  neg/pos = {scale_pos_weight:.1f}")
        print(f"{'-' * 60}")

        for name, model in model_specs.items():
            try:
                self._train_one(name, model, X_train, y_train, X_val, y_val)
            except (ValueError, RuntimeError, OSError) as exc:
                self.failures[name] = str(exc)
                print(f"FAILED: {exc}")

        # Select winner
        if not self._val_auprc:
            raise RuntimeError(
                "All configured models failed: "
                + "; ".join(f"{name}: {error}" for name, error in self.failures.items())
            )
        self.best_model_name = max(self._val_auprc, key=self._val_auprc.get)
        self.best_model = self._models[self.best_model_name]
        self.best_threshold = self._thresholds[self.best_model_name]

        print(f"\n{'=' * 60}")
        print("  Model Comparison (Validation AUPRC)")
        print(f"  {'Model':<22} {'AUPRC':>8}  {'Threshold':>10}")
        print(f"  {'-' * 44}")
        for name in model_specs:
            marker = " <- WINNER" if name == self.best_model_name else ""
            print(
                f"  {name:<22} {self._val_auprc[name]:>8.4f}"
                f"  {self._thresholds[name]:>10.4f}{marker}"
            )
        print(f"{'=' * 60}\n")

        return self._val_auprc

    def _build_model_specs(self, scale_pos_weight: float) -> Dict:
        """Return an ordered dict of name → unfitted estimator."""
        return {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                solver="lbfgs",
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=300,
                scale_pos_weight=scale_pos_weight,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="aucpr",
                random_state=42,
                verbosity=0,
            ),
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=300,
                scale_pos_weight=scale_pos_weight,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                metric="average_precision",
                random_state=42,
                verbosity=-1,
            ),
        }

    def _train_one(
        self,
        name: str,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Fit a single model, compute val AUPRC, log to MLflow."""
        print(f"  [+] Training {name}...", end=" ", flush=True)

        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)

            val_proba = model.predict_proba(X_val)[:, 1]
            auprc = average_precision_score(y_val, val_proba)
            auc_roc = roc_auc_score(y_val, val_proba)
            threshold = find_optimal_threshold(y_val.values, val_proba)

            # Log hyperparameters and metrics
            mlflow.log_params(self._extract_params(model, name))
            mlflow.log_metrics({
                "val_auprc": auprc,
                "val_auc_roc": auc_roc,
                "optimal_threshold": threshold,
            })

            # Tag the model type for easy filtering in the MLflow UI
            mlflow.set_tag("model_type", name)

        self._models[name] = model
        self._val_auprc[name] = auprc
        self._thresholds[name] = threshold
        print(f"AUPRC={auprc:.4f}")

    @staticmethod
    def _extract_params(model, name: str) -> Dict:
        """Pull out a small set of loggable hyperparameters."""
        common = {"model_name": name}
        try:
            params = model.get_params()
            keys = [
                "n_estimators", "max_depth", "learning_rate",
                "scale_pos_weight", "class_weight",
                "num_leaves", "subsample", "colsample_bytree",
            ]
            common.update({k: str(params[k]) for k in keys if k in params})
        except Exception:
            pass
        return common

    # ── Test evaluation ───────────────────────────────────────────────────────

    def evaluate_test(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Evaluate the best model on the held-out test set using
        the optimal threshold found on validation data.

        Returns a dict of metrics logged by train.py.
        """
        proba = self.best_model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= self.best_threshold).astype(int)

        auprc   = average_precision_score(y_test, proba)
        auc_roc = roc_auc_score(y_test, proba)
        prec    = precision_score(y_test, y_pred, zero_division=0)
        rec     = recall_score(y_test, y_pred, zero_division=0)
        f1      = f1_score(y_test, y_pred, zero_division=0)
        from sklearn.metrics import confusion_matrix
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

        metrics = {
            "model":     self.best_model_name,
            "threshold": self.best_threshold,
            "auprc":     auprc,
            "auc_roc":   auc_roc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "confusion_matrix": matrix,
            "sample_count": int(len(y_test)),
            "fraud_prevalence": float(np.mean(y_test)),
        }

        print("\n-- Test Set Results -------------------------------------")
        print(f"  Model:     {self.best_model_name}")
        print(f"  Threshold: {self.best_threshold:.4f}")
        print(f"  AUPRC:     {auprc:.4f}")
        print(f"  AUC-ROC:   {auc_roc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1:        {f1:.4f}")
        print("-" * 56)

        return metrics

    # ── Save ─────────────────────────────────────────────────────────────────

    def save(self, model_dir: str = "models", feature_names=None) -> None:
        """
        Persist the winning model and its metadata to *model_dir*.

        Saves
        -----
        model.pkl       — the fitted estimator (joblib)
        metadata.json   — model name, threshold, validation AUPRCs
        """
        out = Path(model_dir)
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.best_model, out / "model.pkl")

        metadata = {
            "best_model":      self.best_model_name,
            "best_threshold":  self.best_threshold,
            "val_auprc_all":   self._val_auprc,
            "feature_names":   list(feature_names) if feature_names is not None else None,
            "failed_models":   self.failures,
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved '{self.best_model_name}' -> {out}/model.pkl")
        print(f"  Metadata     -> {out}/metadata.json")
